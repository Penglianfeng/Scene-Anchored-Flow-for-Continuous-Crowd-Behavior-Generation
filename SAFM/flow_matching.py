"""Flow-matching objective and samplers used by SAFE's crowd emitter.

Besides the base conditional flow objective, this module implements SAFE's OD
joint alignment, bidirectional coverage, and scene-anchor energy terms.
"""

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from collections import namedtuple

from einops import rearrange, repeat

from utils.normalization import unnormalize_min_max, unnormalize_sqrt


class LossBuffer:
    def __init__(self, t_min, t_max, num_time_steps):
        self.t_min = t_min
        self.t_max = t_max
        self.num_time_steps = num_time_steps
        self.t_interval = np.linspace(t_min, t_max, num_time_steps)
        self.loss_data = [[] for _ in range(self.num_time_steps)]
        self.last_epoch = -1

    def record_loss(self, t, loss, epoch_id):
        flag_reset = False
        if epoch_id != self.last_epoch:
            self.last_epoch = epoch_id
            self.reset()
            flag_reset = epoch_id > 0

        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()

        idx = np.digitize(t, self.t_interval) - 1
        idx = np.clip(idx, 0, self.num_time_steps - 1)
        for i, l in zip(idx, loss):
            self.loss_data[i].append(l)

        return flag_reset

    def reset(self):
        self.loss_data = [[] for _ in range(self.num_time_steps)]

    def get_average_loss(self):
        avg_loss_per_level = [np.mean(l) if len(l) > 0 else 0.0 for l in self.loss_data]
        return {t: l for t, l in zip(self.t_interval, avg_loss_per_level)}

ModelPrediction = namedtuple('ModelPrediction', ['pred_vel', 'pred_data', 'pred_score'])


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class FlowMatcher(nn.Module):
    def __init__(
        self,
        cfg,
        model,
        logger
    ):
        super().__init__()

        # init
        self.cfg = cfg
        self.model = model
        self.logger = logger

        self.num_agents = self._cfg_get('agents', 1)
        if hasattr(cfg, 'MODEL') and cfg.MODEL is not None:
            self.out_dim = getattr(cfg.MODEL, 'MODEL_OUT_DIM', self._cfg_get('model_out_dim', 7))
        else:
            self.out_dim = self._cfg_get('model_out_dim', 7)
        self.feature_dim = self._cfg_get('feature_dim', 2)
        self.future_frames = self._cfg_get('future_frames', 1)

        self.objective = self._cfg_get('objective', 'pred_data')
        self.sampling_steps = self._cfg_get('sampling_steps', 20)
        self.solver = cfg.get('solver', 'euler')

        assert self.objective in {'pred_vel', 'pred_data'}, 'objective must be either pred_vel or pred_data'
        assert self.cfg.get('LOSS_VELOCITY', False) == False, 'Velocity loss is not supported yet.'

        # special normalization params
        if self.cfg.get('data_norm', None) == 'sqrt':
            self.sqrt_a_ = torch.tensor([self.cfg.sqrt_x_a, self.cfg.sqrt_y_a], device=self.device)
            self.sqrt_b_ = torch.tensor([self.cfg.sqrt_x_b, self.cfg.sqrt_y_b], device=self.device)

        # set up the loss buffer
        self.loss_buffer = LossBuffer(t_min=0, t_max=1.0, num_time_steps=100)

        self._print_flow_config_once()

    @property
    def device(self):
        if hasattr(self.cfg, 'device'):
            return self.cfg.device
        return next(self.model.parameters()).device

    def _cfg_get(self, key, default_value):
        if hasattr(self.cfg, 'get'):
            return self.cfg.get(key, default_value)
        return getattr(self.cfg, key, default_value)

    def _print_flow_config_once(self):
        t_schedule = self._cfg_get('t_schedule', 'uniform')
        items = {
            'K': self._cfg_get('denoising_head_preds', 1),
            'wrapper': self._cfg_get('fm_wrapper', 'velocity'),
            't_schedule': t_schedule,
            'logit_norm': t_schedule == 'logit_normal',
            'logit_norm_mean': self._cfg_get('logit_norm_mean', 0.0),
            'logit_norm_std': self._cfg_get('logit_norm_std', 1.0),
            'tied_noise': self._cfg_get('tied_noise', False),
            'loss_nn_mode': self._cfg_get('LOSS_NN_MODE', 'scene'),
        }
        msg = '[FlowMatcher config] ' + ', '.join(f'{k}={v}' for k, v in items.items())
        print(msg, flush=True)
        if self.logger is not None and hasattr(self.logger, 'info'):
            self.logger.info(msg)

    def _default_pred_score(self, model_out):
        if model_out.dim() == 4:
            return torch.zeros(model_out.shape[:3], device=model_out.device, dtype=model_out.dtype)
        if model_out.dim() == 3:
            return torch.zeros(model_out.shape[:2], device=model_out.device, dtype=model_out.dtype)
        raise ValueError(f'Unsupported output rank: {model_out.dim()}')

    def _model_forward(self, y_t_in, t, x_data):
        mask = x_data.get('agent_mask', x_data.get('output_mask', x_data.get('mask', None))) if isinstance(x_data, dict) else None
        condition_tokens = x_data.get('condition_tokens', None) if isinstance(x_data, dict) else None

        try:
            outputs = self.model(y_t_in, t, condition=condition_tokens, mask=mask, x_data=x_data, return_dict=False)
        except TypeError:
            outputs = self.model(y_t_in, t, x_data=x_data)

        if isinstance(outputs, tuple):
            if len(outputs) == 1:
                model_out = outputs[0]
                pred_score = self._default_pred_score(model_out)
            else:
                model_out, pred_score = outputs[0], outputs[1]
            return model_out, pred_score

        if hasattr(outputs, 'sample'):
            model_out = outputs.sample
            pred_score = getattr(outputs, 'cls_logits', self._default_pred_score(model_out))
            return model_out, pred_score

        model_out = outputs
        pred_score = self._default_pred_score(model_out)
        return model_out, pred_score

    def _od_joint(self, y_raw):
        # z = (tau, origin_x, origin_y, goal_x, goal_y).
        z = y_raw[..., [1, 3, 4, 5, 6]]
        if z.dim() == 4:
            if z.shape[-2] == 1:
                z = z.squeeze(-2)
            else:
                z = z.reshape(z.shape[0], -1, z.shape[-1])
        return z

    def _masked_rbf_mmd2(self, x, y, mask, sigmas=(0.05, 0.10, 0.20, 0.40)):
        # Per-window conditional MMD over OD tokens. x,y: [B,N,D], mask: [B,N].
        mask = mask.float()
        xx = torch.cdist(x, x).pow(2)
        yy = torch.cdist(y, y).pow(2)
        xy = torch.cdist(x, y).pow(2)

        k_xx = sum(torch.exp(-xx / (2 * s * s)) for s in sigmas)
        k_yy = sum(torch.exp(-yy / (2 * s * s)) for s in sigmas)
        k_xy = sum(torch.exp(-xy / (2 * s * s)) for s in sigmas)

        mm = mask[:, :, None] * mask[:, None, :]
        denom = mm.sum(dim=(1, 2)).clamp_min(1.0)
        mmd = ((k_xx * mm).sum((1, 2)) + (k_yy * mm).sum((1, 2)) - 2 * (k_xy * mm).sum((1, 2))) / denom
        return mmd.mean()

    def _od_cover_metric_scale(self, z):
        weights = self._cfg_get("od_cover_feature_weights", None)
        if weights is None:
            return z
        weights = z.new_tensor(weights)
        if weights.numel() != z.shape[-1]:
            raise ValueError(f"od_cover_feature_weights must have {z.shape[-1]} values, got {weights.numel()}")
        return z * weights.clamp_min(0.0).sqrt().view(*([1] * (z.dim() - 1)), -1)

    def _soft_bidirectional_od_distance(self, x, y, mask):
        # Inserted in FlowMatcher before p_losses: differentiable diagnostic-aligned
        # OD coverage loss. x,y: [B,N,D] with z=(tau,o_x,o_y,g_x,g_y).
        # This is the strict LogSumExp softmin form:
        # smin_tau(d_i) = -tau * log mean_j exp(-d_ij / tau).
        # The two directions approximate precision (generated->GT) and recall
        # (GT->generated), and torch.logsumexp keeps the computation stable.
        mask_f = mask.to(device=x.device, dtype=x.dtype)
        mask_b = mask_f > 0
        valid_per_scene = mask_f.sum(dim=1)
        if torch.all(valid_per_scene <= 0):
            return x.new_zeros(())

        tau = max(float(self._cfg_get("od_cover_temperature", 0.05)), 1e-6)
        x = self._od_cover_metric_scale(x)
        y = self._od_cover_metric_scale(y)
        dist = torch.cdist(x, y, p=2)

        large = torch.finfo(dist.dtype).max / 4
        y_valid = mask_b[:, None, :]
        x_valid = mask_b[:, :, None]

        logits_xy = (-dist / tau).masked_fill(~y_valid, -large)
        log_count_y = valid_per_scene.clamp_min(1.0).log()[:, None]
        source_to_target = -tau * (torch.logsumexp(logits_xy, dim=2) - log_count_y)

        logits_yx = (-dist / tau).masked_fill(~x_valid, -large)
        log_count_x = valid_per_scene.clamp_min(1.0).log()[:, None]
        target_to_source = -tau * (torch.logsumexp(logits_yx, dim=1) - log_count_x)

        denom = valid_per_scene.clamp_min(1.0)
        loss_xy = (source_to_target * mask_f).sum(dim=1) / denom
        loss_yx = (target_to_source * mask_f).sum(dim=1) / denom
        scene_valid = (valid_per_scene > 0).to(dtype=x.dtype)
        loss_per_scene = 0.5 * (loss_xy + loss_yx)
        return (loss_per_scene * scene_valid).sum() / scene_valid.sum().clamp_min(1.0)

    def _gt_usage_entropy_loss(self, x, y, mask):
        # GT-anchored diversity. The soft assignment p_ij is from generated OD
        # tokens to GT OD tokens; maximizing entropy of u_j = mean_i p_ij
        # spreads generated samples over real modes without pushing them off GT support.
        mask_f = mask.to(device=x.device, dtype=x.dtype)
        mask_b = mask_f > 0
        valid_per_scene = mask_f.sum(dim=1)
        if torch.all(valid_per_scene <= 1):
            return x.new_zeros(())

        tau = max(float(self._cfg_get("usage_entropy_temperature", self._cfg_get("od_cover_temperature", 0.05))), 1e-6)
        eps = float(self._cfg_get("usage_entropy_eps", 1e-8))
        x = self._od_cover_metric_scale(x)
        y = self._od_cover_metric_scale(y)
        dist = torch.cdist(x, y, p=2)

        large = torch.finfo(dist.dtype).max / 4
        y_valid = mask_b[:, None, :]
        x_valid_f = mask_f[:, :, None]
        logits = (-dist / tau).masked_fill(~y_valid, -large)
        assign = torch.softmax(logits, dim=2) * x_valid_f

        denom = valid_per_scene.clamp_min(1.0)[:, None]
        usage = assign.sum(dim=1) / denom
        usage = usage * mask_f
        usage = usage / usage.sum(dim=1, keepdim=True).clamp_min(eps)

        entropy_objective = (usage.clamp_min(eps) * usage.clamp_min(eps).log() * mask_f).sum(dim=1)
        scene_valid = (valid_per_scene > 1).to(dtype=x.dtype)
        return (entropy_objective * scene_valid).sum() / scene_valid.sum().clamp_min(1.0)

    def _sample_map(self, condition_map, xy01, channel):
        # Differentiable lookup for M(o), M(g). xy01 uses normalized image coordinates.
        b = xy01.shape[0]
        token_shape = xy01.shape[1:-1]
        grid = xy01.clamp(0.0, 1.0).mul(2.0).sub(1.0).reshape(b, -1, 1, 2)
        val = F.grid_sample(
            condition_map[:, channel:channel + 1],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return val.squeeze(1).squeeze(-1).reshape(b, *token_shape)

    def _expand_anchor_mask(self, mask, target):
        mask = mask.float()
        if target.dim() == 4:
            return mask[:, None, :, None].expand_as(target)
        if target.dim() == 3:
            if target.shape[1] == mask.shape[1]:
                return mask[:, :, None].expand_as(target)
            if target.shape[2] == mask.shape[1]:
                return mask[:, None, :].expand_as(target)
        if target.dim() == 2:
            return mask.expand_as(target)
        raise ValueError(f"Unsupported target shape {tuple(target.shape)} for agent mask {tuple(mask.shape)}")

    def _anchor_energy(self, y_raw, condition_map, mask):
        # E_C(o,g) = -log pi_C(o,g), using appearance/population condition channels.
        app_ch = int(self._cfg_get("appearance_channel", -1))
        pop_ch = int(self._cfg_get("population_channel", -1))
        if condition_map is None or app_ch < 0 or pop_ch < 0:
            return y_raw.new_zeros(())
        if condition_map.dim() != 4:
            return y_raw.new_zeros(())
        if app_ch >= condition_map.shape[1] or pop_ch >= condition_map.shape[1]:
            return y_raw.new_zeros(())

        condition_map = condition_map.to(device=y_raw.device, dtype=y_raw.dtype)
        origin = y_raw[..., [3, 4]]
        goal = y_raw[..., [5, 6]]

        ao = self._sample_map(condition_map, origin, app_ch)
        ag = self._sample_map(condition_map, goal, app_ch)
        po = self._sample_map(condition_map, origin, pop_ch)
        pg = self._sample_map(condition_map, goal, pop_ch)

        eps = float(self._cfg_get("anchor_eps", 1e-4))
        appearance_weight = float(self._cfg_get("anchor_appearance_weight", 1.0))
        rho = float(self._cfg_get("anchor_population_power", 0.5))
        energy = (
            -appearance_weight * (ao + eps).log()
            -appearance_weight * (ag + eps).log()
            -rho * (po + eps).log()
            -rho * (pg + eps).log()
        )
        mask_weight = self._expand_anchor_mask(mask, energy)
        return (energy * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)
    
    def get_precond_coef(self, t):
        """
        Get preconditioned wrapper coefficients.
        D_theta = alpha_t * x_t + beta_t * F_theta
        @param t: [B]
        """
        sigma_data = self._cfg_get('sigma_data', 1.0)
        coef_1 = t.pow(2) * sigma_data ** 2 + (1-t).pow(2)
        alpha_t = t * sigma_data ** 2 / coef_1
        beta_t = (1 - t) * sigma_data / coef_1.sqrt()

        return alpha_t, beta_t
    
    def get_input_scaling(self, t):
        """
        Get the input scaling factor.
        """
        sigma_data = self._cfg_get('sigma_data', 1.0)
        var_x_t = sigma_data ** 2 * t.pow(2) + (1 - t).pow(2)
        return 1.0 / var_x_t.sqrt().clip(min=1e-4, max=1e4)

    def fm_wrapper_func(self, x_t, t, model_out):  # D_θ,数据预测器
        """
        denoised_y = self.fm_wrapper_func(y_t, t, model_out)

        Build wrapper for network regression output. We don't modify the classification logits.
        We aim to let the wrapper to match the data prediction (x_1 in the flow model).
        @param x_t: 		[B, K, A, F * D]
        @param t: 			[B]
        @param model_out: 	[B, K, A, F * D]
        """
        wrapper = self._cfg_get('fm_wrapper', 'velocity')
        if wrapper == 'direct':
            return model_out
        elif wrapper == 'velocity':
            t = pad_t_like_x(t, x_t)
            return x_t + (1 - t) * model_out # D_θ = Y^t  + (1 − t) v_θ (Y^t, C, t)
        elif wrapper == 'precond':
            t = pad_t_like_x(t, x_t)
            alpha_t, beta_t = self.get_precond_coef(t)
            return alpha_t * x_t + beta_t * model_out
        raise ValueError(f'Unknown fm_wrapper: {wrapper}')


    def predict_vel_from_data(self, x1, xt, t):
        """
        (y_data_at_t, y_t, t)
        Predict the velocity field from the predicted data.
        """
        t = pad_t_like_x(t, x1)
        den = (1 - t).clamp(min=1e-4)
        v = (x1 - xt) / den
        return v

    def predict_data_from_vel(self, v, xt, t):
        """
        Predict the data from the predicted velocity field.
        """
        t = pad_t_like_x(t, xt)
        x1 = xt + v * (1 - t)
        return x1

    def _expand_agent_mask(self, mask, x):
        if mask is None:
            return None

        mask = mask.to(device=x.device)
        if mask.dim() == 2:
            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[2]:
                raise ValueError(f"mask must be [B,A], got {tuple(mask.shape)} for x={tuple(x.shape)}")
            mask = mask[:, None, :, None]
        elif mask.dim() == 3:
            if mask.shape[0] != x.shape[0] or mask.shape[2] != x.shape[2]:
                raise ValueError(f"mask must be [B,K,A], got {tuple(mask.shape)} for x={tuple(x.shape)}")
            mask = mask[:, :, :, None]
        elif mask.dim() == 4:
            if mask.shape[0] != x.shape[0] or mask.shape[2] != x.shape[2]:
                raise ValueError(f"mask must be broadcastable to [B,K,A,1], got {tuple(mask.shape)} for x={tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported mask rank: {mask.dim()}")

        return mask.to(dtype=x.dtype)

    def compute_masked_dot_product(self, a, b, mask=None):
        """
        Compute a mode-level scene dot product for [B,K,A,7] trajectory-emitter tensors.

        Only physical spatial fields [..., 3:7] participate in the coefficient
        estimate. The returned shape is [B,K,1,1], so it can broadcast back to
        the full 7D agent parameter tensor.
        """
        if a.shape != b.shape:
            raise ValueError(f"dot-product shape mismatch: a={tuple(a.shape)}, b={tuple(b.shape)}")
        if a.dim() != 4:
            raise ValueError(f"Expected [B,K,A,F], got {tuple(a.shape)}")
        if a.shape[-1] < 7:
            raise ValueError(f"Expected at least 7 feature dims, got {a.shape[-1]}")

        a_spatial = a[..., 3:7]
        b_spatial = b[..., 3:7]
        dot = a_spatial * b_spatial

        agent_mask = self._expand_agent_mask(mask, a)
        if agent_mask is not None:
            dot = dot * agent_mask

        dot = dot.sum(dim=(-1, -2))
        return dot.unsqueeze(-1).unsqueeze(-1)

    def projection_coefficient(self, tensor_a, tensor_b, mask=None, tensor_b_dot_product=None, eps=1e-8):
        """
        Project tensor_a onto tensor_b using only origin/goal xy dimensions.
        """
        cross_corr = self.compute_masked_dot_product(tensor_a, tensor_b, mask=mask)
        if tensor_b_dot_product is None:
            var_b = self.compute_masked_dot_product(tensor_b, tensor_b, mask=mask)
        else:
            var_b = tensor_b_dot_product

        coeff = cross_corr / var_b.clamp_min(eps)
        coeff = torch.nan_to_num(coeff, nan=0.0, posinf=2.0, neginf=-2.0)
        return torch.clamp(coeff, min=-2.0, max=2.0)

    def _tensor_stats(self, x):
        x = x.detach().float().reshape(-1)
        if x.numel() == 0:
            return {}
        qs = torch.quantile(x, torch.tensor([0.5, 0.9, 0.95, 0.99], device=x.device))
        return {
            'mean': float(x.mean().cpu()),
            'std': float(x.std(unbiased=False).cpu()),
            'min': float(x.min().cpu()),
            'max': float(x.max().cpu()),
            'p50': float(qs[0].cpu()),
            'p90': float(qs[1].cpu()),
            'p95': float(qs[2].cpu()),
            'p99': float(qs[3].cpu()),
        }

    def fwd_sample_t(self, x0, x1, t):
        """
        (x0=noise, x1=y_start_k, t=t) x0: [B, K, A, F * D]
        Sample the latent space at time t.
        """
        t = pad_t_like_x(t, x0)
        # t:[B]->t:[B, 1, 1, 1]
        xt = t * x1 + (1 - t) * x0      # simple linear interpolation
        ut = x1 - x0                    # xt derivative w.r.t. t
        return xt, ut

    def get_reweighting(self, t, wrapper=None):
        wrapper = default(wrapper, self._cfg_get('fm_wrapper', 'velocity'))
        if wrapper == 'direct':
            l_weight = torch.ones_like(t) # 传入一个数组作为参数，返回一个与该数组形状相同且元素全为1的张量
        elif wrapper == 'velocity': 
            l_weight = 1.0 / (1 - t) ** 2 
        elif wrapper == 'precond':
            alpha_t, beta_t = self.get_precond_coef(t)
            l_weight = 1.0 / beta_t ** 2
        if self._cfg_get('fm_rew_sqrt', False):
            l_weight = l_weight.sqrt()
        l_weight = l_weight.clamp(min=1e-4, max=1e4)
        return l_weight
    
    def get_loss_input(self, y_start_k):
        """
        y_start_k: [B, K, A, F * D]
        Prepare the input for the flow matching model training.

        """

        # random time steps to inject noise
        bs = y_start_k.shape[0]
        t_schedule = self._cfg_get('t_schedule', 'uniform')
        if t_schedule == 'uniform':
            t = torch.rand((bs, ), device=self.device)  # 生成一个形状为 (bs,) 的张量，元素在 [0, 1) 范围内均匀分布
        elif t_schedule == 'logit_normal':
            # note: this is logit-normal (not log-normal)
            mean_ = self._cfg_get('logit_norm_mean', 0.0)
            std_ = self._cfg_get('logit_norm_std', 1.0)
            t_normal_ = torch.randn((bs, ), device=self.device) * std_ + mean_
            t = torch.sigmoid(t_normal_)
        else:
            if '==' in t_schedule:
                # constant_t
                t = float(t_schedule.split('==')[1]) * torch.ones((bs, ), device=self.device)
            else:
                # custom two-stage uniform distribution
                # e.g., 't0.5_p0.3' means with 30% probability, sample from [0, 0.5] uniformly, and with 70% probability, sample from [0.5, 1] uniformly
                cutoff_t = float(t_schedule.split('_')[0][1:])
                prob_1 = float(t_schedule.split('_')[1][1:])

                t_1 = torch.rand((bs, ), device=self.device) * cutoff_t
                t_2 = cutoff_t + torch.rand((bs, ), device=self.device) * (1 - cutoff_t)
                rand_num = torch.rand((bs, ), device=self.device)

                t = t_1 * (rand_num < prob_1) + t_2 * (rand_num >= prob_1)



        assert t.min() >= 0 and t.max() <= 1


        #噪声采样
        # noise sample
        if self._cfg_get('tied_noise', False):
            # 所有K个预测头共用同一个初始噪声
            noise = torch.randn_like(y_start_k[:, 0:1])                                  # [B, 1, T, D]
            # [,0:1]第二个维度做切片取索引0到1
            noise = noise.expand(-1, self._cfg_get('denoising_head_preds', 1), -1, -1)              # [B, K, T, D]
        else:
            # 否则，为每个样本和每个头采样独立的噪声
            noise = torch.randn_like(y_start_k)                                          # [B, K, T, D]
            '''
            randn_like 函数用于生成与给定张量形状相同的张量,且元素服从标准正态分布(均值为0,标准差为1)
            在flow matching模型中,这里是为了生成与y_start_k形状相同的标准正态噪声
            生成的随机数会围绕 0 对称分布。
            大约 68% 的数值会落在 [-1, 1] 的区间内。
            大约 95% 的数值会落在 [-2, 2] 的区间内。
            '''

        # sample the latent space at time t
        x_t, u_t = self.fwd_sample_t(x0=noise, x1=y_start_k, t=t)                        # [B, K, T, D] * 2
        # return Y^t, U^t

        #确定训练目标
        # target:[B,K,A,F*D]
        if self.objective == 'pred_data':  # 目标是预测最终真值Y^1(即D_θ方法)
            target = y_start_k
        elif self.objective == 'pred_vel': # 目标是预测速度U^t
            target = u_t
        else:
            raise ValueError(f'unknown objective {self.objective}')

        l_weight = self.get_reweighting(t)

        return t, x_t, u_t, target, l_weight

    def model_predictions(self, y_t, x, t, flag_print):
        # (y_t, x_data, batched_tp[B,t], flag_print)
        if self._cfg_get('fm_in_scaling', False):
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t) # 缩放输入
        else:
            y_t_in = y_t # 不缩放输入

        model_out, pred_score = self._model_forward(y_t_in, t, x) # pred_score->logits
        # S_i(D_θ的输出)
        y_data_at_t = self.fm_wrapper_func(y_t, t, model_out)            # [B, K, A, F * D]

        if self.objective == 'pred_vel':  # 模型预测常速度场
            raise NotImplementedError

        elif self.objective == 'pred_data': # 模型直接预测最终的真实数据
            # ground truth data
            gt_y_data = None
            if isinstance(x, dict) and 'fut_traj' in x:
                gt_y_data = rearrange(x['fut_traj'], 'b a f d -> b 1 a (f d)')

            this_t = round(t.unique().item(), 4)

            if flag_print and gt_y_data is not None:
                y_data_ = rearrange(y_data_at_t, 'b k a (f d) -> (b a) k f d', f=self.future_frames)
                gt_y_data = rearrange(gt_y_data, 'b k a (f d) -> (b a) k f d', f=self.future_frames) # ready the ground truth data for metric calculation
                # 将 Batch 和 Agent 维度合并 (b a)，因为评估指标通常是针对每个智能体的
                data_norm = self._cfg_get('data_norm', 'original')
                if data_norm == 'min_max':
                    # 利用数据集的最大值 (fut_traj_max) 和最小值 (fut_traj_min)，将预测值从 [-1, 1] 映射回真实的物理单位
                    y_data_metric = unnormalize_min_max(y_data_, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                    gt_y_data_metric = unnormalize_min_max(gt_y_data, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
                elif data_norm == 'sqrt':
                    
                    y_data_metric = unnormalize_sqrt(y_data_, self.sqrt_a_, self.sqrt_b_)
                    gt_y_data_metric = unnormalize_sqrt(gt_y_data, self.sqrt_a_, self.sqrt_b_)
                elif data_norm == 'original':
                    y_data_metric = y_data_
                    gt_y_data_metric = gt_y_data
                else:
                    y_data_metric = y_data_
                    gt_y_data_metric = gt_y_data

                error_metric = (y_data_metric - gt_y_data_metric).abs()  # [B * A, K, F, D]
                batch_min_ade_approx = error_metric.norm(dim=-1, p=2).mean(dim=-1).min(dim=-1).values.mean()
                '''
                .norm(dim=-1, p=2):在最后一个维度(dim=-1,即 D 维度)上计算 L2 范数(欧几里得距离),现在我们有了每一帧的欧式距离
                [B * A, K, F, D] -> [B * A, K, F]
                .mean(dim=-1):在新的最后一个维度(即 F 维度,时间帧)上求平均值.计算ADE将未来12帧的误差平均,得到每条预测轨迹的平均误差
                [B * A, K, F] -> [B * A, K]
                min(dim=-1):在最后一个维度(即 K 维度,模态)上找最小值.
                [B * A, K] -> [B * A]
                .values:从 .min() 返回的元组中提取数值部分
                .mean():对所有元素求平均
                [B * A] -> Scalar(标量，一个数)
                '''
                if this_t == 0.0:
                    if self.logger is not None:
                        self.logger.info("{}".format("-" * 50))
                # self.logger.info("Sampling time step: {:.3f}, batch minADE approx: {:.4f}".format(this_t, batch_min_ade_approx))
                if self.logger is not None:
                    self.logger.info("Sampling time step: {:.3f}".format(this_t))

            # 从预测的终点y_data_at_t(S_i)和当前位置y_t，反推出当前需要的"方向"v_θ
            pred_vel = self.predict_vel_from_data(y_data_at_t, y_t, t) #v_θ^((i) )=(S_i−Y_i^t)/(1−t)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_vel, y_data_at_t, pred_score) #将v_θ 、S_i 、logits打包返回

    @torch.inference_mode()

    def bwd_sample_t(self, y_t: torch.tensor, t: int, dt: float, x_data: dict, flag_print: bool=False):
        # bwd_sample_t(y_t, cur_t, cur_dt, x_data, flag_print)
        B, K, T, D = y_t.shape

        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        model_preds = self.model_predictions(y_t, x_data, batched_t, flag_print)

        y_next = y_t + model_preds.pred_vel * dt # Y^(t+1)  = Y^t  + v_θ (Y^t,C,t)
        return y_next, model_preds.pred_data, model_preds

    @torch.no_grad()
    # 测试/生成时入口
    def sample(self, x_data, num_trajs, return_all_states=False):
        """
        Sample from the model.
        """
        # start with y_T ~ N(0,I), reversed MC to conditionally denoise the traj
        denoising_head_preds = self._cfg_get('denoising_head_preds', 1)
        assert num_trajs == denoising_head_preds, 'num_trajs must be equal to denoising_head_preds = {}'.format(denoising_head_preds)
        y_data = None

        batch_size = x_data['batch_size'] 
        num_agents = int(x_data.get('num_agents', self.num_agents))
        y_t = torch.randn((batch_size, num_trajs, num_agents, self.out_dim), device=self.device) # 一开始全是噪声 [B,K,A,F*D]
        # 所有K个预测头共用同一个初始噪声
        if self._cfg_get('tied_noise', False):
            y_t = y_t[:, :1].expand(-1, denoising_head_preds, -1, -1) # [B,1,A,F*D] -> [B,K,A,F*D]，将切出来的第0个头的噪声扩展成K个头

        # sampling loop
        y_data_at_t_ls = [] # 每一步t都会输出一个y_data(S_i),走几步，这个列表就从几个张量
        t_ls = []
        y_t_ls = []
        store_intermediate = bool(x_data.get('store_intermediate', False))

        if self.solver == 'euler':
            dt = 1.0 / self.sampling_steps # Δt= 1/N，sampling_steps=10
            t_ls = (dt * np.arange(self.sampling_steps)).tolist()  # 时间点列表，[0, Δt, 2Δt, ..., (N-1)Δt]，告诉去噪过程进行到哪一步了
            dt_ls = (dt * np.ones(self.sampling_steps)).tolist()   # 步长列表，[Δt, Δt, ..., Δt]，用于求解器中，计算下一步的状态更新量

        elif self.solver == 'evodiff':
            evodiff_time_schedule = self._cfg_get('evodiff_time_schedule', 'time_uniform')
            if evodiff_time_schedule in {'time_uniform', 'uniform'}:
                t_points = np.linspace(0.0, 1.0, self.sampling_steps + 1)
            elif evodiff_time_schedule in {'time_quadratic', 'quadratic'}:
                t_points = np.linspace(0.0, 1.0, self.sampling_steps + 1) ** 2
            else:
                raise NotImplementedError(f"Unknown evodiff_time_schedule: {evodiff_time_schedule}")
            t_ls = t_points[:-1].tolist()
            dt_ls = np.diff(t_points).tolist()

        elif self.solver == 'lin_poly':
            # linear time growth in the first half with small dt
            # polinomial growth of dt in the second half
            lin_poly_long_step = self.cfg.lin_poly_long_step
            lin_poly_p = self.cfg.lin_poly_p

            n_steps_lin = self.sampling_steps // 2 # 
            n_steps_poly = self.sampling_steps - n_steps_lin

            dt_lin = 1.0 / lin_poly_long_step
            t_lin_ls = dt_lin * np.arange(n_steps_lin)

            def _polynomially_spaced_points(a, b, N, p=2):
                # Generate N points in the interval [a, b] with spacing determined by the power p.
                points = [a + (b - a) * ((i - 1) ** p) / ((N - 1) ** p) for i in range(1, N + 1)]
                return points

            t_poly_start = t_lin_ls[-1] + dt_lin
            t_poly_end = 1.0
            t_poly_ls_ = _polynomially_spaced_points(t_poly_start, t_poly_end, n_steps_poly + 1, p=lin_poly_p)
            dt_poly = np.diff(t_poly_ls_)

            dt_ls = np.concatenate([dt_lin * np.ones(n_steps_lin), dt_poly]).tolist()
            t_ls = np.concatenate([t_lin_ls, t_poly_ls_[:-1]]).tolist()

        else:
            raise NotImplementedError(f"Unknown solver: {self.solver}")

        # define the time steps to print
        num_prints = 10
        if len(t_ls) > num_prints:
            print_stride = max(1, self.sampling_steps // num_prints)
            print_times = t_ls[::print_stride]
            if t_ls[-1] not in print_times:
                print_times.append(t_ls[-1])
        else:
            print_times = t_ls # 10

        agent_mask = None
        if isinstance(x_data, dict):
            agent_mask = x_data.get('agent_mask', x_data.get('output_mask', x_data.get('mask', None)))

        m0_dot_m0 = None
        m1_dot_m1 = None
        m0_dot_m1 = None
        model_prev_list = []
        time_prev_list = []
        evodiff_diagnostics = x_data.get('evodiff_diagnostics', None) if isinstance(x_data, dict) else None

        for idx_step, (cur_t, cur_dt) in enumerate(zip(t_ls, dt_ls)):
            # idx_step 拿走了最外层的索引（比如 0）。(cur_t, cur_dt) 拿走了里面的那个元组（比如 (0.0, 0.1)）
            flag_print = cur_t in print_times
            # 随着 t 增加，y_t 越来越像真实数据，模型的猜测 y_data 也会变得越来越精准和稳定
            # 执行单步更新
            # 输入：当前状态 y_t，当前时间 cur_t，步长 cur_dt
            # 输出：
            #   y_t: 更新后的下一步状态 (y_{t+dt})
            #   y_data: 模型当前预测的最终终点 (y_1 的估计值)
            #   model_preds: 包含速度 v 和分类分数 logits 的对象            

            if self.solver == 'evodiff':
                batched_t = torch.full((batch_size,), cur_t, device=self.device, dtype=torch.float)
                model_preds = self.model_predictions(y_t, x_data, batched_t, flag_print)
                y_data = model_preds.pred_data

                next_t = min(float(cur_t + cur_dt), 1.0)
                sigma_cur = torch.tensor(1.0 - float(cur_t), device=y_t.device, dtype=y_t.dtype).clamp_min(1e-4)
                sigma_next = torch.tensor(1.0 - next_t, device=y_t.device, dtype=y_t.dtype).clamp_min(0.0)
                sigma_ratio = sigma_next / sigma_cur
                x_euler = sigma_ratio * y_t + (float(cur_dt) / sigma_cur) * y_data

                use_euler = len(model_prev_list) < 2 or idx_step == len(t_ls) - 1 or sigma_cur.item() <= 1e-4
                if use_euler:
                    y_t = x_euler
                else:
                    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
                    t_prev_1, t_prev_0 = time_prev_list[-2], time_prev_list[-1]

                    sigma_prev_1 = torch.tensor(1.0 - float(t_prev_1), device=y_t.device, dtype=y_t.dtype).clamp_min(1e-4)
                    sigma_prev_0 = torch.tensor(1.0 - float(t_prev_0), device=y_t.device, dtype=y_t.dtype).clamp_min(1e-4)
                    sigma_cur_over_prev_0 = sigma_cur / sigma_prev_0
                    sigma_prev_0_over_prev_1 = sigma_prev_0 / sigma_prev_1
                    balance_base = torch.sqrt((sigma_cur_over_prev_0 / sigma_prev_0_over_prev_1).clamp_min(1e-4))

                    if m0_dot_m0 is None:
                        m0_dot_m0 = self.compute_masked_dot_product(model_prev_0, model_prev_0, mask=agent_mask)
                        m1_dot_m1 = self.compute_masked_dot_product(model_prev_1, model_prev_1, mask=agent_mask)
                        m0_dot_m1 = self.compute_masked_dot_product(model_prev_0, model_prev_1, mask=agent_mask)

                    t_normalized = torch.tensor(cur_t, device=y_t.device, dtype=y_t.dtype)
                    weight_t = 0.5 * (1.0 - t_normalized.pow(2))

                    r_01_pc = torch.clamp(m0_dot_m1 / m1_dot_m1.clamp_min(1e-8), min=-2.0, max=2.0)
                    r1_balance = (1.0 - weight_t) * balance_base + weight_t * r_01_pc
                    D1_0 = model_prev_0 - r1_balance * model_prev_1

                    mt_dot_mt = self.compute_masked_dot_product(y_data, y_data, mask=agent_mask)
                    mt_dot_m0 = self.compute_masked_dot_product(y_data, model_prev_0, mask=agent_mask)
                    r_t0_pc = torch.clamp(mt_dot_m0 / m0_dot_m0.clamp_min(1e-8), min=-2.0, max=2.0)
                    r2_balance = (1.0 - weight_t) * balance_base + weight_t * r_t0_pc
                    D2_0 = y_data - r2_balance * model_prev_0

                    dt_prev_0 = max(float(cur_t - t_prev_0), 1e-4)
                    dt_prev_1 = max(float(t_prev_0 - t_prev_1), 1e-4)
                    r_base = min(max(dt_prev_0 / dt_prev_1, 0.25), 1.5)
                    ri_temperature = float(self._cfg_get('evodiff_ri_temperature', 0.0))
                    if ri_temperature > 0.0:
                        ri_scale = torch.sigmoid(ri_temperature * r1_balance.abs())
                    else:
                        ri_scale = torch.ones_like(r1_balance)
                    r_i = torch.clamp(ri_scale * r_base, min=0.25, max=1.5)

                    B_pre_i_i = D1_0 / dt_prev_1
                    B_next_i_i = D2_0 / dt_prev_0
                    eta_star = 0.5 * self.projection_coefficient(
                        B_next_i_i + B_pre_i_i,
                        B_next_i_i - B_pre_i_i,
                        mask=agent_mask,
                    )
                    eta = 0.5 * torch.sigmoid(torch.abs(eta_star))
                    eta_1, eta_2 = -eta, 1.0 - eta
                    B_theta = eta_1 / r_i * D1_0 + r_i * eta_2 * D2_0

                    zeta_star = self.projection_coefficient(D2_0, D1_0, mask=agent_mask)
                    shift_mu = float(self._cfg_get('evodiff_shift_mu', 0.5))
                    zeta = torch.sigmoid(-(torch.abs(zeta_star) - shift_mu))
                    zeta = torch.clamp(
                        zeta,
                        min=float(self._cfg_get('evodiff_zeta_min', 0.1)),
                        max=float(self._cfg_get('evodiff_zeta_max', 1.0)),
                    )

                    correction = 0.5 * float(cur_dt) * (1.0 / zeta) * B_theta
                    correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
                    correction_scale = float(self._cfg_get('evodiff_correction_scale', 1.0))
                    correction = correction * correction_scale

                    correction_scope = self._cfg_get('evodiff_correction_scope', 'spatial')
                    if correction_scope == 'spatial':
                        correction_full = torch.zeros_like(correction)
                        correction_full[..., 3:7] = correction[..., 3:7]
                    elif correction_scope == 'all':
                        correction_full = correction
                    else:
                        raise ValueError(f"Unknown evodiff_correction_scope: {correction_scope}")

                    if evodiff_diagnostics is not None:
                        correction_spatial = correction_full[..., 3:7]
                        evodiff_diagnostics.append({
                            'step': int(idx_step),
                            't': float(cur_t),
                            'dt': float(cur_dt),
                            'scope': correction_scope,
                            'correction_scale': correction_scale,
                            'correction_spatial': self._tensor_stats(correction_spatial),
                            'correction_spatial_abs': self._tensor_stats(correction_spatial.abs()),
                            'zeta': self._tensor_stats(zeta),
                            'eta': self._tensor_stats(eta),
                            'ri_temperature': ri_temperature,
                            'ri_scale': self._tensor_stats(ri_scale),
                            'r_i': self._tensor_stats(r_i),
                            'B_theta_spatial': self._tensor_stats(B_theta[..., 3:7]),
                            'B_theta_spatial_abs': self._tensor_stats(B_theta[..., 3:7].abs()),
                            'pred_data_spatial': self._tensor_stats(y_data[..., 3:7]),
                            'pred_vel_spatial': self._tensor_stats(model_preds.pred_vel[..., 3:7]),
                        })
                    y_t = x_euler + correction_full

                    m1_dot_m1 = m0_dot_m0
                    m0_dot_m0 = mt_dot_mt
                    m0_dot_m1 = mt_dot_m0

                if agent_mask is not None:
                    expanded_mask = self._expand_agent_mask(agent_mask, y_t).bool()
                    y_t = y_t.masked_fill(~expanded_mask, 0.0)

                model_prev_list.append(y_data.detach())
                time_prev_list.append(float(cur_t))
                if len(model_prev_list) > 2:
                    model_prev_list = model_prev_list[-2:]
                    time_prev_list = time_prev_list[-2:]
            else:
                y_t, y_data, model_preds = self.bwd_sample_t(y_t, cur_t, cur_dt, x_data, flag_print)
                # return y_next, model_preds.pred_data, model_preds
                # model_preds-> return ModelPrediction(pred_vel, y_data_at_t, pred_score)

            if store_intermediate:
                y_data_at_t_ls.append(y_data)
            if return_all_states:
                y_t_ls.append(y_t)

        if store_intermediate:
            y_data_at_t_ls = torch.stack(y_data_at_t_ls, dim=1)     # [B, S, K, A, F * D]
        else:
            y_data_at_t_ls = y_data.unsqueeze(1)
        t_ls = torch.tensor(t_ls, device=self.device)   # [S]
        if return_all_states:
            y_t_ls = torch.stack(y_t_ls, dim=1)  # [B, S, K, A, F * D]

        return y_t, y_data_at_t_ls, t_ls, y_t_ls, model_preds.pred_score

    # 训练时入口
    def p_losses(self, x_data, log_dict=None):
        """Compute the base flow loss and enabled SAFE scene constraints."""

        # init
        B, A = x_data['fut_traj'].shape[:2] # batch size, num agents
        K = self._cfg_get('denoising_head_preds', 1)   # num predicted trajectories
        T = self.future_frames          # 未来轨迹的时间步长（frames）
        assert self.objective == 'pred_data', 'only pred_data is supported for now' # 检查是否直接预测数据的目标
        

        # forward process to create noisy samples
        fut_traj_normalized = repeat(x_data['fut_traj'], 'b a f d -> b k a (f d)', k=K) # 把真值轨迹复制K次，作为多个预测头的目标 [B, K, A, F * D]
        t, y_t, u_t, _, l_weight = self.get_loss_input(y_start_k = fut_traj_normalized) 
        # return t, x_t, u_t, target, l_weight

        
        # model pass
        if self._cfg_get('fm_in_scaling', False):
            y_t_in = y_t * pad_t_like_x(self.get_input_scaling(t), y_t) # 将输入按时间步缩放
        else:
            y_t_in = y_t # 直接使用原始输入

        if self.training and self._cfg_get('drop_method', None) == 'input':
            assert self.cfg.get('drop_logi_k', None) is not None and self.cfg.get('drop_logi_m', None) is not None
            m, k = self.cfg.drop_logi_m, self.cfg.drop_logi_k 
            p_m = 1 / (1 + torch.exp(-k * (t - m)))
            p_m = p_m[:, None, None, None]
            y_t_in = y_t_in.masked_fill(torch.rand_like(p_m) < p_m, 0.)

        model_out, denoiser_cls = self._model_forward(y_t_in, t, x_data)  # [B, K, A, T * D] + [B, K, A]
        denoised_y = self.fm_wrapper_func(y_t, t, model_out)

        # component selection
        denoised_y = rearrange(denoised_y, 'b k a (f d) -> b k a f d', f = self.future_frames, d=self.feature_dim)
        fut_traj_normalized = fut_traj_normalized.view(B, K, A, T, self.feature_dim)
        # 反归一化
        data_norm = self._cfg_get('data_norm', 'original')
        if data_norm == 'min_max':
            # 利用记录下来的最大值 (fut_traj_max) 和最小值 (fut_traj_min) 进行还原
            denoised_y_metric = unnormalize_min_max(denoised_y, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 		 # [B, K, A, T, D]
            fut_traj_metric = unnormalize_min_max(fut_traj_normalized, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)  # [B, K, A, T, D]
        elif data_norm == 'sqrt':
            denoised_y_metric = unnormalize_sqrt(denoised_y, self.sqrt_a_, self.sqrt_b_)            # [B, K, A, T, D]
            fut_traj_metric = unnormalize_sqrt(fut_traj_normalized, self.sqrt_a_, self.sqrt_b_)     # [B, K, A, T, D]
        elif data_norm == 'original':
            denoised_y_metric = denoised_y
            fut_traj_metric = fut_traj_normalized
        else:
            raise ValueError(f"Unknown data normalization method: {data_norm}")

        if self.cfg.get('LOSS_VELOCITY', False):
            raise NotImplementedError
            denoised_y_metric = rearrange(denoised_y_metric, 'b k a (f d) -> b k a f d', f=self.future_frames, d=4)
            denoised_y_metric_xy, denoised_y_metric_v = denoised_y_metric[..., :2], denoised_y_metric[..., 2:4]

            gt_traj_vel = x_data['fut_traj_vel'][:, None].expand(-1, K, -1, -1, -1)  # [B, K, A, T, 2]
            loss_reg_vel = F.l1_loss(denoised_y_metric_v, gt_traj_vel, reduction='none').mean()
        else:
            denoised_y_metric_xy = denoised_y_metric
            loss_reg_vel = torch.zeros(1).to(self.device)

        # 计算误差
        denoising_error_per_agent = (denoised_y_metric_xy - fut_traj_metric).view(B, K, A, T, self.feature_dim).norm(dim=-1)  	 # [B, K, A, T]

        if self.cfg.get('LOSS_REG_SQUARED', False):
            denoising_error_per_agent = denoising_error_per_agent ** 2

        denoising_error_per_scene = denoising_error_per_agent.mean(dim=-2)  								 	 # [B, K, T]

        if self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'mean':
            denoising_error_per_scene = denoising_error_per_scene.mean(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.mean(dim=-1)
        elif self.cfg.get('LOSS_REG_REDUCTION', 'mean') == 'sum':
            denoising_error_per_scene = denoising_error_per_scene.sum(dim=-1)
            denoising_error_per_agent = denoising_error_per_agent.sum(dim=-1)
        else:
            raise ValueError(f"Unknown reduction method: {self.cfg.get('LOSS_REG_REDUCTION', 'mean')}")

        loss_nn_mode = self._cfg_get('LOSS_NN_MODE', 'scene')
        if loss_nn_mode == 'scene':
            # scene-level selection
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            loss_reg_b = denoising_error_per_scene.gather(1, selected_components[:, None]).squeeze(1)  		# [B]

            cls_logits = denoiser_cls.mean(dim=-1)  # [B, K]
            loss_cls_b = F.cross_entropy(input=cls_logits, target=selected_components, reduction='none')	# [B]
        elif loss_nn_mode == 'agent':
            # agent-level selection
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A]
            loss_reg_b = loss_reg_b.mean(dim=-1)  # [B]

            cls_logits = rearrange(denoiser_cls, 'b k a -> (b a) k')	# [B * A, K]
            cls_labels = selected_components.view(-1)					# [B * A]
            loss_cls_b = F.cross_entropy(input=cls_logits, target=cls_labels, reduction='none')	 # [B * A]
            loss_cls_b = loss_cls_b.view(B, A).mean(dim=-1)  	# [B]
        elif loss_nn_mode == 'both':
            # scene-level selection
            selected_components = denoising_error_per_scene.argmin(dim=1)  # [B]
            loss_reg_b_scene = denoising_error_per_scene.gather(1, selected_components[:, None]).squeeze(1)  		# [B] 

            # agent-level selection
            selected_components = denoising_error_per_agent.argmin(dim=1)  # [B, A]
            loss_reg_b = denoising_error_per_agent.gather(1, selected_components[:, None, :]).squeeze(1)  	# [B, A]
            loss_reg_b_agent = loss_reg_b.mean(dim=-1)  # [B]
            optimization_cfg = self._cfg_get('OPTIMIZATION', {})
            loss_weights = optimization_cfg.get('LOSS_WEIGHTS', {}) if hasattr(optimization_cfg, 'get') else {}
            loss_reg_b = loss_weights.get('omega', 1.0) * loss_reg_b_scene + loss_reg_b_agent

            ## dummy input for loss_cls_b
            loss_cls_b = torch.zeros_like(loss_reg_b)


        # loss computation
        loss_reg = (loss_reg_b * l_weight).mean()  # scalar

        loss_cls = loss_cls_b.mean()

        optimization_cfg = self._cfg_get('OPTIMIZATION', {})
        loss_weights = optimization_cfg.get('LOSS_WEIGHTS', {}) if hasattr(optimization_cfg, 'get') else {}
        weight_reg = loss_weights.get('reg', 1.0)
        weight_cls = loss_weights.get('cls', 1.0)
        weight_vel = loss_weights.get('vel', 0.2)

        loss = weight_reg * loss_reg.mean() + weight_cls * loss_cls.mean() + weight_vel * loss_reg_vel.mean()

        loss_joint = torch.zeros((), device=self.device, dtype=loss.dtype)
        loss_mmd = torch.zeros((), device=self.device, dtype=loss.dtype)
        loss_cover = torch.zeros((), device=self.device, dtype=loss.dtype)
        loss_usage_entropy = torch.zeros((), device=self.device, dtype=loss.dtype)
        loss_anchor = torch.zeros((), device=self.device, dtype=loss.dtype)
        agent_mask = x_data.get('agent_mask', x_data.get('output_mask', None))
        if agent_mask is not None:
            agent_mask = agent_mask.to(device=denoised_y.device)
            pred_raw = self.model.denormalize_features(denoised_y)
            gt_raw = self.model.denormalize_features(fut_traj_normalized)

            if float(self._cfg_get("flow_weight_joint", 0.0)) != 0.0:
                feat_w = pred_raw.new_tensor([1.0, 3.0, 1.0, 6.0, 6.0, 6.0, 6.0])
                joint_err = ((pred_raw - gt_raw).pow(2) * feat_w).sum(dim=-1).sqrt()
                mask_weight = self._expand_agent_mask(agent_mask, joint_err)
                loss_joint = (joint_err * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)

            if float(self._cfg_get("flow_weight_od_mmd", 0.0)) != 0.0:
                pred_z = self._od_joint(pred_raw[:, 0])
                gt_z = self._od_joint(gt_raw[:, 0])
                loss_mmd = self._masked_rbf_mmd2(pred_z, gt_z, agent_mask)

            if float(self._cfg_get("flow_weight_od_cover", 0.0)) != 0.0:
                pred_z = self._od_joint(pred_raw[:, 0])
                gt_z = self._od_joint(gt_raw[:, 0])
                loss_cover = self._soft_bidirectional_od_distance(pred_z, gt_z, agent_mask)

            if float(self._cfg_get("flow_weight_usage_entropy", 0.0)) != 0.0:
                pred_z = self._od_joint(pred_raw[:, 0])
                gt_z = self._od_joint(gt_raw[:, 0])
                loss_usage_entropy = self._gt_usage_entropy_loss(pred_z, gt_z, agent_mask)

            if float(self._cfg_get("flow_weight_anchor", 0.0)) != 0.0:
                loss_anchor = self._anchor_energy(pred_raw, x_data.get("condition_map", None), agent_mask)

        loss = loss + float(self._cfg_get("flow_weight_joint", 0.0)) * loss_joint
        loss = loss + float(self._cfg_get("flow_weight_od_mmd", 0.0)) * loss_mmd
        loss = loss + float(self._cfg_get("flow_weight_od_cover", 0.0)) * loss_cover
        loss = loss + float(self._cfg_get("flow_weight_usage_entropy", 0.0)) * loss_usage_entropy
        loss = loss + float(self._cfg_get("flow_weight_anchor", 0.0)) * loss_anchor

        # record the loss for each denoising level
        if log_dict is not None:
            log_dict.update({
                'loss_joint': float(loss_joint.detach().cpu()),
                'loss_od_mmd': float(loss_mmd.detach().cpu()),
                'loss_od_cover': float(loss_cover.detach().cpu()),
                'loss_usage_entropy': float(loss_usage_entropy.detach().cpu()),
                'loss_anchor': float(loss_anchor.detach().cpu()),
            })
            epoch_id = log_dict.get('cur_epoch', 0) if hasattr(log_dict, 'get') else 0
            flag_reset = self.loss_buffer.record_loss(t, loss_reg_b.detach(), epoch_id=epoch_id)
            if flag_reset:
                dict_loss_per_level = self.loss_buffer.get_average_loss()
                log_dict.update({
                    'denoiser_loss_per_level': dict_loss_per_level
                })

        return loss, loss_reg.mean(), loss_cls.mean(), loss_reg_vel.mean()

    def forward(self, x, log_dict=None):
        return self.p_losses(x, log_dict)
