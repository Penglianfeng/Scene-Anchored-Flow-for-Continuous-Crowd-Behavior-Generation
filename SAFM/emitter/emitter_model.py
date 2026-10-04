"""Transformer emitter used by SAFE.

The public class keeps its historical CrowdES name so existing Diffusers
checkpoints remain loadable.  SAFE conditions flow matching on a global scene
encoding and on time-gated appearance/population values sampled at each
candidate origin and goal.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


@dataclass
class CrowdESEmitterModelOutput(BaseOutput):
    sample: torch.Tensor
    cls_logits: Optional[torch.Tensor] = None


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H], cond: [B, S, C]
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ada1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.gate1 = nn.Linear(cond_dim, hidden_dim)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ada2 = AdaLayerNorm(hidden_dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.gate2 = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ada1(self.ln1(x), cond)
        h_attn, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + torch.sigmoid(self.gate1(cond)) * h_attn

        h2 = self.ada2(self.ln2(x), cond)
        h_mlp = self.mlp(h2)
        x = x + torch.sigmoid(self.gate2(cond)) * h_mlp
        return x


class CrowdESEmitterModel(ModelMixin, ConfigMixin):
    # Older checkpoints contain this never-used projection.  Ignoring it lets
    # the cleaned model load those checkpoints without changing predictions.
    _keys_to_ignore_on_load_unexpected = [r"external_condition_proj\.(weight|bias)"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        max_num_agents: int = 64,
        num_classes: int = 7,
        hidden_dim: int = 256,
        condition_size: Tuple[int, int] = (64, 64),
        condition_channels: int = 4,
        condition_hidden_dim: int = 64,
        condition_embedding_dim: int = 256,
        condition_token_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        time_embedding_type: str = "fourier",
        time_embedding_dim: int = 64,
        use_timestep_embedding: bool = True,
        act_fn: str = "silu",
        latent_len_multiplier: int = 1,
        use_local_condition: bool = False,
        local_condition_time_gamma: float = 1.0,
        local_condition_zero_init: bool = True,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.max_num_agents = max_num_agents
        self.num_classes = num_classes
        self.latent_len_multiplier = latent_len_multiplier

        self.norm_mean = nn.Parameter(torch.zeros(num_classes))
        self.norm_std = nn.Parameter(torch.ones(num_classes))

        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=time_embedding_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=True,
            )
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(time_embedding_dim, downscale_freq_shift=0.0, flip_sin_to_cos=True)
        else:
            raise ValueError(f"Unknown time embedding type: {time_embedding_type}")

        self.time_mlp = None
        if use_timestep_embedding:
            self.time_mlp = TimestepEmbedding(
                in_channels=time_embedding_dim,
                time_embed_dim=time_embedding_dim,
                act_fn=act_fn,
            )

        # The global branch summarizes scene semantics, appearance, population,
        # and active occupancy before conditioning every emitted-agent token.
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, condition_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=condition_hidden_dim),
            nn.Conv2d(condition_hidden_dim, condition_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=condition_hidden_dim),
            nn.Conv2d(condition_hidden_dim, condition_hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(num_groups=4, num_channels=condition_hidden_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(condition_hidden_dim, condition_embedding_dim),
        )

        # Sequence backbone: input and output are always [B, S, F].
        self.input_proj = nn.Linear(num_classes, hidden_dim)
        self.token_condition_proj = nn.Linear(num_classes, condition_token_dim)
        self.local_condition_proj = None
        if use_local_condition:
            self.local_condition_proj = nn.Sequential(
                nn.Linear(condition_channels * 2, condition_token_dim),
                nn.SiLU(),
                nn.Linear(condition_token_dim, condition_token_dim),
            )
            if local_condition_zero_init:
                nn.init.zeros_(self.local_condition_proj[-1].weight)
                nn.init.zeros_(self.local_condition_proj[-1].bias)
        self.global_condition_proj = nn.Linear(condition_embedding_dim + time_embedding_dim, condition_token_dim)

        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    hidden_dim=hidden_dim,
                    cond_dim=condition_token_dim,
                    num_heads=num_heads,
                    mlp_ratio=4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, num_classes)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def _normalize_timestep(self, timestep: Union[torch.Tensor, float, int], batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            t = torch.full((batch_size,), float(timestep), device=device, dtype=torch.float32)
        else:
            t = timestep.to(device=device)
            if t.dim() == 0:
                t = t[None]
            if t.numel() == 1 and batch_size > 1:
                t = t.repeat(batch_size)
            if t.shape[0] != batch_size:
                raise ValueError(f"timestep batch mismatch: t.shape={t.shape}, batch_size={batch_size}")
            t = t.to(dtype=torch.float32)
        return t

    def _to_sequence(self, sample: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        if sample.dim() == 3:
            b, s, f = sample.shape
            if f != self.num_classes:
                raise ValueError(f"Expected feature dim {self.num_classes}, got {f}")
            return sample, (3, b, 1, s, f)

        if sample.dim() == 4:
            b, k, a, f = sample.shape
            if f != self.num_classes:
                raise ValueError(f"Expected feature dim {self.num_classes}, got {f}")
            sample = sample.reshape(b, k * a, f)
            return sample, (4, b, k, a, f)

        raise ValueError(f"Expected sample rank 3/4, got {sample.dim()}")

    def _build_token_mask(self, mask: Optional[torch.Tensor], b: int, k: int, a: int, s: int, device: torch.device) -> torch.Tensor:
        if mask is None:
            return torch.ones((b, s), dtype=torch.bool, device=device)

        mask = mask.to(device=device)
        if mask.dim() == 2 and mask.shape == (b, a):
            if k > 1:
                mask = mask[:, None, :].expand(-1, k, -1).reshape(b, s)
            else:
                mask = mask.reshape(b, s)
            return mask.bool()

        if mask.dim() == 3 and mask.shape == (b, k, a):
            return mask.reshape(b, s).bool()

        if mask.dim() == 2 and mask.shape == (b, s):
            return mask.bool()

        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)} for (b={b}, k={k}, a={a}, s={s})")

    def _encode_condition(
        self,
        seq_input: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor],
        x_data: Optional[dict],
    ) -> torch.Tensor:
        b, s, _ = seq_input.shape
        device = seq_input.device

        condition_map = None
        if x_data is not None:
            condition_map = x_data.get("condition_map", None)
            if condition_map is None:
                condition_map = x_data.get("condition", None)
            if condition_map is None:
                condition_map = x_data.get("input_data", None)

        if condition is None and x_data is not None:
            condition = x_data.get("condition_tokens", None)

        if condition_map is not None:
            if condition_map.dim() != 4:
                raise ValueError(f"condition_map must be [B,C,H,W], got {tuple(condition_map.shape)}")
            map_embed = self.condition_encoder(condition_map.to(device=device))
        else:
            map_embed = torch.zeros((b, self.config.condition_embedding_dim), device=device)

        t_embed = self.time_proj(timestep)
        if self.time_mlp is not None:
            t_embed = self.time_mlp(t_embed)

        global_cond = torch.cat([map_embed, t_embed], dim=-1)
        global_cond = self.global_condition_proj(global_cond).unsqueeze(1).expand(-1, s, -1)

        token_cond = self.token_condition_proj(seq_input)
        local_cond = torch.zeros_like(token_cond)
        if self.config.use_local_condition and condition_map is not None and self.local_condition_proj is not None:
            # Sample scene evidence at origin/goal and apply g(t)=t^gamma.  The
            # gate suppresses unreliable spatial coordinates early in the flow
            # path while restoring full scene anchoring near clean data.
            seq_raw = self.denormalize_features(seq_input)
            xy = seq_raw[..., [3, 4, 5, 6]].clamp(0.0, 1.0).reshape(b, s, 2, 2)
            grid = xy.mul(2.0).sub(1.0).reshape(b, s * 2, 1, 2)
            sampled = torch.nn.functional.grid_sample(
                condition_map.to(device=device, dtype=seq_input.dtype),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled.reshape(b, self.config.condition_channels, s, 2)
            sampled = sampled.permute(0, 2, 3, 1).reshape(b, s, -1)
            gamma = max(float(self.config.local_condition_time_gamma), 0.0)
            gate = timestep.clamp(0.0, 1.0).pow(gamma).view(b, 1, 1)
            local_cond = gate * self.local_condition_proj(sampled)

        if condition is not None:
            condition = condition.to(device=device)
            if condition.dim() == 2:
                condition = condition.unsqueeze(1).expand(-1, s, -1)
            if condition.dim() != 3 or condition.shape[:2] != (b, s):
                raise ValueError(f"condition must be [B,S,C], got {tuple(condition.shape)}")
            if condition.shape[-1] != self.config.condition_token_dim:
                raise ValueError(
                    f"condition token dim mismatch: expected {self.config.condition_token_dim}, got {condition.shape[-1]}"
                )
        else:
            condition = torch.zeros((b, s, self.config.condition_token_dim), device=device)

        return global_cond + token_cond + local_cond + condition

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        condition: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        x_data: Optional[dict] = None,
        return_dict: bool = True,
    ) -> Union[CrowdESEmitterModelOutput, Tuple[torch.Tensor, torch.Tensor]]:
        seq_input, (input_rank, b, k, a, f) = self._to_sequence(sample)
        s = seq_input.shape[1]

        token_mask = self._build_token_mask(mask=mask, b=b, k=k, a=a, s=s, device=seq_input.device)
        key_padding_mask = ~token_mask

        t = self._normalize_timestep(timestep=timestep, batch_size=b, device=seq_input.device)
        cond_tokens = self._encode_condition(seq_input=seq_input, timestep=t, condition=condition, x_data=x_data)

        x = self.input_proj(seq_input)
        for block in self.blocks:
            x = block(x, cond_tokens, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        out_seq = self.out_proj(x)
        cls_seq = self.cls_head(x).squeeze(-1)

        out_seq = out_seq.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        cls_seq = cls_seq.masked_fill(~token_mask, -1e4)

        if input_rank == 4:
            out = out_seq.reshape(b, k, a, f)
            cls = cls_seq.reshape(b, k, a)
        else:
            out = out_seq
            cls = cls_seq

        if not return_dict:
            return out, cls

        return CrowdESEmitterModelOutput(sample=out, cls_logits=cls)

    def VAEEncoder(self, input_batch: torch.Tensor) -> torch.Tensor:
        input_batch = (input_batch - self.norm_mean.detach()) / (self.norm_std.detach() + 1e-8)
        if self.latent_len_multiplier <= 1:
            return input_batch
        input_batch = input_batch.unsqueeze(-2).repeat(1, 1, self.latent_len_multiplier, 1)
        return input_batch.view(input_batch.size(0), self.max_num_agents, -1)

    def VAEDecoder(self, output_batch: torch.Tensor) -> torch.Tensor:
        if output_batch.dim() == 3 and output_batch.shape[-1] == self.num_classes:
            out = output_batch
        else:
            out = output_batch.view(-1, self.max_num_agents, self.latent_len_multiplier, self.num_classes).mean(dim=2)
        return out * self.norm_std.detach() + self.norm_mean.detach()

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.norm_mean.detach()) / (self.norm_std.detach() + 1e-8)

    def denormalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.norm_std.detach() + self.norm_mean.detach()

    def set_norm_mean_std(self, mean: torch.Tensor, std: torch.Tensor):
        self.norm_mean.data = mean
        self.norm_std.data = std.clamp_min(1e-6)
