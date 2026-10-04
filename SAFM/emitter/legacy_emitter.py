"""Legacy CrowdES diffusion emitter used by the released baseline checkpoints.

The current ``emitter_model`` module contains the SAFE flow-matching network.
Keeping the legacy architecture under a separate class name lets visualization
tools load both checkpoint families in one process without mutating imports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import BaseOutput

from CrowdES.layers import ConcatScaleLinear, ConcatSquashLinear, GroupNormLinear, social_transformer
from CrowdES.emitter.emitter_pipeline import CrowdESEmitterPipeline


@dataclass
class LegacyCrowdESEmitterModelOutput(BaseOutput):
    sample: torch.Tensor


class LegacyCrowdESEmitterModel(ModelMixin, ConfigMixin):
    """Architecture stored in ``checkpoints/<scene>/emitter``."""

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        max_num_agents: int = 64,
        num_classes: int = 7,
        hidden_dim: int = 256,
        condition_size: Tuple[int, int] = (64, 64),
        condition_channels: int = 10,
        condition_hidden_dim: int = 64,
        condition_embedding_dim: int = 1024,
        time_embedding_type: str = "fourier",
        time_embedding_dim: int = 8,
        use_timestep_embedding: bool = True,
        act_fn: str = "silu",
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
        vae_latent_dim: int = 32,
        vae_hidden_dim: int = 2048,
        latent_len_multiplier: int = 16,
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
            self.time_proj = Timesteps(
                time_embedding_dim,
                downscale_freq_shift=0.0,
                flip_sin_to_cos=True,
            )
        else:
            raise ValueError(f"Unknown time embedding type: {time_embedding_type}")

        if use_timestep_embedding:
            self.time_mlp = TimestepEmbedding(
                in_channels=time_embedding_dim,
                time_embed_dim=time_embedding_dim,
                act_fn=act_fn,
            )

        condition_blocks = []
        for index in range(3):
            input_channels = condition_channels if index == 0 else condition_hidden_dim
            condition_blocks.extend(
                [
                    nn.Conv2d(input_channels, condition_hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(num_groups=4, num_channels=condition_hidden_dim),
                ]
            )
        condition_blocks.append(nn.Flatten())
        flattened_condition_size = (
            condition_size[0] // 8 * condition_size[1] // 8 * condition_hidden_dim
        )
        condition_blocks.append(nn.Linear(flattened_condition_size, condition_embedding_dim))
        self.condition_block = nn.Sequential(*condition_blocks)

        self.encoders = nn.ModuleList(
            [
                ConcatSquashLinear(num_classes, hidden_dim, num_classes),
                ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim),
                ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim),
            ]
        )
        self.self_attentions = nn.ModuleList(
            [social_transformer(hidden_dim, hidden_dim, n_head=4, n_layers=2)]
        )
        self.self_attentions_norm = GroupNormLinear(num_groups=4, num_channels=hidden_dim)

        self.cross_attentions = nn.ModuleList(
            [
                ConcatSquashLinear(
                    hidden_dim,
                    hidden_dim,
                    condition_embedding_dim + time_embedding_dim,
                )
            ]
        )
        self.cross_attention_norm = GroupNormLinear(num_groups=4, num_channels=hidden_dim)

        self.decoders = nn.ModuleList(
            [
                ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim),
                ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim),
                ConcatScaleLinear(hidden_dim, num_classes, hidden_dim),
            ]
        )

        self.noise_norm = GroupNormLinear(num_groups=1, num_channels=num_classes)
        self.noise_predictors = nn.ModuleList(
            [
                ConcatSquashLinear(latent_len_multiplier, hidden_dim, latent_len_multiplier),
                ConcatScaleLinear(hidden_dim, latent_len_multiplier, hidden_dim),
            ]
        )

    def forward(
        self,
        sample: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ):
        sample_initial = sample
        sample = sample.view(
            -1,
            self.max_num_agents,
            self.latent_len_multiplier,
            self.num_classes,
        ).mean(dim=2)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)

        condition_embed = self.condition_block(condition)
        condition_embed = torch.cat([condition_embed, timestep_embed], dim=-1).unsqueeze(1)

        for encoder in self.encoders:
            sample = encoder(sample, sample)

        for self_attention in self.self_attentions:
            sample = self_attention(sample, padding_mask=mask)
        sample = self.self_attentions_norm(sample)

        for cross_attention in self.cross_attentions:
            sample = cross_attention(condition_embed, sample)
        sample = self.cross_attention_norm(sample)

        for decoder in self.decoders:
            sample = decoder(sample, sample)

        sample = sample.unsqueeze(2).repeat_interleave(
            repeats=self.latent_len_multiplier,
            dim=2,
        )

        noise = sample_initial.view(
            -1,
            self.max_num_agents,
            self.latent_len_multiplier,
            self.num_classes,
        )
        noise = self.noise_norm(noise - sample).permute(0, 1, 3, 2)
        for noise_predictor in self.noise_predictors:
            noise = noise_predictor(noise, noise)
        noise = noise.permute(0, 1, 3, 2).reshape(
            -1,
            self.max_num_agents,
            self.latent_len_multiplier * self.num_classes,
        )

        if not return_dict:
            return (noise,)
        return LegacyCrowdESEmitterModelOutput(sample=noise)

    def VAEEncoder(self, input_batch: torch.Tensor) -> torch.Tensor:
        normalized = (input_batch - self.norm_mean.detach()) / (
            self.norm_std.detach() + 1e-8
        )
        return normalized.unsqueeze(-2).repeat(
            1,
            1,
            self.latent_len_multiplier,
            1,
        ).view(input_batch.size(0), self.max_num_agents, -1).detach()

    def VAEDecoder(self, output_batch: torch.Tensor) -> torch.Tensor:
        decoded = output_batch.view(
            -1,
            self.max_num_agents,
            self.latent_len_multiplier,
            self.num_classes,
        ).mean(dim=2)
        return decoded * self.norm_std.detach() + self.norm_mean.detach()

    def set_norm_mean_std(self, mean: torch.Tensor, std: torch.Tensor):
        self.norm_mean.data = mean
        self.norm_std.data = std


def load_legacy_emitter(
    checkpoint_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> LegacyCrowdESEmitterModel:
    """Load a released CrowdES baseline emitter and verify every state key."""

    checkpoint_dir = Path(checkpoint_dir)
    model = LegacyCrowdESEmitterModel.load_config(checkpoint_dir)
    model = LegacyCrowdESEmitterModel.from_config(model)
    weights_path = checkpoint_dir / "diffusion_pytorch_model.bin"
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    return model


@torch.no_grad()
def sample_legacy_emitter(
    model: LegacyCrowdESEmitterModel,
    condition: torch.Tensor,
    num_agents: int,
    seed: int,
    diffusion_steps: int = 50,
    leapfrog_steps: int = 5,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Sample normalized ``[type, time, speed, origin, goal]`` rows.

    ``condition`` must be ``[1, C, H, W]`` and should contain the same ten
    channels used during baseline training. Coordinates remain in ``[0, 1]``;
    callers can map them to the original image without losing the raw output.
    """

    if not 1 <= num_agents <= model.config.max_num_agents:
        raise ValueError(
            f"num_agents must be in [1, {model.config.max_num_agents}], got {num_agents}"
        )
    if condition.ndim != 4 or condition.shape[0] != 1:
        raise ValueError(f"condition must have shape [1,C,H,W], got {tuple(condition.shape)}")
    if condition.shape[1] != model.config.condition_channels:
        raise ValueError(
            "condition channel mismatch: "
            f"expected {model.config.condition_channels}, got {condition.shape[1]}"
        )
    expected_size = tuple(model.config.condition_size)
    if tuple(condition.shape[-2:]) != expected_size:
        raise ValueError(
            f"condition spatial size must be {expected_size}, got {tuple(condition.shape[-2:])}"
        )
    if not 1 <= leapfrog_steps < diffusion_steps:
        raise ValueError(
            "The released baseline requires 1 <= leapfrog_steps < diffusion_steps "
            "so that leapfrog OD initialization is defined."
        )

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device).eval()
    condition = condition.to(device=device, dtype=next(model.parameters()).dtype)

    scheduler = DDIMScheduler(num_train_timesteps=diffusion_steps, beta_schedule="linear")
    pipeline = CrowdESEmitterPipeline(unet=model, scheduler=scheduler)

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        output = pipeline(
            batch_size=1,
            condition=condition,
            num_crowd=num_agents,
            device=device,
            seed=seed,
            num_inference_steps=diffusion_steps,
            leapfrog_steps=leapfrog_steps,
            eta=1.0,
        ).crowd_emission

    return output[0].detach()
