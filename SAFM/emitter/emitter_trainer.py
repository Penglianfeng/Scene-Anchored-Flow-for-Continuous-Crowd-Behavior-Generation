"""Training entry point for the SAFE scene-conditioned flow emitter."""

import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from CrowdES.emitter.emitter_model import CrowdESEmitterModel
from CrowdES.flow_matching import FlowMatcher
from CrowdES.safe_config import build_safe_flow_config
from utils.dataloader.emitter_dataloader import EmitterDataset
from utils.utils import reproducibility_settings


def batch_to_flow_x_data(model, batch, device):
    condition_map = batch["input_data"].to(device)
    target_pad = batch["output_data"].to(device)
    target_mask = batch["output_mask"].to(device)

    # Keep strict sequence dimension: [B, A, 1, F].
    target_norm = model.normalize_features(target_pad)
    fut_traj = target_norm.unsqueeze(2)

    x_data = {
        "fut_traj": fut_traj,
        "condition_map": condition_map,
        "agent_mask": target_mask,
        "output_mask": target_mask,
        "batch_size": fut_traj.shape[0],
        "num_agents": fut_traj.shape[1],
        "store_intermediate": False,
    }
    return x_data


def main(config):
    emitter_cfg = config.crowd_emitter.emitter
    reproducibility_settings(seed=emitter_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Runtime knobs for Ampere GPUs (e.g., RTX 3090): default to faster kernels.
    cudnn_deterministic = bool(emitter_cfg.get("cudnn_deterministic", False))
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = bool(emitter_cfg.get("cudnn_benchmark", not cudnn_deterministic))
    allow_tf32 = bool(emitter_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    num_workers = int(emitter_cfg.get("num_workers", 4))
    pin_memory = bool(emitter_cfg.get("pin_memory", True))
    persistent_workers = bool(emitter_cfg.get("persistent_workers", True)) and num_workers > 0

    dataset_train = EmitterDataset(config, "train")
    dataset_val = EmitterDataset(config, "train")
    loader_train = DataLoader(
        dataset_train,
        batch_size=emitter_cfg.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=emitter_cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    input_dim, output_dim = dataset_train.input_dim, dataset_train.output_dim
    image_size = dataset_train.image_size
    max_num_agents_pad = dataset_train.max_num_agents_pad

    model = CrowdESEmitterModel(
        max_num_agents=max_num_agents_pad,
        num_classes=output_dim,
        condition_size=image_size,
        condition_channels=input_dim,
        time_embedding_type="fourier",
        latent_len_multiplier=1,
        hidden_dim=emitter_cfg.get("flow_hidden_dim", 256),
        condition_embedding_dim=emitter_cfg.get("flow_condition_embedding_dim", 256),
        condition_token_dim=emitter_cfg.get("flow_condition_token_dim", 256),
        num_layers=emitter_cfg.get("flow_num_layers", 4),
        num_heads=emitter_cfg.get("flow_num_heads", 8),
        use_local_condition=emitter_cfg.get("use_local_condition", False),
        local_condition_time_gamma=emitter_cfg.get("local_condition_time_gamma", 1.0),
        local_condition_zero_init=emitter_cfg.get("local_condition_zero_init", True),
    )
    model.to(device)

    # Build feature normalization stats on non-padded tokens with streaming moments.
    feat_sum = torch.zeros(output_dim, dtype=torch.float64)
    feat_sq_sum = torch.zeros(output_dim, dtype=torch.float64)
    feat_count = 0
    for batch in tqdm(loader_train, desc="Prepare mean and std"):
        target_pad_batch = batch["output_data"]
        target_mask_batch = batch["output_mask"]
        target_batch = target_pad_batch[target_mask_batch, :]
        if target_batch.numel() == 0:
            continue
        target_batch = target_batch.to(torch.float64)
        feat_sum += target_batch.sum(dim=0)
        feat_sq_sum += (target_batch * target_batch).sum(dim=0)
        feat_count += int(target_batch.shape[0])

    feat_count = max(feat_count, 1)
    mean = feat_sum / feat_count
    var = feat_sq_sum / feat_count - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    model.set_norm_mean_std(mean.to(torch.float32).to(device), std.to(torch.float32).to(device))

    flow_cfg = build_safe_flow_config(
        config=config,
        max_num_agents=max_num_agents_pad,
        out_dim=output_dim,
        device=device,
    )
    flow_matcher = FlowMatcher(cfg=flow_cfg, model=model, logger=None).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=emitter_cfg.learning_rate,
        weight_decay=float(emitter_cfg.get("weight_decay", 0.0)),
    )
    gradient_accumulation_steps = emitter_cfg.gradient_accumulation_steps
    grad_norm_clip = float(emitter_cfg.get("grad_norm_clip", 1.0))

    # Validation/runtime controls.
    val_every_n_epochs = max(1, int(emitter_cfg.get("val_every_n_epochs", 1)))
    max_val_batches = int(emitter_cfg.get("max_val_batches", -1))
    save_best_only = bool(emitter_cfg.get("save_best_only", False))
    save_every_n_epochs = max(1, int(emitter_cfg.get("save_every_n_epochs", 1)))

    num_update_steps_per_epoch = math.ceil(len(loader_train) / gradient_accumulation_steps)
    num_train_epochs = emitter_cfg.num_train_epochs
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = emitter_cfg.num_warmup_steps
    lr_scheduler_type = emitter_cfg.lr_scheduler_type
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    checkpoint_dir = emitter_cfg.checkpoint_dir.format(config.dataset.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = float("inf")
    last_val_loss = float("inf")
    last_val_reg = 0.0
    last_val_cls = 0.0
    training_log = []

    for epoch in range(num_train_epochs):
        model.train()
        flow_matcher.train()

        total_loss = 0.0
        total_reg = 0.0
        total_cls = 0.0
        num_train_batches = 0

        train_bar = tqdm(loader_train, desc=f"Train Epoch {epoch}", leave=False)
        for step, batch in enumerate(train_bar):
            x_data = batch_to_flow_x_data(model=model, batch=batch, device=device)
            loss, loss_reg, loss_cls, _ = flow_matcher(x_data, log_dict={"cur_epoch": epoch})

            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss.backward()

            total_loss += loss.detach().float().item()
            total_reg += loss_reg.detach().float().item()
            total_cls += loss_cls.detach().float().item()
            num_train_batches += 1

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(loader_train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        do_validate = (epoch % val_every_n_epochs == 0) or (epoch == num_train_epochs - 1)

        val_loss = last_val_loss
        val_reg = last_val_reg
        val_cls = last_val_cls
        num_val_batches = 0
        is_best = False

        if do_validate:
            model.eval()
            flow_matcher.eval()
            val_loss_list = []
            val_reg_list = []
            val_cls_list = []

            val_bar = tqdm(loader_val, desc=f"Valid Epoch {epoch}", leave=False)
            with torch.no_grad():
                for val_step, batch in enumerate(val_bar):
                    x_data = batch_to_flow_x_data(model=model, batch=batch, device=device)
                    loss, loss_reg, loss_cls, _ = flow_matcher(x_data, log_dict={"cur_epoch": epoch})
                    val_loss_list.append(loss.detach().cpu().item())
                    val_reg_list.append(loss_reg.detach().cpu().item())
                    val_cls_list.append(loss_cls.detach().cpu().item())
                    num_val_batches += 1
                    if max_val_batches > 0 and num_val_batches >= max_val_batches:
                        break

            val_loss = float(np.mean(val_loss_list)) if len(val_loss_list) > 0 else 0.0
            val_reg = float(np.mean(val_reg_list)) if len(val_reg_list) > 0 else 0.0
            val_cls = float(np.mean(val_cls_list)) if len(val_cls_list) > 0 else 0.0
            last_val_loss = val_loss
            last_val_reg = val_reg
            last_val_cls = val_cls

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                is_best = True

        train_loss = total_loss / max(1, num_train_batches)
        train_reg = total_reg / max(1, num_train_batches)
        train_cls = total_cls / max(1, num_train_batches)

        eval_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_reg": float(train_reg),
            "train_cls": float(train_cls),
            "val_loss": float(val_loss),
            "val_reg": float(val_reg),
            "val_cls": float(val_cls),
            "validated": bool(do_validate),
            "num_val_batches": int(num_val_batches),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "min_val_loss": float(min_val_loss),
        }
        training_log.append(eval_metrics)

        should_save = False
        if save_best_only:
            should_save = is_best
        else:
            should_save = ((epoch + 1) % save_every_n_epochs == 0) or (epoch == num_train_epochs - 1)

        if should_save:
            model.save_pretrained(checkpoint_dir, safe_serialization=False)

        tqdm.write(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"validated={do_validate}, val_batches={num_val_batches}"
        )

    with open(os.path.join(checkpoint_dir, "all_results.json"), "w") as f:
        json.dump(training_log, f, indent=2)


if __name__ == "__main__":
    import argparse

    from utils.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="./configs/model/CrowdES_gcs.yaml", help="Path to a model config file")
    parser.add_argument("--dataset_config", type=str, default=None, help="Path to a dataset config file (optional)")
    parser.add_argument("--trainer_config", type=str, default=None, help="Path to a trainer config file (optional)")
    args = parser.parse_args()
    config = get_config(args.model_config, args.dataset_config, args.trainer_config)

    main(config)
