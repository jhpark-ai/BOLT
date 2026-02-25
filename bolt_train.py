"""BOLT: Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation

Learns sigma parameters on SVD-decomposed task vector bases
to adapt a pretrained model to a target dataset using few-shot data.
"""

import os
import time
import copy
import json
import logging
import argparse

import numpy as np
import torch
import torchvision
from typing import Optional
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.nn.utils.stateless import functional_call

from src.utils.variables_and_paths import (
    ALL_DATASETS,
    get_finetuned_path,
    get_zeroshot_path,
)
from src.datasets import get_dataloader, maybe_dictionarize, get_dataset
from src.models import ImageClassifier, ImageEncoder, get_classification_head
from src.utils.sigma_param import SigmaParametrization
from src.models.task_vectors import NonLinearTaskVector
from src.utils.utils import cosine_lr, load_checkpoint_safe, sample_k_shot_indices
from src.eval.eval_comparison import evaluate_encoder_with_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_value(val):
    if isinstance(val, float):
        val = f"{val:.6g}"
    elif isinstance(val, bool):
        val = int(val)
    return "".join(
        ch if ch.isalnum() or ch in ["-", "_"] else "_"
        for ch in str(val).replace(".", "p")
    )


def build_config_tag(cfg) -> str:
    """Build a unique directory tag from key hyperparameters."""
    parts = [
        "bolt",
        _sanitize_value(len(cfg.DATASETS_ALL) - 1),
        _sanitize_value(cfg.sigma_lr),
        _sanitize_value(getattr(cfg, "svd_keep_topk", 12)),
        _sanitize_value(getattr(cfg, "initialize_sigma", "average")),
        _sanitize_value(getattr(cfg, "warmup_ratio", 0.1)),
        _sanitize_value(getattr(cfg, "sigma_wd", 0.0)),
    ]
    return "_".join(parts)


def load_config(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def setup_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# K-shot sampling utilities
# ---------------------------------------------------------------------------

def save_k_shot_indices(indices, save_dir, dataset_name, k, seed):
    """Save k-shot indices to a JSON file for reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(
            {"indices": indices, "dataset": dataset_name, "k": k, "seed": seed}, f
        )
    return path


def load_k_shot_indices(save_dir, k, seed):
    """Load previously saved k-shot indices if available."""
    path = os.path.join(save_dir, f"k_shot_indices_k{k}_seed{seed}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["indices"]
    return None


def subsample_from_larger_k(larger_indices, dataset, target_k, seed):
    """Deterministically subsample target_k indices per class from a larger k-shot set."""
    from torch.utils.data import Subset

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    labels = []

    if hasattr(base_dataset, "targets"):
        all_targets = base_dataset.targets
        if torch.is_tensor(all_targets):
            all_targets = all_targets.cpu().numpy()
        elif not isinstance(all_targets, np.ndarray):
            all_targets = np.array(all_targets)
        labels = [int(all_targets[idx]) for idx in larger_indices]

    elif hasattr(base_dataset, "data") and hasattr(base_dataset.data, "iloc"):
        try:
            all_targets = base_dataset.data["target"].values
            labels = [int(all_targets[idx]) - 1 for idx in larger_indices]
        except Exception:
            labels = []

    elif hasattr(base_dataset, "samples") and base_dataset.samples is not None:
        try:
            labels = [int(base_dataset.samples[idx][1]) for idx in larger_indices]
        except (TypeError, IndexError, ValueError):
            labels = []

    elif hasattr(base_dataset, "_labels"):
        all_labels = base_dataset._labels
        if torch.is_tensor(all_labels):
            all_labels = all_labels.cpu().numpy()
        elif not isinstance(all_labels, np.ndarray):
            all_labels = np.array(all_labels)
        labels = [int(all_labels[idx]) for idx in larger_indices]

    if not labels:
        for idx in larger_indices:
            _, label = base_dataset[idx]
            if torch.is_tensor(label):
                label = label.item()
            elif isinstance(label, np.ndarray):
                label = int(label)
            labels.append(int(label))

    class_to_indices = {}
    for idx, label in zip(larger_indices, labels):
        class_to_indices.setdefault(label, []).append(idx)

    selected = []
    for label in sorted(class_to_indices.keys()):
        selected.extend(class_to_indices[label][:target_k])
    return selected


def _apply_k_shot_sampling(cfg, dataset_train, train_loader, test_ds, logger):
    """Apply k-shot sampling and return the updated dataloader."""
    k = int(cfg.train_k)
    if k <= 0:
        return train_loader

    logger.info(f"Applying {k}-shot sampling...")
    seed = int(getattr(cfg, "seed", 1))
    val_dataset_name = test_ds + "Val"
    indices_save_dir = os.path.join(cfg.model_location, cfg.model, val_dataset_name)

    selected_indices = load_k_shot_indices(indices_save_dir, k, seed)
    if selected_indices is not None:
        logger.info(f"Loaded existing {k}-shot indices (seed={seed})")
    else:
        larger_k = 16
        if k < larger_k:
            larger_indices = load_k_shot_indices(indices_save_dir, larger_k, seed)
            if larger_indices is not None:
                base_ds = getattr(dataset_train, "train_dataset", dataset_train)
                selected_indices = subsample_from_larger_k(
                    larger_indices, base_ds, k, seed
                )
                save_k_shot_indices(
                    selected_indices, indices_save_dir, val_dataset_name, k, seed
                )
        if selected_indices is None:
            selected_indices = sample_k_shot_indices(
                dataset_train,
                k,
                seed=seed,
                verbose=True,
                progress_desc=f"{test_ds} {k}-shot",
            )
            save_k_shot_indices(
                selected_indices, indices_save_dir, val_dataset_name, k, seed
            )

    base_dataset = getattr(dataset_train, "train_dataset", None) or getattr(
        train_loader, "dataset", None
    )
    if base_dataset is not None:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(base_dataset, selected_indices),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=getattr(train_loader, "collate_fn", None),
        )
        logger.info(f"Created {k}-shot dataloader ({len(selected_indices)} samples)")
    else:
        logger.warning("Could not create k-shot subset; using full loader")

    return train_loader


# ---------------------------------------------------------------------------
# SVD basis construction
# ---------------------------------------------------------------------------

SIGMA_EPOCHS_PER_DATASET = {
    "DTD": 20, "GTSRB": 20, "MNIST": 20, "SVHN": 20,
    "CIFAR10": 20, "CIFAR100": 20, "STL10": 20, "Food101": 20,
    "Flowers102": 20, "PCAM": 20, "OxfordIIITPet": 20,
    "RenderedSST2": 20, "EMNIST": 20, "FashionMNIST": 20,
    "FGVCAircraft": 20, "CUB200": 20, "Country211": 20,
}


def compute_svd_basis(task_vectors, config, sigma_reduce: str = "mean"):
    """Compute orthogonalized SVD bases and aggregated sigma from multiple task vectors.

    For each 2D weight matrix, gathers top-k singular components from each task,
    orthogonalizes them, and re-projects to obtain initial sigma values.

    Args:
        task_vectors: List of task vectors (state-dict deltas).
        config: Config with device, DATASETS, svd_keep_topk, etc.
        sigma_reduce: Aggregation for sigma diagonal ('mean', 'max', or 'sum').

    Returns:
        Dict mapping parameter names to either averaged tensors (non-2D)
        or [U_orth, Sigma_diag, V_orth] lists (2D weights).
    """
    device = config.device
    num_tasks = len(list(config.DATASETS))
    desired_k = max(1, int(getattr(config, "svd_keep_topk", 12)))
    sigma_reduce = str(sigma_reduce).lower()

    skip_patterns = ("text_projection", "positional", "token_embedding")

    def is_matrix_key(tv0, key):
        return tv0.vector[key].ndim == 2 and all(p not in key for p in skip_patterns)

    with torch.no_grad():
        new_vector = {}
        tv0 = task_vectors[0]

        for key in tv0.vector:
            if not is_matrix_key(tv0, key):
                avg = None
                for i, tv in enumerate(task_vectors):
                    vec = tv.vector[key].to(device)
                    avg = vec.clone() if i == 0 else avg + (vec - avg) / (i + 1)
                new_vector[key] = avg
                continue

            vec0 = tv0.vector[key].to(device)
            u0, s0, vh0 = torch.linalg.svd(vec0, full_matrices=False)
            m, r, n = int(u0.shape[0]), int(s0.shape[0]), int(vh0.shape[1])

            if r == 0:
                new_vector[key] = torch.zeros_like(vec0)
                continue

            num_used = min(num_tasks, r)
            k = min(desired_k, max(1, r // num_used))
            chunks = k * num_used

            sum_u = torch.zeros((m, chunks), device=device, dtype=u0.dtype)
            sum_v = torch.zeros((chunks, n), device=device, dtype=vh0.dtype)

            for i, tv in enumerate(task_vectors[:num_used]):
                vec = tv.vector[key].to(device)
                u, s, vh = torch.linalg.svd(vec, full_matrices=False)
                k_i = min(k, int(s.shape[0]))
                start = i * k
                sum_u[:, start : start + k_i] = u[:, :k_i]
                sum_v[start : start + k_i, :] = vh[:k_i, :]

            # Orthogonalize gathered axes
            u_u, _, vh_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, vh_v = torch.linalg.svd(sum_v, full_matrices=False)
            U_orth = u_u @ vh_u
            V_orth = u_v @ vh_v

            # Re-project each task vector onto the orthogonal basis
            all_sigma_diags = []
            U_orth_T, V_orth_T = U_orth.T, V_orth.T
            for tv in task_vectors:
                M_i = tv.vector[key].to(device)
                Sigma_i = (U_orth_T @ M_i) @ V_orth_T
                all_sigma_diags.append(torch.diag(Sigma_i))

            if not all_sigma_diags:
                Sigma = torch.zeros((chunks, chunks), device=device, dtype=u0.dtype)
            else:
                stacked = torch.stack(all_sigma_diags, dim=0)
                if sigma_reduce in ("mean", "average"):
                    agg = stacked.mean(dim=0)
                elif sigma_reduce == "max":
                    agg = stacked.max(dim=0).values
                elif sigma_reduce == "sum":
                    agg = stacked.sum(dim=0)
                else:
                    agg = stacked.mean(dim=0)
                Sigma = torch.diag(agg)

            new_vector[key] = [U_orth, Sigma, V_orth]

    return new_vector


# ---------------------------------------------------------------------------
# Alpha grid search
# ---------------------------------------------------------------------------

def grid_search_sigma_alpha(
    sigma_modules: torch.nn.ModuleDict,
    sigma_key_map: dict,
    base_params: dict,
    base_buffers: dict,
    model,
    train_loader,
    device,
    alphas=None,
    max_batches: int = None,
    logger: Optional[logging.Logger] = None,
):
    """Search for the best global scaling factor for initial sigma values.

    Multiplies all sigma diagonals by each candidate alpha, evaluates on the
    training data, then permanently applies the best scaling.

    Returns:
        (best_alpha, best_accuracy)
    """
    if alphas is None:
        alphas = [1, 3, 5, 7, 10]
    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()
    best_alpha, best_acc = None, -1.0

    with torch.no_grad():
        for alpha in alphas:
            correct, total = 0.0, 0.0
            for b_idx, batch in enumerate(train_loader):
                if max_batches and b_idx >= max_batches:
                    break
                batch = maybe_dictionarize(batch)
                inputs = batch["images"].to(device)
                labels = batch["labels"].to(device)

                delta_map = {}
                for safe_key, module in sigma_modules.items():
                    orig_key = sigma_key_map.get(safe_key, safe_key)
                    if orig_key in base_params and module.sigma.numel() > 0:
                        sigma_vec = torch.relu(module.sigma) * float(alpha)
                        delta = module.U @ torch.diag(sigma_vec) @ module.V
                        if delta.shape == base_params[orig_key].shape:
                            delta_map[orig_key] = delta

                params_map = {
                    name: (p + delta_map[name] if name in delta_map else p)
                    for name, p in base_params.items()
                }
                merged = {}
                merged.update(base_buffers)
                merged.update(params_map)
                features = functional_call(model.image_encoder, merged, (inputs,))
                logits = model.classification_head(features)
                correct += float(logits.argmax(1).eq(labels).sum().item())
                total += float(labels.size(0))

            acc = correct / total if total > 0 else 0.0
            logger.info(f"  alpha={alpha} -> accuracy={acc * 100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

    if best_alpha is None:
        best_alpha = 1.0

    for _, module in sigma_modules.items():
        if module.sigma.numel() > 0:
            module.sigma.data.mul_(float(best_alpha))

    logger.info(f"Selected alpha={best_alpha} (accuracy={best_acc * 100:.2f}%)")
    return best_alpha, best_acc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_bolt(cfg: DictConfig) -> None:
    """Run BOLT training: learn sigma parameters on SVD bases for a target dataset."""
    logger = setup_logger("bolt")

    with open_dict(cfg):
        test_ds = cfg.test_dataset
        if test_ds in SIGMA_EPOCHS_PER_DATASET and cfg.sigma_epochs is None:
            cfg.sigma_epochs = SIGMA_EPOCHS_PER_DATASET[test_ds]
        if cfg.sigma_epochs is None:
            cfg.sigma_epochs = 20
        if not cfg.config_tag:
            cfg.config_tag = build_config_tag(cfg)

    test_ds = cfg.test_dataset

    # Determine basis datasets (leave-one-out)
    if hasattr(cfg, "DATASETS_ALL") and cfg.DATASETS_ALL:
        base_list = list(cfg.DATASETS_ALL)
    else:
        base_list = ALL_DATASETS[: cfg.num_tasks]
    if test_ds in base_list:
        base_list = [d for d in base_list if d != test_ds]

    cfg.DATASETS = base_list
    cfg.num_tasks = len(base_list)
    cfg.DATASETS_VAL = [d + "Val" for d in base_list]
    cfg.data_location = os.path.expanduser(cfg.data_location)
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Target dataset: {test_ds}")
    logger.info(f"Basis datasets ({len(base_list)}): {base_list}")

    # ------------------------------------------------------------------
    # Load checkpoints and build task vectors
    # ------------------------------------------------------------------
    ft_checks = []
    for dataset in cfg.DATASETS_VAL:
        path = get_finetuned_path(cfg.model_location, dataset, model=cfg.model)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ft_checks.append(load_checkpoint_safe(path, map_location="cpu"))

    first_dataset = cfg.DATASETS_VAL[0] if cfg.DATASETS_VAL else "dummy"
    zeroshot_path = get_zeroshot_path(
        cfg.model_location, first_dataset, model=cfg.model
    )
    ptm_check = load_checkpoint_safe(zeroshot_path, map_location="cpu")

    task_vectors = [
        NonLinearTaskVector(cfg.model, ptm_check, check) for check in ft_checks
    ]

    # ------------------------------------------------------------------
    # SVD basis computation
    # ------------------------------------------------------------------
    init_mode = getattr(cfg, "initialize_sigma", "average").lower()
    sigma_reduce = {"average": "mean", "mean": "mean", "max": "max", "sum": "sum"}.get(
        init_mode, "mean"
    )

    logger.info(f"Computing SVD bases (reduce={sigma_reduce})...")
    t0 = time.time()
    svd_dict = compute_svd_basis(task_vectors, cfg, sigma_reduce=sigma_reduce)
    logger.info(f"SVD bases computed in {time.time() - t0:.1f}s")

    # Extract basis components
    basis = {}
    for key, value in svd_dict.items():
        if isinstance(value, list) and len(value) == 3:
            U_orth, diag_s, V_orth = value
            basis[key] = {
                "U": U_orth.detach().cpu(),
                "V": V_orth.detach().cpu(),
                "sigma": torch.diagonal(diag_s).detach().cpu(),
            }

    # ------------------------------------------------------------------
    # Build learnable sigma modules
    # ------------------------------------------------------------------
    sigma_modules = torch.nn.ModuleDict()
    sigma_key_map = {}
    for key, fv in basis.items():
        U, V, sigma = fv["U"], fv["V"], fv["sigma"]
        if U.ndim == 2 and V.ndim == 2 and sigma.ndim == 1:
            safe_key = key.replace(".", "_")
            if safe_key in sigma_key_map:
                suffix = 1
                while f"{safe_key}_{suffix}" in sigma_key_map:
                    suffix += 1
                safe_key = f"{safe_key}_{suffix}"
            sigma_key_map[safe_key] = key
            sigma_modules[safe_key] = SigmaParametrization(U, V, sigma)
    sigma_modules = sigma_modules.cuda()

    trainable_params = sum(
        p.numel() for p in sigma_modules.parameters() if p.requires_grad
    )
    logger.info(
        f"Trainable sigma parameters: {trainable_params:,} ({len(sigma_modules)} modules)"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Prepare dataset, model, and data loaders
    # ------------------------------------------------------------------
    val_dataset_name = test_ds + "Val"
    k = int(cfg.train_k)
    shot_folder = f"{k}shots" if k > 0 else "fullshots"

    with open_dict(cfg):
        if "save_dir" not in cfg:
            cfg.save_dir = os.path.join(cfg.model_location, cfg.model)

    image_encoder = ImageEncoder(cfg.model).cuda()
    train_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                size=224,
                scale=(0.5, 1.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ]
        + image_encoder.train_preprocess.transforms[-3:]
    )

    dataset_train = get_dataset(
        test_ds, train_preprocess, location=cfg.data_location, batch_size=cfg.batch_size
    )
    classification_head = get_classification_head(cfg, test_ds)
    model = ImageClassifier(image_encoder, classification_head).cuda()
    model.freeze_head()

    train_loader = get_dataloader(
        dataset_train, is_train=True, args=cfg, image_encoder=None
    )
    train_loader = _apply_k_shot_sampling(
        cfg, dataset_train, train_loader, test_ds, logger
    )

    val_dataset = get_dataset(
        test_ds,
        image_encoder.val_preprocess,
        location=cfg.data_location,
        batch_size=cfg.batch_size,
    )
    val_loader = get_dataloader(val_dataset, is_train=False, args=cfg, image_encoder=None)

    # ------------------------------------------------------------------
    # Optimizer, scheduler, and frozen base parameters
    # ------------------------------------------------------------------
    save_dir = os.path.join(
        cfg.model_location, cfg.model, val_dataset_name, cfg.config_tag, shot_folder
    )
    os.makedirs(save_dir, exist_ok=True)

    params = [p for p in sigma_modules.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.sigma_lr, weight_decay=cfg.sigma_wd)
    num_batches = len(train_loader)
    total_steps = int(cfg.sigma_epochs) * num_batches
    scheduler = cosine_lr(
        optimizer, cfg.sigma_lr, int(cfg.warmup_ratio * total_steps), total_steps
    )

    base_params = {
        n: p.detach().clone() for n, p in model.image_encoder.named_parameters()
    }
    base_buffers = {
        n: b.detach().clone() for n, b in model.image_encoder.named_buffers()
    }

    # ------------------------------------------------------------------
    # Alpha grid search for initial sigma scaling
    # ------------------------------------------------------------------
    alpha_candidates = getattr(cfg, "sigma_alpha_candidates", [1, 3, 5, 7, 10])
    max_eval_batches = int(getattr(cfg, "sigma_alpha_eval_batches", 0)) or None
    logger.info(f"Alpha grid search ({alpha_candidates})...")
    grid_search_sigma_alpha(
        sigma_modules=sigma_modules,
        sigma_key_map=sigma_key_map,
        base_params=base_params,
        base_buffers=base_buffers,
        model=model,
        train_loader=train_loader,
        device=cfg.device,
        alphas=alpha_candidates,
        max_batches=max_eval_batches,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Sigma fine-tuning loop
    # ------------------------------------------------------------------
    logger.info(
        f"Training for {cfg.sigma_epochs} epochs "
        f"({num_batches} steps/epoch, {len(train_loader.dataset)} samples)"
    )

    best_epoch_loss = float("inf")
    best_epoch_idx = -1
    best_sigma_state = None

    model.train()
    for epoch in range(int(cfg.sigma_epochs)):
        epoch_loss_sum = 0.0
        epoch_count = 0

        for i, batch in enumerate(train_loader):
            step = epoch * num_batches + i
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].cuda()
            labels = batch["labels"].cuda()

            delta_map = {}
            for safe_key, module in sigma_modules.items():
                orig_key = sigma_key_map.get(safe_key, safe_key)
                if orig_key in base_params and module.sigma.numel() > 0:
                    delta = module()
                    if delta.shape == base_params[orig_key].shape:
                        delta_map[orig_key] = delta

            params_map = {
                name: (p.detach() + delta_map[name] if name in delta_map else p.detach())
                for name, p in base_params.items()
            }

            merged = {}
            merged.update(base_buffers)
            merged.update(params_map)
            features = functional_call(model.image_encoder, merged, (inputs,))
            logits = model.classification_head(features)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler(step)

            epoch_loss_sum += loss.item()
            epoch_count += 1

        avg_loss = epoch_loss_sum / max(epoch_count, 1)
        logger.info(
            f"  Epoch {epoch}/{cfg.sigma_epochs - 1} | "
            f"loss={avg_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            best_epoch_idx = epoch
            best_sigma_state = {
                k: copy.deepcopy(m.sigma.data) for k, m in sigma_modules.items()
            }

    # ------------------------------------------------------------------
    # Restore best sigma and materialize into encoder
    # ------------------------------------------------------------------
    if best_sigma_state is not None:
        logger.info(
            f"Restoring best sigma from epoch {best_epoch_idx} "
            f"(loss={best_epoch_loss:.4f})"
        )
        with torch.no_grad():
            for safe_key, saved_sigma in best_sigma_state.items():
                if safe_key in sigma_modules:
                    sigma_modules[safe_key].sigma.data.copy_(saved_sigma)

    with torch.no_grad():
        materialized = {name: p.clone() for name, p in base_params.items()}
        for safe_key, module in sigma_modules.items():
            orig_key = sigma_key_map.get(safe_key, safe_key)
            if orig_key in materialized and module.sigma.numel() > 0:
                delta = module().to(materialized[orig_key].device)
                if materialized[orig_key].shape == delta.shape:
                    materialized[orig_key] += delta
        model.image_encoder.load_state_dict(materialized, strict=False)

    # ------------------------------------------------------------------
    # Final evaluation and model saving
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        metrics = evaluate_encoder_with_dataloader(
            model.image_encoder, classification_head, val_loader, cfg.device
        )
    final_acc = metrics["top1"]
    logger.info(f"Final accuracy: {final_acc * 100:.2f}%")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(f"Peak GPU memory: {peak_mb:.0f} MB")

    model_path = os.path.join(save_dir, "encoder.pt")
    torch.save(model.image_encoder.state_dict(), model_path)
    logger.info(f"Saved encoder to {model_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BOLT: Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test_dataset", type=str, default="CIFAR10",
        help="Target dataset for leave-one-out training",
    )
    parser.add_argument(
        "--config_file", type=str, default="config/config_bolt.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--model", default="ViT-B-32", type=str,
        help="Vision backbone (e.g. ViT-B-32, ViT-B-16)",
    )
    parser.add_argument("--sigma_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--sigma_lr", type=float, help="Learning rate for sigma")
    parser.add_argument("--sigma_wd", type=float, help="Weight decay for sigma")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--k", type=int, dest="train_k", default=4,
        help="K-shot samples per class (0=fullshot)",
    )
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio for LR schedule")
    parser.add_argument(
        "--svd_keep_topk", default=12, type=int,
        help="Singular vectors to keep per task",
    )
    parser.add_argument(
        "--initialize_sigma", type=str, choices=["average", "sum"],
        help="Sigma initialization strategy",
    )
    parser.add_argument("--config_tag", type=str, help="Custom tag for output directory")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for k-shot sampling")

    args = parser.parse_args()
    cfg = load_config(args.config_file)

    cli_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k != "config_file"
    }
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(cli_overrides))

    if not cfg.get("test_dataset"):
        parser.error("--test_dataset is required")

    OmegaConf.set_struct(cfg, True)
    run_bolt(cfg)
