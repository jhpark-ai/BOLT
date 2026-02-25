import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from tqdm.auto import tqdm

from src.utils.variables_and_paths import TQDM_BAR_FORMAT


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint_safe(path, map_location='cpu'):
    """
    Load checkpoint and automatically convert to state_dict format.

    Handles two checkpoint formats:
    1. State dict only (e.g., CIFAR10): torch.save(model.state_dict(), path)
    2. Full model object (e.g., Caltech): torch.save(model, path)

    Returns state_dict in both cases.
    """
    if 'src.modeling' not in sys.modules:
        try:
            import src.models.modeling as modeling_mod
            sys.modules['src.modeling'] = modeling_mod
            try:
                import torch.serialization as _ts
                encoder_cls = getattr(modeling_mod, 'ImageEncoder', None)
                if hasattr(_ts, 'add_safe_globals') and encoder_cls is not None:
                    _ts.add_safe_globals([encoder_cls])
            except Exception:
                pass
        except ImportError:
            pass

    try:
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)

        if isinstance(checkpoint, dict):
            return checkpoint
        else:
            if hasattr(checkpoint, 'state_dict'):
                return checkpoint.state_dict()
            return checkpoint
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}. Error: {e}")


def torch_save(model, save_path, save_state_dict=True):
    if save_state_dict and isinstance(model, torch.nn.Module):
        model = model.state_dict()

    if isinstance(model, dict):
        cpu_state = {}
        for k, v in model.items():
            if hasattr(v, "detach"):
                try:
                    cpu_state[k] = v.detach().cpu()
                except Exception:
                    cpu_state[k] = v
            else:
                cpu_state[k] = v
        model = cpu_state

    dir_name = os.path.dirname(save_path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    try:
        torch.save(model, save_path)
    except Exception:
        try:
            torch.save(model, save_path, _use_new_zipfile_serialization=False)
        except Exception as e:
            raise e


def torch_load(save_path, device=None):
    try:
        model = torch.load(save_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(save_path, map_location="cpu")
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_logits(inputs, classifier, dataset_name=None):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


# ---------------------------------------------------------------------------
# K-shot sampling
# ---------------------------------------------------------------------------

def sample_k_shot_indices(dataset, k, seed=0, verbose=True, progress_desc=None):
    """
    Sample exactly *k* examples per class from a dataset.

    Uses numpy random seed for reproducibility across runs.

    Args:
        dataset: Dataset object (should have ``train_dataset`` attribute or be
                 a raw dataset).
        k: Number of samples per class.
        seed: Random seed (default: 0).
        verbose: Print detailed information.
        progress_desc: Optional prefix for tqdm progress bars.

    Returns:
        List of selected indices.
    """
    dataset_name = getattr(dataset, "dataset_name", None)
    if hasattr(dataset, 'train_dataset'):
        base_dataset = dataset.train_dataset
    else:
        base_dataset = dataset
        dataset_name = dataset_name or getattr(base_dataset, "dataset_name", None)

    progress_prefix = progress_desc or (dataset_name or "k-shot")

    # ------------------------------------------------------------------
    def _extract_targets(ds):
        if hasattr(ds, 'targets') and ds.targets is not None:
            return np.array(ds.targets)
        if hasattr(ds, 'labels') and ds.labels is not None:
            return np.array(ds.labels)
        if hasattr(ds, 'y') and ds.y is not None:
            return np.array(ds.y)
        # CUB2011 style: pandas DataFrame with 'target' column
        if hasattr(ds, 'data') and hasattr(ds.data, 'iloc'):
            try:
                return ds.data['target'].values - 1
            except Exception:
                pass
        if hasattr(ds, 'samples') and ds.samples is not None:
            try:
                return np.array([item[1] for item in ds.samples])
            except Exception:
                return None
        return None

    def _resolve_subset_targets(subset_obj):
        if not isinstance(subset_obj, Subset):
            return None

        def _get_indices(obj):
            idx = getattr(obj, 'indices', None)
            if idx is None:
                idx = getattr(obj, '_indices', None)
            return None if idx is None else np.asarray(idx, dtype=int)

        indices = _get_indices(subset_obj)
        if indices is None:
            return None

        parent = subset_obj.dataset
        while isinstance(parent, Subset):
            parent_indices = _get_indices(parent)
            if parent_indices is None:
                return None
            indices = parent_indices[indices]
            parent = parent.dataset

        parent_labels = _extract_targets(parent)
        if parent_labels is None:
            return None
        parent_labels = np.asarray(parent_labels)
        try:
            return parent_labels[indices]
        except Exception:
            parent_list = parent_labels.tolist()
            return np.array([parent_list[i] for i in indices])
    # ------------------------------------------------------------------

    labels = _extract_targets(base_dataset)

    if labels is None and isinstance(base_dataset, Subset):
        labels = _resolve_subset_targets(base_dataset)

    if labels is None and hasattr(base_dataset, 'data') and not hasattr(base_dataset.data, 'iloc'):
        try:
            labels = np.array([item[1] for item in base_dataset.data])
        except Exception:
            labels = None

    if labels is None:
        if verbose:
            print("Extracting labels from dataset by iteration...")
        labels_list = []
        iterator = range(len(base_dataset))
        if verbose:
            iterator = tqdm(
                iterator,
                desc=f"{progress_prefix}: label extraction",
                total=len(base_dataset),
                leave=False,
                bar_format=TQDM_BAR_FORMAT,
            )
        for i in iterator:
            try:
                _, label = base_dataset[i]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels_list.append(label)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to get label for index {i}: {e}")
                continue
        labels = np.array(labels_list)

    if len(labels) == 0:
        raise ValueError("Could not extract labels from dataset")

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    if verbose:
        print(f"Found {num_classes} classes in dataset")
        print(f"Sampling {k} examples per class (seed={seed})...")

    np.random.seed(seed)
    selected_indices = []
    class_sample_counts = {}

    class_iterator = unique_classes
    if verbose:
        class_iterator = tqdm(
            unique_classes,
            desc=f"{progress_prefix}: sampling classes",
            total=num_classes,
            leave=False,
            bar_format=TQDM_BAR_FORMAT,
        )

    for cls in class_iterator:
        cls_indices = np.where(labels == cls)[0]

        if len(cls_indices) < k:
            if verbose:
                print(f"  Class {cls}: has only {len(cls_indices)} samples (requested {k}), using all")
            selected_indices.extend(cls_indices.tolist())
            class_sample_counts[int(cls)] = len(cls_indices)
        else:
            sampled = np.random.choice(cls_indices, size=k, replace=False)
            selected_indices.extend(sampled.tolist())
            class_sample_counts[int(cls)] = k

    if verbose:
        print(f"\nK-shot sampling summary:")
        print(f"  Total classes: {num_classes}")
        print(f"  Requested samples per class: {k}")
        print(f"  Total selected samples: {len(selected_indices)}")

    return selected_indices
