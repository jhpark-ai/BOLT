# BOLT: Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation

**[CVPR 2026]** Official PyTorch implementation of

> **Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation**
>
> Junghwan Park,&ensp;Woojin Cho,&ensp;Junhyuk Heo,&ensp;Darongsae Kwon,&ensp;Kookjin Lee
>
> [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_Basis-Oriented_Low-rank_Transfer_for_Few-Shot_and_Test-Time_Adaptation_CVPR_2026_paper.pdf)

<p align="center">
  <img src="figures/model_archi.png" width="90%" alt="BOLT pipeline"/>
</p>

## Abstract

Adapting large pre-trained models to unseen tasks under tight data and compute budgets remains challenging. Meta-learning approaches explicitly learn good initializations, but they require an additional meta-training phase over many tasks, incur high training cost, and can be unstable. At the same time, the number of task-specific pre-trained models continues to grow, yet the question of how to transfer them to new tasks with minimal additional training remains relatively underexplored. We propose BOLT (Basis-Oriented Low-rank Transfer), a framework that reuses existing fine-tuned models not by merging weights, but instead by extracting an orthogonal, task-informed spectral basis and adapting within that subspace. In the offline phase, BOLT collects dominant singular directions from multiple task vectors and orthogonalizes them per layer to form reusable bases. In the online phase, we freeze these bases and train only a small set of diagonal coefficients per layer for the new task, yielding a rank-controlled update with very few trainable parameters. This design provides (i) a strong, training-free initialization for unseen tasks, obtained by pooling source-task coefficients—along with a lightweight rescaling step—while leveraging the shared orthogonal bases, and (ii) a parameter-efficient fine-tuning (PEFT) path that, in our experiments, achieves robust performance compared to common PEFT baselines as well as a representative meta-learned initialization. Our results show that constraining adaptation to a task-informed orthogonal subspace provides an effective alternative for unseen-task transfer.

## Prerequisites

### 1. Create a conda environment

```bash
conda env create -f environment.yml
conda activate bolt
```

### 2. Add the project root to `PYTHONPATH`

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/bolt"
```

### 3. Prepare datasets

Most datasets are downloaded automatically via `torchvision`.
For datasets requiring manual setup (e.g., DTD), please follow the instructions in the [task_arithmetic](https://github.com/mlfoundations/task_vectors) or [task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors) repositories.

Datasets are expected at the path specified by `data_location` in the config file (default: `./datasets`).

**Supported general-domain datasets (17):**
DTD, GTSRB, MNIST, SVHN, STL10, OxfordIIITPet, Flowers102, CIFAR100, PCAM, CIFAR10, Food101, FashionMNIST, RenderedSST2, EMNIST, FGVCAircraft, CUB200, Country211

### 4. Prepare checkpoints

BOLT requires (a) a zero-shot CLIP encoder and (b) fine-tuned checkpoints for each source task.
These can be obtained by following the fine-tuning protocol from the [aTLAS](https://github.com/fredzzhang/atlas) or [task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors) repository.

Place all checkpoints under `model_location` (default: `./models/checkpoints`).

```
models/checkpoints/
├── ViT-B-32/
│   ├── DTDVal/
│   │   ├── finetuned.pt
│   │   └── zeroshot.pt
│   ├── GTSRBVal/
│   │   ├── finetuned.pt
│   │   └── zeroshot.pt
│   └── ...
```

## Configuration

Default hyperparameters are defined in `config/config_bolt.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `ViT-B-32` | CLIP backbone (`ViT-B-32`, `ViT-B-16`, `ViT-L-14`) |
| `sigma_lr` | `1e-3` | Learning rate for sigma |
| `sigma_wd` | `0.0` | Weight decay for sigma |
| `sigma_epochs` | `20` | Training epochs |
| `svd_keep_topk` | `12` | Singular vectors to keep per task |
| `initialize_sigma` | `average` | Sigma initialization (`average` or `sum`) |
| `batch_size` | `32` | Training batch size |
| `warmup_ratio` | `0.1` | LR warmup ratio |
| `train_k` | `0` | K-shot per class (`0` = fullshot) |

All config values can be overridden from the command line.

## Training

### Few-shot adaptation

Run BOLT with different k-shot settings on a target dataset:

```bash
# 4-shot on CIFAR10 with ViT-B/32
python bolt_train.py --test_dataset CIFAR10 --model ViT-B-32 --k 4

# 1-shot on DTD
python bolt_train.py --test_dataset DTD --k 1

# 16-shot on Flowers102
python bolt_train.py --test_dataset Flowers102 --k 16
```

### Custom hyperparameters

```bash
python bolt_train.py \
    --test_dataset CIFAR10 \
    --model ViT-B-16 \
    --k 4 \
    --sigma_lr 1e-3 \
    --sigma_epochs 20 \
    --svd_keep_topk 12 \
    --batch_size 32 \
    --seed 1
```

### Output

The trained encoder is saved at:

```
models/checkpoints/{model}/{dataset}Val/{config_tag}/{k}shots/encoder.pt
```

## How it works

1. **Task vector construction:** Compute per-task weight deltas between fine-tuned and zero-shot CLIP encoders.
2. **SVD basis construction (offline):** For each 2D weight matrix, gather top-k singular directions from all source tasks, then orthogonalize them to form shared spectral bases (U, V).
3. **Coefficient initialization:** Project each task vector onto the shared basis, extract diagonal coefficients, and pool them across tasks. A lightweight alpha grid search rescales the pooled coefficients.
4. **Sigma fine-tuning (online):** Freeze the orthogonal bases and train only the diagonal sigma coefficients on the target dataset using cross-entropy loss with cosine LR scheduling.
5. **Materialization:** Apply the learned sigma deltas to the base encoder and save the final model.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{park2026basis,
  title={Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation},
  author={Park, Junghwan and Cho, Woojin and Heo, Junhyuk and Kwon, Darongsae and Lee, Kookjin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={860--870},
  year={2026}
}
```


## Acknowledgement

This repository builds upon the following excellent codebases:

- [aTLAS](https://github.com/fredzzhang/atlas) (Zhang et al., NeurIPS 2024) — Task vector composition with learned anisotropic scaling
- [Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors) (Gargiulo et al., CVPR 2025) — SVD-based task vector analysis and merging
- [Task Arithmetic](https://github.com/mlfoundations/task_vectors) (Ilharco et al., 2022) — Editing models with task arithmetic

We thank the authors for making their code and checkpoints publicly available.

## License

This project is released under the [Apache 2.0 License](LICENSE).
