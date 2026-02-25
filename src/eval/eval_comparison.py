"""Evaluation utilities for BOLT training."""

import torch
from typing import Dict

from src.models.modeling import ImageClassifier, ImageEncoder
from src.datasets.common import maybe_dictionarize
from src.utils import utils


def evaluate_encoder_with_dataloader(
    image_encoder: ImageEncoder,
    classification_head,
    dataloader,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate an image encoder using a pre-loaded dataloader.

    Args:
        image_encoder: Image encoder to evaluate
        classification_head: Classification head
        dataloader: Pre-loaded dataloader
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics (top1 accuracy)
    """
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    with torch.no_grad():
        correct, n = 0.0, 0.0
        for _, data in enumerate(dataloader):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    return {"top1": top1}
