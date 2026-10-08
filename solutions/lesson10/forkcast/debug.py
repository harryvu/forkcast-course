"""Lesson 10. Debugging and interpreting a classifier: look at the mistakes, look inside the model, check its confidence.

``reliability_bins`` and ``gallery`` are provided; the rest are exercises.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def class_counts(labels: Tensor, n_classes: int) -> Tensor:
    """How many examples of each class: (N,) int64 -> (n_classes,) int64. torch.bincount with minlength."""
    return torch.bincount(labels, minlength=n_classes)


def class_weights(counts: Tensor) -> Tensor:
    """Inverse-frequency weights: total / (n_classes * count), with empty classes clamped to 1.

    Every class then carries the same total weight (count × weight is equal for all classes), so a mistake
    on a rare class costs more than one on a common class when passed to nn.CrossEntropyLoss(weight=...).
    """
    counts = counts.float()
    return counts.sum() / (len(counts) * counts.clamp_min(1))


def misclassified(logits: Tensor, y: Tensor, k: int = 16) -> list[tuple[int, int, int, float]]:
    """The k most confidently wrong predictions as (index, true, predicted, confidence), highest confidence first.

    confidence is the softmax probability of the predicted class.
    """
    probs = torch.softmax(logits, dim=1)
    confidence, predicted = probs.max(dim=1)
    wrong = (predicted != y).nonzero().flatten()
    order = wrong[confidence[wrong].argsort(descending=True)][:k]
    return [(int(i), int(y[i]), int(predicted[i]), float(confidence[i])) for i in order]


def record_activation(model: nn.Module, layer: nn.Module, x: Tensor) -> Tensor:
    """Run ``model(x)`` and return what ``layer`` produced on the way, using a forward hook.

    A forward hook is a function ``hook(module, inputs, output)`` registered with
    ``layer.register_forward_hook``; it is called every time the layer runs. Remove the hook
    afterwards with the handle it returns, or it will fire on every later forward pass forever.
    """
    captured = {}

    def hook(module, inputs, output):
        captured["out"] = output.detach()

    handle = layer.register_forward_hook(hook)
    try:
        model(x)
    finally:
        handle.remove()                                 # never leave hooks behind
    return captured["out"]


class GradCAM:
    """Where did the model look? A heatmap over ``layer``'s spatial grid for the chosen class.

    A forward hook captures the layer's activations (N, C, H, W). For the class score s, the
    gradient ds/dactivations is averaged over H and W to give one weight per channel; the
    heatmap is relu(sum over channels of weight * activation), normalised to [0, 1] per image.
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model, self.layer, self.activations = model, layer, None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.activations = output                       # kept in the graph so we can differentiate through it

    def __call__(self, x: Tensor, class_idx: Tensor | None = None) -> Tensor:
        """(N, 3, H, W) -> (N, h, w) heatmaps in [0, 1]. ``class_idx`` defaults to the predicted class."""
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)          # so activations are in the graph even when the backbone is frozen
            logits = self.model(x)                      # fills self.activations through the hook
            if class_idx is None:
                class_idx = logits.argmax(dim=1)
            score = logits.gather(1, class_idx.view(-1, 1)).sum()
            grads, = torch.autograd.grad(score, self.activations)       # (N, C, h, w): how each activation moves the score
        weights = grads.mean(dim=(2, 3), keepdim=True)                  # one importance per channel
        cam = torch.relu((weights * self.activations).sum(dim=1))       # (N, h, w): positive evidence only
        cam = cam / cam.flatten(1).max(dim=1).values.clamp_min(1e-8)[:, None, None]
        return cam.detach()

    def remove(self) -> None:
        self.handle.remove()


def reliability_bins(probs: Tensor, y: Tensor, n_bins: int = 10) -> tuple[Tensor, Tensor, Tensor]:
    """Bin predictions by confidence. Returns (mean confidence, accuracy, count) per bin, each (n_bins,).

    ``probs`` is (N, C) after softmax. Empty bins have count 0 and confidence/accuracy 0.
    """
    confidence, predicted = probs.max(dim=1)
    correct = (predicted == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bin_idx = torch.bucketize(confidence, edges[1:-1], right=False).clamp(0, n_bins - 1)
    count = torch.bincount(bin_idx, minlength=n_bins).float()
    conf_sum = torch.bincount(bin_idx, weights=confidence, minlength=n_bins)
    acc_sum = torch.bincount(bin_idx, weights=correct, minlength=n_bins)
    safe = count.clamp_min(1)
    return conf_sum / safe, acc_sum / safe, count


def expected_calibration_error(probs: Tensor, y: Tensor, n_bins: int = 10) -> float:
    """Weighted mean, over bins, of |accuracy - confidence|, weights = fraction of predictions in the bin.

    0 means the model's confidence is exactly as often right as it claims. Use reliability_bins.
    """
    confidence, accuracy, count = reliability_bins(probs, y, n_bins)
    return (((count / count.sum()) * (accuracy - confidence).abs()).sum()).item()


def gallery(images: Tensor, picks: list[tuple[int, int, int, float]], nrow: int = 8) -> Tensor:
    """A make_grid tensor of the picked images (for plt.imshow(...permute(1, 2, 0)))."""
    from torchvision.utils import make_grid
    idx = torch.tensor([p[0] for p in picks], dtype=torch.long)
    return make_grid(images[idx].clamp(0, 1), nrow=nrow)
