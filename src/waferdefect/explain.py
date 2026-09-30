import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import img_size

stage_weights = (0.1, 0.2, 0.3, 0.4)


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


@torch.no_grad()
def stage_activation_map(model, img):
    # mean activation per swin stage, upsampled and blended. not real attention rollout
    device = next(model.parameters()).device
    outs = []
    hooks = [s.register_forward_hook(lambda m, i, o: outs.append(o.detach().cpu())) for s in model.backbone.layers]
    try:
        model.eval()
        model(img.unsqueeze(0).to(device))
    finally:
        for h in hooks:
            h.remove()

    maps = []
    for o in outs:
        act = o[0].mean(dim=-1)
        if act.dim() == 1:  # older timm returns tokens flat
            side = int(act.numel() ** 0.5)
            act = act.reshape(side, side)
        maps.append(cv2.resize(norm01(act.numpy()), (img_size, img_size), interpolation=cv2.INTER_CUBIC))
    return norm01(sum(w * m for w, m in zip(stage_weights, maps)))


@torch.no_grad()
def reconstruction_diff(model, img):
    device = next(model.parameters()).device
    model.eval()
    _, recon = model(img.unsqueeze(0).to(device))
    orig = img[0].cpu().numpy() * 0.5 + 0.5
    rec = recon[0, 0].float().cpu().numpy()
    return orig, rec, norm01(np.abs(orig - rec))


def plot_explanation(model, img, true_name, pred_name, save_path=None):
    act = stage_activation_map(model, img)
    orig, rec, diff = reconstruction_diff(model, img)
    overlay = np.clip(0.5 * np.stack([orig] * 3, -1) + 0.5 * plt.cm.jet(act)[..., :3], 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(orig, "gray", "wafer"), (overlay, None, "stage activation"),
              (rec, "gray", "reconstruction"), (diff, "hot", "recon diff")]
    for ax, (im, cmap, title) in zip(axes, panels):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"true: {true_name} | pred: {pred_name}", color="green" if true_name == pred_name else "red")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
