"""Grad-CAM (Selvaraju et al., 2017) with plain forward/backward hooks, no extra dependency.

For a single-logit model the explained "class score" is the PNEUMONIA logit:
    alpha_k = mean_{i,j} d logit / d A_k[i,j]        (global-average-pooled gradients)
    cam     = ReLU( sum_k alpha_k * A_k )             (weighted feature-map sum)
then upsampled to the input size and min-max scaled to [0, 1] per image.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [target_layer.register_forward_hook(self._forward_hook)]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # Tensor hook: fires with dlogit/doutput during backward.
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return (cams [B,H,W] in [0,1], probabilities [B])."""
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            self.model.zero_grad(set_to_none=True)
            logits.sum().backward()  # each image's logit only depends on that image
        A, G = self.activations, self.gradients
        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * A).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
        cam = ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(cam)
        self.model.train(was_training)
        return cam.cpu().numpy(), torch.sigmoid(logits).detach().cpu().numpy()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


# ----------------------------------------------------------------------------- audit heuristics
def border_mass(cam: np.ndarray, frac: float = 0.12) -> float:
    """Share of the CAM's total mass lying in the outer `frac` band of the image.

    In these films the lung fields sit in the central region; mass in the outer band
    (collimation edges, laterality letters, scanner text, arms/shoulders) is a red flag.
    """
    h, w = cam.shape
    bh, bw = max(1, int(h * frac)), max(1, int(w * frac))
    inner = cam[bh : h - bh, bw : w - bw].sum()
    total = cam.sum()
    return float(1 - inner / total) if total > 0 else float("nan")


def peak_location(cam: np.ndarray) -> tuple[float, float]:
    """(x, y) of the CAM maximum, normalised to [0, 1]."""
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return float(x / (cam.shape[1] - 1)), float(y / (cam.shape[0] - 1))


def auto_flag(cam: np.ndarray, frac: float = 0.12, mass_threshold: float = 0.6) -> str:
    """Heuristic first pass, NOT a substitute for looking.

    'empty'        - no positive evidence for PNEUMONIA anywhere (typical for confident negatives)
    'off-anatomy?' - the peak lies in the outer border band, or > mass_threshold of the mass does
    'central'      - otherwise
    """
    if float(cam.max()) < 1e-6:
        return "empty"
    px, py = peak_location(cam)
    peak_in_border = min(px, py, 1 - px, 1 - py) < frac
    off = peak_in_border or border_mass(cam, frac) > mass_threshold
    return "off-anatomy?" if off else "central"


# ----------------------------------------------------------------------------- CLI runner
CASE_TYPES = ("True positives", "False negatives", "False positives", "True negatives")


def select_cases(pred: pd.DataFrame, thr: float, per_case: int) -> dict:
    """Pick cases per type. FNs: lowest probability first (the confidently wrong ones);
    FPs: highest probability first; TPs / TNs: most confident first."""
    df = pred.copy()
    df["pred"] = (df["score"] >= thr).astype(int)
    tp = df[(df.label == 1) & (df.pred == 1)].sort_values("score", ascending=False)
    fn = df[(df.label == 1) & (df.pred == 0)].sort_values("score", ascending=True)
    fp = df[(df.label == 0) & (df.pred == 1)].sort_values("score", ascending=False)
    tn = df[(df.label == 0) & (df.pred == 0)].sort_values("score", ascending=True)
    return dict(
        zip(
            CASE_TYPES, (tp.head(per_case), fn.head(per_case), fp.head(per_case), tn.head(per_case))
        )
    )


def run_gradcam(cfg: dict, per_case: int = 3, device: str | None = None):
    import json
    from pathlib import Path

    import pandas as pd

    from . import plots
    from .config import run_dir
    from .data import CXRDataset, build_transforms, denormalize, resolve_root
    from .evaluate import load_checkpoint
    from .metrics import choose_threshold
    from .utils import get_device

    name = cfg["experiment"]["name"]
    results_dir = Path(cfg["paths"]["results_dir"])
    pred_dir = results_dir / "predictions"
    test_p, val_p = pred_dir / f"{name}_test.csv", pred_dir / f"{name}_val.csv"
    if not test_p.exists():
        raise FileNotFoundError(f"{test_p} missing - run `pneumonia evaluate` first.")
    dev = get_device(device or cfg["train"].get("device", "auto"))
    model, _ = load_checkpoint(run_dir(cfg) / "best.pt", dev)
    pv, pt = pd.read_csv(val_p), pd.read_csv(test_p)
    thr = choose_threshold(pv["label"], pv["score"], float(cfg["selection"]["target_recall"]))[
        "threshold"
    ]
    root = resolve_root(cfg)
    ds = CXRDataset(pt, root, build_transforms(cfg, train=False))
    gc_cfg = cfg["eval"].get("gradcam", {})
    frac = float(gc_cfg.get("border_frac", 0.12))
    mass_thr = float(gc_cfg.get("border_mass_threshold", 0.6))

    groups, audit = {}, []
    with GradCAM(model, model.features) as cam:
        for ct, sel in select_cases(pt, thr, per_case).items():
            items = []
            for idx, r in sel.iterrows():
                x, _, _ = ds[idx]
                cams, probs = cam(x.unsqueeze(0).to(dev))
                c = cams[0]
                flag = auto_flag(c, frac, mass_thr)
                px, py = peak_location(c)
                items.append(
                    {
                        "image": denormalize(x),
                        "cam": c,
                        "prob": float(probs[0]),
                        "caption": f"{r['subtype']} · {Path(r['path']).name[:22]}",
                    }
                )
                audit.append(
                    {
                        "case_type": ct,
                        "path": r["path"],
                        "subtype": r["subtype"],
                        "prob": round(float(probs[0]), 4),
                        "border_mass": round(border_mass(c, frac), 3),
                        "peak_x": round(px, 3),
                        "peak_y": round(py, 3),
                        "auto_flag": flag,
                        "manual_location": "",
                        "manual_notes": "",
                    }
                )
            groups[ct] = items
    fig = plots.gradcam_panel(
        groups,
        results_dir / "figures" / "gradcam_panel.png",
        title=f"Grad-CAM, {name}, test set (threshold {thr:.3f})",
    )
    audit_df = pd.DataFrame(audit)
    audit_path = results_dir / "gradcam_audit.csv"
    if audit_path.exists():  # keep any manual labels already typed in
        old = pd.read_csv(audit_path).fillna("")
        keep = old.set_index("path")[["manual_location", "manual_notes"]]
        audit_df = audit_df.set_index("path")
        audit_df.update(keep)
        audit_df = audit_df.reset_index()
    audit_df.to_csv(audit_path, index=False)
    summary = {
        "n_maps": len(audit_df),
        "threshold": thr,
        "auto_off_anatomy": int((audit_df["auto_flag"] == "off-anatomy?").sum()),
        "by_case_type": audit_df.groupby("case_type")["auto_flag"]
        .apply(lambda s: int((s == "off-anatomy?").sum()))
        .to_dict(),
        "mean_border_mass": float(audit_df["border_mass"].mean()),
        "uniform_map_border_mass": round(1 - (1 - 2 * frac) ** 2, 3),
        "note": "auto_flag is a geometric heuristic; fill manual_location (lung / outside / "
        "mixed) in gradcam_audit.csv after looking at the panel, then run `pneumonia report`.",
    }
    (results_dir / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[gradcam] wrote {fig} and {audit_path}; auto off-anatomy flags: "
        f"{summary['auto_off_anatomy']}/{summary['n_maps']}"
    )
    return fig
