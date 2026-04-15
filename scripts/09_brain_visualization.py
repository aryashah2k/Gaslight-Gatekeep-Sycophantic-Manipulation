"""
09_brain_visualization.py

Publication-ready brain surface visualizations for the VLM Brain Alignment
research project.

Generates four distinct, self-contained figures:
  1. ROI Atlas          – ROI regions coloured on the cortex, with legend
  2. Model × ROI Matrix – heatmap of all 12 models' brain alignment scores
  3. Group Comparison   – resistant vs susceptible VLMs on the brain surface
  4. Summary Bar Chart  – ROI scores with correlation statistics

Brain panels are rendered individually (fast), auto-cropped, then
composited into matplotlib figures.  Typography: Helvetica / Arial.

Outputs: results/brain_visualizations/*.{png,pdf}
"""

import json, os, sys, tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image as PILImage

# ── Conference-paper typography ─────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  9,
    "figure.titlesize": 12,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

from nilearn import datasets, plotting

# ────────────────────────────────────────────────────────────────
# Paths & constants
# ────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR          = PROJECT_ROOT / "data"
RESULTS_DIR       = PROJECT_ROOT / "results"
ROI_SCORES_DIR    = RESULTS_DIR / "roi_brain_scores"
SYC_DIR           = RESULTS_DIR / "sycophancy_v2"
MIXED_DIR         = RESULTS_DIR / "mixed_effects_analysis"
OUTPUT_DIR        = RESULTS_DIR / "brain_visualizations"
REFERENCE_SUBJECT = "subj01"
RESISTANT_THRESH  = 0.5

ROI_CATEGORIES = [
    "prf-visualrois", "floc-bodies", "floc-faces",
    "floc-places",    "floc-words",  "streams",
]

ROI_LABEL_SHORT = {
    "prf-visualrois": "Early Visual\n(V1–V4)",
    "floc-bodies":    "Body-\nselective",
    "floc-faces":     "Face-\nselective",
    "floc-places":    "Place-\nselective",
    "floc-words":     "Word-\nselective",
    "streams":        "Anat.\nStreams",
}

ROI_LABEL = {
    "prf-visualrois": "Early Visual (V1–V4)",
    "floc-bodies":    "Body-selective",
    "floc-faces":     "Face-selective",
    "floc-places":    "Place-selective",
    "floc-words":     "Word-selective",
    "streams":        "Anat. Streams",
}

ROI_COLOR = {
    "prf-visualrois": "#E63946",
    "floc-bodies":    "#457B9D",
    "floc-faces":     "#F4A261",
    "floc-places":    "#2A9D8F",
    "floc-words":     "#9B5DE5",
    "streams":        "#00BBF9",
}

MODEL_DISPLAY = {
    "qwen2vl_2b":     "Qwen2-VL 2B",
    "qwen25vl_3b":    "Qwen2.5-VL 3B",
    "gemma3_1b":      "Gemma 3 1B",
    "llava_7b":       "LLaVA 7B",
    "paligemma2_10b": "PaliGemma2 10B",
    "blip2_opt27b":   "BLIP-2 OPT 2.7B",
    "lfm2vl_8b":      "LFM2-VL 8B",
    "lfm25vl_1b":     "LFM2.5-VL 1B",
    "idefics2_8b":    "IDEFICS2 8B",
    "phi35_vision":   "Phi-3.5 Vision",
    "smolvlm_256m":   "SmolVLM 256M",
    "smolvlm_500m":   "SmolVLM 500M",
}

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
_fsavg_cache = None

def _fsavg():
    global _fsavg_cache
    if _fsavg_cache is None:
        _fsavg_cache = datasets.fetch_surf_fsaverage("fsaverage")
    return _fsavg_cache

def _mask_dir():
    return DATA_DIR / REFERENCE_SUBJECT / "roi_masks"

def _load_json(p):
    with open(p) as f:
        return json.load(f)

def _n_verts(hemi):
    return len(np.load(_mask_dir() / f"{hemi[0]}h.all-vertices_fsaverage_space.npy"))

def _model_groups():
    syc = _load_json(SYC_DIR / "summary_v2.json")
    res, sus = [], []
    for m, d in syc.items():
        (res if d.get("final_sycophancy_rate", 1) < RESISTANT_THRESH else sus).append(m)
    return res, sus

def _avg_scores(models, summary):
    return {roi: float(np.mean([summary[m][roi] for m in models if m in summary]))
            for roi in ROI_CATEGORIES}

# ── Surface map builders ──────────────────────────────────────

# Paint order: streams FIRST (broad), then specific functional ROIs on top
_ATLAS_PAINT_ORDER = [
    "streams",          # 1 — broad anatomical regions, painted first
    "prf-visualrois",   # 2 — overwrites streams where they overlap
    "floc-bodies",      # 3
    "floc-faces",       # 4
    "floc-places",      # 5
    "floc-words",       # 6
]

def _roi_class_map(hemi):
    """Paint streams first (broad) so specific functional ROIs overwrite it."""
    h = hemi[0]
    out = np.zeros(_n_verts(hemi), dtype=float)
    for idx, rc in enumerate(_ATLAS_PAINT_ORDER, 1):
        fp = _mask_dir() / f"{h}h.{rc}_fsaverage_space.npy"
        if fp.exists():
            out[np.load(fp) > 0] = idx
    return out

def _score_map(scores, hemi):
    """Paint streams first so specific functional ROIs overwrite overlapping vertices."""
    h = hemi[0]
    out = np.full(_n_verts(hemi), np.nan)
    for rc in _ATLAS_PAINT_ORDER:
        if rc not in scores:
            continue
        fp = _mask_dir() / f"{h}h.{rc}_fsaverage_space.npy"
        if fp.exists():
            out[np.load(fp) > 0] = scores[rc]
    return out

# ── Render / crop / embed ─────────────────────────────────────

def _render(stat_map, hemi="left", view="lateral", cmap="YlOrRd",
            vmin=None, vmax=None, threshold=1e-14, colorbar=False, title=None):
    fs = _fsavg()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    plotting.plot_surf_stat_map(
        surf_mesh=fs[f"infl_{hemi}"],
        stat_map=stat_map,
        bg_map=fs[f"sulc_{hemi}"],
        cmap=cmap, vmin=vmin, vmax=vmax,
        threshold=threshold, colorbar=colorbar,
        view=view, title=title,
        output_file=tmp.name,
    )
    return tmp.name

def _autocrop(path, border=4):
    img = PILImage.open(path).convert("RGB")
    arr = np.array(img)
    mask = np.any(arr < 245, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - border); rmax = min(arr.shape[0]-1, rmax + border)
        cmin = max(0, cmin - border); cmax = min(arr.shape[1]-1, cmax + border)
        img = img.crop((cmin, rmin, cmax+1, rmax+1))
    img.save(path)
    return path

def _embed(ax, path, title=None):
    path = _autocrop(path)
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
    try:
        os.unlink(path)
    except OSError:
        pass

def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}",
                    dpi=300, bbox_inches="tight",
                    pad_inches=0.03, facecolor="white")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 1 — ROI Atlas with embedded legend
# ════════════════════════════════════════════════════════════════

def figure_roi_atlas():
    print("[Fig 1] ROI atlas …")
    from matplotlib.colors import ListedColormap

    bg = (0.85, 0.85, 0.85, 1.0)
    # Colormap order must match _ATLAS_PAINT_ORDER
    clist = [bg] + [matplotlib.colors.to_rgba(ROI_COLOR[rc])
                    for rc in _ATLAS_PAINT_ORDER]
    cmap = ListedColormap(clist)

    views = [("left", "lateral"), ("left", "medial"),
             ("right", "lateral"), ("right", "medial")]

    panels = [_render(_roi_class_map(h), h, v, cmap=cmap, threshold=0.5)
              for h, v in views]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    fig.subplots_adjust(wspace=0.01, hspace=0)
    labels = ["Left Lateral", "Left Medial", "Right Lateral", "Right Medial"]
    for ax, p, lab in zip(axes, panels, labels):
        _embed(ax, p, title=lab)

    # Legend uses _ATLAS_PAINT_ORDER so colours match the map
    handles = [Patch(fc=ROI_COLOR[rc], ec="grey", lw=0.5,
                     label=ROI_LABEL[rc]) for rc in _ATLAS_PAINT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               frameon=True, fancybox=True, edgecolor="grey",
               bbox_to_anchor=(0.5, -0.01),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

    fig.suptitle("Visual Cortex ROI Atlas (fsaverage)",
                 fontsize=12, fontweight="bold", y=1.01)
    _save(fig, "fig1_roi_atlas")
    print("  ✓ fig1_roi_atlas")


# ════════════════════════════════════════════════════════════════
# Figure 2 — Model × ROI Heatmap Matrix
# ════════════════════════════════════════════════════════════════

def figure_model_roi_matrix():
    """
    Annotated heatmap: rows = 12 VLMs (sorted by sycophancy rate),
    columns = 6 ROI categories.  Y-axis labels are colour-coded
    (teal = resistant, coral = susceptible).
    """
    print("[Fig 2] Model × ROI matrix …")
    summary = _load_json(ROI_SCORES_DIR / "summary.json")
    syc = _load_json(SYC_DIR / "summary_v2.json")

    # Sort models by sycophancy rate (ascending → resistant on top)
    models = sorted(summary.keys(),
                    key=lambda m: syc.get(m, {}).get("final_sycophancy_rate", 1))
    n_models = len(models)
    n_rois   = len(ROI_CATEGORIES)

    # Build matrix
    matrix = np.zeros((n_models, n_rois))
    for i, m in enumerate(models):
        for j, roi in enumerate(ROI_CATEGORIES):
            matrix[i, j] = summary[m].get(roi, 0)

    syc_rates = [syc.get(m, {}).get("final_sycophancy_rate", 0) for m in models]
    divider_idx = sum(1 for r in syc_rates if r < RESISTANT_THRESH) - 0.5

    fig, ax = plt.subplots(figsize=(9, 5.5))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())

    # Annotate cells with values
    for i in range(n_models):
        for j in range(n_rois):
            val = matrix[i, j]
            txt_col = "white" if val > (matrix.max() - matrix.min()) * 0.65 + matrix.min() else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color=txt_col, fontweight="medium")

    # Column headers (rotated)
    ax.set_xticks(range(n_rois))
    ax.set_xticklabels([ROI_LABEL[r] for r in ROI_CATEGORIES],
                       fontsize=8.5, rotation=35, ha="left")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Row labels — colour-coded by group
    y_labels = []
    y_colors = []
    for m, rate in zip(models, syc_rates):
        display = MODEL_DISPLAY.get(m, m)
        y_labels.append(f"{display}  ({rate:.0%})")
        y_colors.append("#2A9D8F" if rate < RESISTANT_THRESH else "#E76F51")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(y_labels, fontsize=8)
    for tick_label, color in zip(ax.get_yticklabels(), y_colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    # Divider line
    ax.axhline(divider_idx, color="black", lw=2, ls="-")

    # Compact inline legend (no floating text far to the left)
    handles = [Patch(fc="#2A9D8F", label="Resistant (syc < 50%)"),
               Patch(fc="#E76F51", label="Susceptible (syc ≥ 50%)")]
    ax.legend(handles=handles, loc="upper center", fontsize=8,
              frameon=True, fancybox=True, edgecolor="grey",
              bbox_to_anchor=(0.5, -0.04), ncol=2,
              handlelength=1, handletextpad=0.4)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Brain Alignment Score (Pearson r)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Brain Alignment Scores Across All VLMs and ROIs",
                 fontsize=11, fontweight="bold", pad=14)

    fig.tight_layout()
    _save(fig, "fig2_model_roi_matrix")
    print("  ✓ fig2_model_roi_matrix")


# ════════════════════════════════════════════════════════════════
# Figure 3 — Resistant vs Susceptible (brain surface)
# ════════════════════════════════════════════════════════════════

def figure_group_comparison():
    print("[Fig 3] Group comparison …")
    summary = _load_json(ROI_SCORES_DIR / "summary.json")
    res, sus = _model_groups()
    print(f"  Resistant  ({len(res)}): {res}")
    print(f"  Susceptible ({len(sus)}): {sus}")

    res_sc = _avg_scores(res, summary)
    sus_sc = _avg_scores(sus, summary)

    all_v = list(res_sc.values()) + list(sus_sc.values())
    vmin, vmax = min(all_v), max(all_v)

    views = [("left", "lateral"), ("left", "medial"),
             ("right", "lateral"), ("right", "medial")]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.01, hspace=0.15)

    groups = [
        (f"Resistant (n={len(res)}, syc < 50%)", res_sc),
        (f"Susceptible (n={len(sus)}, syc ≥ 50%)", sus_sc),
    ]

    for row, (label, scores) in enumerate(groups):
        for col, (hemi, view) in enumerate(views):
            sm = _score_map(scores, hemi)
            p = _render(sm, hemi, view, vmin=vmin, vmax=vmax,
                        colorbar=(col == 3))
            col_lab = f"{hemi[0].upper()}H {view.title()}" if row == 0 else None
            _embed(axes[row, col], p, title=col_lab)

        axes[row, 0].text(-0.04, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight="bold",
                          va="center", ha="right", rotation=90)

    fig.suptitle("Brain Alignment: Sycophancy-Resistant vs Susceptible VLMs",
                 fontsize=12, fontweight="bold", y=1.0)
    _save(fig, "fig3_group_comparison")
    print("  ✓ fig3_group_comparison")


# ════════════════════════════════════════════════════════════════
# Figure 4 — Bar chart with correlation overlay
# ════════════════════════════════════════════════════════════════

def figure_bar_chart():
    print("[Fig 4] Bar chart + correlation …")
    summary = _load_json(ROI_SCORES_DIR / "summary.json")
    mixed = _load_json(MIXED_DIR / "mixed_effects_results.json")
    res, sus = _model_groups()

    res_sc = _avg_scores(res, summary)
    sus_sc = _avg_scores(sus, summary)

    x = np.arange(len(ROI_CATEGORIES))
    w = 0.35
    res_v = [res_sc[r] for r in ROI_CATEGORIES]
    sus_v = [sus_sc[r] for r in ROI_CATEGORIES]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x - w/2, res_v, w, color="#2A9D8F", edgecolor="white",
            label=f"Resistant (n={len(res)}, syc < 50%)", zorder=3)
    ax1.bar(x + w/2, sus_v, w, color="#E76F51", edgecolor="white",
            label=f"Susceptible (n={len(sus)}, syc ≥ 50%)", zorder=3)
    ax1.set_ylabel("Mean Brain Alignment Score", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ROI_LABEL_SHORT[r] for r in ROI_CATEGORIES], fontsize=8)
    ax1.set_ylim(0.25, 0.52)
    ax1.grid(axis="y", alpha=0.25, zorder=0)

    # ── Correlation overlay ──
    ax2 = ax1.twinx()
    reg = mixed.get("regression", {})
    r_mid, clo, chi = [], [], []
    for roi in ROI_CATEGORIES:
        lo = reg.get(roi, {}).get("bootstrap_ci_lower", 0)
        hi = reg.get(roi, {}).get("bootstrap_ci_upper", 0)
        r_mid.append((lo + hi) / 2)
        clo.append(lo); chi.append(hi)

    r_mid = np.array(r_mid); clo = np.array(clo); chi = np.array(chi)

    for i, roi in enumerate(ROI_CATEGORIES):
        sig = clo[i] > 0 or chi[i] < 0
        c = "#264653" if sig else "#ADB5BD"
        ax2.errorbar(x[i], r_mid[i],
                     yerr=[[r_mid[i]-clo[i]], [chi[i]-r_mid[i]]],
                     fmt="D", ms=7, capsize=5, capthick=1.5,
                     color=c, ecolor=c, zorder=5)
        if sig:
            pp = reg.get(roi, {}).get("permutation_p", 1.0)
            ax2.annotate(f"r={r_mid[i]:.2f}, p={pp:.3f}",
                         xy=(x[i], chi[i]),
                         xytext=(x[i]+0.25, chi[i]+0.06),
                         fontsize=8, ha="center", color="#264653",
                         fontweight="bold",
                         arrowprops=dict(arrowstyle="->",
                                         color="#264653", lw=0.8))

    ax2.axhline(0, ls="--", color="grey", alpha=0.5, zorder=1)
    ax2.set_ylabel("Pearson r (brain score vs sycophancy)", fontsize=10)
    ax2.set_ylim(-0.6, 0.3)

    h1, _ = ax1.get_legend_handles_labels()
    diamond = Line2D([], [], marker="D", color="#264653", ls="none", ms=7,
                     label="Pearson r (bootstrap 95% CI)")
    ax1.legend(handles=h1 + [diamond], loc="upper right", fontsize=9,
               frameon=True, fancybox=True, edgecolor="grey")

    ax1.set_title("ROI Brain Alignment by Sycophancy Group\n"
                  "with Brain-Score–Sycophancy Correlation",
                  fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, "fig4_bar_chart")
    print("  ✓ fig4_bar_chart")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Brain ROI Visualization Script")
    print("=" * 60)
    print(f"Output → {OUTPUT_DIR}\n")

    if not _mask_dir().exists():
        sys.exit(f"ERROR: masks not found at {_mask_dir()}")

    figure_roi_atlas()
    figure_model_roi_matrix()
    figure_group_comparison()
    figure_bar_chart()

    print(f"\nAll figures saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
