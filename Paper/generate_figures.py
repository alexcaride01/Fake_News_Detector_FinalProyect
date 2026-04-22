"""
We generate Figure 1 (system pipeline) and Figure 2 (MobileNetV3-Small architecture)
for the Fake News Detector paper.

We save both images in the same folder as this script:
    figure1_pipeline.png
    figure2_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

# We resolve the output directory from the script's own location
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# We define the colour palette used across both figures
C_BLUE   = "#1E3A5F"   # we use dark blue for titles and borders
C_TEAL   = "#028090"   # we use teal for the text module
C_CNN    = "#1C6EA4"   # we use medium blue for the visual module
C_ENGINE = "#B85042"   # we use terracotta for the decision engine
C_ARROW  = "#555555"   # we use dark grey for arrows
C_LIGHT  = "#F0F4F8"   # we use light grey as a neutral background
C_FROZEN = "#CADCFC"   # we use light blue to indicate frozen layers
C_THAW   = "#FCD5B4"   # we use light orange to indicate unfrozen layers
C_HEAD   = "#D4EDDA"   # we use light green for the classifier head
WHITE    = "#FFFFFF"   # we use white as the figure background


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor=C_BLUE,
             fontsize=9, textcolor="white", bold=False, radius=0.03,
             subtext=None):
    """We draw a rounded rectangle with centred text and an optional subtitle."""
    # We create the rounded patch and add it to the axes
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.4,
                         zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    # We render the main label and, when provided, a smaller italic subtitle
    if subtext:
        ax.text(x, y + 0.015, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight,
                zorder=4)
        ax.text(x, y - 0.022, subtext, ha="center", va="center",
                fontsize=fontsize - 1.5, color=textcolor, style="italic",
                zorder=4)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, head=8):
    """We draw an annotated arrow between two points."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"->,head_width={head/100},head_length={head/100*0.6}",
                                color=color, lw=lw),
                zorder=5)


def make_figure1():
    """We build and save Figure 1: the full system pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # We place the input image block on the far left
    draw_box(ax, 0.09, 0.50, 0.13, 0.14, "Input\nImage", C_BLUE,
             fontsize=9.5, bold=True)

    # We draw the main horizontal arrow and fork it into the two processing branches
    arrow(ax, 0.155, 0.50, 0.195, 0.50)
    ax.annotate("", xy=(0.22, 0.73), xytext=(0.195, 0.50),
                arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.5), zorder=5)
    ax.annotate("", xy=(0.22, 0.27), xytext=(0.195, 0.50),
                arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.5), zorder=5)

    # We render the visual module block in the upper branch
    VY = 0.73
    draw_box(ax, 0.335, VY, 0.20, 0.13,
             "Visual Module", C_CNN,
             subtext="MobileNetV3-Small", fontsize=9.5, bold=True)
    arrow(ax, 0.22, VY, 0.235, VY)
    arrow(ax, 0.435, VY, 0.46, VY)

    # We label the probability output of the CNN
    ax.text(0.449, VY + 0.08, "p_fake ∈ [0,1]",
            fontsize=7.5, color=C_ARROW, ha="center")

    # We build the text module in the lower branch as four sequential steps
    TY = 0.27
    steps = [
        (0.255, TY, "OCR\n(Tesseract)",    C_TEAL),
        (0.365, TY, "NER +\nKeywords",     C_TEAL),
        (0.475, TY, "Wikipedia\nRetrieval",C_TEAL),
        (0.585, TY, "LLM\n(Mistral)",      C_TEAL),
    ]
    prev_x = 0.22
    for (sx, sy, slabel, sc) in steps:
        draw_box(ax, sx, sy, 0.085, 0.13, slabel, sc, fontsize=8.2, bold=False)
        arrow(ax, prev_x, sy, sx - 0.0425, sy)
        prev_x = sx + 0.0425

    # We draw a dashed bounding box around the text module to group its steps visually
    bx = FancyBboxPatch((0.205, TY - 0.10), 0.435, 0.21,
                         boxstyle="round,pad=0,rounding_size=0.02",
                         facecolor="none", edgecolor=C_TEAL,
                         linewidth=1.2, linestyle="--", zorder=2)
    ax.add_patch(bx)
    ax.text(0.422, TY + 0.115, "Text Module",
            ha="center", va="center", fontsize=9, color=C_TEAL,
            fontweight="bold")

    # We annotate the LLM output with the three possible verdicts
    arrow(ax, 0.628, TY, 0.66, TY)
    ax.text(0.644, TY + 0.04, "support /\nrefute / unknown",
            fontsize=7, color=C_ARROW, ha="center")

    # We join both branches into the decision engine with converging lines
    ax.annotate("", xy=(0.695, 0.50), xytext=(0.46, VY),
                arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.5), zorder=5)
    ax.annotate("", xy=(0.695, 0.50), xytext=(0.66, TY),
                arrowprops=dict(arrowstyle="-", color=C_ARROW, lw=1.5), zorder=5)

    # We place the decision engine block at the confluence of both branches
    draw_box(ax, 0.775, 0.50, 0.155, 0.20,
             "Decision Engine", C_ENGINE,
             subtext="Rule-based fusion", fontsize=9.5, bold=True)
    arrow(ax, 0.695, 0.50, 0.697, 0.50)

    # We show the four possible output labels on the right side
    arrow(ax, 0.853, 0.50, 0.88, 0.50)
    verdicts = ["FAKE", "REAL", "DOUBTFUL", "UNKNOWN"]
    colors   = ["#C0392B", "#27AE60", "#E67E22", "#7F8C8D"]
    for i, (v, c) in enumerate(zip(verdicts, colors)):
        vy = 0.73 - i * 0.155
        draw_box(ax, 0.940, vy, 0.10, 0.11, v, c, fontsize=9, bold=True)
        ax.annotate("", xy=(0.940 - 0.05, vy), xytext=(0.88, 0.50),
                    arrowprops=dict(arrowstyle="->,head_width=0.05,head_length=0.03",
                                    color=c, lw=1.2), zorder=5)

    # We add the figure caption at the top of the plot
    ax.text(0.50, 0.965, "Figure 1. Overview of the proposed multimodal fake news detection pipeline.",
            ha="center", va="top", fontsize=10, color=C_BLUE, style="italic")

    plt.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "figure1_pipeline.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"Saved: {out}")


def make_figure2():
    """We build and save Figure 2: the MobileNetV3-Small architecture diagram."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    CY = 0.52   # we place all main blocks along this vertical centre

    def rbox(x, y, w, h, label, fc, ec=C_BLUE, fs=8, tc="black",
             bold=False, sub=None):
        """We draw a single layer block with optional subtitle."""
        b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0,rounding_size=0.02",
                           facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=3)
        ax.add_patch(b)
        wt = "bold" if bold else "normal"
        if sub:
            ax.text(x, y + 0.025, label, ha="center", va="center",
                    fontsize=fs, color=tc, fontweight=wt, zorder=4)
            ax.text(x, y - 0.025, sub, ha="center", va="center",
                    fontsize=fs - 1.5, color=tc, style="italic", zorder=4)
        else:
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=fs, color=tc, fontweight=wt, zorder=4)

    def arr(x0, y0, x1, y1, c=C_ARROW):
        """We draw a short connecting arrow between two adjacent blocks."""
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->,head_width=0.05,head_length=0.03",
                                    color=c, lw=1.4), zorder=5)

    # We draw the input image block on the left
    rbox(0.045, CY, 0.07, 0.20, "Image\n224×224", C_BLUE, tc="white",
         fs=8.5, bold=True)
    arr(0.08, CY, 0.10, CY)

    # We render the frozen backbone blocks shared by both training phases
    frozen_blocks = [
        (0.155, "Stem\nConv"),
        (0.235, "Blocks\n1–3"),
        (0.315, "Blocks\n4–9"),
    ]
    prev = 0.10
    for (bx, blabel) in frozen_blocks:
        rbox(bx, CY, 0.07, 0.20, blabel, C_FROZEN, ec="#5A8CC2", tc=C_BLUE, fs=8)
        arr(prev, CY, bx - 0.035, CY)
        prev = bx + 0.035

    # We annotate the frozen region with a bracket above the blocks
    ax.text(0.23, CY + 0.185, "Frozen in Phase 1 & 2",
            ha="center", va="bottom", fontsize=8.5, color="#5A8CC2",
            fontweight="bold")
    brace_x0, brace_x1 = 0.11, 0.352
    ax.annotate("", xy=(brace_x0, CY + 0.18), xytext=(brace_x1, CY + 0.18),
                arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2",
                                color="#5A8CC2", lw=1.2))

    # We render the last three blocks that we unfreeze during Phase 2
    unfrozen_blocks = [
        (0.400, "Block\n10"),
        (0.477, "Block\n11"),
        (0.554, "Block\n12"),
    ]
    for (bx, blabel) in unfrozen_blocks:
        rbox(bx, CY, 0.065, 0.20, blabel, C_THAW, ec="#D4813A", tc="#6B3A00", fs=8)
        arr(prev, CY, bx - 0.0325, CY)
        prev = bx + 0.0325

    # We annotate the unfrozen region with a bracket above
    ax.text(0.477, CY + 0.185, "Unfrozen in Phase 2",
            ha="center", va="bottom", fontsize=8.5, color="#D4813A",
            fontweight="bold")
    brace_x0u, brace_x1u = 0.366, 0.588
    ax.annotate("", xy=(brace_x0u, CY + 0.18), xytext=(brace_x1u, CY + 0.18),
                arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2",
                                color="#D4813A", lw=1.2))

    # We add the average pooling block that reduces the feature map to 576-d
    rbox(0.628, CY, 0.065, 0.20, "Avg\nPool\n576-d", "#E8EEF5",
         ec=C_BLUE, tc=C_BLUE, fs=8)
    arr(prev, CY, 0.628 - 0.0325, CY)
    prev = 0.628 + 0.0325

    # We place the four classifier head layers that we always train
    head_layers = [
        (0.706, "Linear\n576→128"),
        (0.772, "BN +\nReLU"),
        (0.838, "Dropout\n0.3"),
        (0.904, "Linear\n128→2"),
    ]
    for (hx, hlabel) in head_layers:
        rbox(hx, CY, 0.058, 0.20, hlabel, C_HEAD,
             ec="#2D6A4F", tc="#1B4332", fs=7.8)
        arr(prev, CY, hx - 0.029, CY)
        prev = hx + 0.029

    # We annotate the classifier head region with a bracket below the blocks
    ax.text(0.805, CY - 0.19, "Classifier Head (always trained)",
            ha="center", va="top", fontsize=8.5, color="#2D6A4F",
            fontweight="bold")
    ax.annotate("", xy=(0.675, CY - 0.18), xytext=(0.934, CY - 0.18),
                arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2",
                                color="#2D6A4F", lw=1.2))

    # We draw the final output block with the two class labels
    arr(prev, CY, 0.965, CY)
    rbox(0.978, CY, 0.038, 0.20, "Fake\nReal", C_ENGINE,
         ec=C_ENGINE, tc="white", fs=8, bold=True)

    # We add a colour legend at the bottom to explain the three block categories
    legend_items = [
        (C_FROZEN, "#5A8CC2", "Frozen (Phase 1 & 2)"),
        (C_THAW,   "#D4813A", "Fine-tuned (Phase 2 only)"),
        (C_HEAD,   "#2D6A4F", "Classifier Head"),
    ]
    lx = 0.115
    for i, (fc, ec, label) in enumerate(legend_items):
        b = FancyBboxPatch((lx + i * 0.245, 0.06), 0.03, 0.06,
                           boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor=fc, edgecolor=ec, linewidth=1.1)
        ax.add_patch(b)
        ax.text(lx + i * 0.245 + 0.038, 0.09, label,
                ha="left", va="center", fontsize=8.5, color="#333333")

    # We add the figure caption at the top of the plot
    ax.text(0.50, 0.975,
            "Figure 2. MobileNetV3-Small architecture with custom classifier head.\n"
            "Blue = frozen layers · Orange = unfrozen in Phase 2 · Green = classifier head.",
            ha="center", va="top", fontsize=9, color=C_BLUE, style="italic")

    plt.tight_layout(pad=0.3)
    out = os.path.join(OUT_DIR, "figure2_architecture.png")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"Saved: {out}")


# We run both figure generators when this script is executed directly
if __name__ == "__main__":
    make_figure1()
    make_figure2()