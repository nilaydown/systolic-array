"""Render the 3x3 systolic matmul as an animated GIF."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
import io

from systolic import SystolicArray, stagger_inputs


A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

B = [[1, 0, 2],
     [0, 1, 1],
     [3, 1, 0]]

N = 3
OUT_GIF = Path(__file__).parent / "demo.gif"

# Palette
BG = "#ffffff"
CELL_BG = "#f4f1ea"
CELL_BORDER = "#2d3340"
ACC_COLOR = "#b8860b"
ACC_DONE = "#cdebc1"
A_COLOR = "#2980b9"
B_COLOR = "#c0392b"
TEXT = "#222222"
MUTED = "#666666"
TITLE = "#111111"


def fmt_val(v):
    if v is None:
        return ""
    return f"{int(v) if float(v).is_integer() else v}"


def draw_frame(arr, a_streams, b_streams, tick, n, done=False):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cell_size = 1.0
    pad = 0.15
    grid_origin_x = 2.0
    grid_origin_y = 2.0

    # Cell positions: (i, j) → (x, y)
    def cell_xy(i, j):
        x = grid_origin_x + j * (cell_size + pad)
        y = grid_origin_y + (n - 1 - i) * (cell_size + pad)
        return x, y

    # Draw cells with acc values
    for i in range(n):
        for j in range(n):
            x, y = cell_xy(i, j)
            color = ACC_DONE if done else CELL_BG
            rect = Rectangle((x, y), cell_size, cell_size,
                             facecolor=color, edgecolor=CELL_BORDER, linewidth=1.8)
            ax.add_patch(rect)
            acc = arr.cells[i][j].acc
            ax.text(x + cell_size / 2, y + cell_size / 2 + 0.05,
                    f"{int(acc) if acc.is_integer() else f'{acc:.1f}'}",
                    ha="center", va="center", fontsize=22,
                    color=ACC_COLOR if acc > 0 else MUTED, fontweight="bold")
            ax.text(x + cell_size / 2, y + 0.12,
                    f"c{i}{j}", ha="center", va="center", fontsize=9, color=MUTED)

    # Draw the a values that are entering each row (left side) at this tick
    # arr.tick_count is incremented after tick(), so for the just-completed tick
    # we look at index (tick_count - 1)
    show_tick = tick
    for i in range(n):
        if 0 <= show_tick < len(a_streams[i]):
            v = a_streams[i][show_tick]
            if v is not None:
                x, y = cell_xy(i, 0)
                ax.text(x - 0.45, y + cell_size / 2, fmt_val(v),
                        ha="right", va="center", fontsize=16, color=A_COLOR, fontweight="bold")
                arrow = FancyArrowPatch((x - 0.4, y + cell_size / 2),
                                        (x - 0.05, y + cell_size / 2),
                                        arrowstyle="->", mutation_scale=14,
                                        color=A_COLOR, linewidth=1.8)
                ax.add_patch(arrow)

    # Draw the b values entering each column (top side)
    for j in range(n):
        if 0 <= show_tick < len(b_streams[j]):
            v = b_streams[j][show_tick]
            if v is not None:
                x, y = cell_xy(0, j)
                ax.text(x + cell_size / 2, y + cell_size + 0.45, fmt_val(v),
                        ha="center", va="bottom", fontsize=16, color=B_COLOR, fontweight="bold")
                arrow = FancyArrowPatch((x + cell_size / 2, y + cell_size + 0.4),
                                        (x + cell_size / 2, y + cell_size + 0.05),
                                        arrowstyle="->", mutation_scale=14,
                                        color=B_COLOR, linewidth=1.8)
                ax.add_patch(arrow)

    # Headers / matrices
    ax.text(grid_origin_x + (n * (cell_size + pad)) / 2 - pad / 2,
            grid_origin_y + n * (cell_size + pad) + 1.0,
            "Systolic matrix multiply",
            ha="center", fontsize=18, color=TITLE, fontweight="bold")

    subtitle = f"Tick {tick + 1}/{3*n - 2}" if not done else "Done — result matches A @ B"
    ax.text(grid_origin_x + (n * (cell_size + pad)) / 2 - pad / 2,
            grid_origin_y + n * (cell_size + pad) + 0.55,
            subtitle, ha="center", fontsize=13, color=MUTED)

    # Side legend
    ax.text(grid_origin_x - 1.7, grid_origin_y + n * (cell_size + pad) - 0.2,
            "A rows →", color=A_COLOR, fontsize=11, fontweight="bold")
    ax.text(grid_origin_x + n * (cell_size + pad) + 0.2,
            grid_origin_y + n * (cell_size + pad) + 0.7,
            "B cols ↓", color=B_COLOR, fontsize=11, fontweight="bold")

    # Show A and B matrices small at bottom
    def render_matrix(label, M, origin_x, origin_y, color):
        ax.text(origin_x + 0.6, origin_y + 1.4, label,
                color=color, fontsize=12, fontweight="bold", ha="center")
        for i in range(n):
            for j in range(n):
                ax.text(origin_x + j * 0.4, origin_y + (n - 1 - i) * 0.4,
                        str(M[i][j]), color=TEXT, fontsize=10, ha="center")

    render_matrix("A", A, 0.4, 0.2, A_COLOR)
    render_matrix("B", B, 6.0, 0.2, B_COLOR)

    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    arr = SystolicArray(N)
    a_streams = stagger_inputs(A, axis="rows")
    b_streams = stagger_inputs(B, axis="cols")
    total_ticks = 3 * N - 2

    frames = []
    # opening frame (tick 0, all zeros)
    frames.append(draw_frame(arr, a_streams, b_streams, tick=-1, n=N))

    for t in range(total_ticks):
        arr.feed(a_streams, b_streams)
        arr.tick()
        frames.append(draw_frame(arr, a_streams, b_streams, tick=t, n=N))

    # Hold the final frame
    final = draw_frame(arr, a_streams, b_streams, tick=total_ticks - 1, n=N, done=True)
    for _ in range(6):
        frames.append(final)

    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_GIF} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
