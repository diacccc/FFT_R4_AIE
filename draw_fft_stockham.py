#!/usr/bin/env python3
"""
Draw a radix-4 Stockham FFT butterfly picture.

This follows the same stage/index mapping used by fft_stockham_f32.cc:
  in:  idx_r = q + s * (p + r*m), r in {0,1,2,3}
  out: out_r = q + s * (4*p + r)
with
  n = N / 4^stage, s = 4^stage, m = n / 4.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


@dataclass(frozen=True)
class StageParams:
    stage: int
    n: int
    s: int
    m: int


def is_power_of_four(n: int) -> bool:
    if n < 1:
        return False
    while n > 1:
        if n % 4 != 0:
            return False
        n //= 4
    return True


def stockham_stage_params(n_fft: int) -> list[StageParams]:
    n_stages = int(round(math.log(n_fft, 4)))
    out: list[StageParams] = []
    for stage in range(n_stages):
        n = n_fft // (4**stage)
        s = 4**stage
        m = n // 4
        out.append(StageParams(stage=stage, n=n, s=s, m=m))
    return out


def draw_stockham_network(
    n_fft: int,
    output: str,
    label_twiddles: bool,
    clear_dft4: bool,
) -> None:
    if not is_power_of_four(n_fft):
        raise ValueError(f"N must be a power of 4, got {n_fft}")

    stages = stockham_stage_params(n_fft)
    n_stages = len(stages)

    fig, ax = plt.subplots(figsize=(2.8 * (n_stages + 1), max(9, n_fft * 0.15)))

    colors = {
        0: "#2E3440",  # a path
        1: "#1D4ED8",  # b path
        2: "#DC2626",  # c path
        3: "#059669",  # d path
    }
    dft4_palette = [
        "#1D4ED8",
        "#DC2626",
        "#059669",
        "#D97706",
        "#7C3AED",
        "#0F766E",
        "#BE185D",
        "#4B5563",
        "#0284C7",
        "#65A30D",
    ]

    # Draw node columns (input + stage outputs).
    for col in range(n_stages + 1):
        for idx in range(n_fft):
            y = n_fft - 1 - idx
            ax.plot(col, y, "o", ms=2.3, color="#111827", alpha=0.8)

    # Draw edges per stage.
    for st in stages:
        x0 = st.stage
        x1 = st.stage + 1
        xm = (x0 + x1) * 0.5

        for p in range(st.m):
            for q in range(st.s):
                in_idx = [q + st.s * (p + r * st.m) for r in range(4)]
                out_idx = [q + st.s * (4 * p + r) for r in range(4)]
                block_id = st.stage * n_fft + p * st.s + q
                dft4_color = dft4_palette[block_id % len(dft4_palette)]

                if clear_dft4:
                    y_in = [n_fft - 1 - idx for idx in in_idx]
                    y_out = [n_fft - 1 - idx for idx in out_idx]
                    y_min = min(min(y_in), min(y_out))
                    y_max = max(max(y_in), max(y_out))

                    box_h = max(1.8, y_max - y_min + 0.9)
                    box_y = (y_min + y_max) * 0.5 - box_h * 0.5
                    box_w = 0.14
                    box_x = xm - box_w * 0.5

                    dft4_box = FancyBboxPatch(
                        (box_x, box_y),
                        box_w,
                        box_h,
                        boxstyle="round,pad=0.02",
                        linewidth=0.55,
                        edgecolor="#111827",
                        facecolor="#F9FAFB",
                        alpha=0.9,
                    )
                    ax.add_patch(dft4_box)

                    if st.s <= 4 and p < 2:
                        ax.text(
                            xm,
                            box_y + box_h * 0.5,
                            "DFT4",
                            fontsize=5.8,
                            color=dft4_color,
                            rotation=90,
                            ha="center",
                            va="center",
                        )

                    # Draw all 16 computation connections inside one DFT4 block
                    # using the same color for that block.
                    for rin in range(4):
                        y0i = n_fft - 1 - in_idx[rin]
                        for rout in range(4):
                            y1o = n_fft - 1 - out_idx[rout]
                            ax.plot(
                                [xm - 0.07, xm + 0.07],
                                [y0i, y1o],
                                color=dft4_color,
                                lw=0.5,
                                alpha=0.8,
                            )

                for r in range(4):
                    y0 = n_fft - 1 - in_idx[r]
                    y1 = n_fft - 1 - out_idx[r]

                    if clear_dft4:
                        # Draw input -> DFT4 box -> output in two segments.
                        ax.plot(
                            [x0, xm - 0.07],
                            [y0, y0],
                            color=dft4_color,
                            lw=0.6,
                            alpha=0.85,
                        )
                        ax.plot(
                            [xm + 0.07, x1],
                            [y1, y1],
                            color=dft4_color,
                            lw=0.6,
                            alpha=0.85,
                        )
                    else:
                        ax.plot(
                            [x0, x1],
                            [y0, y1],
                            color=colors[r],
                            lw=0.75,
                            alpha=0.9,
                        )

                    # Twiddle labels for stage > 0, r in {1,2,3}
                    if label_twiddles and st.stage > 0 and r > 0:
                        # In the kernel comments:
                        # b: W_N^(q*m), c: W_N^(2*q*m), d: W_N^(3*q*m)
                        exp_mul = r
                        exp = exp_mul * q * st.m
                        if exp != 0 and st.s <= 4:
                            xm = (x0 + x1) * 0.5
                            ym = (y0 + y1) * 0.5
                            ax.text(
                                xm,
                                ym,
                                f"W^{exp}",
                                fontsize=6,
                                color=dft4_color if clear_dft4 else colors[r],
                                alpha=0.9,
                                ha="center",
                                va="center",
                                bbox=dict(
                                    boxstyle="round,pad=0.12",
                                    facecolor="white",
                                    edgecolor="none",
                                    alpha=0.6,
                                ),
                            )

    # Axis labels and stage annotations.
    for st in stages:
        ax.text(
            st.stage + 0.5,
            n_fft + 2.0,
            f"Stage {st.stage}\\ns={st.s}, m={st.m}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111827",
        )

    ax.text(0, n_fft + 6.0, "Input", ha="center", va="bottom", fontsize=10)
    ax.text(n_stages, n_fft + 6.0, "Output", ha="center", va="bottom", fontsize=10)

    title_suffix = "Clear DFT-4 blocks" if clear_dft4 else "Wire view"
    ax.set_title(
        f"Radix-4 Stockham FFT Butterfly Network (N={n_fft}) - {title_suffix}",
        fontsize=12,
    )
    ax.set_xlim(-0.25, n_stages + 0.25)
    ax.set_ylim(-2, n_fft + 9)
    ax.set_xticks(range(n_stages + 1))
    ax.set_xticklabels([f"C{i}" for i in range(n_stages + 1)])
    ax.set_yticks([n_fft - 1 - i for i in range(n_fft)])
    ax.set_yticklabels([str(i) for i in range(n_fft)], fontsize=6)
    ax.set_ylabel("Sample index")
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig.savefig(output, dpi=220)
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw radix-4 Stockham FFT network")
    parser.add_argument("--N", type=int, default=64, help="FFT size (power of 4)")
    parser.add_argument(
        "--output",
        type=str,
        default="fft_stockham_N64.png",
        help="Output image path",
    )
    parser.add_argument(
        "--label-twiddles",
        action="store_true",
        help="Add some twiddle labels (can make the plot busy)",
    )
    parser.add_argument(
        "--clear-dft4",
        action="store_true",
        help="Draw explicit DFT-4 butterfly blocks between stage columns",
    )
    args = parser.parse_args()

    draw_stockham_network(
        n_fft=args.N,
        output=args.output,
        label_twiddles=args.label_twiddles,
        clear_dft4=args.clear_dft4,
    )


if __name__ == "__main__":
    main()
