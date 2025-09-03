import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curve(
    agg,
    x0: float,
    lowest: float,
    highest: float,
    max_len: float,
    stat: str = "mean",            # "mean" or "median"
    band: str = "ci",              # "ci" for 95% CI, "iqr" for interquartile (25–75%)
    alpha: float = 0.05,
    n_boot: int = 2000,
    description: str = "GA Convergence (central tendency ± variability)",
    ylabel: str = "Fitness",
    xlabel: str = "Generation",
    annotate_counts: bool = True,  # (kept for compatibility)
    color: str = "C0",             # (kept for compatibility)
    color_left: str = "tab:blue",
    color_right: str = "tab:orange",
    vline_caption: str | None = None,
    vline_kw: dict | None = None,
    text_kw: dict | None = None,
):
    gen = np.asarray(agg["gen"], dtype=float)
    center = np.asarray(agg["center"], dtype=float)
    low = np.asarray(agg["lower"], dtype=float)
    up = np.asarray(agg["upper"], dtype=float)

    # Ensure increasing x for interpolation safety
    order = np.argsort(gen)
    gen, center, low, up = gen[order], center[order], low[order], up[order]

    # Trim NaNs from both ends based on center
    valid = ~np.isnan(center)
    if not valid.any():
        raise ValueError("All aggregated values are NaN. Check your `runs` input.")
    first, last = np.argmax(valid), len(valid) - np.argmax(valid[::-1]) - 1
    gen, center, low, up = gen[first:last+1], center[first:last+1], low[first:last+1], up[first:last+1]

    if not (gen.min() <= x0 <= gen.max()):
        raise ValueError(f"x0={x0} is outside the generation range [{gen.min()}, {gen.max()}].")

    # Interpolate y at the split so colored segments meet cleanly
    y0 = float(np.interp(x0, gen, center))

    #start
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1])

    ax = fig.add_subplot(gs[0, 0])  # main plot as before
    ax.xaxis.labelpad = 6
    cap_ax = fig.add_subplot(gs[1, 0])  # caption row
    cap_ax.axis('off')
    cap_ax.text(0.5, 0.5, description, ha='center', va='center', wrap=True)

    # Variability band (single color over full domain)
    ax.fill_between(gen, low, up, alpha=0.2,
                    label=("95% CI" if band == "ci" else "IQR (25–75%)"))

    # Split central tendency line into two colored segments
    left_mask = gen <= x0
    right_mask = gen >= x0

    x_left = np.r_[gen[left_mask], x0]
    y_left = np.r_[center[left_mask], y0]
    x_right = np.r_[x0, gen[right_mask]]
    y_right = np.r_[y0, center[right_mask]]

    line_left, = ax.plot(x_left, y_left, color=color_left, lw=2,
                         label=f"{stat.capitalize()} fitness (generations ≤ {round(x0):g})")
    line_right, = ax.plot(x_right, y_right, color=color_right, lw=2,
                          label=f"{stat.capitalize()} fitness (generations ≥ {round(x0):g})")

    # Limits and labels
    ax.set_xlim(0 - round(max_len / 20), max_len)
    lowest = lowest - ((highest - lowest) / 20)   # small bottom margin
    ax.set_ylim(lowest, highest)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("GA Convergence (central tendency ± variability)")

    # Vertical line from curve at x0 down to the x-axis (bottom of current y-limits)
    vk = dict(linestyle='--', linewidth=1.5, color='0.35')
    if vline_kw: vk.update(vline_kw)
    y_bottom = ax.get_ylim()[0]
    ax.vlines(x0, y_bottom, y0, **vk)

    # Caption next to the vertical line
    if vline_caption is None:
        vline_caption = f"Avg\ngenerations = {round(x0):g}"
    tk = dict(ha='left', va='center')
    if text_kw: tk.update(text_kw)
    # If x0 is near the right edge, place caption to the left
    place_left = x0 > (gen.min() + 0.8 * (gen.max() - gen.min()))
    dx = -6 if place_left else 6
    y_mid = 0.5 * (y0 + y_bottom)
    ax.annotate(vline_caption, xy=(x0, y_mid), xytext=(dx, 0),
                textcoords='offset points', **tk)

    ax.legend()

    if annotate_counts:
        fig2, ax2 = plt.subplots()
        ax2.plot(agg["gen"], agg["n_at_gen"], color=color)
        ax2.set_xlim(0 - round(max_len / 20), max_len)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("# runs contributing")
        ax2.set_title("Contributing runs per generation")

        # Vertical line at x0
        try:
            ax2.axvline(x0, **vk)  # reuse main plot style
        except NameError:
            ax2.axvline(x0, linestyle='--', linewidth=1.5, color='0.35')

        # --- Text "Average generations = {x0}" beside the vline ---
        gen_arr = np.asarray(agg["gen"], dtype=float)
        n_arr = np.asarray(agg["n_at_gen"], dtype=float)
        order = np.argsort(gen_arr)
        gen_arr, n_arr = gen_arr[order], n_arr[order]
        y_at_x0 = float(np.interp(x0, gen_arr, n_arr))  # count at x0 (interp)

        y_min, y_max = ax2.get_ylim()
        y_text = 0.5 * (y_at_x0 + y_min)  # place halfway to baseline

        place_left = x0 > (gen_arr.min() + 0.8 * (gen_arr.max() - gen_arr.min()))
        dx = -6 if place_left else 6

        ax2.annotate(f"Avg\ngenerations = {round(x0):g}",
                     xy=(x0, y_text), xytext=(dx, 0),
                     textcoords='offset points', ha='left', va='center')

        # Caption under the plot
        fig2.subplots_adjust(bottom=0.22)
        fig2.text(0.5, 0.04, description, ha="center", va="bottom", wrap=True)

        plt.show()

