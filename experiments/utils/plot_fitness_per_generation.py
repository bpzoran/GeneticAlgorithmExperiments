import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_curve(
    agg,
    x0,
    lowest: float,
    highest: float,
    max_len: float,
    stat: str = "mean",            # "mean" or "median"
    band: str = "ci",              # "ci" for 95% CI, "iqr" for interquartile (25–75%)
    description: str = "GA Convergence (central tendency ± variability)",
    ylabel: str = "Fitness",
    xlabel: str = "Generation",
    annotate_counts: bool = True,  # plot diagnostic figure with #runs per generation
    vline_kw: dict | None = None,
    text_kw: dict | None = None,
):
    """
    agg:
      - single: {"gen","center","lower","upper"}
      - multi:  {label: {"gen","center","lower","upper","n_at_gen"(opt)}}
    x0:
      - single: float
      - multi:  {label: float}
    """
    description = transform_function_string(description)
    # Normalize to multi-series dict: label -> series_dict
    is_single = isinstance(agg, dict) and {"gen", "center", "lower", "upper"} <= set(agg.keys())
    if is_single:
        series_dict = {"Series": agg}
        x0_map = {"Series": float(x0)}
    else:
        if not isinstance(x0, dict):
            raise TypeError("When agg contains multiple series, x0 must be a dict keyed like agg.")
        series_dict = agg
        x0_map = {k: float(v) for k, v in x0.items()}

    # --- Figure with a caption row ---
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.xaxis.labelpad = 6
    cap_ax = fig.add_subplot(gs[1, 0])
    cap_ax.axis('off')
    cap_ax.text(0.5, 0.5, description, ha='center', va='center', wrap=True)

    # Global limits
    xpad_left = round(max_len / 20)
    ypad_bottom = (highest - lowest) / 20
    ax.set_xlim(0 - xpad_left, max_len)
    ax.set_ylim(lowest - ypad_bottom, highest)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("GA Convergence (central tendency ± variability)")

    # Common styles
    vk = dict(linestyle='--', linewidth=1.5, color='0.35')
    if vline_kw: vk.update(vline_kw)
    tk = dict(ha='left', va='center')
    if text_kw: tk.update(text_kw)

    # Plot all series; store colors and x0 for stacked annotations later
    colors_by_label: dict[str, str] = {}
    vline_info: list[tuple[str, float, str]] = []  # (label, x0_val, color)

    for label, s in series_dict.items():
        gen = np.asarray(s["gen"], dtype=float)
        center = np.asarray(s["center"], dtype=float)
        low = np.asarray(s["lower"], dtype=float)
        up = np.asarray(s["upper"], dtype=float)

        # sort by gen and trim NaNs
        order = np.argsort(gen)
        gen, center, low, up = gen[order], center[order], low[order], up[order]
        valid = ~np.isnan(center)
        if not valid.any():
            continue
        first = np.argmax(valid)
        last = len(valid) - np.argmax(valid[::-1]) - 1
        gen, center, low, up = gen[first:last+1], center[first:last+1], low[first:last+1], up[first:last+1]

        # band & line
        ax.fill_between(gen, low, up, alpha=0.15, label="_nolegend_")
        line, = ax.plot(gen, center, lw=2, label=label)
        line_color = line.get_color()
        colors_by_label[label] = line_color

        # vline in same color
        if label in x0_map:
            x0_val = x0_map[label]
            if gen.min() <= x0_val <= gen.max():
                ax.axvline(x0_val, **{**vk, "color": line_color})
                vline_info.append((label, x0_val, line_color))

    ax.legend()

    # ---- Stack "Avg gen =" annotations so they don't overlap ----
    if vline_info:
        # Sort by x0 so stacking is deterministic (left -> right)
        vline_info.sort(key=lambda t: t[1])

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        top_anchor_frac = 0.95         # start near the top
        stack_band_frac = 0.25         # use top 25% of the plot for stacking
        n = len(vline_info)
        dy_frac = stack_band_frac / max(n, 1)  # vertical step in fraction of y-range

        for i, (label, x0_val, color) in enumerate(vline_info):
            # place labels one under another within the top band
            y_text = y_min + (top_anchor_frac - i * dy_frac) * y_range

            # place left of the line if too close to right edge
            x_min, x_max = ax.get_xlim()
            place_left = x0_val > (x_min + 0.85 * (x_max - x_min))
            dx = -6 if place_left else 6

            ax.annotate(f"Avg gen = {round(x0_val):g}",
                        xy=(x0_val, y_text), xytext=(dx, 0),
                        textcoords='offset points',
                        color=color, ha='left', va='center')

    # --- Diagnostic plot: #runs per generation for all series ---
    if annotate_counts:
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0 - xpad_left, max_len)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("# runs contributing")
        ax2.set_title("Contributing runs per generation")

        for label, s in series_dict.items():
            gen = np.asarray(s["gen"], dtype=float)
            n_at_gen = np.asarray(s["n_at_gen"], dtype=float) if "n_at_gen" in s else None
            if n_at_gen is None:
                continue
            order = np.argsort(gen)
            gen, n_at_gen = gen[order], n_at_gen[order]
            line2, = ax2.plot(gen, n_at_gen, label=label)
            c = colors_by_label.get(label, line2.get_color())

            # vline + stacked annotation on diagnostic too (optional: same order)
            if label in x0_map:
                x0_val = x0_map[label]
                ax2.axvline(x0_val, **{**vk, "color": c})

        ax2.legend()
        fig2.subplots_adjust(bottom=0.22)
        fig2.text(0.5, 0.04, description, ha="center", va="bottom", wrap=True)
        plt.show()

    plt.show()

def transform_function_string(text: str) -> str:
    words = text.split()
    new_words = []
    for word in words:
        if "_func" in word:
            word = word.replace("func", "Function")
        new_words.append(word)
    # replace underscores with spaces
    result = " ".join(new_words).replace("_", " ")
    # capitalize every word
    return result.title()