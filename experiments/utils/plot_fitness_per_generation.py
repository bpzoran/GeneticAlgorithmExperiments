# ---- MUST be first: pick a headless-safe backend BEFORE importing pyplot ----
import os

# Respect a user-specified backend if set; otherwise default to Agg (safe for RDP/headless).
if "MPLBACKEND" not in os.environ:
    # Setting via matplotlib.use() must happen before importing pyplot.
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend avoids Tk/Qt crashes on headless/RDP
else:
    import matplotlib  # still import to access get_backend later if needed

# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

from utils.experiment_utils import transform_function_string

def compute_afi_percent(metrics_by_strategy, baseline, ref="diversity mutation", eps=1e-12):
    """
    AFI (%) = ((f_base_min - f_div_min) / (f_first - f_div_min)) * 100
    where:
      f_first    = metrics_by_strategy[ref]['avg_fitness_after_first']
      f_div_min  = metrics_by_strategy[ref]['avg_min_fitness']
      f_base_min = metrics_by_strategy[baseline]['avg_min_fitness']
    """
    f_first   = float(metrics_by_strategy[ref]["avg_fitness_after_first"])
    f_div_min = float(metrics_by_strategy[ref]["avg_min_fitness"])
    f_base_min= float(metrics_by_strategy[baseline]["avg_min_fitness"])

    denom = f_first - f_div_min
    if abs(denom) < eps:
        return np.nan  # undefined or negligible margin from first gen to div minimum

    return 100.0 * (f_base_min - f_div_min) / denom

def plot_convergence_curve(
    agg,
    x0,
    lowest: float,
    highest: float,
    max_len: float,
    #description: str = "GA Convergence (central tendency ± variability)",
    description: str = "GA Convergence (central tendency)",
    ylabel: str = "Fitness",
    xlabel: str = "Generation",
    annotate_counts: bool = True,  # plot diagnostic figure with #runs per generation
    vline_kw: dict | None = None,
    text_kw: dict | None = None,
    # --- NEW ---
    save: bool = False,
    outdir: str | None = None,
    basename: str = "convergence",
    formats: tuple[str, ...] = ("png",),
    metrics_by_strategy: dict | None = None,
):
    """
    agg:
      - single: {"gen","center","lower","upper"}
      - multi:  {label: {"gen","center","lower","upper","n_at_gen"(opt)}}
    x0:
      - single: float
      - multi:  {label: float}

    Saving:
      If save=True, figures are saved to `outdir` using `basename` and `formats`
      (e.g., basename.png, basename_counts.png) and not shown. Returns the list
      of saved file paths. If save=False, figures are shown and [] is returned.
    """
    description = transform_function_string(description)
    len_border = round(max(x0.values())) + 1

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
    xpad_left = round(len_border / 20)
    ypad_bottom = (highest - lowest) / 20
    ax.set_xlim(0 - xpad_left, len_border)
    ax.set_ylim(lowest - ypad_bottom, highest)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("GA Convergence (central tendency)")

    # --- Shade the REC window: first n = 25% of the smallest avg-generations ---
    min_avg_gens = min(x0_map.values()) if x0_map else None
    if min_avg_gens is not None:
        n_rec = int(round(0.25 * min_avg_gens))
        if n_rec > 0:
            ax.axvspan(0, n_rec, alpha=0.08, color='0.5', label="_nolegend_")
            ax.text(n_rec, ax.get_ylim()[0], f" REC window: 0–{n_rec} gen",
                    va='bottom', ha='right', fontsize=8, color='0.3')

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

        # After plotting each strategy's main line:
        final_x0 = float(gen[-1])  # that series' last available generation
        final_y = float(center[-1])  # that series' final mean

        # draw a short dotted segment to the right, e.g., +5% of the axis span but capped
        x_left = final_x0 - 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        x_right = final_x0 + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        # ensure we don't overshoot the axis
        x_left = max(x_left, ax.get_xlim()[0])
        x_right = min(x_right, ax.get_xlim()[1])

        ax.hlines(final_y, x_left, x_right, linestyles='dotted', linewidth=1.5, color=line_color)


    ax.legend()

    # ---- Stack "Avg gen =" annotations so they don't overlap ----
    if vline_info:
        vline_info.sort(key=lambda t: t[1])  # left -> right
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        top_anchor_frac = 0.95
        stack_band_frac = 0.25
        n = len(vline_info)
        dy_frac = stack_band_frac / max(n, 1)

        for i, (label, x0_val, color) in enumerate(vline_info):
            y_text = y_min + (top_anchor_frac - i * dy_frac) * y_range
            x_min, x_max = ax.get_xlim()
            place_left = x0_val > (x_min + 0.85 * (x_max - x_min))
            dx = -6 if place_left else 6
            ax.annotate(f"Avg gen = {round(x0_val):g}",
                        xy=(x0_val, y_text), xytext=(dx, 0),
                        textcoords='offset points',
                        color=color, ha='left', va='center')
    afi_vs_adaptive = compute_afi_percent(metrics_by_strategy, baseline="adaptive mutation")
    afi_vs_random = compute_afi_percent(metrics_by_strategy, baseline="random mutation")

    afi_text = (
        f"AFI vs adaptive: {afi_vs_adaptive:.2f}%\n"
        f"AFI vs random:   {afi_vs_random:.2f}%"
    )
    # After you've drawn the curves and set titles/labels:
    ax.text(
        0.01, 0.99, afi_text,
        transform=ax.transAxes, va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9)
    )

    saved_paths: list[str] = []

    # --- Diagnostic plot: #runs per generation for all series ---
    fig2, ax2 = None, None
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
            if label in x0_map:
                x0_val = x0_map[label]
                ax2.axvline(x0_val, **{**vk, "color": c})

        ax2.legend()
        fig2.subplots_adjust(bottom=0.22)
        fig2.text(0.5, 0.04, description, ha="center", va="bottom", wrap=True)

    # --- Save or Show ---
    if save:
        outdir_final = outdir or os.getcwd()
        os.makedirs(outdir_final, exist_ok=True)

        # sanitize basename lightly (no path separators)
        safe_base = basename.replace(os.sep, "_").replace("/", "_")

        # main figure
        for ext in formats:
            path = os.path.join(outdir_final, f"{safe_base}.{ext.lstrip('.')}")
            fig.savefig(path, dpi=200, bbox_inches="tight")
            saved_paths.append(path)

        # diagnostic figure
        if annotate_counts and fig2 is not None:
            for ext in formats:
                path = os.path.join(outdir_final, f"{safe_base}_counts.{ext.lstrip('.')}")
                fig2.savefig(path, dpi=200, bbox_inches="tight")
                saved_paths.append(path)
                try:
                    import csv
                    for label, s in series_dict.items():
                        data_path = os.path.join(outdir_final, f"{safe_base}_{label.replace(' ', '_')}.csv")
                        with open(data_path, "w", newline="") as f:
                            w = csv.writer(f)
                            w.writerow(["gen", "center", "lower", "upper",
                                        "n_at_gen" if "n_at_gen" in s else "n_at_gen (NA)"])
                            ncol_default = np.full_like(s["gen"], 0)
                            ncol = s.get("n_at_gen", ncol_default)
                            for i in range(len(s["gen"])):
                                w.writerow([s["gen"][i], s["center"][i], s["lower"][i], s["upper"][i],
                                            ncol[i] if i < len(ncol) else ""])
                        saved_paths.append(data_path)
                except Exception:
                    pass

        plt.close(fig)
        if fig2 is not None:
            plt.close(fig2)
        return saved_paths

    # default behavior: show (no-op on Agg, but safe)
    if annotate_counts and fig2 is not None:
        plt.show()
    plt.show()
    return []
