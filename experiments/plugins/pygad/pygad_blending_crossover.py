import numpy as np
import pygad

def _get_per_gene(arr_or_scalar, n):
    if arr_or_scalar is None:
        return [None] * n
    if np.isscalar(arr_or_scalar):
        return [arr_or_scalar] * n
    if len(arr_or_scalar) != n:
        raise ValueError(f"Expected {n} step values, got {len(arr_or_scalar)}.")
    return list(arr_or_scalar)

def _extract_bounds_and_steps_from_gene_space(ga_instance, n_genes):
    lows  = [None] * n_genes
    highs = [None] * n_genes
    steps = [None] * n_genes

    gs = getattr(ga_instance, "gene_space", None)
    if gs is None:
        return lows, highs, steps

    # PyGAD allows per-gene gene_space (list); we only use dict entries with low/high/step
    if isinstance(gs, list) and len(gs) == n_genes:
        for i, spec in enumerate(gs):
            if isinstance(spec, dict):
                lows[i]  = spec.get("low",  lows[i])
                highs[i] = spec.get("high", highs[i])
                steps[i] = spec.get("step", steps[i])
    elif isinstance(gs, dict):
        # same dict applied to all genes
        for i in range(n_genes):
            lows[i]  = gs.get("low",  lows[i])
            highs[i] = gs.get("high", highs[i])
            steps[i] = gs.get("step", steps[i])

    return lows, highs, steps

def _snap_to_step(x, step, base=None):
    if step is None or step == 0:
        return x
    if base is None:
        base = 0.0
    return base + round((x - base) / step) * step

def blend_crossover_pygad(parents, offspring_size, ga_instance):
    """
    BLX-α crossover with per-gene step quantization and optional bounds.
    Read per-gene steps from:
      1) ga_instance.crossover_steps (scalar or list), or
      2) ga_instance.gene_space[i]['step'] if present.
    Bounds (low/high) are taken from gene_space if provided.
    """
    alpha = getattr(ga_instance, "blend_alpha", 0.5)  # allow overriding via attribute
    n_offspring, n_genes = offspring_size
    offspring = np.empty(offspring_size, dtype=float)

    # Gather bounds & default steps from gene_space
    lows, highs, gs_steps = _extract_bounds_and_steps_from_gene_space(ga_instance, n_genes)

    # User-provided steps (override gene_space steps if given)
    user_steps = getattr(ga_instance, "crossover_steps", None)
    steps = _get_per_gene(user_steps, n_genes) if user_steps is not None else gs_steps

    for k in range(n_offspring):
        p1 = parents[k % parents.shape[0], :]
        p2 = parents[(k + 1) % parents.shape[0], :]
        child = np.empty(n_genes, dtype=float)

        for j in range(n_genes):
            a, b = p1[j], p2[j]
            d = abs(a - b)
            low_blx  = min(a, b) - alpha * d
            high_blx = max(a, b) + alpha * d

            # Sample within BLX interval
            val = np.random.uniform(low_blx, high_blx)

            # Snap to per-gene step grid.
            # Choose a sensible base for quantization:
            #   - Prefer the lower bound if provided,
            #   - else use the smaller parent gene value (keeps grid aligned to parents).
            base = lows[j] if lows[j] is not None else min(a, b)
            val = _snap_to_step(val, steps[j], base=base)

            # Enforce bounds if present
            if lows[j] is not None:
                val = max(val, lows[j])
            if highs[j] is not None:
                val = min(val, highs[j])

            child[j] = val

        offspring[k, :] = child

    return offspring
