"""Plotly chart JSON generators for logging review HTML report."""
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.outliers import OUTLIER_DISPLAY_THRESHOLD


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex #rrggbb to rgba(r, g, b, alpha) for Plotly fill."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _convex_hull_2d(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return closed polygon (x, y) of the 2D convex hull (Gift wrapping / Jarvis march).
    Requires at least 3 non-collinear points; otherwise returns all points closed.
    """
    n = len(x)
    if n < 3:
        xc = np.append(x, x[0]) if n == 2 else x
        yc = np.append(y, y[0]) if n == 2 else y
        return xc, yc
    points = np.column_stack((x, y))
    # Start with leftmost (min x), then lowest y on tie
    start = np.lexsort((y, x))[0]
    hull = [start]
    current = start
    while True:
        next_idx = (current + 1) % n  # any index different from current
        for i in range(n):
            if i == current:
                continue
            # Cross product: (p - current) x (cand - current); >0 means cand is left of p
            dx1 = points[i, 0] - points[current, 0]
            dy1 = points[i, 1] - points[current, 1]
            dx2 = points[next_idx, 0] - points[current, 0]
            dy2 = points[next_idx, 1] - points[current, 1]
            cross = dx1 * dy2 - dy1 * dx2
            if cross > 0 or (cross == 0 and (dx1 * dx1 + dy1 * dy1) > (dx2 * dx2 + dy2 * dy2)):
                next_idx = i
        if next_idx == start:
            break
        hull.append(next_idx)
        current = next_idx
    hull = np.array(hull)
    x_hull = np.append(points[hull, 0], points[hull[0], 0])
    y_hull = np.append(points[hull, 1], points[hull[0], 1])
    return x_hull, y_hull


def _plotly_pie_json(labels: List[str], values: List[float], title: str) -> Tuple[str, str]:
    """Return (data_json, layout_json) for a Plotly pie chart."""
    colors = {"Match": "#2f7d61", "Mismatch": "#c9382a", "Pending Assays": "#5d6672"}
    data = [{
        "type": "pie",
        "labels": labels,
        "values": values,
        "marker": {"colors": [colors.get(label, "#5d6672") for label in labels]},
        "textinfo": "label+percent",
        "hovertemplate": "%{label}: %{value} (%{percent})<extra></extra>",
    }]
    layout = {
        "title": {"text": title},
        "margin": {"t": 40, "b": 20, "l": 20, "r": 20},
        "showlegend": True,
        "height": 280,
        "autosize": True,
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_stacked_bar_json(
    quarters: List[str], match: List[float], mismatch: List[float], pending: List[float], title: str
) -> Tuple[str, str]:
    """Return (data_json, layout_json) for stacked bar by quarter (absolute counts)."""
    data = [
        {"type": "bar", "x": quarters, "y": pending, "name": "Pending Assays", "marker": {"color": "#5d6672"}},
        {"type": "bar", "x": quarters, "y": mismatch, "name": "Mismatch", "marker": {"color": "#c9382a"}},
        {"type": "bar", "x": quarters, "y": match, "name": "Match", "marker": {"color": "#2f7d61"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "stack",
        "margin": {"t": 40, "b": 60},
        "height": 280,
        "autosize": True,
        "xaxis": {"type": "category", "tickangle": -45},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_stacked_bar_pct_json(
    quarters: List[str],
    match: List[float],
    mismatch: List[float],
    pending: List[float],
    title: str,
) -> Tuple[str, str]:
    """Stacked bar by quarter as % of quarter (volume-neutral). Each bar sums to 100%."""
    n = len(quarters)
    match_pct = [0.0] * n
    mismatch_pct = [0.0] * n
    pending_pct = [0.0] * n
    for i in range(n):
        total = (match[i] if i < len(match) else 0) + (mismatch[i] if i < len(mismatch) else 0) + (pending[i] if i < len(pending) else 0)
        if total > 0:
            match_pct[i] = 100.0 * (match[i] if i < len(match) else 0) / total
            mismatch_pct[i] = 100.0 * (mismatch[i] if i < len(mismatch) else 0) / total
            pending_pct[i] = 100.0 * (pending[i] if i < len(pending) else 0) / total
    data = [
        {"type": "bar", "x": quarters, "y": pending_pct, "name": "Pending Assays", "marker": {"color": "#5d6672"}},
        {"type": "bar", "x": quarters, "y": mismatch_pct, "name": "Mismatch", "marker": {"color": "#c9382a"}},
        {"type": "bar", "x": quarters, "y": match_pct, "name": "Match", "marker": {"color": "#2f7d61"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "stack",
        "margin": {"t": 40, "b": 60},
        "height": 280,
        "autosize": True,
        "xaxis": {"type": "category", "tickangle": -45},
        "yaxis": {"title": {"text": "% of quarter"}, "range": [0, 100], "ticksuffix": "%"},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_strat_grouped_bar_json(
    strat_codes: List[str],
    logger_counts: List[int],
    team_counts: List[int],
    title: str,
) -> Tuple[str, str]:
    """Grouped bar: Logger vs Team strat code counts (allows missing values – 0 when not used)."""
    data = [
        {"type": "bar", "x": strat_codes, "y": logger_counts, "name": "Logger", "marker": {"color": "#0d5b88"}},
        {"type": "bar", "x": strat_codes, "y": team_counts, "name": "Team", "marker": {"color": "#94a3b8"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 80},
        "height": 320,
        "autosize": True,
        "xaxis": {"tickangle": -45},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_strat_grouped_bar_pct_json(
    strat_codes: List[str],
    logger_counts: List[int],
    team_counts: List[int],
    title: str,
) -> Tuple[str, str]:
    """Grouped bar: Logger vs Team strat code as % of intervals (relative distribution)."""
    total_logger = max(sum(logger_counts), 1)
    total_team = max(sum(team_counts), 1)
    logger_pct = [x / total_logger * 100 for x in logger_counts]
    team_pct = [x / total_team * 100 for x in team_counts]
    data = [
        {"type": "bar", "x": strat_codes, "y": logger_pct, "name": "Logger", "marker": {"color": "#0d5b88"}},
        {"type": "bar", "x": strat_codes, "y": team_pct, "name": "Team", "marker": {"color": "#94a3b8"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 80},
        "height": 320,
        "autosize": True,
        "xaxis": {"tickangle": -45},
        "yaxis": {"title": {"text": "% of intervals"}},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_zonation_bar_json(
    categories: List[str], correct: List[int], incorrect: List[int], title: str
) -> Tuple[str, str]:
    """Clustered bar: Correctly vs Incorrectly logged by zonation category."""
    data = [
        {"type": "bar", "x": categories, "y": correct, "name": "Correct", "marker": {"color": "#2f7d61"}},
        {"type": "bar", "x": categories, "y": incorrect, "name": "Incorrect", "marker": {"color": "#c9382a"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 40},
        "height": 280,
        "autosize": True,
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_zonation_bar_pct_json(
    categories: List[str], correct: List[int], incorrect: List[int], title: str
) -> Tuple[str, str]:
    """Clustered bar: Correct % vs Incorrect % by zonation category (each category sums to 100%)."""
    n = len(categories)
    correct_pct = []
    incorrect_pct = []
    for i in range(n):
        c = correct[i] if i < len(correct) else 0
        inc = incorrect[i] if i < len(incorrect) else 0
        total = c + inc
        if total > 0:
            correct_pct.append(100.0 * c / total)
            incorrect_pct.append(100.0 * inc / total)
        else:
            correct_pct.append(0.0)
            incorrect_pct.append(0.0)
    data = [
        {"type": "bar", "x": categories, "y": correct_pct, "name": "Correct %", "marker": {"color": "#2f7d61"}},
        {"type": "bar", "x": categories, "y": incorrect_pct, "name": "Incorrect %", "marker": {"color": "#c9382a"}},
    ]
    layout = {
        "title": {"text": title},
        "barmode": "group",
        "margin": {"t": 40, "b": 40},
        "height": 280,
        "autosize": True,
        "yaxis": {"title": {"text": "%"}, "range": [0, 100], "ticksuffix": "%"},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_outlier_box_json(
    team_df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    max_strats: int = 20,
) -> Tuple[str, str]:
    """
    Return (data_json, layout_json) for a Plotly box plot: team geochem ranges by strat.
    One trace per element (Fe, SiO2, Al2O3, etc.); x = strat code, y = value.
    """
    if team_df.empty or not strat_col or strat_col not in team_df.columns or not chem_cols:
        return "[]", "{}"
    cols_in_df = [c for c in chem_cols if c in team_df.columns][:4]
    if not cols_in_df:
        return "[]", "{}"
    strats = team_df[strat_col].astype(str).dropna().unique().tolist()
    if len(strats) > max_strats:
        # Keep strats with most data
        counts = team_df[strat_col].astype(str).value_counts()
        strats = counts.head(max_strats).index.tolist()
    subset = team_df[team_df[strat_col].astype(str).isin(strats)].copy()
    if subset.empty:
        return "[]", "{}"
    data = []
    colors = ["#0d5b88", "#2f7d61", "#c9802a", "#5d6672"]
    short_names = {"fe_pct": "Fe %", "sio2_pct": "SiO2 %", "al2o3_pct": "Al2O3 %", "mgo_pct": "MgO %", "cao_pct": "CaO %", "s_pct": "S %", "p_pct": "P %"}
    for i, col in enumerate(cols_in_df):
        name = short_names.get(col.lower(), col)
        valid = subset[[strat_col, col]].dropna()
        if valid.empty:
            continue
        data.append({
            "type": "box",
            "x": valid[strat_col].astype(str).tolist(),
            "y": valid[col].tolist(),
            "name": name,
            "marker": {"color": colors[i % len(colors)]},
            "boxpoints": "outliers",
        })
    if not data:
        return "[]", "{}"
    layout = {
        "title": {"text": "Team geochem ranges by strat"},
        "boxmode": "group",
        "margin": {"t": 40, "b": 80, "l": 50, "r": 20},
        "height": 320,
        "autosize": True,
        "xaxis": {"tickangle": -45, "type": "category"},
        "yaxis": {"title": {"text": "%"}},
        "showlegend": True,
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_pca_biplot_json(
    assay_df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    max_points: int = 2000,
    min_strat_count: int = 10,
) -> Tuple[str, str]:
    """
    Return (data_json, layout_json) for a PCA biplot in CLR space.

    Points colored by strat, with loading vectors showing which elements
    drive separation. Outliers (if scored) shown with distinct marker.
    Hulls are drawn only for strats with at least min_strat_count samples; all strats' points are still shown.
    """
    if assay_df.empty or not strat_col or strat_col not in assay_df.columns or not chem_cols:
        return "[]", "{}"

    cols_in_df = [c for c in chem_cols if c in assay_df.columns]
    if len(cols_in_df) < 3:
        return "[]", "{}"

    # Prepare data; use display_outlier (strat-mismatch only) when present so charts highlight reviewable intervals
    has_outlier = "outlier_score" in assay_df.columns
    has_display_outlier = "display_outlier" in assay_df.columns
    keep_cols = [strat_col] + cols_in_df
    if has_outlier:
        keep_cols.append("outlier_score")
    if has_display_outlier:
        keep_cols.append("display_outlier")
    df = assay_df[keep_cols].dropna(subset=cols_in_df)
    if df.empty or len(df) < 10:
        return "[]", "{}"

    # CLR transform (same as outlier scoring)
    raw = df[cols_in_df].copy().astype(float)
    raw = raw.clip(lower=1e-6)  # Replace zeros
    log_data = np.log(raw.values)
    gm = log_data.mean(axis=1, keepdims=True)
    clr = log_data - gm

    # PCA via SVD
    clr_centered = clr - clr.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(clr_centered, full_matrices=False)
        pc1 = U[:, 0] * S[0]
        pc2 = U[:, 1] * S[1]
        loadings = Vt[:2, :].T  # (n_features, 2)
        var_explained = (S[:2] ** 2) / (S ** 2).sum() * 100
    except Exception:
        return "[]", "{}"

    df = df.copy()
    df["PC1"] = pc1
    df["PC2"] = pc2

    # Subsample if too large
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    # Highlight only strat-mismatch outliers when display_outlier present, else score > threshold
    if has_display_outlier and "display_outlier" in df.columns:
        is_outlier = df["display_outlier"].fillna(False).astype(bool)
    else:
        is_outlier = df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD if has_outlier else pd.Series(False, index=df.index)
    non_outliers = df[~is_outlier]
    outliers_df = df[is_outlier]

    strats = non_outliers[strat_col].astype(str).unique().tolist()
    data = []
    palette = ["#0d5b88", "#2f7d61", "#c9802a", "#5d6672", "#94a3b8", "#64748b", "#0f766e", "#b45309",
               "#7c3aed", "#db2777", "#059669", "#d97706", "#4f46e5", "#be185d"]

    # Convex hull outlines only for strats with enough samples (hulls need min_strat_count; still need 3+ points)
    for idx, strat in enumerate(strats):
        sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
        if len(sub) < 3 or len(sub) < min_strat_count:
            continue
        x_pts = sub["PC1"].values
        y_pts = sub["PC2"].values
        x_hull, y_hull = _convex_hull_2d(x_pts, y_pts)
        color = palette[idx % len(palette)]
        data.append({
            "type": "scatter",
            "mode": "lines",
            "x": x_hull.tolist(),
            "y": y_hull.tolist(),
            "name": str(strat),
            "line": {"color": color, "width": 2, "dash": "solid"},
            "fill": "toself",
            "fillcolor": _hex_to_rgba(color, 0.12),
            "showlegend": False,
            "hoverinfo": "skip",
        })

    for idx, strat in enumerate(strats):
        sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
        if sub.empty:
            continue
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": sub["PC1"].tolist(),
            "y": sub["PC2"].tolist(),
            "name": str(strat),
            "marker": {"size": 6, "color": palette[idx % len(palette)], "opacity": 0.7},
            "hovertemplate": f"{strat}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
        })

    # Outliers
    if not outliers_df.empty:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": outliers_df["PC1"].tolist(),
            "y": outliers_df["PC2"].tolist(),
            "name": "Outliers",
            "marker": {"size": 10, "symbol": "x", "color": "#dc2626", "line": {"width": 2}},
            "hovertemplate": "Outlier<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        })

    # Loading vectors (scaled for visibility)
    max_pc = max(abs(pc1).max(), abs(pc2).max(), 1)
    load_scale = max_pc * 0.6 / max(abs(loadings).max(), 0.01)
    short_names = {"fe_pct": "Fe", "sio2_pct": "SiO2", "al2o3_pct": "Al2O3",
                   "mgo_pct": "MgO", "cao_pct": "CaO", "s_pct": "S", "p_pct": "P",
                   "loi_pct": "LOI", "mn_pct": "Mn", "tio2_pct": "TiO2", "k2o_pct": "K2O", "na2o_pct": "Na2O"}

    for i, col in enumerate(cols_in_df):
        lx, ly = loadings[i, 0] * load_scale, loadings[i, 1] * load_scale
        label = short_names.get(col.lower(), col.replace("_pct", "").replace("_", " "))
        # Arrow line
        data.append({
            "type": "scatter",
            "mode": "lines",
            "x": [0, lx],
            "y": [0, ly],
            "line": {"color": "#374151", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip",
        })
        # Label at arrow tip
        data.append({
            "type": "scatter",
            "mode": "text",
            "x": [lx * 1.15],
            "y": [ly * 1.15],
            "text": [label],
            "textfont": {"size": 11, "color": "#1f2937", "weight": "bold"},
            "showlegend": False,
            "hoverinfo": "skip",
        })

    if not data:
        return "[]", "{}"

    layout = {
        "title": {"text": f"PCA biplot (CLR) — PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%"},
        "margin": {"t": 50, "b": 60, "l": 50, "r": 20},
        "height": 450,
        "autosize": True,
        "xaxis": {"title": {"text": f"PC1 ({var_explained[0]:.1f}%)"}, "zeroline": True, "zerolinecolor": "#d1d5db"},
        "yaxis": {"title": {"text": f"PC2 ({var_explained[1]:.1f}%)"}, "zeroline": True, "zerolinecolor": "#d1d5db", "scaleanchor": "x"},
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "top", "y": -0.12, "xanchor": "center", "x": 0.5},
    }
    return json.dumps(data), json.dumps(layout)


def _plotly_outlier_scatter_grid_json(
    assay_df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_strat_count: int = 10,
    max_pairs: int = 6,
    max_points_per_plot: int = 500,
) -> Tuple[str, str]:
    """
    Return (data_json, layout_json) for a 2x3 grid of scatter plots (axis pairs)
    with convex hulls per strat. Each subplot uses different x/y chemistry columns,
    so aspect ratios are natural per pair. Hulls are drawn only for strats with at
    least min_strat_count samples; all strats' points are still shown.
    """
    if assay_df.empty or not strat_col or strat_col not in assay_df.columns or not chem_cols:
        return "[]", "{}"
    cols_in_df = [c for c in chem_cols if c in assay_df.columns]
    if len(cols_in_df) < 2:
        return "[]", "{}"

    # Build axis pairs from first 4 cols: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    n_cols = min(4, len(cols_in_df))
    pairs = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pairs.append((cols_in_df[i], cols_in_df[j]))
    pairs = pairs[:max_pairs]
    if not pairs:
        return "[]", "{}"

    has_outlier = "outlier_score" in assay_df.columns
    has_display_outlier = "display_outlier" in assay_df.columns
    all_cols = list(dict.fromkeys([strat_col] + [c for p in pairs for c in p] + (["outlier_score"] if has_outlier else []) + (["display_outlier"] if has_display_outlier else [])))
    df = assay_df[all_cols].dropna(subset=[c for p in pairs for c in p])
    if df.empty:
        return "[]", "{}"

    if has_display_outlier and "display_outlier" in df.columns:
        is_outlier = df["display_outlier"].fillna(False).astype(bool)
    else:
        is_outlier = df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD if has_outlier else pd.Series(False, index=df.index)
    non_outliers = df[~is_outlier]
    strats = non_outliers[strat_col].astype(str).unique().tolist()

    palette = ["#0d5b88", "#2f7d61", "#c9802a", "#5d6672", "#94a3b8", "#64748b", "#0f766e", "#b45309",
               "#7c3aed", "#db2777", "#059669", "#d97706", "#4f46e5", "#be185d"]
    short_names = {"fe_pct": "Fe", "sio2_pct": "SiO2", "al2o3_pct": "Al2O3", "mgo_pct": "MgO",
                   "cao_pct": "CaO", "s_pct": "S", "p_pct": "P", "loi_pct": "LOI", "mn_pct": "Mn"}

    rows, cols = 2, 3
    n_subplots = min(len(pairs), rows * cols)
    # Subplot domains: 2 rows x 3 cols, with small gaps (no overlap)
    x_gap, y_gap = 0.02, 0.04
    cell_w = (1.0 - (cols + 1) * x_gap) / cols
    cell_h = (1.0 - (rows + 1) * y_gap) / rows
    # Row 0: top half; row 1: bottom half
    def y_dom_for_row(r: int) -> List[float]:
        if r == 0:
            return [1.0 - y_gap - cell_h, 1.0 - y_gap]
        return [y_gap, y_gap + cell_h]

    layout = {
        "margin": {"t": 44, "b": 28, "l": 44, "r": 20},
        "height": 420,
        "autosize": True,
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "top", "y": -0.08, "xanchor": "center", "x": 0.5},
    }

    data = []
    for subplot_idx, (x_col, y_col) in enumerate(pairs):
        if subplot_idx >= n_subplots:
            break
        row, col = subplot_idx // cols, subplot_idx % cols
        x_dom = [x_gap + col * (cell_w + x_gap), x_gap + col * (cell_w + x_gap) + cell_w]
        y_dom = y_dom_for_row(row)
        # Plotly: layout keys are xaxis, yaxis, xaxis2, yaxis2; trace refs are x, y, x2, y2
        x_layout_key = "xaxis" if subplot_idx == 0 else f"xaxis{subplot_idx + 1}"
        y_layout_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
        x_trace_ref = "x" if subplot_idx == 0 else f"x{subplot_idx + 1}"
        y_trace_ref = "y" if subplot_idx == 0 else f"y{subplot_idx + 1}"

        layout[x_layout_key] = {"domain": x_dom, "anchor": y_trace_ref, "title": {"text": short_names.get(x_col.lower(), x_col.replace("_pct", ""))}}
        layout[y_layout_key] = {"domain": y_dom, "anchor": x_trace_ref, "title": {"text": short_names.get(y_col.lower(), y_col.replace("_pct", ""))}}

        # Hulls only for strats with enough samples
        for idx, strat in enumerate(strats):
            sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
            if len(sub) < 3 or len(sub) < min_strat_count:
                continue
            x_pts = sub[x_col].values
            y_pts = sub[y_col].values
            x_hull, y_hull = _convex_hull_2d(x_pts, y_pts)
            color = palette[idx % len(palette)]
            data.append({
                "type": "scatter", "mode": "lines",
                "x": x_hull.tolist(), "y": y_hull.tolist(),
                "xaxis": x_trace_ref, "yaxis": y_trace_ref,
                "line": {"color": color, "width": 1.5}, "fill": "toself", "fillcolor": _hex_to_rgba(color, 0.12),
                "showlegend": False, "hoverinfo": "skip",
            })

        # Points (show legend only for first subplot to avoid duplicates)
        show_leg = subplot_idx == 0
        for idx, strat in enumerate(strats):
            sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
            if sub.empty:
                continue
            if len(sub) > max_points_per_plot:
                sub = sub.sample(n=max_points_per_plot, random_state=42)
            data.append({
                "type": "scatter", "mode": "markers",
                "x": sub[x_col].tolist(), "y": sub[y_col].tolist(),
                "xaxis": x_trace_ref, "yaxis": y_trace_ref,
                "name": str(strat),
                "marker": {"size": 5, "color": palette[idx % len(palette)], "opacity": 0.7},
                "hovertemplate": f"{strat} %{{x:.2f}}, %{{y:.2f}}<extra></extra>",
                "showlegend": show_leg,
            })

        # Outliers for this pair
        out_sub = df[is_outlier].dropna(subset=[x_col, y_col])
        if not out_sub.empty:
            data.append({
                "type": "scatter", "mode": "markers",
                "x": out_sub[x_col].tolist(), "y": out_sub[y_col].tolist(),
                "xaxis": x_trace_ref, "yaxis": y_trace_ref,
                "name": "Outliers",
                "marker": {"size": 9, "symbol": "x", "color": "#dc2626", "line": {"width": 1.5}},
                "hovertemplate": "%{x:.2f}, %{y:.2f} (outlier)<extra></extra>",
                "showlegend": show_leg,
            })

    if not data:
        return "[]", "{}"
    return json.dumps(data), json.dumps(layout)


def _plotly_outlier_scatter_json(
    assay_df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    max_points: int = 800,
    min_strat_count: int = 10,
) -> Tuple[str, str]:
    """
    Return (data_json, layout_json) for a single Plotly scatter (one axis pair).
    Kept for compatibility; prefer _plotly_outlier_scatter_grid_json for the report.
    """
    if assay_df.empty or not strat_col or strat_col not in assay_df.columns or not chem_cols:
        return "[]", "{}"
    cols_in_df = [c for c in chem_cols if c in assay_df.columns]
    if len(cols_in_df) < 2:
        return "[]", "{}"
    x_col = x_col or cols_in_df[0]
    y_col = y_col or cols_in_df[1]
    if x_col not in assay_df.columns or y_col not in assay_df.columns:
        return "[]", "{}"
    has_outlier = "outlier_score" in assay_df.columns
    df = assay_df[[strat_col, x_col, y_col] + (["outlier_score"] if has_outlier else [])].dropna(subset=[x_col, y_col])
    if df.empty:
        return "[]", "{}"
    is_outlier = df["outlier_score"] > OUTLIER_DISPLAY_THRESHOLD if has_outlier else pd.Series(False, index=df.index)
    non_outliers = df[~is_outlier]
    strats = non_outliers[strat_col].astype(str).unique().tolist()
    data = []
    palette = ["#0d5b88", "#2f7d61", "#c9802a", "#5d6672", "#94a3b8", "#64748b", "#0f766e", "#b45309"]
    for idx, strat in enumerate(strats):
        sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=42)
        if sub.empty:
            continue
        data.append({
            "type": "scatter", "mode": "markers",
            "x": sub[x_col].tolist(), "y": sub[y_col].tolist(),
            "name": str(strat),
            "marker": {"size": 6, "color": palette[idx % len(palette)], "opacity": 0.7},
            "hovertemplate": "%{x:.2f}, %{y:.2f}<extra></extra>",
        })
    if not df[is_outlier].empty:
        data.append({
            "type": "scatter", "mode": "markers",
            "x": df.loc[is_outlier, x_col].tolist(), "y": df.loc[is_outlier, y_col].tolist(),
            "name": "Outliers",
            "marker": {"size": 12, "symbol": "x", "color": "#c9382a", "line": {"width": 2}},
            "hovertemplate": "%{x:.2f}, %{y:.2f} (outlier)<extra></extra>",
        })
    if not data:
        return "[]", "{}"
    x_label = x_col.replace("_pct", " %").replace("_", " ").title()
    y_label = y_col.replace("_pct", " %").replace("_", " ").title()
    layout = {
        "title": {"text": "Geochem clusters and outlier positions"},
        "margin": {"t": 40, "b": 50, "l": 50, "r": 20},
        "height": 360,
        "autosize": True,
        "xaxis": {"title": {"text": x_label}},
        "yaxis": {"title": {"text": y_label}},
        "showlegend": True,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.15},
    }
    return json.dumps(data), json.dumps(layout)


