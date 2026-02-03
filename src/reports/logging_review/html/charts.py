"""Plotly chart JSON generators for logging review HTML report."""
import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..data.outliers import OUTLIER_DISPLAY_THRESHOLD

def _plotly_pie_json(labels: List[str], values: List[float], title: str) -> Tuple[str, str]:
    """Return (data_json, layout_json) for a Plotly pie chart."""
    colors = {"Match": "#2f7d61", "Mismatch": "#c9382a", "Pending Assays": "#5d6672"}
    data = [{
        "type": "pie",
        "labels": labels,
        "values": values,
        "marker": {"colors": [colors.get(l, "#5d6672") for l in labels]},
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


def _plotly_outlier_scatter_json(
    assay_df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    max_points: int = 800,
) -> Tuple[str, str]:
    """
    Return (data_json, layout_json) for a Plotly scatter: Fe vs SiO2 (or first two chem cols),
    points coloured by strat, outliers highlighted with distinct marker.
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
    outliers_df = df[is_outlier]
    data = []
    strats = non_outliers[strat_col].astype(str).unique().tolist()
    palette = ["#0d5b88", "#2f7d61", "#c9802a", "#5d6672", "#94a3b8", "#64748b", "#0f766e", "#b45309"]
    for idx, strat in enumerate(strats):
        sub = non_outliers[non_outliers[strat_col].astype(str) == strat]
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=42)
        if sub.empty:
            continue
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": sub[x_col].tolist(),
            "y": sub[y_col].tolist(),
            "name": str(strat),
            "marker": {"size": 6, "color": palette[idx % len(palette)], "opacity": 0.7},
            "hovertemplate": f"%{{x:.2f}}, %{{y:.2f}}<extra></extra>",
        })
    if not outliers_df.empty:
        data.append({
            "type": "scatter",
            "mode": "markers",
            "x": outliers_df[x_col].tolist(),
            "y": outliers_df[y_col].tolist(),
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


