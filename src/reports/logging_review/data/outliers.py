"""Outlier scoring and strat prediction for logging review report (CLR-based)."""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold above which a sample is shown as outlier in scatter plots and report lists.
# ~3-sigma equivalent; only genuinely anomalous samples are highlighted.
OUTLIER_DISPLAY_THRESHOLD = 3.0


def _clr_transform(data: pd.DataFrame, fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Centered log-ratio (CLR) transform for compositional data.

    Geochemical data is compositional (e.g. percentages that sum to ~100%). CLR opens
    the simplex so that covariance and Mahalanobis distance are valid:
    clr(x_i) = log(x_i) - mean(log(x)) over the composition.

    Zeros and negatives are replaced with a small positive value before log.
    """
    out = data.copy().astype(float)
    if out.empty:
        return out
    if fill_value is None:
        positive = out[out > 0].min().min()
        fill_value = min(positive / 2.0, 1e-6) if pd.notna(positive) and positive > 0 else 1e-6
    out = out.replace(0, np.nan).fillna(fill_value)
    out = out.clip(lower=1e-12)  # ensure strictly positive for log
    log_data = np.log(out.values)
    gm = np.nanmean(log_data, axis=1, keepdims=True)
    gm = np.where(np.isnan(gm), 0.0, gm)
    clr_values = log_data - gm
    return pd.DataFrame(clr_values, index=data.index, columns=data.columns)


def _robust_scale(series: pd.Series) -> pd.Series:
    median = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return series * 0.0
        return (series - median) / std
    robust_sigma = iqr / 1.349
    return (series - median) / robust_sigma


def compute_hybrid_outlier_scores(
    df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Compute hybrid outlier scores per strat using CLR-transformed chemistry for
    Mahalanobis distance (compositional best practice) and IQR flags on raw values.

    Geochemical data is compositional; CLR (centered log-ratio) transform is applied
    before covariance/distance so results are not dominated by closure effects.

    Returns a DataFrame aligned to df with columns:
    - outlier_score
    - outlier_reason
    """
    result = pd.DataFrame(index=df.index)
    result["outlier_score"] = 0.0
    result["outlier_reason"] = ""
    result["outlier_elements"] = ""

    if strat_col not in df.columns:
        logger.warning("Strat column '%s' not found for outlier scoring.", strat_col)
        return result

    for strat_value, group in df.groupby(strat_col):
        if len(group) < min_group_size:
            continue

        available_cols = [c for c in chem_cols if c in group.columns]
        if not available_cols:
            continue

        # CLR transform for compositional data (Mahalanobis part)
        raw_block = group[available_cols].copy()
        try:
            clr_block = _clr_transform(raw_block.fillna(0.0))
        except Exception:
            clr_block = raw_block.apply(_robust_scale)
        clr_block = clr_block.replace([np.inf, -np.inf], np.nan)
        scaled = clr_block
        z_means = clr_block.mean(axis=0, skipna=True)
        z_centered = clr_block - z_means

        # Compute robust covariance with simple shrinkage to avoid instability
        cov = np.cov(z_centered.fillna(0.0).T, bias=True)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]
        if dim == 0:
            continue

        if len(group) < 10:
            alpha = 0.5
        elif len(group) < 30:
            alpha = 0.2
        else:
            alpha = 0.05
        cov_shrink = (1 - alpha) * cov + alpha * np.eye(dim)
        inv_cov = np.linalg.pinv(cov_shrink)

        mahal = []
        for idx, row in z_centered.iterrows():
            vec = row.values.astype(float)
            if np.all(np.isnan(vec)):
                mahal.append(0.0)
                continue
            mask = ~np.isnan(vec)
            vec = vec[mask]
            if vec.size == 0:
                mahal.append(0.0)
                continue
            sub_cov = inv_cov[np.ix_(mask, mask)]
            mahal.append(float(np.sqrt(vec.T @ sub_cov @ vec)))

        mahal = pd.Series(mahal, index=group.index)
        # Normalize by 95th percentile so scores are comparable across strats.
        # Score of 1.0 = 95th percentile, >1 = more extreme (no median centering).
        mahal_p95 = mahal.quantile(0.95)
        if mahal_p95 == 0 or np.isnan(mahal_p95):
            mahal_norm = mahal * 0.0
        else:
            mahal_norm = mahal / mahal_p95

        # IQR outlier flags and per-element severity
        flag_score = pd.Series(0.0, index=group.index)
        reason_text = pd.Series("", index=group.index)
        top_elements = pd.Series("", index=group.index)
        for col in available_cols:
            series = group[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            z = _robust_scale(series)
            is_outlier = (series < lower) | (series > upper)
            flag_score[is_outlier] += z[is_outlier].abs()
            for idx in series[is_outlier].index:
                current = reason_text.at[idx]
                direction = "high" if series.at[idx] > upper else "low"
                reason = f"{col} {direction} (value {series.at[idx]:.2f}, IQR {q1:.2f}-{q3:.2f})"
                reason_text.at[idx] = reason if current == "" else f"{current}; {reason}"

        # Track top elements by absolute robust z-score
        z_abs = scaled[available_cols].abs()
        for idx, row in z_abs.iterrows():
            top = row.sort_values(ascending=False).head(3).index.tolist()
            top_elements.at[idx] = ", ".join(top)

        result.loc[group.index, "outlier_score"] = (mahal_norm + flag_score).fillna(0.0)
        result.loc[group.index, "outlier_reason"] = reason_text.fillna("")
        result.loc[group.index, "outlier_elements"] = top_elements.fillna("")

    return result


def predict_most_likely_strat(
    df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_group_size: int = 10,
) -> pd.Series:
    """
    Predict the most likely strat classification for each row based on multivariate
    distance in CLR (centered log-ratio) space.

    Uses Mahalanobis distance to strat group centroids in CLR-transformed chemistry,
    so compositional closure does not dominate. The strat with the smallest distance
    is the "most likely" classification.

    Args:
        df: DataFrame with strat and chemistry columns
        strat_col: Name of the strat/classification column
        chem_cols: List of chemistry column names to use for distance calculation
        min_group_size: Minimum samples in a strat group to include it as a candidate

    Returns:
        Series with predicted "most likely" strat for each row
    """
    result = pd.Series(index=df.index, dtype=object)
    result[:] = "-"

    if strat_col not in df.columns:
        logger.warning("Strat column '%s' not found for prediction.", strat_col)
        return result

    available_cols = [c for c in chem_cols if c in df.columns]
    if not available_cols:
        logger.warning("No chemistry columns available for strat prediction.")
        return result

    # Build strat group statistics in CLR space (centroids and covariances)
    strat_stats = {}
    for strat_value, group in df.groupby(strat_col):
        if pd.isna(strat_value) or len(group) < min_group_size:
            continue

        raw_data = group[available_cols].fillna(0.0)
        if len(raw_data) < min_group_size:
            continue

        try:
            clr_data = _clr_transform(raw_data)
        except Exception:
            clr_data = raw_data.apply(lambda s: _robust_scale(s) if s.std() > 0 else s * 0)
        clr_data = clr_data.replace([np.inf, -np.inf], np.nan).fillna(0)

        centroid = clr_data.median()
        cov = np.cov(clr_data.T, bias=True)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]

        if dim == 0:
            continue

        alpha = 0.3 if len(clr_data) < 30 else 0.1
        cov_shrink = (1 - alpha) * cov + alpha * np.eye(dim)

        try:
            inv_cov = np.linalg.pinv(cov_shrink)
        except Exception:
            continue

        strat_stats[strat_value] = {
            "centroid": centroid,
            "inv_cov": inv_cov,
            "count": len(group),
        }

    if not strat_stats:
        return result

    # For each row, CLR-transform and find the strat with minimum Mahalanobis distance
    for idx, row in df.iterrows():
        row_raw = row[available_cols]
        if row_raw.isna().all():
            continue
        try:
            row_clr = _clr_transform(pd.DataFrame([row_raw.fillna(0.0)], index=[idx], columns=available_cols))
            row_values = row_clr.iloc[0]
        except Exception:
            row_values = row_raw.fillna(0.0)

        current_strat = row.get(strat_col)
        min_dist = float("inf")
        best_strat = current_strat if pd.notna(current_strat) else "-"

        for strat_value, stats in strat_stats.items():
            centroid = stats["centroid"]
            inv_cov = stats["inv_cov"]

            diff = row_values - centroid
            mask = ~diff.isna()
            if mask.sum() == 0:
                continue

            diff_vec = diff[mask].values.astype(float)
            mask_idx = np.where(mask.values)[0]
            sub_cov = inv_cov[np.ix_(mask_idx, mask_idx)]

            try:
                dist = float(np.sqrt(diff_vec.T @ sub_cov @ diff_vec))
            except Exception:
                continue

            if dist < min_dist:
                min_dist = dist
                best_strat = strat_value

        result.at[idx] = best_strat if pd.notna(best_strat) else "-"

    return result
