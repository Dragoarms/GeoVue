"""Outlier scoring and strat prediction for logging review report (CLR-based)."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold above which a sample is shown as outlier in scatter plots and report lists.
# Scores are ratios to each component's robust upper fence (Q3 + 1.5 IQR).
# A score above 1.0 means either the multivariate CLR distance or strongest
# univariate robust-z signal exceeds its own reference for that strat.
OUTLIER_DISPLAY_THRESHOLD = 1.0


def _clr_transform(data: pd.DataFrame, fill_value: Optional[float] = None) -> pd.DataFrame:
    """
    Centered log-ratio (CLR) transform for compositional data.

    Geochemical data is compositional (e.g. percentages that sum to ~100%). CLR opens
    the simplex so that covariance and Mahalanobis distance are valid:
    clr(x_i) = log(x_i) - mean(log(x)) over the composition.

    Zeros and negatives are replaced with a scale-relative positive value before log.
    Missing values remain missing and are excluded from the row geometric mean.
    """
    out = data.copy().astype(float)
    if out.empty:
        return out

    if fill_value is None:
        positive = out.where(out > 0)
        positive_values = positive.stack()
        global_positive = positive_values.min() if not positive_values.empty else np.nan
        fallback = global_positive / 2.0 if pd.notna(global_positive) and global_positive > 0 else 1e-6
        col_fill = positive.min(axis=0) / 2.0
        col_fill = col_fill.where((col_fill > 0) & col_fill.notna(), fallback).fillna(fallback)
    else:
        col_fill = pd.Series(float(fill_value), index=out.columns)

    out = out.mask(out <= 0, col_fill, axis=1)
    out = out.clip(lower=1e-12)

    values = out.to_numpy(dtype=float)
    log_data = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    log_data[valid] = np.log(values[valid])

    valid_counts = np.isfinite(log_data).sum(axis=1, keepdims=True)
    log_sums = np.nansum(log_data, axis=1, keepdims=True)
    gm = np.divide(
        log_sums,
        valid_counts,
        out=np.full_like(log_sums, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    clr_values = log_data - gm
    clr_values[~np.isfinite(log_data)] = np.nan
    return pd.DataFrame(clr_values, index=data.index, columns=data.columns)


def _robust_scale(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = numeric.median()
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        std = numeric.std(ddof=0)
        if std == 0 or np.isnan(std):
            return numeric * 0.0
        return (numeric - median) / std
    robust_sigma = iqr / 1.349
    return (numeric - median) / robust_sigma


def _robust_upper_reference(series: pd.Series) -> float:
    """Return a positive robust upper-fence reference for ratio-scaled scores."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 1.0
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0 and not np.isnan(iqr):
        ref = q3 + 1.5 * iqr
    else:
        median = values.median()
        std = values.std(ddof=0)
        ref = median + 3.0 * std if std > 0 and not np.isnan(std) else values.max()
    if not np.isfinite(ref) or ref <= 0:
        ref = values.max()
    return float(ref) if np.isfinite(ref) and ref > 0 else 1.0


def _shrink_covariance(cov: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink covariance toward a scaled identity target, preserving data scale."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    dim = cov.shape[0]
    if dim == 0:
        return cov
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.diag(cov)
    scale = float(np.nanmean(diag)) if diag.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (1.0 - alpha) * cov + alpha * scale * np.eye(dim)


def _mahalanobis_distance_masked(
    diff: np.ndarray,
    covariance: np.ndarray,
    present_mask: np.ndarray,
) -> float:
    """Mahalanobis distance using the covariance restricted to present dimensions."""
    present_mask = np.asarray(present_mask, dtype=bool)
    if not present_mask.any():
        return float("nan")
    diff_vec = np.asarray(diff, dtype=float)[present_mask]
    sub_cov = np.asarray(covariance, dtype=float)[np.ix_(present_mask, present_mask)]
    inv_sub_cov = np.linalg.pinv(sub_cov)
    dist_sq = float(diff_vec.T @ inv_sub_cov @ diff_vec)
    return float(np.sqrt(max(dist_sq, 0.0)))


def compute_hybrid_outlier_scores(
    df: pd.DataFrame,
    strat_col: str,
    chem_cols: List[str],
    min_group_size: int = 10,
) -> pd.DataFrame:
    """
    Compute hybrid outlier scores per strat.

    The multivariate component is Mahalanobis distance in CLR-transformed chemistry.
    The univariate component is the maximum absolute robust z-score across raw assay
    columns. Each component is divided by its own robust upper fence (Q3 + 1.5 IQR)
    within the strat group, so a score of 1.0 means "beyond the robust upper
    reference" on either scale.

    Returns a DataFrame aligned to df with columns:
    - outlier_score
    - outlier_mahal_score
    - outlier_univariate_score
    - outlier_reason
    - outlier_elements
    """
    result = pd.DataFrame(index=df.index)
    result["outlier_score"] = 0.0
    result["outlier_mahal_score"] = 0.0
    result["outlier_univariate_score"] = 0.0
    result["outlier_reason"] = ""
    result["outlier_elements"] = ""

    if strat_col not in df.columns:
        logger.warning("Strat column '%s' not found for outlier scoring.", strat_col)
        return result

    for _strat_value, group in df.groupby(strat_col):
        if len(group) < min_group_size:
            continue

        available_cols = [c for c in chem_cols if c in group.columns]
        if not available_cols:
            continue

        raw_block = group[available_cols].apply(pd.to_numeric, errors="coerce")
        try:
            clr_block = _clr_transform(raw_block)
        except Exception:
            clr_block = raw_block.apply(_robust_scale)
        clr_block = clr_block.replace([np.inf, -np.inf], np.nan)
        z_means = clr_block.mean(axis=0, skipna=True)
        z_centered = clr_block - z_means

        # Missing dimensions are mean-imputed after centering, contributing no
        # distance on that dimension.
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
        cov_shrink = _shrink_covariance(cov, alpha)
        inv_cov = np.linalg.pinv(cov_shrink)

        X = z_centered.fillna(0.0).values
        mahal_sq = np.einsum("ij,jk,ik->i", X, inv_cov, X)
        mahal_arr = np.sqrt(np.maximum(mahal_sq, 0.0))
        all_nan = z_centered.isna().all(axis=1).values
        mahal_arr[all_nan] = 0.0
        mahal = pd.Series(mahal_arr, index=group.index)
        mahal_norm = mahal / _robust_upper_reference(mahal)

        robust_z = raw_block.apply(_robust_scale)
        univar_raw = robust_z.abs().max(axis=1, skipna=True).fillna(0.0)
        flag_score = univar_raw / _robust_upper_reference(univar_raw)

        reason_text = pd.Series("", index=group.index)
        top_elements = pd.Series("", index=group.index)
        for col in available_cols:
            series = raw_block[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            is_outlier_mask = (series < lower) | (series > upper)
            for idx in series[is_outlier_mask].index:
                current = reason_text.at[idx]
                direction = "high" if series.at[idx] > upper else "low"
                reason = f"{col} {direction} (value {series.at[idx]:.2f}, IQR {q1:.2f}-{q3:.2f})"
                reason_text.at[idx] = reason if current == "" else f"{current}; {reason}"

        z_abs = robust_z[available_cols].abs()
        for idx, row in z_abs.iterrows():
            row = row.dropna()
            top = row.sort_values(ascending=False).head(3).index.tolist()
            top_elements.at[idx] = ", ".join(top)

        multi_only = (mahal_norm > OUTLIER_DISPLAY_THRESHOLD) & reason_text.eq("")
        for idx in reason_text[multi_only].index:
            reason_text.at[idx] = f"multivariate chemistry distance high (score {mahal_norm.at[idx]:.2f})"

        result.loc[group.index, "outlier_mahal_score"] = mahal_norm.fillna(0.0)
        result.loc[group.index, "outlier_univariate_score"] = flag_score.fillna(0.0)
        result.loc[group.index, "outlier_score"] = np.maximum(mahal_norm, flag_score).fillna(0.0)
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
    distance in population CLR (centered log-ratio) space.

    Uses Mahalanobis distance to strat group centroids in CLR-transformed chemistry,
    so compositional closure does not dominate. Rows with missing assay dimensions
    are compared using the covariance matrix restricted to the present dimensions;
    this avoids using a block of the joint precision matrix as a marginal precision.
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

    raw_all = df[available_cols].apply(pd.to_numeric, errors="coerce")
    try:
        clr_all = _clr_transform(raw_all)
    except Exception:
        clr_all = raw_all.apply(_robust_scale)
    clr_all = clr_all.replace([np.inf, -np.inf], np.nan)

    strat_stats: Dict[object, Dict[str, object]] = {}
    for strat_value, group in df.groupby(strat_col):
        if pd.isna(strat_value) or len(group) < min_group_size:
            continue

        clr_data = clr_all.loc[group.index]
        clr_data = clr_data.loc[clr_data.notna().any(axis=1)]
        if len(clr_data) < min_group_size:
            continue

        centroid = clr_data.median()
        z_centered = clr_data - centroid
        cov = np.cov(z_centered.fillna(0.0).T, bias=True)
        cov = np.atleast_2d(cov)
        dim = cov.shape[0]
        if dim == 0:
            continue

        alpha = 0.3 if len(clr_data) < 30 else 0.1
        cov_shrink = _shrink_covariance(cov, alpha)

        try:
            inv_cov = np.linalg.pinv(cov_shrink)
        except Exception:
            continue

        strat_stats[strat_value] = {
            "centroid": centroid,
            "cov": cov_shrink,
            "inv_cov": inv_cov,
            "count": len(group),
        }

    if not strat_stats:
        return result

    row_values = clr_all.to_numpy(dtype=float)
    n_rows = len(df)
    strat_values = list(strat_stats.keys())
    distances = np.full((n_rows, len(strat_values)), np.inf, dtype=float)

    for col_idx, strat_value in enumerate(strat_values):
        stats = strat_stats[strat_value]
        centroid = stats["centroid"].reindex(available_cols).to_numpy(dtype=float)
        covariance = np.asarray(stats["cov"], dtype=float)
        inv_cov = np.asarray(stats["inv_cov"], dtype=float)

        diff = row_values - centroid
        present = np.isfinite(diff)
        if not present.any():
            continue

        for mask in np.unique(present, axis=0):
            if not mask.any():
                continue
            row_mask = np.all(present == mask, axis=1)
            if not row_mask.any():
                continue
            diff_block = diff[row_mask][:, mask]
            try:
                if mask.all():
                    inv = inv_cov
                else:
                    sub_cov = covariance[np.ix_(mask, mask)]
                    inv = np.linalg.pinv(sub_cov)
                dist_sq = np.einsum("ij,jk,ik->i", diff_block, inv, diff_block)
                distances[row_mask, col_idx] = np.sqrt(np.maximum(dist_sq, 0.0))
            except Exception:
                continue

    finite_rows = np.isfinite(distances).any(axis=1)
    if finite_rows.any():
        best_indices = np.argmin(distances[finite_rows], axis=1)
        best_values = [strat_values[i] for i in best_indices]
        result.loc[df.index[finite_rows]] = best_values

    return result
