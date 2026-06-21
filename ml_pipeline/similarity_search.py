"""Hybrid visual, chemical, and spatial similarity ranking for GeoVue.

This module deliberately keeps the similarity math outside the Tk logging-review
UI.  The UI supplies candidate intervals/images, optional visual scores from the
image embedding index, and optional Snowflake-derived XYZ positions; this module
turns those inputs into ranked, explainable results.

The three signals are intentionally separate:
- visual similarity: image embedding cosine score, usually from embedding_store
- chemical similarity: compositional assay/geochemistry vectors
- spatial similarity: surveyed 3D proximity when collar/survey data is loaded,
  otherwise same-hole depth proximity

Continuity is a fourth, derived signal.  It rewards candidates whose local
same-hole neighbourhood also looks/assays like the query, reducing one-off noisy
matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, sqrt
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from processing.DataManager.column_aliases import ColumnResolver
except Exception:  # pragma: no cover - lets this module run in minimal ML-only contexts.
    ColumnResolver = None  # type: ignore[assignment]


DEFAULT_CHEMISTRY_COLUMNS: tuple[str, ...] = (
    "fe_pct",
    "sio2_pct",
    "al2o3_pct",
    "p_pct",
    "s_pct",
    "loi_pct",
    "mn_pct",
    "cao_pct",
    "mgo_pct",
    "k2o_pct",
    "na2o_pct",
    "tio2_pct",
)

DEFAULT_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "visual": {"visual": 1.0},
    "chemical": {"chemical": 1.0},
    "spatial": {"spatial": 1.0},
    "hybrid": {"visual": 0.45, "chemical": 0.45, "spatial": 0.10},
    "continuity": {
        "visual": 0.25,
        "chemical": 0.35,
        "spatial": 0.20,
        "continuity": 0.20,
    },
}


def path_key(path: object) -> str:
    """Normalize paths for case-insensitive matching on Windows."""
    return str(path or "").replace("\\", "/").lower()


@dataclass(frozen=True)
class SimilarityCandidate:
    """One image/interval that can participate in similarity ranking.

    `chemistry` is expected to be the merged GeoVue row data for the interval
    keyed by whatever column names the source supplies.  Column aliases are
    resolved later so callers do not need to pre-normalize Snowflake/CSV names.
    """

    image_path: str
    hole_id: str
    depth_from: Optional[float]
    depth_to: Optional[float]
    chemistry: Mapping[str, Any] = field(default_factory=dict)
    xyz: Optional[tuple[float, float, float]] = None
    label: str = ""

    @property
    def key(self) -> str:
        return path_key(self.image_path)

    @property
    def hole_key(self) -> str:
        return str(self.hole_id or "").strip().upper()

    @property
    def depth_midpoint(self) -> Optional[float]:
        """Return the interval midpoint, falling back to depth_to."""
        if self.depth_from is not None and self.depth_to is not None:
            return (float(self.depth_from) + float(self.depth_to)) / 2.0
        if self.depth_to is not None:
            return float(self.depth_to)
        return None


@dataclass(frozen=True)
class SimilaritySearchConfig:
    """Configuration for one similarity search.

    The defaults are conservative for Logging Review: hybrid uses visual and
    chemistry primarily, with spatial as a plausibility prior rather than a hard
    gate.  `use_xyz` enables surveyed 3D proximity when caller supplied XYZ.
    """

    mode: str = "hybrid"
    chemistry_columns: tuple[str, ...] = DEFAULT_CHEMISTRY_COLUMNS
    chemistry_weights: Mapping[str, float] = field(default_factory=dict)
    component_weights: Mapping[str, float] = field(default_factory=dict)
    spatial_range_m: float = 50.0
    depth_range_m: float = 10.0
    continuity_window_m: float = 3.0
    min_chemical_coverage: float = 0.5
    use_xyz: bool = True
    top_k: Optional[int] = None

    def weights_for_mode(self) -> dict[str, float]:
        """Return component weights, allowing explicit overrides."""
        mode_key = str(self.mode or "hybrid").strip().lower()
        base = dict(DEFAULT_MODE_WEIGHTS.get(mode_key, DEFAULT_MODE_WEIGHTS["hybrid"]))
        for key, value in self.component_weights.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric >= 0:
                base[str(key).lower()] = numeric
        return base


@dataclass(frozen=True)
class SimilaritySearchResult:
    """One ranked similarity hit with explainable component scores."""

    candidate: SimilarityCandidate
    combined_score: float
    rank: int = 0
    visual_score: Optional[float] = None
    chemical_score: Optional[float] = None
    spatial_score: Optional[float] = None
    continuity_score: Optional[float] = None
    chemical_coverage: float = 0.0
    spatial_distance_m: Optional[float] = None
    depth_delta_m: Optional[float] = None
    explanation: str = ""


@dataclass(frozen=True)
class ChemicalSpace:
    """Resolved and transformed chemistry matrix for a candidate set."""

    columns: tuple[str, ...]
    actual_columns: tuple[str, ...]
    matrix: np.ndarray
    weights: np.ndarray


def _finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _safe_positive_fill(values: pd.DataFrame) -> pd.Series:
    """Return per-column positive replacement values for CLR transform.

    Assays can include zeros, below-detection placeholders, and blanks.  CLR
    cannot take log(0), so each non-positive value is replaced by half the
    smallest positive value in that column, with a dataset-level fallback.
    """
    positive = values.where(values > 0)
    stacked = positive.stack()
    global_min = stacked.min() if not stacked.empty else np.nan
    fallback = global_min / 2.0 if pd.notna(global_min) and global_min > 0 else 1e-6
    fills = positive.min(axis=0) / 2.0
    return fills.where((fills > 0) & fills.notna(), fallback).fillna(fallback)


def clr_transform(values: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio transform for compositional geochemistry.

    Missing values stay missing.  Positive components are transformed as
    log(x_i) minus the row mean log value, which avoids misleading raw-percent
    Euclidean distances caused by compositional closure.
    """
    numeric = values.apply(pd.to_numeric, errors="coerce").astype(float)
    if numeric.empty:
        return numeric

    fills = _safe_positive_fill(numeric)
    positive = numeric.mask(numeric <= 0, fills, axis=1).clip(lower=1e-12)
    arr = positive.to_numpy(dtype=float)
    original_valid = np.isfinite(numeric.to_numpy(dtype=float))

    logs = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr) & (arr > 0) & original_valid
    logs[valid] = np.log(arr[valid])

    valid_counts = np.isfinite(logs).sum(axis=1, keepdims=True)
    log_sums = np.nansum(logs, axis=1, keepdims=True)
    row_mean = np.divide(
        log_sums,
        valid_counts,
        out=np.full_like(log_sums, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    out = logs - row_mean
    out[~np.isfinite(logs)] = np.nan
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def robust_standardize(values: pd.DataFrame) -> pd.DataFrame:
    """Robustly scale transformed chemistry columns to comparable units."""
    out = values.copy().astype(float)
    for col in out.columns:
        series = out[col].replace([np.inf, -np.inf], np.nan)
        median = series.median(skipna=True)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            scale = iqr / 1.349
        else:
            std = series.std(skipna=True, ddof=0)
            scale = std if pd.notna(std) and std > 0 else 1.0
        out[col] = (series - median) / scale
    return out


def _resolve_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """Resolve one requested standard/actual column name against a DataFrame."""
    if df.empty:
        return None
    requested_text = str(requested).strip()
    if not requested_text:
        return None

    lower_to_actual = {str(col).lower(): col for col in df.columns}
    exact = lower_to_actual.get(requested_text.lower())
    if exact is not None:
        return str(exact)

    if ColumnResolver is not None:
        resolved = ColumnResolver(df).get(requested_text)
        if resolved:
            return str(resolved)
    return None


def build_chemical_space(
    candidates: Sequence[SimilarityCandidate],
    config: SimilaritySearchConfig,
) -> ChemicalSpace:
    """Build a transformed chemistry matrix for the candidate list.

    Columns are resolved once against the union of candidate chemistry mappings.
    The output matrix keeps NaNs for missing values so pairwise scoring can use
    only dimensions present on both query and candidate.
    """
    if not candidates:
        return ChemicalSpace((), (), np.empty((0, 0), dtype=float), np.empty(0, dtype=float))

    raw_rows = [dict(candidate.chemistry or {}) for candidate in candidates]
    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        return ChemicalSpace((), (), np.empty((len(candidates), 0), dtype=float), np.empty(0, dtype=float))

    resolved_standard: list[str] = []
    resolved_actual: list[str] = []
    for standard in config.chemistry_columns:
        actual = _resolve_column(raw_df, standard)
        if actual and actual not in resolved_actual:
            resolved_standard.append(str(standard))
            resolved_actual.append(actual)

    if not resolved_actual:
        return ChemicalSpace((), (), np.empty((len(candidates), 0), dtype=float), np.empty(0, dtype=float))

    selected = raw_df[resolved_actual].apply(pd.to_numeric, errors="coerce")
    # Drop dimensions with no usable values.  Query-level coverage is checked
    # later, but globally empty columns only add noise and divide-by-zero risk.
    usable = selected.notna().any(axis=0)
    selected = selected.loc[:, usable]
    resolved_standard = [std for std, keep in zip(resolved_standard, usable.tolist()) if keep]
    resolved_actual = [col for col, keep in zip(resolved_actual, usable.tolist()) if keep]
    if selected.empty or not resolved_actual:
        return ChemicalSpace((), (), np.empty((len(candidates), 0), dtype=float), np.empty(0, dtype=float))

    transformed = robust_standardize(clr_transform(selected))
    weights = []
    configured_weights = {str(k).lower(): float(v) for k, v in config.chemistry_weights.items()}
    for standard, actual in zip(resolved_standard, resolved_actual):
        weight = configured_weights.get(str(standard).lower(), configured_weights.get(str(actual).lower(), 1.0))
        weights.append(max(0.0, float(weight)))

    return ChemicalSpace(
        columns=tuple(resolved_standard),
        actual_columns=tuple(resolved_actual),
        matrix=transformed.to_numpy(dtype=float),
        weights=np.asarray(weights, dtype=float),
    )


def _weighted_masked_similarity(
    query: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    min_coverage: float,
) -> tuple[Optional[float], float]:
    """Return weighted chemistry similarity and weighted coverage."""
    if query.size == 0 or candidate.size == 0 or weights.size == 0:
        return None, 0.0

    query_present = np.isfinite(query) & (weights > 0)
    candidate_present = np.isfinite(candidate) & query_present
    denominator = float(weights[query_present].sum())
    if denominator <= 0:
        return None, 0.0

    coverage = float(weights[candidate_present].sum() / denominator)
    if coverage < min_coverage or not candidate_present.any():
        return None, coverage

    diff = candidate[candidate_present] - query[candidate_present]
    w = weights[candidate_present]
    distance = sqrt(float(np.sum(w * diff * diff) / max(float(w.sum()), 1e-12)))
    return 1.0 / (1.0 + distance), coverage


def _euclidean_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _decay_score(distance: Optional[float], scale: float) -> Optional[float]:
    """Smooth 0..1 proximity score with a conservative Gaussian-like decay."""
    if distance is None or scale <= 0 or not isfinite(distance):
        return None
    return exp(-((distance / scale) ** 2))


def spatial_similarity(
    query: SimilarityCandidate,
    candidate: SimilarityCandidate,
    config: SimilaritySearchConfig,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return spatial score, XYZ distance, and depth delta.

    Surveyed 3D distance is preferred when both samples have XYZ coordinates and
    config.use_xyz is enabled.  If not, same-hole measured-depth continuity is a
    legitimate fallback for RC chip compartments and unloaded collar/survey data.
    """
    depth_delta = None
    if query.depth_midpoint is not None and candidate.depth_midpoint is not None:
        depth_delta = abs(float(candidate.depth_midpoint) - float(query.depth_midpoint))

    if config.use_xyz and query.xyz is not None and candidate.xyz is not None:
        distance = _euclidean_distance(query.xyz, candidate.xyz)
        return _decay_score(distance, config.spatial_range_m), distance, depth_delta

    if query.hole_key and query.hole_key == candidate.hole_key and depth_delta is not None:
        return _decay_score(depth_delta, config.depth_range_m), None, depth_delta

    return None, None, depth_delta


def _depth_window_indices(
    candidates: Sequence[SimilarityCandidate],
    window_m: float,
) -> dict[int, list[tuple[int, float]]]:
    """Build same-hole neighbourhood lists for continuity scoring."""
    by_hole: dict[str, list[tuple[int, float]]] = {}
    for idx, candidate in enumerate(candidates):
        depth = candidate.depth_midpoint
        if candidate.hole_key and depth is not None:
            by_hole.setdefault(candidate.hole_key, []).append((idx, float(depth)))

    out: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(len(candidates))}
    if window_m <= 0:
        return out

    for rows in by_hole.values():
        rows.sort(key=lambda item: item[1])
        for pos, (idx, depth) in enumerate(rows):
            neighbours: list[tuple[int, float]] = []
            left = pos
            while left >= 0 and depth - rows[left][1] <= window_m:
                n_idx, n_depth = rows[left]
                neighbours.append((n_idx, abs(depth - n_depth)))
                left -= 1
            right = pos + 1
            while right < len(rows) and rows[right][1] - depth <= window_m:
                n_idx, n_depth = rows[right]
                neighbours.append((n_idx, abs(depth - n_depth)))
                right += 1
            out[idx] = neighbours
    return out


def _local_average_score(
    scores: Sequence[Optional[float]],
    neighbours: Sequence[tuple[int, float]],
    window_m: float,
) -> Optional[float]:
    """Average query-relative scores over a same-hole local window."""
    weighted_sum = 0.0
    weight_sum = 0.0
    for idx, distance in neighbours:
        score = scores[idx]
        if score is None:
            continue
        weight = exp(-((distance / max(window_m, 1e-6)) ** 2))
        weighted_sum += float(score) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return weighted_sum / weight_sum


def _component_average(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _weighted_component_score(
    components: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
) -> Optional[float]:
    total = 0.0
    denom = 0.0
    for name, weight in weights.items():
        score = components.get(name)
        if score is None or weight <= 0:
            continue
        total += float(score) * float(weight)
        denom += float(weight)
    if denom <= 0:
        return None
    return total / denom


def _score_text(label: str, score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    return f"{label} {score:.2f}"


class HybridSimilarityRanker:
    """Rank candidates using visual, chemical, spatial, and continuity signals."""

    def __init__(
        self,
        candidates: Sequence[SimilarityCandidate],
        *,
        visual_scores_by_key: Optional[Mapping[str, float]] = None,
        config: Optional[SimilaritySearchConfig] = None,
    ) -> None:
        self.candidates = list(candidates)
        self.config = config or SimilaritySearchConfig()
        self.visual_scores_by_key = {
            path_key(key): float(value)
            for key, value in (visual_scores_by_key or {}).items()
            if value is not None
        }

    def rank(self, query: SimilarityCandidate) -> list[SimilaritySearchResult]:
        """Return ranked candidates for a query candidate.

        The query itself is excluded by normalized image path.  Component scores
        that cannot be computed are omitted and remaining component weights are
        re-normalized, so the service degrades cleanly when chemistry or XYZ is
        not loaded.
        """
        candidates = self.candidates
        if not candidates:
            return []

        query_key = query.key
        chem_space = build_chemical_space(candidates, self.config)
        query_index = next((idx for idx, c in enumerate(candidates) if c.key == query_key), None)
        query_vector = chem_space.matrix[query_index] if query_index is not None and chem_space.matrix.size else None

        visual_scores: list[Optional[float]] = []
        chemical_scores: list[Optional[float]] = []
        chemical_coverage: list[float] = []
        spatial_scores: list[Optional[float]] = []
        spatial_distances: list[Optional[float]] = []
        depth_deltas: list[Optional[float]] = []

        for idx, candidate in enumerate(candidates):
            visual_scores.append(self.visual_scores_by_key.get(candidate.key))

            if query_vector is not None and chem_space.matrix.size:
                chem_score, coverage = _weighted_masked_similarity(
                    query_vector,
                    chem_space.matrix[idx],
                    chem_space.weights,
                    self.config.min_chemical_coverage,
                )
            else:
                chem_score, coverage = None, 0.0
            chemical_scores.append(chem_score)
            chemical_coverage.append(coverage)

            spatial, distance, depth_delta = spatial_similarity(query, candidate, self.config)
            spatial_scores.append(spatial)
            spatial_distances.append(distance)
            depth_deltas.append(depth_delta)

        neighbour_index = _depth_window_indices(candidates, self.config.continuity_window_m)
        continuity_scores: list[Optional[float]] = []
        for idx in range(len(candidates)):
            neighbours = neighbour_index.get(idx, [])
            visual_local = _local_average_score(visual_scores, neighbours, self.config.continuity_window_m)
            chem_local = _local_average_score(chemical_scores, neighbours, self.config.continuity_window_m)
            continuity_scores.append(_component_average((visual_local, chem_local)))

        component_weights = self.config.weights_for_mode()
        ranked: list[SimilaritySearchResult] = []
        for idx, candidate in enumerate(candidates):
            if candidate.key == query_key:
                continue

            components = {
                "visual": visual_scores[idx],
                "chemical": chemical_scores[idx],
                "spatial": spatial_scores[idx],
                "continuity": continuity_scores[idx],
            }
            combined = _weighted_component_score(components, component_weights)
            if combined is None:
                continue

            parts = [
                _score_text("V", visual_scores[idx]),
                _score_text("C", chemical_scores[idx]),
                _score_text("S", spatial_scores[idx]),
                _score_text("N", continuity_scores[idx]),
            ]
            explanation = ", ".join(part for part in parts if part)
            if spatial_distances[idx] is not None:
                explanation += f", {spatial_distances[idx]:.1f} m XYZ"
            elif depth_deltas[idx] is not None and candidate.hole_key == query.hole_key:
                explanation += f", {depth_deltas[idx]:.1f} m depth"

            ranked.append(
                SimilaritySearchResult(
                    candidate=candidate,
                    combined_score=combined,
                    visual_score=visual_scores[idx],
                    chemical_score=chemical_scores[idx],
                    spatial_score=spatial_scores[idx],
                    continuity_score=continuity_scores[idx],
                    chemical_coverage=chemical_coverage[idx],
                    spatial_distance_m=spatial_distances[idx],
                    depth_delta_m=depth_deltas[idx],
                    explanation=explanation,
                )
            )

        ranked.sort(key=lambda result: result.combined_score, reverse=True)
        if self.config.top_k and self.config.top_k > 0:
            ranked = ranked[: self.config.top_k]
        return [
            SimilaritySearchResult(
                candidate=result.candidate,
                combined_score=result.combined_score,
                rank=rank,
                visual_score=result.visual_score,
                chemical_score=result.chemical_score,
                spatial_score=result.spatial_score,
                continuity_score=result.continuity_score,
                chemical_coverage=result.chemical_coverage,
                spatial_distance_m=result.spatial_distance_m,
                depth_delta_m=result.depth_delta_m,
                explanation=result.explanation,
            )
            for rank, result in enumerate(ranked, start=1)
        ]
