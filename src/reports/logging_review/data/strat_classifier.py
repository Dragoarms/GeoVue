"""
Experimental strat group classifier using sklearn RandomForest.

This module is not used by the logging-review report path. The canonical report
predictor is reports.logging_review.data.outliers.predict_most_likely_strat,
which is deterministic, dependency-light, and trained once on the prepared
population frame.

Trains on the dataset's own geochemistry + strat labels to learn the actual
population boundaries, then predicts the "most likely" major strat group for
each interval.  Provides per-prediction geochemical justification.

Designed to be trained once per report session and cached.  Training on 190k
rows takes ~3-5 seconds with no GPU.

Standalone experimentation:
    classifier = StratClassifier()
    classifier.train(merged_df, feature_cols=chem_actual_cols)
    predictions = classifier.predict(assay_logger_df)
    # predictions is a DataFrame with columns:
    #   predicted_group, probability, rig_group, interp_group,
    #   explanation, top_features

Architecture:
    BESTSTRAT / STRATSUM  -->  major group mapping  -->  RandomForest training
    Per-interval chemistry -->  predict + explain    -->  evidence table rows
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from processing.DataManager.column_aliases import ColumnResolver

logger = logging.getLogger(__name__)


# ── Strat code → major group mapping ─────────────────────────────────────
# Applied to both BESTSTRAT (interpreter) and STRATSUM (rig geologist).
# Unknown codes fall through to "OTHER".

STRAT_GROUP_MAP: Dict[str, str] = {
    # ── BID — eBIF Mineralised (<10% gangue) ────────────────────────────
    "BID":  "BID",
    "BIDf": "BID",      # friable
    "BIDh": "BID",      # hard (massive ore)
    "BIDm": "BID",      # semi-hard (banded)

    # ── BIDs — eBIF + shale/schist/phyllite interbeds (mineralised or not) ──
    "BIDs": "BIDs",

    # ── BIF — Unmineralised BIF (>15% chert/quartz gangue) ─────────────
    "BIF":  "BIF",
    "BIFf": "BIF",      # friable (powdery cherts)
    "BIFh": "BIF",      # hard (layered cherts)
    "BIFhm":"BIF",      # hard-medium
    "BIFm": "BIF",      # medium hardness
    "BIFN": "BIF",      # BIF-N variant

    # ── BIFs — BIF + shale/schist/phyllite interbeds (unmineralised) ───
    "BIFs": "BIFs",

    # ── PHY — Phyllite (strongly foliated metapelite) ──────────────────
    "PHY":  "PHY",

    # ── AMP — Amphibolite (speckled black-green, hornblende/actinolite) ─
    "AMP":  "AMP",
    "AMp":  "AMP",

    # ── SCH — Schist (moderately foliated metapelite) ──────────────────
    "SCH":  "SCH",
    "PCS":  "SCH",      # Pebble Conglomerate Schist — metapelitic, grouped with SCH

    # ── QV — Quartz vein / quartzite ───────────────────────────────────
    "QT":   "QV",       # Quartzite (monomineralic granular silica)
    "QV":   "QV",       # Quartz vein

    # ── DE — Detrital cover (regolith, shallow) ────────────────────────
    "DE":   "DE",
    "DEi":  "DE",       # unmineralised, >50% gangue, loose fragments
    "DEir": "DE",
    "DEm":  "DE",       # mineralised, <10% gangue, loose fragments
    "DEs":  "DE",       # mineralised, 10-20% gangue, loose fragments
    "DEc":  "DE",       # unmineralised, >50% clays

    # ── REGO — Regolith / laterite / canga ─────────────────────────────
    "LA":   "REGO",     # Laterite (eluvial soils)
    "LAc":  "REGO",     # Laterite, consolidated fragments
    "LAt":  "REGO",     # Laterite, fully decomposed
    "CGi":  "REGO",     # Canga, mineralised in-situ breccia
    "CGs":  "REGO",     # Scanga, unmineralised cemented material

    # ── OTHER — intrusive, structural, voids, unclassified ─────────────
    "Ag":   "OTHER",    # Archaean Granite
    "PEG":  "OTHER",    # Pegmatite
    "CF":   "OTHER",    # Cavity Fill
    "VO":   "OTHER",    # Void
    "ZF":   "OTHER",    # Fault Zone - Broken Ground
    "OTHER":"OTHER",
}

# Reverse: which strat codes belong to each group (for display)
STRAT_GROUP_MEMBERS: Dict[str, List[str]] = {}
for _code, _group in STRAT_GROUP_MAP.items():
    STRAT_GROUP_MEMBERS.setdefault(_group, []).append(_code)

# Feature columns in priority order (standard names resolved via ColumnResolver)
DEFAULT_FEATURE_STANDARDS = [
    "fe_pct", "sio2_pct", "al2o3_pct", "p_pct", "loi_pct",
    "tio2_pct", "mgo_pct", "cao_pct", "k2o_pct", "na2o_pct",
    "s_pct", "mn_pct",
    "loi_425_pct", "loi_1000_pct",  # LOI sub-fractions (goethite, magnetite indicators)
]

# Display-friendly short names
SHORT_NAMES = {
    "fe_pct": "Fe", "sio2_pct": "SiO2", "al2o3_pct": "Al2O3",
    "p_pct": "P", "loi_pct": "LOI", "tio2_pct": "TiO2",
    "mgo_pct": "MgO", "cao_pct": "CaO", "k2o_pct": "K2O",
    "na2o_pct": "Na2O", "s_pct": "S", "mn_pct": "Mn",
    "loi_1000_pct": "LOI1000", "loi_425_pct": "LOI425",
    "loi_650_pct": "LOI650", "zn_pct": "Zn",
}

# Major group display names and descriptions (for report)
MAJOR_GROUP_INFO: Dict[str, str] = {
    "BID":  "eBIF Mineralised (<10% gangue)",
    "BIDs": "eBIF + shale/phyllite interbeds",
    "BIF":  "BIF Unmineralised (>15% chert/quartz)",
    "BIFs": "BIF + shale/phyllite interbeds",
    "PHY":  "Phyllite (strongly foliated metapelite)",
    "AMP":  "Amphibolite (hornblende/actinolite)",
    "SCH":  "Schist / metapelite (moderately foliated)",
    "QV":   "Quartz vein / quartzite",
    "DE":   "Detrital cover (regolith)",
    "REGO": "Laterite / canga (regolith)",
    "OTHER":"Granite, pegmatite, voids, faults",
}


def map_to_major_group(strat_code: str) -> str:
    """Map a strat code to its major group. Unknown codes → 'OTHER'."""
    if not strat_code or pd.isna(strat_code):
        return "OTHER"
    code = str(strat_code).strip()
    return STRAT_GROUP_MAP.get(code, "OTHER")


class ClassStats:
    """Per-class feature statistics for generating explanations."""

    def __init__(self):
        self.medians: Dict[str, Dict[str, float]] = {}   # group -> {feature: median}
        self.q25: Dict[str, Dict[str, float]] = {}       # group -> {feature: q25}
        self.q75: Dict[str, Dict[str, float]] = {}       # group -> {feature: q75}
        self.counts: Dict[str, int] = {}                  # group -> sample count

    def build(self, df: pd.DataFrame, group_col: str, feature_cols: List[str]) -> None:
        """Compute median and IQR per group per feature."""
        for group, sub in df.groupby(group_col):
            self.counts[group] = len(sub)
            self.medians[group] = {}
            self.q25[group] = {}
            self.q75[group] = {}
            for col in feature_cols:
                series = pd.to_numeric(sub[col], errors="coerce").dropna()
                if series.empty:
                    continue
                self.medians[group][col] = float(series.median())
                self.q25[group][col] = float(series.quantile(0.25))
                self.q75[group][col] = float(series.quantile(0.75))

    def typical_range_str(self, group: str, feature: str) -> str:
        """Return 'Q25-Q75' string for a group/feature."""
        q25 = self.q25.get(group, {}).get(feature)
        q75 = self.q75.get(group, {}).get(feature)
        if q25 is None or q75 is None:
            return "n/a"
        return f"{q25:.1f}-{q75:.1f}"

    def median_val(self, group: str, feature: str) -> Optional[float]:
        return self.medians.get(group, {}).get(feature)


class StratClassifier:
    """
    RandomForest strat group classifier with per-prediction explanations.

    Train on the full dataset, predict for individual loggers, explain each
    classification with geochemical justification.

    Thread-safe for read after training (predict/explain are read-only).
    """

    def __init__(self):
        self._model: Optional["RandomForestClassifier"] = None
        self._label_encoder: Optional["LabelEncoder"] = None
        self._feature_cols: List[str] = []        # Actual column names
        self._feature_standards: List[str] = []   # Standard names (for SHORT_NAMES)
        self._class_stats: Optional[ClassStats] = None
        self._is_trained: bool = False
        self._training_n: int = 0
        self._training_groups: List[str] = []
        self._feature_importances: Dict[str, float] = {}
        self._depth_col: Optional[str] = None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def groups(self) -> List[str]:
        """List of major groups the model knows about."""
        return list(self._training_groups)

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Feature importances (display name → importance)."""
        return dict(self._feature_importances)

    def train(
        self,
        df: pd.DataFrame,
        strat_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        include_depth: bool = True,
        min_group_size: int = 30,
        n_estimators: int = 200,
        max_depth: int = 15,
        random_state: int = 42,
    ) -> bool:
        """
        Train the classifier on the full dataset.

        Args:
            df: Full merged DataFrame (all loggers, all holes)
            strat_col: Column containing strat codes (auto-detected if None)
            feature_cols: Actual chemistry column names (auto-resolved if None)
            include_depth: Whether to include depth_to as a feature (helps DE*)
            min_group_size: Minimum samples for a group to be included
            n_estimators: Number of trees in the forest
            max_depth: Max tree depth (prevents overfitting)
            random_state: For reproducibility

        Returns:
            True if training succeeded
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available — strat classifier disabled")
            return False

        resolver = ColumnResolver(df)

        # Resolve strat column
        if strat_col is None:
            strat_col = resolver.get("strat")
        if not strat_col or strat_col not in df.columns:
            logger.warning("No strat column found for classifier training")
            return False

        # Resolve feature columns
        if feature_cols is None:
            feature_cols = []
            self._feature_standards = []
            for std in DEFAULT_FEATURE_STANDARDS:
                actual = resolver.get(std)
                if actual and actual in df.columns:
                    feature_cols.append(actual)
                    self._feature_standards.append(std)
        else:
            # Map actual columns back to standard names where possible
            self._feature_standards = []
            standard_by_actual = {}
            for std in DEFAULT_FEATURE_STANDARDS:
                actual = resolver.get(std)
                if actual:
                    standard_by_actual[actual.lower()] = std
            for col in feature_cols:
                self._feature_standards.append(standard_by_actual.get(col.lower(), col))

        if len(feature_cols) < 3:
            logger.warning("Too few chemistry columns (%d) for classifier", len(feature_cols))
            return False

        self._feature_cols = feature_cols

        # Optionally add depth
        self._depth_col = None
        if include_depth:
            depth_col = resolver.get("depth_to")
            if depth_col and depth_col in df.columns:
                self._depth_col = depth_col

        # Prepare training data
        work = df[[strat_col] + feature_cols].copy()
        if self._depth_col:
            work["_depth"] = pd.to_numeric(df[self._depth_col], errors="coerce")

        # Map to major groups
        work["_major_group"] = work[strat_col].apply(map_to_major_group)

        # Convert features to numeric
        for col in feature_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        # Drop rows with missing features or strat
        all_feature_cols = feature_cols + (["_depth"] if self._depth_col else [])
        work = work.dropna(subset=all_feature_cols + ["_major_group"])
        work = work[work["_major_group"] != "OTHER"]  # Don't train on OTHER

        # Filter small groups
        group_counts = work["_major_group"].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index.tolist()
        work = work[work["_major_group"].isin(valid_groups)]

        if len(valid_groups) < 2:
            logger.warning("Too few strat groups with sufficient data: %s", valid_groups)
            return False

        logger.info(
            "Training strat classifier: %d rows, %d groups (%s), %d features",
            len(work), len(valid_groups), ", ".join(valid_groups), len(all_feature_cols),
        )

        # Encode labels
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(work["_major_group"])
        X = work[all_feature_cols].values

        # Train
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",  # Handle imbalanced groups
            random_state=random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)

        # Store metadata
        self._is_trained = True
        self._training_n = len(work)
        self._training_groups = list(self._label_encoder.classes_)

        # Feature importances with display names
        importances = self._model.feature_importances_
        self._feature_importances = {}
        for i, col in enumerate(all_feature_cols):
            if col == "_depth":
                display = "Depth"
            elif i < len(self._feature_standards):
                display = SHORT_NAMES.get(self._feature_standards[i], col)
            else:
                display = col
            self._feature_importances[display] = float(importances[i])

        # Sort by importance
        self._feature_importances = dict(
            sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)
        )

        # Compute per-class statistics for explanations
        self._class_stats = ClassStats()
        self._class_stats.build(work, "_major_group", all_feature_cols)

        # Log summary
        logger.info("Strat classifier trained successfully:")
        for grp in self._training_groups:
            logger.info("  %s: %d samples", grp, self._class_stats.counts.get(grp, 0))
        top3 = list(self._feature_importances.items())[:3]
        logger.info("  Top features: %s", ", ".join(f"{n}={v:.3f}" for n, v in top3))

        return True

    def predict(
        self,
        df: pd.DataFrame,
        strat_col: Optional[str] = None,
        stratsum_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict major strat group for each interval and generate explanations.

        Args:
            df: DataFrame to predict on (typically assay_logger_df)
            strat_col: Interpreter strat column (BESTSTRAT)
            stratsum_col: Rig geologist strat column (STRATSUM)

        Returns:
            DataFrame aligned to df.index with columns:
              predicted_group, probability, recorded_group, rig_group,
              interp_group, is_mismatch, explanation, top_features
        """
        result = pd.DataFrame(index=df.index)
        result["predicted_group"] = "OTHER"
        result["probability"] = 0.0
        result["recorded_group"] = "OTHER"
        result["rig_group"] = ""
        result["interp_group"] = ""
        result["is_mismatch"] = False
        result["explanation"] = ""
        result["top_features"] = ""

        if not self._is_trained or self._model is None:
            return result

        resolver = ColumnResolver(df)

        # Resolve strat columns
        if strat_col is None:
            strat_col = resolver.get("strat")
        if stratsum_col is None:
            # Try STRATSUM specifically
            for candidate in ["stratsum", "strat_summary", "stratsummary"]:
                resolved = resolver.get(candidate)
                if resolved and resolved in df.columns:
                    stratsum_col = resolved
                    break

        # Map recorded strats to groups
        if strat_col and strat_col in df.columns:
            result["interp_group"] = df[strat_col].apply(map_to_major_group)
            result["recorded_group"] = result["interp_group"]
        if stratsum_col and stratsum_col in df.columns:
            result["rig_group"] = df[stratsum_col].apply(map_to_major_group)
            # If no interpreter strat, use rig strat as recorded
            mask_no_interp = result["recorded_group"] == "OTHER"
            result.loc[mask_no_interp, "recorded_group"] = result.loc[mask_no_interp, "rig_group"]

        # Prepare features
        all_feature_cols = self._feature_cols.copy()
        X_df = pd.DataFrame(index=df.index)
        for col in self._feature_cols:
            if col in df.columns:
                X_df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                X_df[col] = np.nan

        if self._depth_col:
            all_feature_cols.append("_depth")
            if self._depth_col in df.columns:
                X_df["_depth"] = pd.to_numeric(df[self._depth_col], errors="coerce")
            else:
                X_df["_depth"] = np.nan

        # Only predict rows with enough features
        valid_mask = X_df.notna().sum(axis=1) >= max(3, len(all_feature_cols) // 2)
        if valid_mask.sum() == 0:
            return result

        X_valid = X_df.loc[valid_mask].fillna(0).values
        proba = self._model.predict_proba(X_valid)
        pred_indices = proba.argmax(axis=1)
        pred_groups = self._label_encoder.inverse_transform(pred_indices)
        pred_probs = proba[np.arange(len(pred_indices)), pred_indices]

        result.loc[valid_mask, "predicted_group"] = pred_groups
        result.loc[valid_mask, "probability"] = pred_probs

        # Determine mismatches
        result["is_mismatch"] = (
            (result["predicted_group"] != result["recorded_group"])
            & (result["predicted_group"] != "OTHER")
            & (result["recorded_group"] != "OTHER")
            & (result["probability"] > 0.6)  # Only flag confident predictions
        )

        # Generate explanations for mismatches
        mismatch_indices = result.index[result["is_mismatch"]]
        for idx in mismatch_indices:
            row_features = X_df.loc[idx]
            pred_grp = result.loc[idx, "predicted_group"]
            rec_grp = result.loc[idx, "recorded_group"]
            expl, top_feat = self._explain_prediction(
                row_features, pred_grp, rec_grp, all_feature_cols
            )
            result.loc[idx, "explanation"] = expl
            result.loc[idx, "top_features"] = top_feat

        # Log summary
        n_mismatch = result["is_mismatch"].sum()
        logger.info(
            "Strat predictions: %d intervals, %d mismatches (%.1f%%)",
            len(df), n_mismatch,
            100 * n_mismatch / max(1, len(df)),
        )

        return result

    def _explain_prediction(
        self,
        row_features: pd.Series,
        predicted_group: str,
        recorded_group: str,
        feature_cols: List[str],
    ) -> Tuple[str, str]:
        """
        Generate a geochemical justification for why the model predicted
        a different group than what was recorded.

        Returns:
            (explanation_text, top_features_short)
        """
        if self._class_stats is None:
            return ("", "")

        # For each feature, compute how much closer the value is to the
        # predicted group vs the recorded group (using z-score distance)
        contributions = []
        for i, col in enumerate(feature_cols):
            val = row_features.get(col)
            if pd.isna(val):
                continue
            val = float(val)

            # Get display name
            if col == "_depth":
                display = "Depth"
                std_name = "_depth"
            elif i < len(self._feature_standards):
                std_name = self._feature_standards[i]
                display = SHORT_NAMES.get(std_name, col)
            else:
                display = col
                std_name = col

            # Distance to predicted group median vs recorded group median
            pred_med = self._class_stats.median_val(predicted_group, col)
            rec_med = self._class_stats.median_val(recorded_group, col)
            if pred_med is None or rec_med is None:
                continue

            pred_range = self._class_stats.typical_range_str(predicted_group, col)
            rec_range = self._class_stats.typical_range_str(recorded_group, col)

            # How much closer to predicted than recorded?
            dist_pred = abs(val - pred_med)
            dist_rec = abs(val - rec_med)
            if dist_pred + dist_rec == 0:
                continue

            # Positive = closer to predicted group (supports prediction)
            support = (dist_rec - dist_pred) / (dist_rec + dist_pred)

            contributions.append({
                "feature": display,
                "value": val,
                "support": support,
                "pred_range": pred_range,
                "rec_range": rec_range,
                "pred_median": pred_med,
                "rec_median": rec_med,
            })

        # Sort by support (highest = most supports the prediction)
        contributions.sort(key=lambda x: x["support"], reverse=True)

        # Build explanation from top 3 supporting features
        top = contributions[:3]
        if not top:
            return ("Insufficient chemistry data for explanation", "")

        parts = []
        for c in top:
            parts.append(
                f"{c['feature']}={c['value']:.1f}% "
                f"({predicted_group} typical: {c['pred_range']}%, "
                f"{recorded_group} typical: {c['rec_range']}%)"
            )

        explanation = f"Predicted {predicted_group} because: " + "; ".join(parts)
        top_features = ", ".join(f"{c['feature']}" for c in top)

        return (explanation, top_features)

    def summary_dict(self) -> Dict:
        """Serialisable summary for report metadata."""
        return {
            "is_trained": self._is_trained,
            "training_samples": self._training_n,
            "groups": self._training_groups,
            "group_counts": self._class_stats.counts if self._class_stats else {},
            "feature_importances": self._feature_importances,
            "feature_columns": [
                SHORT_NAMES.get(s, s) for s in self._feature_standards
            ] + (["Depth"] if self._depth_col else []),
        }


# ── Session cache ────────────────────────────────────────────────────────

_cached_classifier: Optional[StratClassifier] = None
_cached_data_hash: Optional[int] = None


def get_or_train_classifier(
    merged_df: pd.DataFrame,
    strat_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    force_retrain: bool = False,
) -> Optional[StratClassifier]:
    """
    Get the cached classifier, or train a new one if needed.

    Uses a hash of the DataFrame shape + columns to detect data changes.
    """
    global _cached_classifier, _cached_data_hash

    # Simple hash to detect if data has changed
    data_hash = hash((len(merged_df), tuple(merged_df.columns[:20])))

    if (
        not force_retrain
        and _cached_classifier is not None
        and _cached_classifier.is_trained
        and _cached_data_hash == data_hash
    ):
        logger.info("Using cached strat classifier (%d groups)", len(_cached_classifier.groups))
        return _cached_classifier

    classifier = StratClassifier()
    success = classifier.train(merged_df, strat_col=strat_col, feature_cols=feature_cols)

    if success:
        _cached_classifier = classifier
        _cached_data_hash = data_hash
        return classifier

    return None


def clear_classifier_cache() -> None:
    """Clear the cached classifier (e.g. when data sources change)."""
    global _cached_classifier, _cached_data_hash
    _cached_classifier = None
    _cached_data_hash = None
