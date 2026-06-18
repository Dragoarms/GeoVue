"""TypedDict definitions for the logging review HTML report data structures."""
from typing import Any, Dict, List, Optional, TypedDict


class ReportMeta(TypedDict):
    """Metadata for a single report (logger, date range, generated timestamp)."""
    logger: str
    date_from: str
    date_to: str
    generated: str


class ComparisonMetric(TypedDict):
    """One comparison metric (value vs median_all / median_project)."""
    value: float
    median_all: Optional[float]
    median_project: Optional[float]


class ReportSummary(TypedDict):
    """Top-level summary counts for the report."""
    assay_intervals: int
    logging_intervals: int
    unique_holes: int
    strat_codes: int
    total_depth_m: float
    match_rate_pct: Optional[Any]


class ReportData(TypedDict, total=False):
    """
    Full report data dict passed to render_html.
    Required keys: meta, overview, summary, comment_stats, comment_stats_logging,
    comment_coverage, comparisons, wordcloud, fines_summary, grouping_summary,
    outliers, mineralisation, profile_zonation, grouping_kpis, grouping_columns_used,
    outlier_kpis, outlier_box_plot_data, outlier_box_plot_layout, outlier_scatter_data,
    outlier_scatter_layout, map, project_codes, has_project_scope, logging_detail_issue_types.
    """
    meta: ReportMeta
    overview: Dict[str, Any]
    summary: ReportSummary
    comment_stats: Dict[str, Any]
    comment_stats_logging: Dict[str, Any]
    comment_coverage: float
    comparisons: Dict[str, ComparisonMetric]
    wordcloud: Dict[str, int]
    fines_summary: List[str]
    grouping_summary: List[str]
    outliers: List[Dict[str, Any]]
    mineralisation: Dict[str, Any]
    profile_zonation: Dict[str, Any]
    grouping_kpis: Dict[str, Any]
    grouping_columns_used: List[str]
    outlier_kpis: Dict[str, Any]
    outlier_box_plot_data: Any
    outlier_box_plot_layout: Any
    outlier_scatter_data: Any
    outlier_scatter_layout: Any
    outlier_pca_data: Any
    outlier_pca_layout: Any
    map: Dict[str, Any]
    project_codes: List[str]
    has_project_scope: bool
    logging_detail_issue_types: List[Dict[str, str]]


class IntervalItem(TypedDict, total=False):
    """One interval row for evidence tables (fines, logging detail, outliers)."""
    hole_id: Optional[str]
    depth_from: Optional[float]
    depth_to: Optional[float]
    strat: Optional[str]
    classified_as: Optional[str]
    issue: Optional[str]
    significance: Optional[str]
    geochem: Dict[str, Any]
    image: Optional[str]
    recorded_as: Optional[str]
    most_likely: Optional[str]
    reason: Optional[str]
    outlier_score: float
    outlier_reason: Optional[str]
    outlier_elements: Optional[str]
    flags: List[str]
    validation: Optional[str]
    logged_as: Optional[str]
    assay_suggests: Optional[str]
    gangue_pct: Optional[float]
    logged_minerals: Optional[str]


class LoggingDetailIntervals(TypedDict):
    """Intervals for the logging-detail tab (fines, clay, magnetite, goethite, carbonate_gangue, sulphide/manganese/mafics/magnesium gangue)."""
    fines: List[IntervalItem]
    clay: List[IntervalItem]
    magnetite: List[IntervalItem]
    goethite: List[IntervalItem]
    carbonate_gangue: List[IntervalItem]
    sulphide_gangue: List[IntervalItem]
    manganese_gangue: List[IntervalItem]
    mafics_gangue: List[IntervalItem]
    magnesium_gangue: List[IntervalItem]


class GroupingGroupItem(TypedDict, total=False):
    """One group in the grouping tab."""
    group_key: str
    strat: Optional[str]
    cv_max: Optional[float]
    mean_interval_m: float
    max_interval_m: float
    significance: str
    intervals: List[IntervalItem]


class IntervalsForReview(TypedDict):
    """
    Intervals passed to render_html for evidence tables and tabs.
    fines: list of interval dicts (fines issues).
    logging_detail: nested dict with fines, magnetite, goethite, carbonate_gangue lists.
    grouping_flat: flat list of grouping issue intervals.
    grouping: list of groups (each with intervals).
    outliers: list of outlier interval dicts.
    """
    fines: List[IntervalItem]
    logging_detail: LoggingDetailIntervals
    grouping_flat: List[IntervalItem]
    grouping: List[GroupingGroupItem]
    outliers: List[IntervalItem]
