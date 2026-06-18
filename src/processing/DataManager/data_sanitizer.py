"""
Data Sanitizer - Handles data quality issues across loaded data sources.

Provides:
- Duplicate column detection and resolution
- Cross-file consistency checks
- Data type validation
- Missing data analysis
- Data quality reporting

This module is used by DataCoordinator during initialization to ensure
data integrity before exposing data to UI components.

Author: George Symonds
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SanitizationSeverity(Enum):
    """Severity levels for data issues."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Potential issue, may need review
    ERROR = "error"         # Serious issue, may cause problems
    CRITICAL = "critical"   # Data cannot be used as-is


class DuplicateResolution(Enum):
    """Strategies for resolving duplicate column names."""
    PREFIX_SOURCE = "prefix_source"     # Add source name prefix: drillhole_data.Fe_pct
    SUFFIX_SOURCE = "suffix_source"     # Add source name suffix: Fe_pct_drillhole_data
    KEEP_FIRST = "keep_first"           # Keep only the first occurrence
    KEEP_LAST = "keep_last"             # Keep only the last occurrence
    RENAME_DUPLICATES = "rename_dups"   # Add numeric suffix to duplicates: Fe_pct, Fe_pct_2


@dataclass
class DataIssue:
    """
    Represents a data quality issue found during sanitization.
    
    Attributes:
        severity: How serious the issue is
        category: Type of issue (duplicate, type_mismatch, missing, etc.)
        source: Which data source has the issue
        column: Which column (if applicable)
        message: Human-readable description
        details: Additional context
        resolution: Suggested fix (if any)
        auto_fixed: Whether the issue was automatically resolved
    """
    severity: SanitizationSeverity
    category: str
    source: str
    column: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolution: Optional[str] = None
    auto_fixed: bool = False
    
    def __str__(self):
        prefix = f"[{self.severity.value.upper()}] "
        loc = f"{self.source}"
        if self.column:
            loc += f".{self.column}"
        return f"{prefix}{loc}: {self.message}"


@dataclass
class SanitizationReport:
    """
    Summary report of data sanitization results.
    
    Attributes:
        issues: List of all issues found
        duplicates_found: Number of duplicate column names
        duplicates_resolved: Number of duplicates that were auto-resolved
        type_mismatches: Number of data type inconsistencies
        sources_processed: Number of data sources checked
        columns_processed: Total columns across all sources
    """
    issues: List[DataIssue] = field(default_factory=list)
    duplicates_found: int = 0
    duplicates_resolved: int = 0
    type_mismatches: int = 0
    sources_processed: int = 0
    columns_processed: int = 0
    
    @property
    def has_critical(self) -> bool:
        """Check if any critical issues were found."""
        return any(i.severity == SanitizationSeverity.CRITICAL for i in self.issues)
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return any(i.severity in (SanitizationSeverity.ERROR, SanitizationSeverity.CRITICAL) 
                   for i in self.issues)
    
    @property
    def error_count(self) -> int:
        """Count of error and critical issues."""
        return sum(1 for i in self.issues 
                   if i.severity in (SanitizationSeverity.ERROR, SanitizationSeverity.CRITICAL))
    
    @property
    def warning_count(self) -> int:
        """Count of warning issues."""
        return sum(1 for i in self.issues if i.severity == SanitizationSeverity.WARNING)
    
    def add_issue(self, issue: DataIssue):
        """Add an issue to the report."""
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: SanitizationSeverity) -> List[DataIssue]:
        """Get all issues of a specific severity."""
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_source(self, source: str) -> List[DataIssue]:
        """Get all issues for a specific data source."""
        return [i for i in self.issues if i.source == source]
    
    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            "=" * 60,
            "DATA SANITIZATION REPORT",
            "=" * 60,
            f"Sources processed: {self.sources_processed}",
            f"Columns processed: {self.columns_processed}",
            "",
            f"Duplicate columns: {self.duplicates_found} found, {self.duplicates_resolved} resolved",
            f"Type mismatches: {self.type_mismatches}",
            "",
            f"Issues: {self.error_count} errors, {self.warning_count} warnings",
            "=" * 60,
        ]
        
        if self.issues:
            lines.append("")
            lines.append("ISSUES:")
            for issue in self.issues:
                lines.append(f"  {issue}")
        
        return "\n".join(lines)


class DataSanitizer:
    """
    Validates and sanitizes data across multiple sources.

    Handles:
    - Duplicate column detection and resolution
    - Data type consistency checks
    - Missing data analysis
    - Cross-source validation (e.g., hole IDs match)

    Usage:
        >>> sanitizer = DataSanitizer()
        >>> report = sanitizer.sanitize_sources(geological_store)
        >>> if report.has_errors:
        >>>     print(report.summary())
    """

    # Standard key columns that are expected to exist in multiple sources
    # These are not flagged as duplicates since they're standard identifiers
    STANDARD_KEY_COLUMNS = {
        # Hole identifiers
        "holeid", "hole_id", "bhid", "drillhole_id",
        # Depth columns
        "sampfrom", "sampto", "geolfrom", "geolto",
        "from", "to", "depth_from", "depth_to",
        "from_m", "to_m", "start_depth", "end_depth",
        # Project/site identifiers
        "projectcode", "project_code", "siteid", "site_id",
    }

    def __init__(
        self,
        duplicate_resolution: DuplicateResolution = DuplicateResolution.PREFIX_SOURCE,
        auto_fix: bool = True,
    ):
        """
        Initialize the data sanitizer.

        Args:
            duplicate_resolution: Strategy for handling duplicate column names
            auto_fix: Whether to automatically fix issues when possible
        """
        self.duplicate_resolution = duplicate_resolution
        self.auto_fix = auto_fix

        # Column name tracking
        self._column_sources: Dict[str, List[str]] = defaultdict(list)  # col_name -> [source1, source2, ...]
        self._qualified_names: Dict[str, str] = {}  # original -> qualified

        logger.debug(f"DataSanitizer initialized with resolution={duplicate_resolution.value}, auto_fix={auto_fix}")
    
    def sanitize_sources(self, geological_store) -> SanitizationReport:
        """
        Run full sanitization on all loaded data sources.
        
        Args:
            geological_store: GeologicalStore with loaded data sources
            
        Returns:
            SanitizationReport with all findings
        """
        report = SanitizationReport()
        
        logger.info("Starting data sanitization...")
        
        # Get all data sources
        sources = geological_store.get_data_sources()
        if not sources:
            logger.warning("No data sources to sanitize")
            return report
        
        report.sources_processed = len(sources)
        
        # Phase 1: Collect column information
        self._column_sources.clear()
        self._qualified_names.clear()
        
        for source_name, indexed_source in sources.items():
            schema = indexed_source.schema
            for col_name in schema.columns:
                self._column_sources[col_name.lower()].append(source_name)
                report.columns_processed += 1
        
        # Phase 2: Check for duplicates
        self._check_duplicates(report, sources)
        
        # Phase 3: Check data types
        self._check_data_types(report, sources)
        
        # Phase 4: Check for missing key columns
        self._check_key_columns(report, sources)
        
        # Phase 5: Check hole ID consistency
        self._check_hole_consistency(report, sources)
        
        # Phase 6: Check for high null rates
        self._check_null_rates(report, sources)
        
        logger.info(f"Sanitization complete: {report.error_count} errors, {report.warning_count} warnings")
        
        return report
    
    def _check_duplicates(self, report: SanitizationReport, sources: Dict):
        """Check for and resolve duplicate column names across sources."""
        for col_name_lower, source_list in self._column_sources.items():
            if len(source_list) > 1:
                # Skip standard key columns - these are expected to be in multiple sources
                if col_name_lower in self.STANDARD_KEY_COLUMNS:
                    logger.debug(f"Skipping standard key column '{col_name_lower}' in duplicate check")
                    continue

                report.duplicates_found += 1

                # Create issue
                issue = DataIssue(
                    severity=SanitizationSeverity.WARNING,
                    category="duplicate_column",
                    source=", ".join(source_list),
                    column=col_name_lower,
                    message=f"Column '{col_name_lower}' exists in {len(source_list)} sources",
                    details={"sources": source_list},
                )

                # Auto-resolve if enabled
                if self.auto_fix:
                    resolved_names = self._resolve_duplicate(col_name_lower, source_list)
                    issue.resolution = f"Qualified names: {resolved_names}"
                    issue.auto_fixed = True
                    report.duplicates_resolved += 1

                report.add_issue(issue)
    
    def _resolve_duplicate(self, col_name: str, sources: List[str]) -> Dict[str, str]:
        """
        Resolve a duplicate column name according to the resolution strategy.
        
        Args:
            col_name: The duplicate column name
            sources: List of sources that have this column
            
        Returns:
            Dictionary mapping source to qualified column name
        """
        resolved = {}
        
        for source in sources:
            if self.duplicate_resolution == DuplicateResolution.PREFIX_SOURCE:
                qualified = f"{source}.{col_name}"
            elif self.duplicate_resolution == DuplicateResolution.SUFFIX_SOURCE:
                qualified = f"{col_name}_{source}"
            elif self.duplicate_resolution == DuplicateResolution.RENAME_DUPLICATES:
                # First one keeps original name
                if source == sources[0]:
                    qualified = col_name
                else:
                    idx = sources.index(source) + 1
                    qualified = f"{col_name}_{idx}"
            else:
                qualified = col_name
            
            resolved[source] = qualified
            self._qualified_names[f"{source}.{col_name}"] = qualified
        
        return resolved
    
    def get_qualified_name(self, source: str, col_name: str) -> str:
        """
        Get the qualified (deduplicated) name for a column.
        
        Args:
            source: Data source name
            col_name: Original column name
            
        Returns:
            Qualified column name (may be same as original if no duplicates)
        """
        key = f"{source}.{col_name.lower()}"
        return self._qualified_names.get(key, col_name)
    
    def _check_data_types(self, report: SanitizationReport, sources: Dict):
        """Check for data type inconsistencies."""
        # Track types for columns that appear in multiple sources
        col_types: Dict[str, Dict[str, str]] = defaultdict(dict)  # col -> {source: type}
        
        for source_name, indexed_source in sources.items():
            schema = indexed_source.schema
            for col_name, col_schema in schema.columns.items():
                col_types[col_name.lower()][source_name] = col_schema.data_type.value
        
        # Check for mismatches
        for col_name, source_types in col_types.items():
            if len(source_types) > 1:
                unique_types = set(source_types.values())
                if len(unique_types) > 1:
                    report.type_mismatches += 1
                    
                    report.add_issue(DataIssue(
                        severity=SanitizationSeverity.WARNING,
                        category="type_mismatch",
                        source=", ".join(source_types.keys()),
                        column=col_name,
                        message=f"Column has different types across sources: {dict(source_types)}",
                        details={"types": source_types},
                    ))
    
    def _check_key_columns(self, report: SanitizationReport, sources: Dict):
        """Check that all sources have required key columns."""
        for source_name, indexed_source in sources.items():
            schema = indexed_source.schema
            
            # Check hole ID column exists
            if not schema.hole_id_column:
                report.add_issue(DataIssue(
                    severity=SanitizationSeverity.ERROR,
                    category="missing_key",
                    source=source_name,
                    column=None,
                    message="No hole ID column identified",
                ))
            
            # Check from/to for interval data
            if schema.dataset_type.value == "interval":
                if not schema.from_column:
                    report.add_issue(DataIssue(
                        severity=SanitizationSeverity.ERROR,
                        category="missing_key",
                        source=source_name,
                        column=None,
                        message="Interval data missing 'from' column",
                    ))
                if not schema.to_column:
                    report.add_issue(DataIssue(
                        severity=SanitizationSeverity.ERROR,
                        category="missing_key",
                        source=source_name,
                        column=None,
                        message="Interval data missing 'to' column",
                    ))
    
    def _check_hole_consistency(self, report: SanitizationReport, sources: Dict):
        """Check that hole IDs are consistent across sources."""
        # Collect all hole IDs from each source
        source_holes: Dict[str, Set[str]] = {}
        
        for source_name, indexed_source in sources.items():
            schema = indexed_source.schema
            df = indexed_source.df
            
            if df is not None and schema.hole_id_column:
                hole_col = schema.hole_id_column
                if hole_col in df.columns:
                    holes = set(df[hole_col].dropna().astype(str).str.upper())
                    source_holes[source_name] = holes
        
        if len(source_holes) < 2:
            return  # Need at least 2 sources to compare
        
        # Find holes that don't appear in all sources
        all_holes = set.union(*source_holes.values()) if source_holes else set()
        
        for source_name, holes in source_holes.items():
            missing = all_holes - holes
            if missing and len(missing) < 100:  # Only report if manageable
                report.add_issue(DataIssue(
                    severity=SanitizationSeverity.INFO,
                    category="missing_holes",
                    source=source_name,
                    column=None,
                    message=f"Missing {len(missing)} holes that exist in other sources",
                    details={"missing_count": len(missing), "sample": list(missing)[:10]},
                ))
    
    def _check_null_rates(self, report: SanitizationReport, sources: Dict):
        """Check for columns with high null rates."""
        HIGH_NULL_THRESHOLD = 0.9  # 90% nulls
        
        for source_name, indexed_source in sources.items():
            df = indexed_source.df
            if df is None:
                continue
            
            for col in df.columns:
                null_rate = df[col].isna().mean()
                
                if null_rate >= HIGH_NULL_THRESHOLD:
                    report.add_issue(DataIssue(
                        severity=SanitizationSeverity.INFO,
                        category="high_nulls",
                        source=source_name,
                        column=col,
                        message=f"Column is {null_rate*100:.1f}% null",
                        details={"null_rate": null_rate},
                    ))


# =============================================================================
# Validation Rules (extensible)
# =============================================================================

class ValidationRule:
    """Base class for custom validation rules."""
    
    name: str = "base_rule"
    description: str = ""
    
    def validate(self, df: pd.DataFrame, schema, report: SanitizationReport) -> None:
        """
        Run validation on a DataFrame.
        
        Args:
            df: DataFrame to validate
            schema: DataSourceSchema for the DataFrame
            report: Report to add issues to
        """
        raise NotImplementedError


class DepthOrderRule(ValidationRule):
    """Validate that from < to for interval data."""
    
    name = "depth_order"
    description = "Check that from depth is less than to depth"
    
    def validate(self, df: pd.DataFrame, schema, report: SanitizationReport) -> None:
        if schema.dataset_type.value != "interval":
            return
        
        from_col = schema.from_column
        to_col = schema.to_column
        
        if not from_col or not to_col:
            return
        
        if from_col not in df.columns or to_col not in df.columns:
            return
        
        # Find rows where from >= to
        invalid = df[df[from_col] >= df[to_col]]
        
        if len(invalid) > 0:
            report.add_issue(DataIssue(
                severity=SanitizationSeverity.ERROR,
                category="invalid_depth_order",
                source=schema.name,
                column=None,
                message=f"{len(invalid)} rows have from >= to depth",
                details={"invalid_count": len(invalid), "sample_holes": invalid[schema.hole_id_column].head(5).tolist()},
            ))


class NegativeDepthRule(ValidationRule):
    """Validate that depths are non-negative."""
    
    name = "negative_depth"
    description = "Check for negative depth values"
    
    def validate(self, df: pd.DataFrame, schema, report: SanitizationReport) -> None:
        depth_cols = []
        if schema.from_column:
            depth_cols.append(schema.from_column)
        if schema.to_column:
            depth_cols.append(schema.to_column)
        if schema.depth_column:
            depth_cols.append(schema.depth_column)
        
        for col in depth_cols:
            if col not in df.columns:
                continue
            
            negative = df[df[col] < 0]
            if len(negative) > 0:
                report.add_issue(DataIssue(
                    severity=SanitizationSeverity.WARNING,
                    category="negative_depth",
                    source=schema.name,
                    column=col,
                    message=f"{len(negative)} rows have negative depth values",
                    details={"count": len(negative)},
                ))


class PercentageRangeRule(ValidationRule):
    """Validate that percentage columns are in 0-100 range."""
    
    name = "percentage_range"
    description = "Check that percentage columns are within 0-100"
    
    PERCENTAGE_PATTERNS = ["_pct", "_percent", "percentage", "%"]
    
    def validate(self, df: pd.DataFrame, schema, report: SanitizationReport) -> None:
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if this looks like a percentage column
            is_pct = any(p in col_lower for p in self.PERCENTAGE_PATTERNS)
            if not is_pct:
                continue
            
            # Check for values outside 0-100
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                out_of_range = ((numeric_col < 0) | (numeric_col > 100)).sum()
                
                if out_of_range > 0:
                    report.add_issue(DataIssue(
                        severity=SanitizationSeverity.WARNING,
                        category="percentage_range",
                        source=schema.name,
                        column=col,
                        message=f"{out_of_range} values outside 0-100% range",
                        details={"out_of_range_count": int(out_of_range)},
                    ))
            except Exception:
                pass  # Skip non-numeric columns


# Default validation rules
DEFAULT_RULES = [
    DepthOrderRule(),
    NegativeDepthRule(),
    PercentageRangeRule(),
]


def run_validation_rules(
    df: pd.DataFrame,
    schema,
    report: SanitizationReport,
    rules: Optional[List[ValidationRule]] = None,
) -> None:
    """
    Run validation rules on a DataFrame.
    
    Args:
        df: DataFrame to validate
        schema: DataSourceSchema for the DataFrame
        report: Report to add issues to
        rules: List of rules to run (defaults to DEFAULT_RULES)
    """
    if rules is None:
        rules = DEFAULT_RULES
    
    for rule in rules:
        try:
            rule.validate(df, schema, report)
        except Exception as e:
            logger.error(f"Error running validation rule '{rule.name}': {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def sanitize_geological_store(geological_store, auto_fix: bool = True) -> SanitizationReport:
    """
    Convenience function to sanitize a geological store.
    
    Args:
        geological_store: GeologicalStore to sanitize
        auto_fix: Whether to auto-fix issues
        
    Returns:
        SanitizationReport
    """
    sanitizer = DataSanitizer(auto_fix=auto_fix)
    return sanitizer.sanitize_sources(geological_store)


def log_sanitization_report(report: SanitizationReport):
    """Log a sanitization report at appropriate log levels."""
    logger.info(f"Data sanitization: {report.sources_processed} sources, "
                f"{report.columns_processed} columns")
    
    if report.duplicates_found > 0:
        logger.info(f"  Duplicates: {report.duplicates_found} found, "
                    f"{report.duplicates_resolved} resolved")
    
    if report.type_mismatches > 0:
        logger.warning(f"  Type mismatches: {report.type_mismatches}")
    
    for issue in report.issues:
        if issue.severity == SanitizationSeverity.CRITICAL:
            logger.critical(str(issue))
        elif issue.severity == SanitizationSeverity.ERROR:
            logger.error(str(issue))
        elif issue.severity == SanitizationSeverity.WARNING:
            logger.warning(str(issue))
        else:
            logger.debug(str(issue))


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Simple test
    print("DataSanitizer module loaded successfully")
    
    # Create a sample report
    report = SanitizationReport()
    report.sources_processed = 3
    report.columns_processed = 50
    report.duplicates_found = 2
    report.duplicates_resolved = 2
    
    report.add_issue(DataIssue(
        severity=SanitizationSeverity.WARNING,
        category="duplicate_column",
        source="drillhole_data, exassay",
        column="fe_pct",
        message="Column 'fe_pct' exists in 2 sources",
        auto_fixed=True,
    ))
    
    print(report.summary())
