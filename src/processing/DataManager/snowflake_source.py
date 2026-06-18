"""
snowflake_source.py - Snowflake data provider for GeoVue's GeologicalStore.

Queries Snowflake tables and returns pandas DataFrames that can be loaded
directly into GeologicalStore via add_source_from_dataframe().

This keeps all Snowflake-specific logic (connection, queries, table mappings)
isolated from the rest of the data layer. The existing CSV pipeline remains
as a fallback when Snowflake is unavailable.

Architecture:
    SnowflakeSource  -->  DataFrame  -->  GeologicalStore.add_source_from_dataframe()
                                              |
    CSV files        -->  DataFrame  -->  GeologicalStore.add_source() (existing)

Authentication:
    - SSO externalbrowser (no passwords or roles stored)
    - User hint from: SNOWFLAKE_USER env var -> Windows username
    - Role: Snowflake assigns the user's default role automatically

Usage:
    from processing.DataManager.snowflake_source import SnowflakeSource

    sf = SnowflakeSource()
    if sf.connect():
        for name, df in sf.fetch_all():
            geological_store.add_source_from_dataframe(df, name=name)
        sf.close()

Requires: snowflake-connector-python
Author: George Symonds
"""

import logging
import os
import getpass
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator

try:
    import snowflake.connector
    import pandas as pd
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── CONNECTION DEFAULTS ──────────────────────────────────────────────
DEFAULT_ACCOUNT = "FMG-WN74261"
DEFAULT_WAREHOUSE = "WH_DA_EXPLORATION"
DEFAULT_AUTHENTICATOR = "externalbrowser"
FMG_EMAIL_DOMAIN = "fortescue.com"


def resolve_user() -> str:
    """
    Build a best-guess Snowflake login hint from the environment.

    The externalbrowser authenticator ultimately resolves identity from
    whoever signs in via SSO — this is just the login_name hint.

    Priority:
      1. SNOWFLAKE_USER environment variable
      2. Windows session username + @fortescue.com
    """
    env_user = os.environ.get("SNOWFLAKE_USER", "").strip()
    if env_user:
        return env_user
    win_user = getpass.getuser()
    return f"{win_user}@{FMG_EMAIL_DOMAIN}"


@dataclass
class SnowflakeTableConfig:
    """
    Configuration for a single Snowflake table to expose as a GeoVue data source.

    Attributes:
        database: Snowflake database name
        schema: Snowflake schema name
        table: Table or view name
        geovue_name: Name to register in GeologicalStore (e.g. "exassay", "collar")
        query_override: Optional custom SQL (if None, SELECT * FROM table)
        hole_filter_column: Column containing hole IDs for optional filtering
        enabled: Whether to include this source
    """
    database: str
    schema: str
    table: str
    geovue_name: str
    query_override: Optional[str] = None
    hole_filter_column: Optional[str] = None
    enabled: bool = True

    @property
    def fqn(self) -> str:
        """Fully qualified table name."""
        return f'"{self.database}"."{self.schema}"."{self.table}"'


# ── DEFAULT TABLE REGISTRY ───────────────────────────────────────────
# Extend this as you discover which tables map to which GeoVue sources.
# Run snowflake_schema_discovery.py to populate column names.
DEFAULT_TABLES: List[SnowflakeTableConfig] = [
    SnowflakeTableConfig(
        database="DA_EXPLORATION",
        schema="STG_PHOTOS",
        table="CORE_PHOTOS_GABON",
        geovue_name="core_photos",
    ),
    # Add more tables here after running schema discovery:
    # SnowflakeTableConfig(
    #     database="AA_EXPLORATION_AFR",
    #     schema="SENS_GAB",
    #     table="<TABLE_NAME>",
    #     geovue_name="exassay",
    # ),
]


class SnowflakeSource:
    """
    Snowflake data provider for GeoVue.

    Manages a single SSO connection and provides DataFrames for each
    configured table. Designed to be used during DataCoordinator.initialize()
    as an alternative/supplement to CSV file loading.

    Connection lifecycle:
        1. connect() - opens SSO browser, authenticates
        2. fetch_table() / fetch_all() - queries tables, returns DataFrames
        3. close() - releases connection

    Thread safety: Not thread-safe. Use from main thread only.
    """

    def __init__(
        self,
        account: str = DEFAULT_ACCOUNT,
        warehouse: str = DEFAULT_WAREHOUSE,
        authenticator: str = DEFAULT_AUTHENTICATOR,
        tables: Optional[List[SnowflakeTableConfig]] = None,
    ):
        self._account = account
        self._warehouse = warehouse
        self._authenticator = authenticator
        self._user_hint = resolve_user()
        self._tables = tables or DEFAULT_TABLES
        self._conn: Optional["snowflake.connector.SnowflakeConnection"] = None
        self._is_connected = False

        # Populated after connect() from actual session
        self._sf_user: Optional[str] = None
        self._sf_role: Optional[str] = None

        logger.debug(
            f"SnowflakeSource initialized: account={account}, "
            f"user_hint={self._user_hint}, tables={len(self._tables)}"
        )

    @property
    def user(self) -> Optional[str]:
        """The actual Snowflake user (populated after connect())."""
        return self._sf_user

    @property
    def role(self) -> Optional[str]:
        """The actual Snowflake role (populated after connect())."""
        return self._sf_role

    @property
    def is_available(self) -> bool:
        """Whether the Snowflake connector package is installed."""
        return SNOWFLAKE_AVAILABLE

    @property
    def is_connected(self) -> bool:
        """Whether an active connection exists."""
        return self._is_connected and self._conn is not None

    def connect(self, timeout: int = 120) -> bool:
        """
        Open a Snowflake connection via SSO.

        This will open a browser window for authentication on the first call.
        Subsequent calls in the same process reuse the cached SSO token.

        Args:
            timeout: Login timeout in seconds

        Returns:
            True if connected successfully, False otherwise
        """
        if not SNOWFLAKE_AVAILABLE:
            logger.error("snowflake-connector-python not installed")
            return False

        if self._is_connected:
            logger.debug("Already connected to Snowflake")
            return True

        logger.info(f"Connecting to Snowflake (hint={self._user_hint})...")
        t0 = time.time()

        try:
            self._conn = snowflake.connector.connect(
                account=self._account,
                user=self._user_hint,
                warehouse=self._warehouse,
                authenticator=self._authenticator,
                login_timeout=timeout,
            )
            self._is_connected = True
            elapsed = time.time() - t0

            # Query actual session identity from Snowflake
            cur = self._conn.cursor()
            cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
            self._sf_user, self._sf_role, sf_wh = cur.fetchone()
            cur.close()

            logger.info(
                f"Connected to Snowflake in {elapsed:.1f}s "
                f"(user={self._sf_user}, role={self._sf_role}, warehouse={sf_wh})"
            )
            return True

        except Exception as e:
            logger.error(f"Snowflake connection failed: {e}")
            self._conn = None
            self._is_connected = False
            return False

    def close(self):
        """Close the Snowflake connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception as e:
                logger.debug(f"Error closing Snowflake connection: {e}")
            finally:
                self._conn = None
                self._is_connected = False
                logger.debug("Snowflake connection closed")

    def fetch_table(
        self,
        config: SnowflakeTableConfig,
        hole_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Optional["pd.DataFrame"]:
        """
        Fetch a single table as a DataFrame.

        Args:
            config: Table configuration
            hole_ids: Optional list of hole IDs to filter by
            limit: Optional row limit (for testing)

        Returns:
            DataFrame or None if query fails
        """
        if not self.is_connected:
            logger.error("Not connected to Snowflake")
            return None

        t0 = time.time()

        try:
            if config.query_override:
                query = config.query_override
                params = ()
            else:
                query = f"SELECT * FROM {config.fqn}"

                where_clauses = []
                params_list = []

                # Optional hole ID filter
                if hole_ids and config.hole_filter_column:
                    placeholders = ", ".join(["%s"] * len(hole_ids))
                    where_clauses.append(
                        f'"{config.hole_filter_column}" IN ({placeholders})'
                    )
                    params_list.extend(hole_ids)

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                if limit:
                    query += f" LIMIT {limit}"

                params = tuple(params_list)

            logger.info(f"Fetching {config.geovue_name} from {config.fqn}...")

            cur = self._conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()

            df = pd.DataFrame(rows, columns=columns)
            elapsed = time.time() - t0

            logger.info(
                f"  Fetched {config.geovue_name}: {len(df):,} rows, "
                f"{len(df.columns)} cols in {elapsed:.1f}s"
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching {config.geovue_name}: {e}")
            return None

    def fetch_all(
        self,
        hole_ids: Optional[List[str]] = None,
        progress_callback=None,
    ) -> Iterator[Tuple[str, "pd.DataFrame"]]:
        """
        Fetch all enabled tables as (name, DataFrame) pairs.

        Yields results as they complete so the caller can load them
        incrementally into GeologicalStore.

        Args:
            hole_ids: Optional list of hole IDs to filter all tables by
            progress_callback: Optional callback(message, count)

        Yields:
            (geovue_name, DataFrame) for each successfully fetched table
        """
        enabled = [t for t in self._tables if t.enabled]

        for i, config in enumerate(enabled):
            if progress_callback:
                progress_callback(
                    f"Fetching {config.geovue_name} from Snowflake...", i
                )

            df = self.fetch_table(config, hole_ids=hole_ids)
            if df is not None and not df.empty:
                yield (config.geovue_name, df)
            else:
                logger.warning(f"Skipping {config.geovue_name}: no data returned")

    def test_access(self) -> Dict[str, str]:
        """
        Quick access test for all configured tables.

        Returns:
            Dict of {geovue_name: status_string}
        """
        if not self.is_connected:
            return {t.geovue_name: "NOT CONNECTED" for t in self._tables}

        results = {}
        for config in self._tables:
            try:
                cur = self._conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {config.fqn}")
                count = cur.fetchone()[0]
                cur.close()
                results[config.geovue_name] = f"OK ({count:,} rows)"
            except Exception as e:
                results[config.geovue_name] = f"FAIL: {e}"

        return results


# ── Module-level convenience ─────────────────────────────────────────

def is_snowflake_available() -> bool:
    """Check if the Snowflake connector is installed."""
    return SNOWFLAKE_AVAILABLE
