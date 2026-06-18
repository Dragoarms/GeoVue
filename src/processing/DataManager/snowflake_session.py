"""
snowflake_session.py  –  Singleton session manager for all Snowflake access in GeoVue.

Design goals:
  • One SSO login per process – the browser opens exactly once.
  • Phase 1 (COLLAR only) runs at startup in a background thread,
    feeds DepthValidator and BulkPhotoRenamer immediately.
  • Phase 2 (all geological tables) runs only when the user clicks
    "Pre-Load Snowflake Data" in the main GUI.
  • Phase 2 results are cached as Parquet in AppData/GeoVue/sf_cache/
    with an 8-hour TTL so subsequent loads are instant.
  • Every failure is silent – CSV data keeps working.

Usage:
    from processing.DataManager.snowflake_session import get_session_manager
    sm = get_session_manager()
    sm.start_phase1(on_status=splash.update_status)
    # Later, in main GUI button:
    sm.start_phase2(data_coordinator, progress_callback=...)

Author: George Symonds
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports – gracefully degrade when not installed / not in build
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    import pyarrow  # noqa: F401  (just checking it's present)
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class SessionState(Enum):
    IDLE            = auto()   # not yet started
    CONNECTING      = auto()   # SSO browser open
    PHASE1_LOADING  = auto()   # fetching COLLAR
    PHASE1_DONE     = auto()   # collar ready, conn still live
    PHASE2_LOADING  = auto()   # fetching all tables
    READY           = auto()   # phase 2 complete
    FAILED          = auto()   # unrecoverable error
    OFFLINE         = auto()   # snowflake pkg missing / user chose not to use it


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------
@dataclass
class SFTable:
    """Describes one Snowflake table to fetch."""
    database:    str
    schema:      str
    table:       str
    geovue_name: str          # key used in DataCoordinator / GeologicalStore
    phase:       int          # 1 = collar-only startup, 2 = user-triggered
    is_collar:   bool = False # True → merged into collar_depth_ranges

    @property
    def fqn(self) -> str:
        return f'"{self.database}"."{self.schema}"."{self.table}"'

    @property
    def cache_filename(self) -> str:
        """Source-specific Parquet cache filename.

        The cache must be keyed by the Snowflake object, not just by
        geovue_name, because a GeoVue source can be re-plumbed to a new
        database/schema/table while keeping the same in-app source name.
        """
        return f"{self.database}__{self.schema}__{self.table}.parquet"

    @property
    def legacy_cache_filename(self) -> str:
        """Previous cache filename format kept only for cleanup."""
        return f"{self.geovue_name}.parquet"


# Explicit table registry – verified against snowflake_schema_catalog.csv 2026-04-17.
# Phase 1 runs at startup; Phase 2 runs on user demand.
#
# Schema sources:
#   AA_EXPLORATION_AFR.SENS_GAB      = Preferred Gabon geological data views
#   AA_EXPLORATION_AFR.SLN_ACQ_AFR_PIVOT = Fallback for views not present in SENS_GAB
#   DA_EXPLORATION.STG_PHOTOS        = Photo management views (Anvil, tray numbering)
TABLE_REGISTRY: List[SFTable] = [
    # ── PHASE 1 + PHASE 2 ──────────────────────────────────────────────────
    # SENS_GAB.COLLAR covers Gabon holes. It is used at startup for fast depth
    # validation and loaded into DataCoordinator during Phase 2 so correlation
    # has actual collar coordinates for the minimap/section selector.
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "COLLAR", "collar_sens_gab", phase=2, is_collar=True),

    # ── PHASE 2 ─────────────────────────────────────────────────────────────
    # --- SENS_GAB (preferred Gabon geology/assay data) ---
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "LITHOLOGY",               "lithology_diamond",       phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "COLLAR_SURVEY",           "collar_survey",           phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "COMPOSITE_GEOLOGY",       "composite_geology",       phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "CORE_RECOVERY",           "core_recovery",           phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "GEOPHYSICSDETAILS",       "geophysicsdetails",       phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "GEOTECHROCKMASS",         "geotechrockmass",         phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "NORMATIVE_MINERALOGY",    "normative_mineralogy",    phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "STRUCTURE",               "structure",               phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "SUMMARY_LOGGING_ASSAYS",  "summary_logging_assays",  phase=2),
    SFTable("AA_EXPLORATION_AFR", "SENS_GAB",      "WEIGHBAR_DENSITY",        "weighbar_density",        phase=2),

    # --- SLN_ACQ_AFR_PIVOT fallback ---
    # SENS_GAB has summary assay fields but no primary SAMPLE_BEST_ASSAYS view.
    SFTable("AA_EXPLORATION_AFR", "SLN_ACQ_AFR_PIVOT", "SAMPLE_BEST_ASSAYS",  "sample_best_assays",      phase=2),

    # --- STG_PHOTOS (photo management) ---
    SFTable("DA_EXPLORATION",     "STG_PHOTOS",    "CORE_PHOTOS_GABON",       "core_photos_gabon",       phase=2),
    SFTable("DA_EXPLORATION",     "STG_PHOTOS",    "TRAYNUMBER",              "traynumber",              phase=2),
    # TRAYSPLITDETAILS removed — 0% HOLEID match to COLLAR, all BGD* holes from a different project
]

# Connection defaults (same as snowflake_source.py)
_DEFAULT_ACCOUNT    = "FMG-WN74261"
_DEFAULT_WAREHOUSE  = "WH_DA_EXPLORATION"
_CACHE_MAX_AGE_SECS = 8 * 3600   # 8 hours


# ---------------------------------------------------------------------------
# Singleton manager
# ---------------------------------------------------------------------------
class SnowflakeSessionManager:
    """
    Manages the Snowflake connection lifecycle and data cache for GeoVue.
    Thread-safe.  Instantiate once via get_session_manager().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = SessionState.IDLE if SNOWFLAKE_AVAILABLE else SessionState.OFFLINE
        self._conn = None
        self._sf_user: Optional[str] = None
        self._sf_role: Optional[str] = None
        self._error_message: Optional[str] = None
        self._user_hint: Optional[str] = None   # set via set_user_hint()

        # Phase 1 output
        self.collar_depth_ranges: Dict[str, Tuple[float, float]] = {}
        self._phase1_complete = threading.Event()

        # Phase 2 output  – geovue_name → DataFrame
        self._phase2_data: Dict[str, "pd.DataFrame"] = {}
        self._phase2_loaded_tables: List[str] = []
        self._phase2_failed_tables: List[str] = []

        # Status callbacks – each callable(state: SessionState, message: str)
        self._status_callbacks: List[Callable] = []

        # Config manager reference (set via set_config_manager)
        self._config_manager = None

        # Cache directory
        self._cache_dir: Optional[Path] = None
        self._init_cache_dir()

        logger.debug(
            f"SnowflakeSessionManager created: "
            f"snowflake_available={SNOWFLAKE_AVAILABLE}, "
            f"parquet_available={PARQUET_AVAILABLE}"
        )

    # ── public properties ────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def is_ready_for_depth_validation(self) -> bool:
        """True once collar data is available (Phase 1 done or better)."""
        return bool(self.collar_depth_ranges)

    @property
    def phase2_dataframes(self) -> Dict[str, "pd.DataFrame"]:
        """Returns dict of geovue_name → DataFrame for loaded Phase 2 tables."""
        with self._lock:
            return dict(self._phase2_data)

    @property
    def user(self) -> Optional[str]:
        return self._sf_user

    @property
    def role(self) -> Optional[str]:
        return self._sf_role

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    @property
    def phase2_summary(self) -> str:
        """Human-readable summary of Phase 2 results."""
        loaded = len(self._phase2_loaded_tables)
        failed = len(self._phase2_failed_tables)
        if not loaded and not failed:
            return "Not loaded"
        parts = [f"{loaded} tables loaded"]
        if failed:
            parts.append(f"{failed} failed")
        if self._phase2_failed_tables:
            parts.append(f"({', '.join(self._phase2_failed_tables[:3])})")
        return " · ".join(parts)

    # ── subscription API ─────────────────────────────────────────────────────

    def add_status_callback(self, cb: Callable) -> None:
        """Register a callback(state, message) invoked on state transitions."""
        with self._lock:
            if cb not in self._status_callbacks:
                self._status_callbacks.append(cb)

    def remove_status_callback(self, cb: Callable) -> None:
        with self._lock:
            self._status_callbacks = [c for c in self._status_callbacks if c is not cb]

    # ── Phase 1 ─────────────────────────────────────────────────────────────

    def start_phase1(
        self,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Launch Phase 1 (COLLAR fetch) in a daemon background thread.

        Args:
            on_status: Optional callback(message, detail) for splash updates.
                       Called on the background thread – use root.after() if
                       you need to update tkinter widgets.
        """
        if not SNOWFLAKE_AVAILABLE:
            self._set_state(SessionState.OFFLINE, "Snowflake connector not installed")
            return

        with self._lock:
            if self._state not in (SessionState.IDLE, SessionState.FAILED, SessionState.OFFLINE):
                logger.debug("Phase 1 already started or complete – skipping")
                return

        t = threading.Thread(
            target=self._run_phase1,
            args=(on_status,),
            daemon=True,
            name="sf-phase1",
        )
        t.start()

    def wait_for_phase1(self, timeout: float = 30.0) -> bool:
        """
        Block until Phase 1 completes or timeout expires.
        Safe to call from any thread.
        Returns True if phase 1 produced data.
        """
        self._phase1_complete.wait(timeout=timeout)
        return bool(self.collar_depth_ranges)

    # ── Phase 2 ─────────────────────────────────────────────────────────────

    def start_phase2(
        self,
        data_coordinator=None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        on_complete: Optional[Callable[[bool, str], None]] = None,
        force_refresh: bool = False,
    ) -> None:
        """
        Launch Phase 2 (all geological tables) in a daemon background thread.

        Args:
            data_coordinator: DataCoordinator to load DataFrames into.
            progress_callback: callback(table_name, done_count, total_count)
            on_complete: callback(success, summary_message) fired when done.
            force_refresh: if True, ignore cache and fetch from Snowflake.
        """
        with self._lock:
            if self._state == SessionState.PHASE2_LOADING:
                logger.warning("Phase 2 already in progress")
                return

        t = threading.Thread(
            target=self._run_phase2,
            args=(data_coordinator, progress_callback, on_complete, force_refresh),
            daemon=True,
            name="sf-phase2",
        )
        t.start()

    # ── cache helpers ────────────────────────────────────────────────────────

    def has_valid_cache(self) -> bool:
        """True if most Phase 2 tables have a non-stale Parquet cache.

        Tables that failed to fetch (e.g. permissions) won't have cache
        files — that's fine, we don't let them block the whole cache.
        """
        if not PARQUET_AVAILABLE or not self._cache_dir:
            return False
        phase2_tables = [t for t in TABLE_REGISTRY if t.phase == 2]
        now = time.time()
        valid_count = 0
        for tbl in phase2_tables:
            path = self._cache_dir / tbl.cache_filename
            if not path.exists():
                continue  # table may have failed on last fetch — skip it
            if now - path.stat().st_mtime > _CACHE_MAX_AGE_SECS:
                return False  # any stale file → full refresh
            valid_count += 1
        # Need at least half the tables cached to consider it valid
        return valid_count >= len(phase2_tables) // 2

    def cache_age_description(self) -> str:
        """Human-readable cache age, e.g. '2h 14m ago' or 'No cache'."""
        if not self._cache_dir:
            return "No cache"
        # Use the oldest file's mtime
        phase2_tables = [t for t in TABLE_REGISTRY if t.phase == 2]
        oldest = None
        for tbl in phase2_tables:
            p = self._cache_dir / tbl.cache_filename
            if p.exists():
                mtime = p.stat().st_mtime
                if oldest is None or mtime < oldest:
                    oldest = mtime
        if oldest is None:
            return "No cache"
        age_s = int(time.time() - oldest)
        if age_s < 60:
            return f"{age_s}s ago"
        if age_s < 3600:
            return f"{age_s // 60}m ago"
        h = age_s // 3600
        m = (age_s % 3600) // 60
        return f"{h}h {m}m ago"

    def clear_cache(self) -> None:
        """Delete all cached Parquet files."""
        if not self._cache_dir:
            return
        for tbl in TABLE_REGISTRY:
            for filename in {tbl.cache_filename, tbl.legacy_cache_filename}:
                p = self._cache_dir / filename
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete cache file {p}: {e}")
        logger.info("Snowflake Parquet cache cleared")

    # ── public – user hint ───────────────────────────────────────────────────

    def set_user_hint(self, email: str) -> None:
        """
        Set the Snowflake login hint (email address).
        Call this before start_phase1() with the value from ConfigManager.
        Priority: this value > SNOWFLAKE_USER env var > Windows username fallback.
        """
        self._user_hint = email.strip() if email else None
        logger.debug(f"Snowflake user hint set: {self._user_hint}")

    # ── internal – connection ────────────────────────────────────────────────

    def _resolve_user(self) -> str:
        """Return the best available Snowflake login hint."""
        # 1. Explicitly configured (from ConfigManager via set_user_hint)
        if self._user_hint:
            return self._user_hint
        # 2. Environment variable override
        env_user = os.environ.get("SNOWFLAKE_USER", "").strip()
        if env_user:
            return env_user
        # 3. Windows username fallback (likely wrong for SSO — just gives a hint)
        import getpass
        return f"{getpass.getuser()}@fortescue.com"

    def _connect(self) -> bool:
        """Open SSO connection if not already open.  Returns True on success."""
        if self._conn is not None:
            try:
                self._conn.cursor().execute("SELECT 1")
                return True          # connection still alive
            except Exception:
                self._conn = None   # stale – reconnect

        self._set_state(SessionState.CONNECTING, "Opening SSO browser…")
        try:
            user_hint = self._resolve_user()
            logger.info(f"Connecting to Snowflake (hint={user_hint})…")
            self._conn = snowflake.connector.connect(
                account=_DEFAULT_ACCOUNT,
                user=user_hint,
                warehouse=_DEFAULT_WAREHOUSE,
                authenticator="externalbrowser",
                login_timeout=120,
            )
            cur = self._conn.cursor()
            cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
            self._sf_user, self._sf_role, sf_wh = cur.fetchone()
            cur.close()
            logger.info(
                f"Snowflake connected: user={self._sf_user}, "
                f"role={self._sf_role}, wh={sf_wh}"
            )
            return True
        except Exception as e:
            self._error_message = str(e)
            logger.error(f"Snowflake connection failed: {e}")
            self._conn = None
            return False

    def _close_connection(self) -> None:
        """Close connection if open."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            logger.debug("Snowflake connection closed")

    # ── internal – Phase 1 ───────────────────────────────────────────────────

    def _run_phase1(self, on_status: Optional[Callable]) -> None:
        self._set_state(SessionState.CONNECTING, "Connecting to Snowflake…")
        _notify(on_status, "Connecting to Snowflake…", "Browser window may open")

        if not self._connect():
            self._set_state(SessionState.FAILED, f"Connection failed: {self._error_message}")
            _notify(on_status, "Snowflake offline", "Using CSV data only")
            self._phase1_complete.set()
            return

        self._set_state(SessionState.PHASE1_LOADING, "Fetching collar depths…")
        _notify(on_status, "Fetching collar data…", "Querying COLLAR views")

        collar_tables = [t for t in TABLE_REGISTRY if t.is_collar]
        raw_depths: Dict[str, float] = {}   # merged max depth per hole
        conflicts = 0

        for tbl in collar_tables:
            result = self._fetch_collar_raw(tbl)
            if result is None:
                continue
            for hole_id, depth in result.items():
                if hole_id in raw_depths:
                    if depth != raw_depths[hole_id]:
                        conflicts += 1
                    raw_depths[hole_id] = max(raw_depths[hole_id], depth)
                else:
                    raw_depths[hole_id] = depth

        if conflicts:
            logger.info(f"Phase 1: {conflicts} depth conflicts resolved by taking max")

        # Round up to nearest 20m
        self.collar_depth_ranges = {
            hole: (0.0, math.ceil(d / 20) * 20)
            for hole, d in raw_depths.items()
        }

        hole_count = len(self.collar_depth_ranges)
        if hole_count:
            msg = f"✓ Collar: {hole_count:,} holes"
            self._set_state(SessionState.PHASE1_DONE, msg)
            _notify(on_status, msg, "Snowflake collar data ready")
            logger.info(f"Phase 1 complete: {hole_count:,} holes")
        else:
            self._set_state(SessionState.FAILED, "No collar data returned")
            _notify(on_status, "Snowflake collar: no data", "")
            self._close_connection()

        self._phase1_complete.set()

    def _fetch_collar_raw(self, tbl: SFTable) -> Optional[Dict[str, float]]:
        """Fetch {HOLEID: max_depth} from one COLLAR view (unrounded)."""
        HOLEID_CANDIDATES = ["HOLEID", "HOLE_ID", "DRILLHOLE", "DHID"]
        DEPTH_CANDIDATES  = ["DEPTH", "EOH", "ENDDEPTH", "MAX_DEPTH", "MAXDEPTH"]
        try:
            cur = self._conn.cursor()
            cur.execute(f"SELECT * FROM {tbl.fqn} LIMIT 0")
            available = [d[0].upper() for d in cur.description]
            cur.close()

            hole_col  = next((c for c in available if c in HOLEID_CANDIDATES), None)
            depth_col = next((c for c in available if c in DEPTH_CANDIDATES),  None)

            if not hole_col or not depth_col:
                logger.warning(
                    f"{tbl.fqn}: required columns not found. "
                    f"Available: {available[:15]}"
                )
                return None

            cur = self._conn.cursor()
            cur.execute(
                f'SELECT "{hole_col}", "{depth_col}" '
                f'FROM {tbl.fqn} '
                f'WHERE "{depth_col}" IS NOT NULL'
            )
            rows = cur.fetchall()
            cur.close()

            raw: Dict[str, float] = {}
            for hole_raw, depth_raw in rows:
                if hole_raw is None:
                    continue
                try:
                    hid = str(hole_raw).strip().upper()
                    d   = float(depth_raw)
                    if hid not in raw or d > raw[hid]:
                        raw[hid] = d
                except (ValueError, TypeError):
                    continue

            logger.info(f"{tbl.fqn}: {len(raw):,} collar rows fetched")
            return raw

        except Exception as e:
            logger.warning(f"{tbl.fqn}: collar fetch failed – {e}")
            return None

    # ── config manager for custom tables ─────────────────────────────────────

    def set_config_manager(self, config_manager) -> None:
        """Set a reference to ConfigManager for reading custom table config."""
        self._config_manager = config_manager

    def _load_custom_tables(self) -> List[SFTable]:
        """Load user-configured custom Snowflake tables from ConfigManager."""
        cm = getattr(self, '_config_manager', None)
        if cm is None:
            return []
        try:
            custom_list = cm.get("snowflake_custom_tables", [])
            tables = []
            for entry in custom_list:
                if not entry.get("enabled", True):
                    continue
                tables.append(SFTable(
                    database=entry["database"],
                    schema=entry["schema"],
                    table=entry["table"],
                    geovue_name=entry["geovue_name"],
                    phase=2,
                ))
            return tables
        except Exception as e:
            logger.warning(f"Could not load custom Snowflake tables from config: {e}")
            return []

    # ── internal – Phase 2 ───────────────────────────────────────────────────

    def _run_phase2(
        self,
        data_coordinator,
        progress_callback: Optional[Callable],
        on_complete: Optional[Callable],
        force_refresh: bool,
    ) -> None:
        self._set_state(SessionState.PHASE2_LOADING, "Loading Snowflake tables…")

        phase2_tables = [t for t in TABLE_REGISTRY if t.phase == 2]

        # Append user-configured custom tables from ConfigManager
        custom = self._load_custom_tables()
        if custom:
            logger.info(f"Adding {len(custom)} custom Snowflake tables from config")
            phase2_tables.extend(custom)

        total = len(phase2_tables)
        loaded, failed = [], []

        # ── try cache first ──────────────────────────────────────────────────
        if not force_refresh and self.has_valid_cache():
            logger.info("Phase 2: loading from Parquet cache")
            for i, tbl in enumerate(phase2_tables):
                cache_path = self._cache_dir / tbl.cache_filename
                if not cache_path.exists():
                    if tbl.is_collar:
                        logger.info(
                            f"No cache for required collar source {tbl.geovue_name}; fetching from Snowflake"
                        )
                        if self._connect():
                            df = self._fetch_table(tbl)
                            if df is not None and not df.empty:
                                self._store_phase2(data_coordinator, tbl, df)
                                self._save_to_cache(tbl, df)
                                loaded.append(tbl.geovue_name)
                            else:
                                failed.append(tbl.geovue_name)
                        else:
                            failed.append(tbl.geovue_name)
                        continue
                    # Table was never fetched (e.g. permissions) — skip, don't count as failure
                    logger.debug(f"No cache for {tbl.geovue_name} — skipping (likely failed on previous fetch)")
                    continue
                _progress(progress_callback, f"Cache: {tbl.geovue_name}", i, total)
                df = self._load_from_cache(tbl)
                if df is not None:
                    self._store_phase2(data_coordinator, tbl, df)
                    loaded.append(tbl.geovue_name)
                else:
                    failed.append(tbl.geovue_name)

            self._finish_phase2(loaded, failed, on_complete, from_cache=True)
            return

        # ── fetch from Snowflake ─────────────────────────────────────────────
        # Ensure connection is live (Phase 1 may have left it open)
        if not self._connect():
            msg = f"Snowflake connection failed: {self._error_message}"
            self._set_state(SessionState.FAILED, msg)
            if on_complete:
                on_complete(False, msg)
            return

        for i, tbl in enumerate(phase2_tables):
            _progress(progress_callback, tbl.geovue_name, i, total)
            df = self._fetch_table(tbl)
            if df is not None and not df.empty:
                self._store_phase2(data_coordinator, tbl, df)
                self._save_to_cache(tbl, df)
                loaded.append(tbl.geovue_name)
            else:
                failed.append(tbl.geovue_name)

        self._close_connection()
        self._finish_phase2(loaded, failed, on_complete, from_cache=False)

    def _fetch_table(self, tbl: SFTable) -> Optional["pd.DataFrame"]:
        """Fetch one table as a DataFrame."""
        if not PANDAS_AVAILABLE:
            return None
        try:
            t0 = time.time()
            cur = self._conn.cursor()
            cur.execute(f"SELECT * FROM {tbl.fqn}")
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            cur.close()
            df = pd.DataFrame(rows, columns=cols)
            logger.info(
                f"{tbl.fqn}: {len(df):,} rows in {time.time()-t0:.1f}s"
            )
            return df
        except Exception as e:
            logger.warning(f"{tbl.fqn}: fetch failed – {e}")
            return None

    def _store_phase2(self, data_coordinator, tbl: SFTable, df: "pd.DataFrame") -> None:
        """Merge a DataFrame into DataCoordinator and local cache dict."""
        with self._lock:
            self._phase2_data[tbl.geovue_name] = df

        if data_coordinator is None:
            return
        try:
            if hasattr(data_coordinator, "add_source_from_dataframe"):
                data_coordinator.add_source_from_dataframe(df, name=tbl.geovue_name)
            elif hasattr(data_coordinator, "_geological_store"):
                gs = data_coordinator._geological_store
                if gs and hasattr(gs, "add_source_from_dataframe"):
                    gs.add_source_from_dataframe(df, name=tbl.geovue_name)
        except Exception as e:
            logger.warning(f"Could not load {tbl.geovue_name} into DataCoordinator: {e}")

    def _finish_phase2(
        self, loaded: List[str], failed: List[str],
        on_complete: Optional[Callable], from_cache: bool,
    ) -> None:
        with self._lock:
            self._phase2_loaded_tables = loaded
            self._phase2_failed_tables = failed

        src = "cache" if from_cache else "Snowflake"
        msg = f"✓ {len(loaded)}/{len(loaded)+len(failed)} tables from {src}"
        if failed:
            msg += f" ({len(failed)} failed)"

        new_state = SessionState.READY if loaded else SessionState.FAILED
        self._set_state(new_state, msg)
        logger.info(f"Phase 2 complete: {msg}")

        if on_complete:
            try:
                on_complete(bool(loaded), msg)
            except Exception as e:
                logger.warning(f"on_complete callback error: {e}")

    # ── internal – cache ─────────────────────────────────────────────────────

    def _init_cache_dir(self) -> None:
        try:
            appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
            cache = Path(appdata) / "GeoVue" / "sf_cache"
            cache.mkdir(parents=True, exist_ok=True)
            self._cache_dir = cache
            logger.debug(f"Snowflake cache dir: {cache}")
        except Exception as e:
            logger.warning(f"Could not create Snowflake cache dir: {e}")
            self._cache_dir = None

    def _save_to_cache(self, tbl: SFTable, df: "pd.DataFrame") -> None:
        if not PARQUET_AVAILABLE or not self._cache_dir:
            return
        try:
            path = self._cache_dir / tbl.cache_filename
            df.to_parquet(path, index=False)
            logger.debug(f"Cached {tbl.geovue_name} → {path}")
        except Exception as e:
            logger.warning(f"Could not cache {tbl.geovue_name}: {e}")

    def _load_from_cache(self, tbl: SFTable) -> Optional["pd.DataFrame"]:
        if not PARQUET_AVAILABLE or not self._cache_dir:
            return None
        try:
            path = self._cache_dir / tbl.cache_filename
            if not path.exists():
                legacy_path = self._cache_dir / tbl.legacy_cache_filename
                if legacy_path.exists():
                    logger.info(
                        f"Ignoring legacy Snowflake cache for {tbl.geovue_name}: "
                        f"{legacy_path.name}"
                    )
                return None
            age = time.time() - path.stat().st_mtime
            if age > _CACHE_MAX_AGE_SECS:
                return None
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Could not read cache for {tbl.geovue_name}: {e}")
            return None

    # ── internal – state ─────────────────────────────────────────────────────

    def _set_state(self, new_state: SessionState, message: str = "") -> None:
        with self._lock:
            self._state = new_state
            callbacks = list(self._status_callbacks)
        logger.debug(f"SnowflakeSession state → {new_state.name}: {message}")
        for cb in callbacks:
            try:
                cb(new_state, message)
            except Exception as e:
                logger.debug(f"Status callback error: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _notify(cb: Optional[Callable], msg: str, detail: str) -> None:
    if cb:
        try:
            cb(msg, detail)
        except Exception:
            pass


def _progress(cb: Optional[Callable], name: str, done: int, total: int) -> None:
    if cb:
        try:
            cb(name, done, total)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_SESSION_MANAGER: Optional[SnowflakeSessionManager] = None
_SESSION_LOCK = threading.Lock()


def get_session_manager() -> SnowflakeSessionManager:
    """Return the process-level SnowflakeSessionManager singleton."""
    global _SESSION_MANAGER
    with _SESSION_LOCK:
        if _SESSION_MANAGER is None:
            _SESSION_MANAGER = SnowflakeSessionManager()
    return _SESSION_MANAGER


def is_snowflake_available() -> bool:
    return SNOWFLAKE_AVAILABLE
