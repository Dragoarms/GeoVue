"""Collar map building and rendering for logging review HTML report."""
import html
import json
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from processing.DataManager.column_aliases import ColumnResolver

logger = logging.getLogger(__name__)

try:
    import pyproj
    PYPROJ_AVAILABLE = True
    PYPROJ_IMPORT_ERROR = None
except ImportError as e:
    PYPROJ_AVAILABLE = False
    PYPROJ_IMPORT_ERROR = str(e)

# One-time log when module loads so console shows pyproj status (e.g. when running from Cursor)
if PYPROJ_AVAILABLE:
    logger.info("Collar map: pyproj available for UTM reprojection.")
else:
    logger.warning(
        "Collar map: pyproj not available (%s). Install with: pip install pyproj",
        PYPROJ_IMPORT_ERROR or "import failed",
    )

def _resolve_coordinate_columns(df):
    """Resolve easting/northing column names using ColumnResolver."""
    resolver = ColumnResolver(df)
    return resolver.get("easting"), resolver.get("northing")

def _build_map_points(
    collar_df: pd.DataFrame,
    hole_col: str,
    logger_holes: set,
    all_holes: set,
    utm_zone: Optional[int] = None,
    utm_north: bool = True,
) -> Dict[str, Any]:
    if collar_df.empty:
        return {"points": [], "has_coords": False}
    resolver = ColumnResolver(collar_df)
    collar_hole_col = resolver.get("hole_id") or hole_col
    east_col, north_col = _resolve_coordinate_columns(collar_df)
    if not east_col or not north_col:
        return {"points": [], "has_coords": False}

    df = collar_df[[collar_hole_col, east_col, north_col]].copy()
    df = df.dropna(subset=[east_col, north_col])
    if df.empty:
        return {"points": [], "has_coords": False}

    df = df[df[collar_hole_col].astype(str).str.upper().str.strip().isin(all_holes)]
    if df.empty:
        # No overlap between collar holes and report holes: show all collar points so map still displays
        df = collar_df[[collar_hole_col, east_col, north_col]].copy()
        df = df.dropna(subset=[east_col, north_col])
    if df.empty:
        return {"points": [], "has_coords": False}

    xs = df[east_col].astype(float)
    ys = df[north_col].astype(float)
    # Drop rows with non-finite coords so reprojection doesn't fail on NaN/Inf
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not finite.all():
        df = df.loc[finite].copy()
        xs = df[east_col].astype(float)
        ys = df[north_col].astype(float)
    if df.empty:
        return {"points": [], "has_coords": False}
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    # Heuristic: coords look like projected (e.g. UTM) if outside WGS84 or in typical UTM range
    outside_wgs84 = min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90
    # Typical UTM: easting 10k–~1M, northing 0–10M (N) or -10M–0 (S); require large easting so WGS84 (e.g. 10,45) is not treated as UTM
    looks_like_utm = (10000 <= min_x <= 1500000 and 0 <= min_y <= 10000000) or (
        10000 <= min_x <= 1500000 and -10000000 <= max_y <= 0
    )
    looks_projected = outside_wgs84 or looks_like_utm

    if looks_projected and not PYPROJ_AVAILABLE:
        logger.warning(
            "Collar coordinates look like UTM but pyproj is not available. "
            "Install with: pip install pyproj. Map will show a warning instead of reprojecting. %s",
            f"(Import error: {PYPROJ_IMPORT_ERROR})" if PYPROJ_IMPORT_ERROR else "",
        )

    # Reproject UTM (easting, northing) to WGS84 (lng, lat) when coords look projected
    if looks_projected and PYPROJ_AVAILABLE:
        if utm_zone is None:
            logger.warning(
                "Collar coordinates look projected, but no UTM zone was supplied. "
                "Map will show the projected CRS warning instead of guessing a zone."
            )
            return {"points": [], "has_coords": True, "bounds": None, "warn_projected": True}

        def try_reproject(zone: int, north: bool, use_swapped_xy: bool = False):
            epsg_utm = 32600 + zone if north else 32700 + zone
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True
            )
            east_vals = df[east_col].astype(float)
            north_vals = df[north_col].astype(float)
            if use_swapped_xy:
                east_vals, north_vals = north_vals.copy(), east_vals.copy()
            valid = np.isfinite(east_vals) & np.isfinite(north_vals)
            if not valid.any():
                raise ValueError("No finite coordinates")
            try:
                lng_arr, lat_arr = transformer.transform(
                    east_vals.values[valid], north_vals.values[valid]
                )
            except Exception as e:
                raise ValueError(f"Transform failed: {e}") from e
            lng_arr = np.asarray(lng_arr)
            lat_arr = np.asarray(lat_arr)
            ok = (
                np.isfinite(lng_arr) & np.isfinite(lat_arr)
                & (np.abs(lat_arr) <= 90)
                & (np.abs(lng_arr) <= 180)
            )
            if not ok.any():
                raise ValueError("No valid points after transform")
            sub = df.loc[valid]
            holes = sub.iloc[ok][collar_hole_col].astype(str).str.upper().str.strip().tolist()
            lng_list = lng_arr[ok].tolist()
            lat_list = lat_arr[ok].tolist()
            rows_ll = list(zip(holes, lng_list, lat_list))
            lngs = [r[1] for r in rows_ll]
            lats = [r[2] for r in rows_ll]
            min_lat, max_lat = min(lats), max(lats)
            min_lng, max_lng = min(lngs), max(lngs)
            span_lat = max(max_lat - min_lat, 1e-6)
            span_lng = max(max_lng - min_lng, 1e-6)
            points = []
            for hole, lng, lat in rows_ll:
                points.append(
                    {
                        "x": (lat - min_lat) / span_lat,
                        "y": (lng - min_lng) / span_lng,
                        "hole_id": hole,
                        "is_logger": hole in logger_holes,
                        "lat": lat,
                        "lng": lng,
                    }
                )
            return {"points": points, "has_coords": True, "bounds": [min_lat, min_lng, max_lat, max_lng], "warn_projected": False}

        # Infer hemisphere from northing: positive => north (326xx), negative => south (327xx)
        inferred_north = min_y >= 0 and max_y <= 10000000
        hemispheres = list(dict.fromkeys([utm_north, inferred_north]))
        last_error = None
        for use_swapped_xy in (False, True):
            for north in hemispheres:
                try:
                    return try_reproject(utm_zone, north, use_swapped_xy=use_swapped_xy)
                except Exception as e:
                    last_error = e
        if last_error is not None:
            logger.warning(
                "UTM reprojection failed for supplied zone/hemisphere and column order (%s); "
                "map will show projected coords warning",
                last_error,
            )
            return {"points": [], "has_coords": True, "bounds": None, "warn_projected": True}

    # No reprojection: pass through as-is (or already WGS84)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    points = []
    for _, row in df.iterrows():
        hole = str(row[collar_hole_col]).upper().strip()
        east_val = float(row[east_col])
        north_val = float(row[north_col])
        x = (east_val - min_x) / span_x
        y = (north_val - min_y) / span_y
        points.append(
            {
                "x": x,
                "y": y,
                "hole_id": hole,
                "is_logger": hole in logger_holes,
                "lat": north_val,
                "lng": east_val,
            }
        )
    bounds = [min_y, min_x, max_y, max_x]
    warn_projected = looks_projected
    return {"points": points, "has_coords": True, "bounds": bounds, "warn_projected": warn_projected}


def _render_map(map_data: Dict[str, Any]) -> str:
    """
    Render an interactive Leaflet map with drill hole locations.

    Uses OpenTopoMap (free, no API key). Team holes as grey circle markers,
    logger holes as accent-colored markers. Map fills the widget; bounds fit
    all collar points. Coordinates are passed as-is (lat/lng); if collar data
    is in a projected CRS, reprojection to WGS84 must be added when building
    map_data.
    """
    if not map_data.get("has_coords"):
        return (
            "<div class=\"empty\" data-i18n-fr=\"Aucune coordonnee de colliers disponible.\" "
            "data-i18n-en=\"No collar coordinates available for map.\">"
            "Aucune coordonnee de colliers disponible.</div>"
        )

    if map_data.get("warn_projected"):
        return (
            "<div class=\"warning-box\" data-i18n-fr=\"Les coordonnees des colliers semblent etre en CRS projete (ex. UTM). "
            "La carte exige WGS84 (lat/lng). Utilisez des coordonnees en degres decimaux ou installez pyproj (pip install pyproj) pour une reprojection automatique.\" "
            "data-i18n-en=\"Collar coordinates appear to be in a projected CRS (e.g. UTM). Map requires WGS84 (lat/lng). "
            "Use decimal-degree coordinates or install pyproj (pip install pyproj) for automatic reprojection.\">"
            "Collar coordinates may be in a projected CRS (e.g. UTM). Map requires WGS84 (lat/lng). "
            "Install the <strong>pyproj</strong> package (pip install pyproj) for automatic reprojection.</div>"
        )

    points = map_data["points"]
    bounds = map_data.get("bounds")  # [min_lat, min_lng, max_lat, max_lng]
    logger_count = sum(1 for p in points if p.get("is_logger"))
    team_count = len(points) - logger_count

    # Serialize points and bounds for JS (escape for HTML attribute)
    points_json = json.dumps([{"lat": p["lat"], "lng": p["lng"], "hole_id": p.get("hole_id", ""), "is_logger": p.get("is_logger", False)} for p in points])
    bounds_json = json.dumps(bounds) if bounds else "null"

    return f'''
        <div class="map-container map-container-leaflet">
            <div id="leaflet-map" class="map-leaflet-viewport" data-map-points="{html.escape(points_json)}" data-map-bounds="{html.escape(bounds_json)}"></div>
            <div class="map-legend map-legend-leaflet">
                <span><span class="legend-dot logger-dot"></span> <span data-i18n-en="Logger holes" data-i18n-fr="Vos forages">Logger holes</span> ({logger_count})</span>
                <span><span class="legend-dot team-dot"></span> <span data-i18n-en="Team holes" data-i18n-fr="Forages equipe">Team holes</span> ({team_count})</span>
            </div>
        </div>
    '''


