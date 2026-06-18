"""
arcpro_drillplan_setup.py
=========================
ArcGIS Pro script to set up the drill plan layers in your
"2 week and 30 day schedule.aprx" template.

Two approaches depending on what works best on your network:

  OPTION A: Snowflake Query Layer (preferred — single source of truth)
  OPTION B: AGOL Feature Layer + Join to Min Summary

Run from ArcGIS Pro Python window or as a standalone script.
"""

import arcpy
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

APRX_PATH = r"C:\path\to\2 week and 30 day schedule.aprx"  # UPDATE THIS
MAP_NAME  = "Map"  # Name of the map frame in the .aprx

# Snowflake connection (for Option A)
SNOWFLAKE_SDE = r"C:\path\to\snowflake_connection.sde"  # UPDATE THIS

# AGOL services (for Option B)
AGOL_PLANNED = "https://services3.arcgis.com/tm6XNpR7uYkeDXYc/arcgis/rest/services/GAB_ACQ_PLANNED_DRILLING/FeatureServer/0"
AGOL_MINSUM  = "https://services3.arcgis.com/tm6XNpR7uYkeDXYc/arcgis/rest/services/GAB_GEO_FMG_AT_ARC_MIN_SUMMARY_acQuire/FeatureServer"
# Check which sublayer has the fields you need:
#   /0 = GAB_ACQ_ASSAYED_MINSUM
#   /1 = GAB_ACQ_LOGGED_MINSUM
AGOL_MINSUM_LAYER = AGOL_MINSUM + "/0"  # adjust as needed


# ══════════════════════════════════════════════════════════════════════════════
# OPTION A: SNOWFLAKE QUERY LAYER
# ══════════════════════════════════════════════════════════════════════════════
#
# This pulls everything from Snowflake via a single SQL query that joins
# COLLAR with the min summary table, giving you Best_Total, IsPadCleared,
# Flagged, Drilled status all in one layer.
#
# PREREQUISITE: You need a Snowflake .sde connection file.
#   - In ArcGIS Pro: Insert > Connections > New Database Connection
#   - Database Platform: Snowflake
#   - Account: FMG-WN74261
#   - Authentication: externalbrowser (SSO)
#   - Database: DA_EXPLORATION
#   - Schema: STG_ACQ_GABON
#   - Warehouse: WH_DA_EXPLORATION
#
# ══════════════════════════════════════════════════════════════════════════════

SNOWFLAKE_SQL = """
SELECT
    c."HOLEID",
    c."PlannedHoleID",
    c."HoleStatus",
    c."PROJECTCODE",
    c."PROSPECT",
    c."Area",
    c."BEST_X",
    c."BEST_Y",
    c."BEST_Z",
    c."WGS84_LONG",
    c."WGS84_LAT",
    c."PlannedDepth",
    c."DRILLEDDEPTH",
    c."RigID",
    c."HolePriority",
    c."DrillingMethod",
    c."STARTDATE",
    c."ENDDATE",
    c."IsPadCleared",
    c."IsRehabilitated",
    c."AZIMUTH",
    c."DIP",

    -- From min summary: total drilled metres and key assay columns
    ms."HOLEDEPTH"          AS "Best_Total",
    ms."FE_PCT_BEST"        AS "Fe_Best",
    ms."SIO2_PCT_BEST"      AS "SiO2_Best",
    ms."AL2O3_PCT_BEST"     AS "Al2O3_Best",
    ms."P_PCT_BEST"          AS "P_Best",
    ms."LOI_PCT_BEST"        AS "LOI_Best",

    -- Derived: is it drilled?
    CASE
        WHEN c."HoleStatus" = 'Drilled' THEN 'Yes'
        ELSE 'No'
    END AS "Drilled",

    -- Flagged placeholder (add real column name if it exists in your collar)
    -- c."Flagged",
    'No' AS "Flagged"

FROM DA_EXPLORATION.STG_ACQ_GABON.COLLAR c
LEFT JOIN DA_EXPLORATION.STG_ACQ_GABON.SUMMARY_LOGGING_ASSAYS ms
    ON c."HOLEID" = ms."HOLEID"
    AND ms."SAMPFROM" = 0  -- top-of-hole summary row, adjust if needed

WHERE c."WGS84_LONG" IS NOT NULL
  AND c."WGS84_LAT"  IS NOT NULL
"""


def setup_snowflake_query_layer():
    """Option A: Add Snowflake query layer to the map."""
    aprx = arcpy.mp.ArcGISProject(APRX_PATH)
    m = aprx.listMaps(MAP_NAME)[0]

    # Create query layer
    result = arcpy.management.MakeQueryLayer(
        input_database=SNOWFLAKE_SDE,
        out_layer_name="DrillPlan_Snowflake",
        query=SNOWFLAKE_SQL,
        oid_fields="HOLEID",
        shape_type="POINT",
        srid="4326",  # WGS84
        spatial_reference=arcpy.SpatialReference(4326),
        # Map x/y to the WGS84 columns
        spatial_properties="DEFINE_SPATIAL_PROPERTIES",
        coords_trust_info="COORDINATES_ARE_TRUSTED",
        x_field="WGS84_LONG",
        y_field="WGS84_LAT"
    )

    lyr = result.getOutput(0)
    m.addLayer(lyr)

    print("✓ Snowflake query layer added. Apply symbology next.")
    aprx.save()
    return aprx


# ══════════════════════════════════════════════════════════════════════════════
# OPTION B: AGOL FEATURE LAYERS + JOIN
# ══════════════════════════════════════════════════════════════════════════════

def setup_agol_layers():
    """Option B: Add AGOL layers and join min summary to planned drilling."""
    aprx = arcpy.mp.ArcGISProject(APRX_PATH)
    m = aprx.listMaps(MAP_NAME)[0]

    # Add planned drilling layer
    planned_lyr = arcpy.management.MakeFeatureLayer(
        AGOL_PLANNED, "GAB_Planned_Drilling"
    ).getOutput(0)
    m.addLayer(planned_lyr)

    # Add min summary as a standalone table (for join)
    minsum_tbl = arcpy.management.MakeTableView(
        AGOL_MINSUM_LAYER, "MinSummary"
    ).getOutput(0)

    # Join min summary to planned drilling on HOLEID
    arcpy.management.AddJoin(
        in_layer_or_view="GAB_Planned_Drilling",
        in_field="HOLEID",          # adjust field name to match your layer
        join_table="MinSummary",
        join_field="HOLEID",        # adjust to match
        join_type="KEEP_ALL"        # keep planned holes even without assays
    )

    print("✓ AGOL layers added with join. Apply symbology next.")
    aprx.save()
    return aprx


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOLOGY - ARCADE EXPRESSION
# ══════════════════════════════════════════════════════════════════════════════
#
# This is your existing Arcade expression, adapted for the query layer field
# names. Apply as "Unique Values" symbology on a custom expression.
#
# In ArcGIS Pro:
#   1. Right-click layer > Symbology
#   2. Choose "Unique Values"
#   3. Click the expression button (calculator icon) next to the field dropdown
#   4. Paste the Arcade below
#   5. Set symbol for each class (see color guide below)
#

ARCADE_EXPRESSION = """
// Drill Status Classification
// For use with Snowflake query layer or joined AGOL layers

// Field names — adjust if using AGOL join (prefix may differ)
var drilled      = $feature.Drilled       // or $feature.HoleStatus
var isPadCleared = $feature.IsPadCleared
var flagged      = $feature.Flagged
var bestTotal    = $feature.Best_Total     // or $feature.HOLEDEPTH

// Undrilled classification
if (drilled == "No" || Lower(Text($feature.HoleStatus)) == "planned") {
    if (isPadCleared == "Yes") {
        return "Cleared"
    } else if (flagged == "Yes") {
        return "Flagged"
    } else {
        return "Planned"
    }
}

// Drilled — graduated by total depth
if (IsEmpty(bestTotal) || bestTotal == null) {
    return "No Data"
} else if (bestTotal <= 1) {
    return "0-1m"
} else if (bestTotal <= 5) {
    return "1-5m"
} else if (bestTotal <= 10) {
    return "5-10m"
} else if (bestTotal <= 25) {
    return "10-25m"
} else if (bestTotal <= 50) {
    return "25-50m"
} else if (bestTotal <= 100) {
    return "50-100m"
} else if (bestTotal <= 200) {
    return "100-200m"
} else {
    return ">200m"
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOLOGY COLOR GUIDE
# ══════════════════════════════════════════════════════════════════════════════
#
# Apply in ArcGIS Pro Symbology pane after pasting the Arcade expression.
# Use "Unique Values" renderer. Suggested symbols:
#
# ┌──────────────┬─────────────────────┬────────┬────────────────────────────┐
# │ Class        │ Symbol              │ Color  │ Notes                      │
# ├──────────────┼─────────────────────┼────────┼────────────────────────────┤
# │ Planned      │ Circle, hollow      │ #888888│ Grey outline, no fill      │
# │ Flagged      │ Triangle, hollow    │ #FFD700│ Gold outline               │
# │ Cleared      │ Square, hollow      │ #00CC66│ Green outline              │
# ├──────────────┼─────────────────────┼────────┼────────────────────────────┤
# │ No Data      │ Circle, filled      │ #CCCCCC│ Light grey                 │
# │ 0-1m         │ Circle, filled  4pt │ #FEE5D9│ Very light red             │
# │ 1-5m         │ Circle, filled  5pt │ #FCBBA1│ Light salmon               │
# │ 5-10m        │ Circle, filled  6pt │ #FC9272│ Salmon                     │
# │ 10-25m       │ Circle, filled  7pt │ #FB6A4A│ Orange-red                 │
# │ 25-50m       │ Circle, filled  8pt │ #EF3B2C│ Red                        │
# │ 50-100m      │ Circle, filled  9pt │ #CB181D│ Dark red                   │
# │ 100-200m     │ Circle, filled 10pt │ #99000D│ Very dark red              │
# │ >200m        │ Circle, filled 11pt │ #67000D│ Deepest red                │
# └──────────────┴─────────────────────┴────────┴────────────────────────────┘
#
# The graduated size + sequential red ramp gives an immediate visual read of
# drilling progress — bigger + darker = deeper holes.
#
# For the planned status symbols, use different SHAPES (not just color) so
# the map is readable in greyscale/printout:
#   - Planned  = circle (ubiquitous, doesn't draw attention)
#   - Flagged  = triangle (warning/attention)
#   - Cleared  = square (actionable/ready)


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMMATIC SYMBOLOGY (if you want to script it)
# ══════════════════════════════════════════════════════════════════════════════

def apply_symbology_programmatic(layer):
    """
    Apply unique values symbology programmatically.
    Note: ArcPy symbology control is limited — for complex Arcade-based
    symbology it's often easier to set up manually once, save as a .lyrx
    file, then apply with arcpy.management.ApplySymbologyFromLayer.
    
    This function creates a .lyrx template approach.
    """
    # Save your manually-configured symbology as a layer file:
    #   Right-click layer > Save As Layer File > drillplan_symbology.lyrx
    #
    # Then apply it programmatically:
    SYMBOLOGY_LYRX = r"C:\path\to\drillplan_symbology.lyrx"  # UPDATE

    arcpy.management.ApplySymbologyFromLayer(
        in_layer=layer,
        in_symbology_layer=SYMBOLOGY_LYRX,
        symbology_fields=[["VALUE_FIELD", "DrillStatus"]],
        update_symbology="MAINTAIN"
    )
    print("✓ Symbology applied from .lyrx template")


# ══════════════════════════════════════════════════════════════════════════════
# MAP EXPORT HELPER (for PPT pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def export_map_to_png(aprx_path, layout_name, output_png, dpi=200):
    """
    Export a layout from the .aprx to PNG.
    Use this in the PPT pipeline to generate map images.
    """
    aprx = arcpy.mp.ArcGISProject(aprx_path)
    layout = aprx.listLayouts(layout_name)[0]
    layout.exportToPNG(output_png, resolution=dpi)
    print(f"✓ Exported {output_png} at {dpi} DPI")
    del aprx


def zoom_to_holes(aprx_path, map_name, layout_name, hole_ids, buffer_pct=0.15):
    """
    Zoom the map frame to fit a set of planned holes.
    
    Parameters:
        hole_ids:   list of HOLEIDs to include in the extent
        buffer_pct: fractional buffer around the extent (0.15 = 15%)
    """
    aprx = arcpy.mp.ArcGISProject(aprx_path)
    m = aprx.listMaps(map_name)[0]
    layout = aprx.listLayouts(layout_name)[0]
    mf = layout.listElements("MAPFRAME_ELEMENT")[0]

    # Query the layer to get extent of selected holes
    lyr = m.listLayers("DrillPlan*")[0]  # adjust layer name
    
    # Build where clause
    ids_str = ",".join(f"'{h}'" for h in hole_ids)
    where = f'"HOLEID" IN ({ids_str})'
    
    # Select and zoom
    arcpy.management.SelectLayerByAttribute(lyr, "NEW_SELECTION", where)
    
    # Get extent from selection
    desc = arcpy.Describe(lyr)
    ext = desc.extent  # This gets the extent of selected features
    
    # Apply buffer
    dx = (ext.XMax - ext.XMin) * buffer_pct
    dy = (ext.YMax - ext.YMin) * buffer_pct
    new_ext = arcpy.Extent(
        ext.XMin - dx, ext.YMin - dy,
        ext.XMax + dx, ext.YMax + dy,
        spatial_reference=ext.spatialReference
    )
    
    mf.camera.setExtent(new_ext)
    
    # Clear selection
    arcpy.management.SelectLayerByAttribute(lyr, "CLEAR_SELECTION")
    
    aprx.save()
    print(f"✓ Zoomed to {len(hole_ids)} holes with {buffer_pct*100}% buffer")
    del aprx


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Drill Plan ArcGIS Pro Setup")
    print("=" * 60)
    print()
    print("Choose your approach:")
    print("  A) Snowflake Query Layer (needs .sde connection)")
    print("  B) AGOL Feature Layers + Join")
    print()
    print("Then apply symbology using the Arcade expression above.")
    print()
    print("Once symbology is set, save as .lyrx for reuse.")
    print("Use zoom_to_holes() + export_map_to_png() for the PPT pipeline.")
