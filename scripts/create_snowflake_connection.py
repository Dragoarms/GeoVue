"""
create_snowflake_connection.py
==============================
Run this ONCE in the ArcGIS Pro Python window (or as a script tool)
to create the Snowflake .sde connection file.

Any team member can run this — no credentials are stored in the file.
Authentication uses SSO (externalbrowser), so it opens a browser window
on first connect.

Usage:
    1. Open ArcGIS Pro
    2. Go to Analysis > Python > Python Window
    3. Paste and run this entire script
    4. A browser SSO window will pop up — sign in with your FMG account
    5. The .sde file is saved to your project folder

The .sde file can also be shared via Teams/SharePoint — it contains
no passwords, just connection parameters.
"""

import arcpy
import os

# ── CONNECTION PARAMETERS ─────────────────────────────────────────────────────
# These are the same for everyone on the team.

SERVER      = "FMG-WN74261.snowflakecomputing.com"  # Your Snowflake account URL
DATABASE    = "DA_EXPLORATION"
WAREHOUSE   = "WH_DA_EXPLORATION"
SCHEMA      = "STG_ACQ_GABON"
ROLE        = "DA_EXPLORATION"  # Role that grants access — adjust if different

# Where to save the .sde file
# Default: current ArcGIS Pro project folder, or user's home if no project open
try:
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    OUT_FOLDER = aprx.homeFolder
except:
    OUT_FOLDER = os.path.expanduser("~")

CONNECTION_NAME = "Snowflake_DA_EXPLORATION.sde"


# ── CREATE THE CONNECTION ─────────────────────────────────────────────────────

def create_connection():
    """
    Create a Snowflake .sde connection file using SSO authentication.
    
    arcpy.management.CreateDatabaseConnection parameters for Snowflake:
      - database_platform: "SNOWFLAKE" (case-insensitive in newer versions)
      - instance: the server URL
      - account_authentication: "OPERATING_SYSTEM_AUTH" triggers externalbrowser/SSO
      - database: the Snowflake database name
      - Additional properties via the connection string or advanced options
    """
    
    out_path = os.path.join(OUT_FOLDER, CONNECTION_NAME)
    
    # Remove existing if present (to avoid "already exists" error)
    if os.path.exists(out_path):
        os.remove(out_path)
        print(f"  Removed existing: {out_path}")
    
    print(f"Creating Snowflake connection...")
    print(f"  Server:    {SERVER}")
    print(f"  Database:  {DATABASE}")
    print(f"  Warehouse: {WAREHOUSE}")
    print(f"  Schema:    {SCHEMA}")
    print(f"  Role:      {ROLE}")
    print(f"  Auth:      SSO (externalbrowser)")
    print(f"  Output:    {out_path}")
    print()
    
    # ── METHOD 1: CreateDatabaseConnection ────────────────────────────────
    # This is the "official" way. However, Snowflake support in Arc Pro's
    # CreateDatabaseConnection can be version-dependent.
    
    try:
        result = arcpy.management.CreateDatabaseConnection(
            out_folder_path=OUT_FOLDER,
            out_name=CONNECTION_NAME,
            database_platform="SNOWFLAKE",
            instance=SERVER,
            account_authentication="OPERATING_SYSTEM_AUTH",
            database=DATABASE,
            # Additional connection properties passed via keyword args
            # These may vary by Arc Pro version
        )
        
        sde_path = result.getOutput(0)
        print(f"✓ Connection created: {sde_path}")
        
        # Now set warehouse, role, schema via connection properties
        # These are set when first used in a query layer
        print()
        print("NOTE: Warehouse, Role, and Schema are set in the query layer SQL:")
        print(f'  USE WAREHOUSE "{WAREHOUSE}";')
        print(f'  USE ROLE "{ROLE}";')
        print(f'  USE SCHEMA "{DATABASE}"."{SCHEMA}";')
        
        return sde_path
        
    except Exception as e:
        print(f"  CreateDatabaseConnection failed: {e}")
        print("  Trying Method 2 (manual .sde file)...")
        return create_connection_manual()


def create_connection_manual():
    """
    METHOD 2: Write the .sde connection file directly.
    
    An .sde file is actually a SQLite database with connection properties.
    If CreateDatabaseConnection doesn't support Snowflake properly in your
    Arc Pro version, we can create it by:
    
    1. Creating a basic connection via the GUI first (any settings)
    2. Then modifying the properties
    
    OR use the ODBC/connection string approach.
    """
    
    out_path = os.path.join(OUT_FOLDER, CONNECTION_NAME)
    
    # Alternative: Use a connection string in the query layer directly
    # This bypasses the .sde file entirely
    print()
    print("=" * 60)
    print("ALTERNATIVE: Skip the .sde file entirely")
    print("=" * 60)
    print()
    print("You can create a Query Layer directly with a connection string.")
    print("In ArcGIS Pro Python window, run:")
    print()
    print("─" * 60)
    print(f"""
import arcpy

# Get current map
aprx = arcpy.mp.ArcGISProject("CURRENT")
m = aprx.listMaps()[0]

# Snowflake ODBC connection string
# Requires Snowflake ODBC driver installed
conn_str = (
    "DRIVER={{Snowflake}};"
    "SERVER={SERVER};"
    "DATABASE={DATABASE};"
    "WAREHOUSE={WAREHOUSE};"
    "SCHEMA={SCHEMA};"
    "ROLE={ROLE};"
    "AUTHENTICATOR=externalbrowser;"
)

sql = \\"\\"\\"
SELECT
    "HOLEID",
    "PlannedHoleID", 
    "HoleStatus",
    "PROJECTCODE",
    "PROSPECT",
    "Area",
    "WGS84_LONG",
    "WGS84_LAT",
    "PlannedDepth",
    "DRILLEDDEPTH",
    "RigID",
    "IsPadCleared",
    "DrillingMethod",
    "STARTDATE",
    "ENDDATE"
FROM DA_EXPLORATION.STG_ACQ_GABON.COLLAR
WHERE "WGS84_LONG" IS NOT NULL
  AND "WGS84_LAT"  IS NOT NULL
\\"\\"\\"

result = arcpy.management.MakeQueryLayer(
    input_database=conn_str,
    out_layer_name="DrillPlan_Snowflake",
    query=sql,
    oid_fields="HOLEID",
    shape_type="POINT",
    srid="4326",
    spatial_reference=arcpy.SpatialReference(4326),
    spatial_properties="DEFINE_SPATIAL_PROPERTIES",
    x_field="WGS84_LONG",
    y_field="WGS84_LAT"
)

lyr = result.getOutput(0)
m.addLayer(lyr)
print("✓ Layer added to map")
aprx.save()
""")
    print("─" * 60)
    print()
    print("This requires the Snowflake ODBC driver to be installed.")
    print("Download from: https://docs.snowflake.com/en/developer-guide/odbc/odbc-download")
    
    return None


# ── SNOWFLAKE ODBC DRIVER CHECK ──────────────────────────────────────────────

def check_odbc_driver():
    """Check if the Snowflake ODBC driver is installed."""
    import subprocess
    try:
        # Windows: check ODBC drivers via registry or odbcconf
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-OdbcDriver | Where-Object {$_.Name -like '*Snowflake*'} | Format-Table Name, Platform"],
            capture_output=True, text=True, timeout=10
        )
        if "Snowflake" in result.stdout:
            print("✓ Snowflake ODBC driver found:")
            print(result.stdout.strip())
            return True
        else:
            print("✗ Snowflake ODBC driver NOT found.")
            print("  Download from: https://docs.snowflake.com/en/developer-guide/odbc/odbc-download")
            print("  Or install via: https://sfc-repo.snowflakecomputing.com/odbc/win64/latest/index.html")
            return False
    except Exception as e:
        print(f"Could not check ODBC drivers: {e}")
        return None


# ── GUI-BASED SETUP INSTRUCTIONS ─────────────────────────────────────────────

def print_gui_instructions():
    """Step-by-step for manual GUI setup if scripting fails."""
    print()
    print("=" * 60)
    print("MANUAL SETUP (GUI)")
    print("=" * 60)
    print("""
    1. Install Snowflake ODBC driver if not already present
    
    2. In ArcGIS Pro:
       Insert > Connections > New Database Connection
       
    3. Fill in the dialog:
       ┌─────────────────────────┬──────────────────────────────────┐
       │ Database Platform       │ Snowflake                        │
       │ Server                  │ FMG-WN74261.snowflakecomputing.com│
       │ Authentication Type     │ User  (see note below)           │
       │ User Name               │ your.name@fmgl.com.au            │
       │ Password                │ (leave blank for SSO — see below)│
       │ Role                    │ DA_EXPLORATION                   │
       │ Database                │ DA_EXPLORATION                   │
       │ Warehouse               │ WH_DA_EXPLORATION                │
       │ Advanced Options        │ authenticator=externalbrowser     │
       └─────────────────────────┴──────────────────────────────────┘
    
    CRITICAL: The "externalbrowser" SSO option
    ─────────────────────────────────────────
    Arc Pro's GUI doesn't have an "externalbrowser" dropdown.
    
    Workaround:
      a) Set Authentication Type to "User"
      b) Enter your FMG email as User Name
      c) Leave Password BLANK
      d) In "Advanced Options" enter:
            authenticator=externalbrowser
      
      This tells the Snowflake ODBC driver to pop up a browser
      window for Okta/SSO sign-in instead of using a password.
    
    4. Click OK — a browser window should open for SSO
    
    5. The connection appears in the Catalog pane under Databases
    
    6. To share: copy the .sde file from your project folder to
       a shared location (Teams/SharePoint). It contains no 
       passwords — each user authenticates via their own SSO.
    """)


# ── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Snowflake Connection Setup for ArcGIS Pro")
    print("=" * 60)
    print()
    
    # Check ODBC driver
    print("Checking for Snowflake ODBC driver...")
    check_odbc_driver()
    print()
    
    # Try automated creation
    sde = create_connection()
    
    if not sde:
        print_gui_instructions()
    
    print()
    print("Done. If you hit issues, the GUI instructions above should get you there.")
