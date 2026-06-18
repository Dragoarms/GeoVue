#!/usr/bin/env python3
"""
Snowflake Connection Test for GeoVue
=====================================
Minimal connectivity and access check for all Snowflake schemas used by GeoVue.

Tests:
  1. SSO authentication via externalbrowser (dynamic user + role)
  2. Read access to each database/schema combination
  3. Row count for each table (verifies SELECT privilege)
  4. Role grants and database role verification

Run this on any FMG machine to verify a user can pull data for GeoVue.

Usage:
    python snowflake_connection_test.py
    python snowflake_connection_test.py --user someone.else@fortescue.com
    python snowflake_connection_test.py --verbose

Requires: pip install snowflake-connector-python
"""

import argparse
import getpass
import os
import sys
import time
from datetime import datetime

try:
    import snowflake.connector
except ImportError:
    print("ERROR: snowflake-connector-python not installed.")
    print("  pip install snowflake-connector-python")
    sys.exit(1)


# ── CONNECTION CONFIG ────────────────────────────────────────────────
ACCOUNT = "FMG-WN74261"
WAREHOUSE = "WH_DA_EXPLORATION"
AUTHENTICATOR = "externalbrowser"
FMG_EMAIL_DOMAIN = "fortescue.com"

# Required database role for read access to exploration data
REQUIRED_DB_ROLE = "data.AA.EXPLORATION.AFR.role.SelfServiceViewer"

# ── SCHEMAS TO TEST ──────────────────────────────────────────────────
# (database, schema, description)
TEST_TARGETS = [
    ("AA_EXPLORATION_AFR", "SENS_GAB", "Gabon sensor/analytical data"),
    ("DA_EXPLORATION", "STG_ACQ_GABON", "Gabon acquisition staging"),
    ("DA_EXPLORATION", "STG_PHOTOS", "Core photo metadata"),
    ("DA_EXPLORATION", "STG_GIS", "GIS spatial data"),
]


def resolve_user(override: str = None) -> str:
    """
    Build a login hint for externalbrowser SSO.

    The actual identity comes from whoever signs in via the browser —
    this is just a hint to Snowflake for the login_name field.
    """
    if override:
        return override
    env_user = os.environ.get("SNOWFLAKE_USER", "").strip()
    if env_user:
        return env_user
    win_user = getpass.getuser()
    return f"{win_user}@{FMG_EMAIL_DOMAIN}"


def test_connection(user_hint: str, verbose: bool = False) -> bool:
    """
    Run the full connection and access test suite.

    Connects via SSO with no role specified — Snowflake assigns the
    user's default role automatically.

    Returns True if all tests pass, False otherwise.
    """
    print("=" * 65)
    print("  GeoVue Snowflake Connection Test")
    print("=" * 65)
    print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Account    : {ACCOUNT}")
    print(f"  User hint  : {user_hint}")
    print(f"  Warehouse  : {WAREHOUSE}")
    print(f"  Auth       : SSO (externalbrowser)")
    print(f"  Targets    : {len(TEST_TARGETS)} schemas")
    print("=" * 65)
    print()

    # ── Step 1: Authenticate ─────────────────────────────────────────
    print("[1/4] Authenticating via SSO...")
    print("      (A browser window should open — sign in with your FMG account)")
    print()

    t0 = time.time()
    try:
        conn = snowflake.connector.connect(
            account=ACCOUNT,
            user=user_hint,
            warehouse=WAREHOUSE,
            authenticator=AUTHENTICATOR,
        )
        auth_time = time.time() - t0
        print(f"  OK  Authenticated in {auth_time:.1f}s")
        print()
    except Exception as e:
        print(f"  FAIL  AUTHENTICATION FAILED: {e}")
        print()
        print("  Troubleshooting:")
        print("    - Are you on the FMG network or VPN?")
        print("    - Is snowflake-connector-python installed?")
        print("    - Try: --user firstname.lastname@fortescue.com")
        return False

    # ── Step 2: Verify session ───────────────────────────────────────
    print("[2/4] Verifying session...")
    try:
        cur = conn.cursor()
        cur.execute("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()")
        sf_user, sf_role, sf_wh = cur.fetchone()
        cur.close()
        print(f"  OK  User      : {sf_user}")
        print(f"  OK  Role      : {sf_role}")
        print(f"  OK  Warehouse : {sf_wh}")
        print()
    except Exception as e:
        print(f"  FAIL  SESSION VERIFICATION FAILED: {e}")
        conn.close()
        return False

    # ── Step 3: Check role grants ────────────────────────────────────
    print("[3/4] Checking database role grants...")
    try:
        cur = conn.cursor()
        cur.execute(f"SHOW GRANTS TO ROLE {sf_role}")
        grants = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        cur.close()

        target_dbs = {t[0] for t in TEST_TARGETS}
        found_db_access = set()

        if verbose:
            print(f"  Grants for role {sf_role}:")

        for row in grants:
            grant_dict = dict(zip(col_names, row))
            privilege = grant_dict.get("privilege", "")
            granted_on = grant_dict.get("granted_on", "")
            name = grant_dict.get("name", "")

            if verbose:
                print(f"    {privilege:>20} on {granted_on}: {name}")

            if privilege == "USAGE" and granted_on in ("DATABASE", "DATABASE ROLE"):
                for db in target_dbs:
                    if db in str(name):
                        found_db_access.add(db)

        for db in target_dbs:
            if db in found_db_access:
                print(f"  OK    USAGE on {db}")
            else:
                print(f"  WARN  No direct USAGE grant on {db} (may be via inherited database role)")

        print()

    except Exception as e:
        print(f"  WARN  Could not check grants: {e}")
        print(f"        (Not fatal — proceeding with access tests)")
        print()

    # ── Step 4: Test each schema ─────────────────────────────────────
    print("[4/4] Testing schema access...")
    print()

    all_passed = True
    results = []

    for db, schema, description in TEST_TARGETS:
        fqn = f"{db}.{schema}"
        print(f"  Testing {fqn}")
        print(f"  ({description})")

        try:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT "TABLE_NAME", "TABLE_TYPE", "ROW_COUNT"
                FROM "{db}"."INFORMATION_SCHEMA"."TABLES"
                WHERE "TABLE_SCHEMA" = '{schema}'
                  AND "TABLE_TYPE" IN ('BASE TABLE', 'VIEW')
                ORDER BY "TABLE_NAME"
            """)
            tables = cur.fetchall()
            cur.close()

            if not tables:
                print(f"    WARN  No tables found (empty schema or no access)")
                results.append((fqn, "WARN", 0, "No tables visible"))
                print()
                continue

            total_rows = 0
            table_details = []
            for tname, ttype, row_count in tables:
                rows = row_count or 0
                total_rows += rows
                ttype_short = "VIEW" if "VIEW" in ttype else "TABLE"
                table_details.append((tname, ttype_short, rows))

            print(f"    OK    {len(tables)} tables/views, ~{total_rows:,} total rows")

            if verbose:
                for tname, ttype_short, rows in table_details:
                    print(f"          [{ttype_short:5}] {tname:<40} {rows:>12,} rows")

            # Quick SELECT test on first table
            test_table = tables[0][0]
            try:
                cur = conn.cursor()
                cur.execute(f'SELECT 1 FROM "{db}"."{schema}"."{test_table}" LIMIT 1')
                cur.fetchone()
                cur.close()
                print(f"    OK    SELECT verified on {test_table}")
            except Exception as e:
                print(f"    FAIL  SELECT failed on {test_table}: {e}")
                all_passed = False

            results.append((fqn, "OK", len(tables), f"~{total_rows:,} rows"))

        except Exception as e:
            print(f"    FAIL  {e}")
            results.append((fqn, "FAIL", 0, str(e)[:60]))
            all_passed = False

        print()

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    for fqn, status, table_count, detail in results:
        icon = {"OK": "[OK]", "WARN": "[!!]", "FAIL": "[XX]"}.get(status, "[??]")
        print(f"  {icon} {fqn:<45} {detail}")
    print("=" * 65)

    if all_passed:
        print()
        print("  All tests passed. This user can connect GeoVue to Snowflake.")
    else:
        print()
        print("  Some tests failed. Check the errors above.")
        print("  You may need this database role granted to your account role:")
        print(f"    {REQUIRED_DB_ROLE}")
        print("  Contact the DA_EXPLORATION team to request access.")

    print()
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test Snowflake connectivity for GeoVue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # auto-detect user + role
  %(prog)s --verbose              # show table details + grants
  %(prog)s --user JOE.BLOGGS@FORTESCUE.COM
  %(prog)s --role EDW_JBLOGGS     # explicit role override
        """,
    )
    parser.add_argument(
        "--user", "-u",
        help="Override Snowflake login hint (default: auto from Windows username)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show individual table details and role grants",
    )
    args = parser.parse_args()

    user_hint = resolve_user(args.user)
    success = test_connection(user_hint, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
