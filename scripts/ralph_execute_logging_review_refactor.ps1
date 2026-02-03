# Ralph-style execution script for Logging Review Report refactor.
# Runs one atomic task at a time via Cursor CLI, with external verification (rg + pytest).
# See docs/plans/logging-review-report-atomic-tasks.md for task definitions.
#
# Prerequisites:
#   - Cursor CLI installed and on PATH (e.g. "cursor" or "agent")
#   - ripgrep (rg) on PATH
#   - Python + pytest for final verification
#
# Usage:
#   .\ralph_execute_logging_review_refactor.ps1
#   .\ralph_execute_logging_review_refactor.ps1 -MaxIterations 50 -AgentModel grok

param(
    [int]   $MaxIterations = 50,
    [string] $AgentModel   = "grok",
    [string] $AgentCmd    = $env:CURSOR_AGENT_CMD,
    [switch] $DryRun      = $false,
    [switch] $SkipPytest = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { (Get-Location).Path }
$TasksDoc = Join-Path $RepoRoot "docs\plans\logging-review-report-atomic-tasks.md"

# Default agent command if not set
if (-not $AgentCmd) { $AgentCmd = "cursor agent" }

# --- Task list: TaskId, Verification(s), and prompt description ---
# Each verification is: [rg pattern, path, expected] where expected = "0" | ">=1" | ">=2" (min matches).
# Path is relative to $RepoRoot.
$TaskList = @(
    @{ Id = "A.5.1";  Desc = "Add project_code, easting, northing to COLUMN_ALIASES"; Verifications = @(@{ Pattern = '"project_code"'; Path = "src/processing/DataManager/column_aliases.py"; Min = 1 }) },
    @{ Id = "A.5.5";  Desc = "Extend depth_from/depth_to for logging columns"; Verifications = @(@{ Pattern = "logging_from|depth_from_geol"; Path = "src/processing/DataManager/column_aliases.py"; Min = 2 }) },
    @{ Id = "A.5.2";  Desc = "Remove PROJECT_CODE_CANDIDATES, EASTING_CANDIDATES, NORTHING_CANDIDATES"; Verifications = @(@{ Pattern = "PROJECT_CODE_CANDIDATES|EASTING_CANDIDATES|NORTHING_CANDIDATES"; Path = "src/processing/logging_review_html_report.py"; Min = 0; Max = 0 }) },
    @{ Id = "A.5.3";  Desc = "Replace _resolve_project_code_column with ColumnResolver"; Verifications = @(@{ Pattern = 'get\("project_code"\)'; Path = "src/processing/logging_review_html_report.py"; Min = 1 }) },
    @{ Id = "A.5.4";  Desc = "Replace _resolve_coordinate_columns with ColumnResolver"; Verifications = @(@{ Pattern = 'get\("easting"\)|get\("northing"\)'; Path = "src/processing/logging_review_html_report.py"; Min = 2 }) },
    @{ Id = "A.5.6";  Desc = "Replace _resolve_logging_interval_columns with ColumnResolver"; Verifications = @(
        @{ Pattern = 'get\("depth_from"\)|get\("depth_to"\)'; Path = "src/processing/logging_review_report.py"; Min = 1 },
        @{ Pattern = "from_candidates|to_candidates"; Path = "src/processing/logging_review_report.py"; Min = 0; Max = 0 }
    )},
    @{ Id = "S.1";    Desc = "Create src/reports/__init__.py"; Verifications = @(@{ Pattern = "reports package"; Path = "src/reports/__init__.py"; Min = 1 }) },
    @{ Id = "S.2";    Desc = "Create src/reports/logging_review/__init__.py"; Verifications = @(@{ Pattern = "logging_review"; Path = "src/reports/logging_review/__init__.py"; Min = 1 }) },
    @{ Id = "S.3";    Desc = "Create src/reports/logging_review/html/__init__.py"; Verifications = @(@{ Pattern = "HTML report"; Path = "src/reports/logging_review/html/__init__.py"; Min = 1 }) },
    @{ Id = "S.4";    Desc = "Create src/reports/logging_review/html/assets/__init__.py"; Verifications = @(@{ Pattern = "CSS and JS"; Path = "src/reports/logging_review/html/assets/__init__.py"; Min = 1 }) },
    @{ Id = "A.1";    Desc = "Create styles.py with CSS_STYLES"; Verifications = @(@{ Pattern = "^CSS_STYLES\s*="; Path = "src/reports/logging_review/html/assets/styles.py"; Min = 1 }) },
    @{ Id = "A.2";    Desc = "Import CSS_STYLES and use in _render_html"; Verifications = @(@{ Pattern = "CSS_STYLES"; Path = "src/processing/logging_review_html_report.py"; Min = 2 }) },
    @{ Id = "A.3";    Desc = "Create scripts.py with JS_SCRIPTS"; Verifications = @(@{ Pattern = "^JS_SCRIPTS\s*="; Path = "src/reports/logging_review/html/assets/scripts.py"; Min = 1 }) },
    @{ Id = "A.4";    Desc = "Import JS_SCRIPTS and use in _render_html"; Verifications = @(@{ Pattern = "JS_SCRIPTS"; Path = "src/processing/logging_review_html_report.py"; Min = 2 }) },
    @{ Id = "B.1";    Desc = "Create html/utils.py with _safe_str, _safe_float, _format_metric"; Verifications = @(@{ Pattern = "def _safe_str|def _safe_float|def _format_metric"; Path = "src/reports/logging_review/html/utils.py"; Min = 3 }) },
    @{ Id = "B.2";    Desc = "Remove those defs from HTML report and add import"; Verifications = @(@{ Pattern = "def _safe_str\("; Path = "src/processing/logging_review_html_report.py"; Min = 0; Max = 0 }) }
    # Phase C, D, E, F, G, H, DOC: add more entries following the same pattern, or run manually / extend script.
)

function Get-RgCount {
    param([string]$Pattern, [string]$Path)
    $fullPath = Join-Path $RepoRoot $Path
    if (-not (Test-Path $fullPath)) { return 0 }
    $out = & rg $Pattern $fullPath 2>$null
    if (-not $out) { return 0 }
    $lines = @($out)
    return $lines.Count
}

function Test-TaskVerified {
    param($Task)
    foreach ($v in $Task.Verifications) {
        $count = Get-RgCount -Pattern $v.Pattern -Path $v.Path
        $min = if ($v.Min -ne $null) { $v.Min } else { 0 }
        $max = if ($v.Max -ne $null) { $v.Max } else { [int]::MaxValue }
        if ($count -lt $min -or $count -gt $max) { return $false }
    }
    return $true
}

function Get-PromptForTask {
    param($Task)
    @"
You are executing exactly ONE atomic task from the Logging Review Report refactor. Do only this task, then stop.

Task ID: $($Task.Id)
Description: $($Task.Desc)

Instructions:
1. Open the file: docs/plans/logging-review-report-atomic-tasks.md
2. Find the section for Task $($Task.Id) (search for "$($Task.Id)").
3. Perform the exact transformation described (Before -> After). Change only what is specified.
4. Do not proceed to the next task. Do not refactor anything else.

Verification: After your edit, the following must hold (script will check):
$(($Task.Verifications | ForEach-Object { "  - rg '$($_.Pattern)' $($_.Path) -> expected: min=$($_.Min)" }) -join "`n")

Workspace root: $RepoRoot
"@
}

$currentIndex = 0
$iteration = 0

Write-Host "Ralph execution: Logging Review Report refactor (max $MaxIterations iterations, agent: $AgentCmd, model: $AgentModel)"
Write-Host "Tasks doc: $TasksDoc"
Write-Host ""

while ($iteration -lt $MaxIterations) {
    # Find next unverified task
    $nextTask = $null
    $nextIndex = -1
    for ($i = 0; $i -lt $TaskList.Count; $i++) {
        if (-not (Test-TaskVerified -Task $TaskList[$i])) {
            $nextTask = $TaskList[$i]
            $nextIndex = $i
            break
        }
    }

    if ($null -eq $nextTask) {
        Write-Host "All defined tasks passed verification."
        if (-not $SkipPytest) {
            Write-Host "Running pytest..."
            Push-Location $RepoRoot
            try {
                & pytest src/processing/tests/test_logging_review_report.py -q 2>&1
                if ($LASTEXITCODE -ne 0) { Write-Warning "pytest reported failures." }
            } finally { Pop-Location }
        }
        Write-Host "SUCCESS: Ralph loop complete (all tasks verified)."
        exit 0
    }

    Write-Host "=== Iteration $iteration: Task $($nextTask.Id) — $($nextTask.Desc) ==="
    $prompt = Get-PromptForTask -Task $nextTask

    if ($DryRun) {
        Write-Host "[DRY RUN] Would spawn agent with prompt (first 500 chars):"
        Write-Host $prompt.Substring(0, [Math]::Min(500, $prompt.Length))
        $iteration++
        continue
    }

    # Spawn Cursor CLI agent. Set $env:CURSOR_AGENT_CMD to your CLI (e.g. "cursor" or "cursor agent").
    # Pass prompt via stdin. If CURSOR_AGENT_CMD has spaces, it is split into executable + arguments.
    $promptFile = [System.IO.Path]::GetTempFileName()
    $prompt | Out-File -FilePath $promptFile -Encoding utf8 -NoNewline
    try {
        $parts = $AgentCmd -split "\s+", 2
        $exe = $parts[0]
        $agentArgs = @()
        if ($parts.Length -gt 1) { $agentArgs += $parts[1] -split "\s+" }
        $agentArgs += "-p", "--model", $AgentModel
        Get-Content -LiteralPath $promptFile -Raw | & $exe @agentArgs
    } catch {
        Write-Warning "Agent invocation failed: $_"
    } finally {
        Remove-Item -LiteralPath $promptFile -ErrorAction SilentlyContinue
    }

    # Re-check verification for this task
    Start-Sleep -Seconds 2
    if (Test-TaskVerified -Task $nextTask) {
        Write-Host "  -> Task $($nextTask.Id) verified."
    } else {
        Write-Host "  -> Task $($nextTask.Id) not yet passing; will retry next iteration."
    }

    $iteration++
}

Write-Host "Reached max iterations ($MaxIterations). Some tasks may remain."
exit 1
