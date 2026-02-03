# Ralph-style execution for Logging Review Report refactor

This folder contains a **two-phase** execution setup:

1. **Plan** (already done): [docs/plans/logging-review-report-atomic-tasks.md](../docs/plans/logging-review-report-atomic-tasks.md) defines atomic tasks with exact before/after and verification commands.
2. **Execute**: The script runs one task at a time via the Cursor CLI, using **external verification** (ripgrep + pytest) to decide when each task is done.

## Prerequisites

- **ripgrep (`rg`)** on PATH — [install](https://github.com/BurntSushi/ripgrep#installation).
- **Cursor CLI** — install and log in so `cursor agent` (or `agent`) works from the shell.
- **Python + pytest** — for final verification (`pytest src/processing/tests/test_logging_review_report.py`).

### Cursor CLI setup

1. Install the Cursor CLI (see [Cursor docs](https://docs.cursor.com/) for current install command), e.g.:
   ```bash
   # Example (check Cursor docs for your platform)
   # Windows: often via Cursor Settings > Install CLI
   ```
2. Authenticate:
   ```bash
   cursor auth
   # or: agent login
   ```
3. Optional: set a custom agent command and model:
   ```powershell
   $env:CURSOR_AGENT_CMD = "cursor agent"
   # Or if you use a separate "agent" binary:
   # $env:CURSOR_AGENT_CMD = "agent"
   ```

## Running the script (Windows PowerShell)

From the repo root or from `scripts/`:

```powershell
cd "c:\Users\georg\OneDrive\Home Python\GeoVue 26_01_27"
.\scripts\ralph_execute_logging_review_refactor.ps1
```

Options:

- `-MaxIterations 50` — stop after 50 agent runs (default 50).
- `-AgentModel grok` — model to use (default grok).
- `-DryRun` — only print which task would run and the prompt; do not call the CLI.
- `-SkipPytest` — do not run pytest when all tasks pass.

Example:

```powershell
.\scripts\ralph_execute_logging_review_refactor.ps1 -MaxIterations 20 -DryRun
```

## How it works

1. The script has a **ordered list of tasks** (A.5.1, A.5.5, A.5.2, … through B.2; Phase C/D/E/F/G/H and docs can be added the same way).
2. Each task has **verification commands** (ripgrep patterns + paths + expected match count).
3. Each iteration:
   - Finds the **next task** whose verification still fails.
   - Builds a **prompt** that tells the agent: “Do only Task X from `docs/plans/logging-review-report-atomic-tasks.md`; then stop.”
   - **Spawns the Cursor CLI** with that prompt (e.g. `cursor agent -p --model grok` with prompt on stdin).
   - Re-runs the **verification** for that task; if it passes, the script moves on.
4. When **all listed tasks** pass verification, the script runs **pytest** (unless `-SkipPytest`) and exits successfully.
5. The loop stops when all tasks pass or **max iterations** is reached.

Important: **Completion is decided only by verification (rg + pytest), not by the LLM saying “done”.**

## Extending the task list

To add Phase C, D, E, etc., edit `ralph_execute_logging_review_refactor.ps1` and append to `$TaskList` using the same structure:

- `Id` — e.g. `"C.1"`, `"DOC.1"`.
- `Desc` — short description for the prompt.
- `Verifications` — array of `@{ Pattern = "rg-pattern"; Path = "path/from/repo/root"; Min = N }` or `Min = 0; Max = 0` for “no matches”.

Use the verification table at the end of [logging-review-report-atomic-tasks.md](../docs/plans/logging-review-report-atomic-tasks.md) as reference.

## If the Cursor CLI does not accept stdin

Some CLI builds take the prompt only as an argument. In that case:

1. Set `$env:CURSOR_AGENT_CMD` to a wrapper that reads from a file, e.g. a small script that does `agent -p --model grok "$(Get-Content $args[1] -Raw)"`, and call it with the temp prompt file path; or
2. Edit the script’s “Spawn Cursor CLI agent” block to pass the prompt as an argument (mind length limits on Windows).

## Troubleshooting

- **`rg` not found** — Install ripgrep and ensure it’s on PATH.
- **Agent command fails** — Run `cursor agent -p --model grok "Hello"` (or your `$env:CURSOR_AGENT_CMD`) manually to confirm the CLI works.
- **Task never verifies** — Check the exact ripgrep pattern and path in the atomic-tasks doc; adjust the `Verifications` entry for that task if the file or pattern changed.
