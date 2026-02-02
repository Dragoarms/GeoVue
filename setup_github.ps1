# GeoVue - Setup and push to GitHub
# Run this script from the GeoVue project folder (or it will cd there).
# Requires: git installed, GitHub auth (HTTPS or SSH) configured.

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
Set-Location $repoRoot

Write-Host "GeoVue GitHub setup - working in: $repoRoot" -ForegroundColor Cyan

# 1. Ensure we have a valid git repo (detect broken .git from failed init)
$ErrorActionPreferenceSave = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$isRepo = (git rev-parse --is-inside-work-tree 2>$null) -eq "true"
$ErrorActionPreference = $ErrorActionPreferenceSave
if (-not $isRepo) {
    if (Test-Path ".git") {
        Write-Host "Removing broken .git folder from previous failed init..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force ".git"
    }
    git init
    Write-Host "Git repository initialized." -ForegroundColor Green
} else {
    Write-Host "Git already initialized." -ForegroundColor Yellow
}

# 2. Add remote (replace if exists)
$remote = "origin"
$url = "https://github.com/Dragoarms/GeoVue.git"
$ErrorActionPreferenceSave = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$currentRemote = git remote get-url $remote 2>$null
$ErrorActionPreference = $ErrorActionPreferenceSave
if ($currentRemote) {
    git remote set-url $remote $url
    Write-Host "Remote '$remote' set to $url" -ForegroundColor Green
} else {
    git remote add $remote $url
    Write-Host "Remote '$remote' added: $url" -ForegroundColor Green
}

# 3. Stage all (respects .gitignore)
git add -A
$status = git status --short
if ($status) {
    Write-Host "Staged files:" -ForegroundColor Cyan
    Write-Host $status
} else {
    Write-Host "Nothing to commit (all ignored or already committed)." -ForegroundColor Yellow
    exit 0
}

# 4. First commit (only if there are changes)
$count = (git status --short | Measure-Object -Line).Lines
if ($count -gt 0) {
    git commit -m "Initial commit: GeoVue chip tray capture and visualisation"
    Write-Host "Committed." -ForegroundColor Green
}

# 5. Branch and push
$branch = "main"
git branch -M $branch
Write-Host "Pushing to $url (branch: $branch)..." -ForegroundColor Cyan
Write-Host "You may be prompted for GitHub credentials." -ForegroundColor Yellow
git push -u $remote $branch

Write-Host "Done. Repository: https://github.com/Dragoarms/GeoVue" -ForegroundColor Green
