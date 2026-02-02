# Pushing GeoVue to GitHub

Your project is set up for Git and the [Dragoarms/GeoVue](https://github.com/Dragoarms/GeoVue) repo.

## What’s already done

1. **`.gitignore`** — Ignores cache, venvs, `.pt` models, large ML datasets (`ml_detection/classifier_images/`, `yolo_dataset/`, etc.), and backup files so they aren’t committed.
2. **`README.md`** — Root readme for the repo.
3. **Remote** — Will be set by the script below to `https://github.com/Dragoarms/GeoVue.git`.

## Option A: Run the setup script (recommended)

In **PowerShell**, from this folder:

```powershell
cd "c:\Users\georg\OneDrive\Home Python\GeoVue 26_01_27"
.\setup_github.ps1
```

If you get an execution policy error:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\setup_github.ps1
```

The script will:

- `git init` (if needed)
- Add remote `origin` → `https://github.com/Dragoarms/GeoVue.git`
- `git add -A` (respecting `.gitignore`)
- Create an initial commit
- Rename branch to `main` and run `git push -u origin main`

You may be asked for your GitHub username and password. For password, use a **Personal Access Token** (Settings → Developer settings → Personal access tokens), not your account password.

## Option B: Do it manually

```powershell
cd "c:\Users\georg\OneDrive\Home Python\GeoVue 26_01_27"

git init
git remote add origin https://github.com/Dragoarms/GeoVue.git
git add -A
git status
git commit -m "Initial commit: GeoVue chip tray capture and visualisation"
git branch -M main
git push -u origin main
```

## If the GitHub repo already has commits

If https://github.com/Dragoarms/GeoVue already has content (e.g. an existing `main`):

**Option 1 — Overwrite with your local (use only if you’re sure):**

```powershell
git push -u origin main --force
```

**Option 2 — Keep existing history and add your code:**

```powershell
git pull origin main --allow-unrelated-histories
# Resolve any merge conflicts, then:
git push -u origin main
```

## Authentication

- **HTTPS:** Use your GitHub username and a [Personal Access Token](https://github.com/settings/tokens) as the password.
- **SSH:** If you use SSH keys, switch the remote and push:

  ```powershell
  git remote set-url origin git@github.com:Dragoarms/GeoVue.git
  git push -u origin main
  ```

## What is not pushed (by `.gitignore`)

- `__pycache__/`, `.pyc`, `.pytest_cache/`
- Virtual envs (e.g. `.venv/`, `venv/`)
- `*.egg-info/`, `dist/`, `build/`
- ML data: `ml_detection/classifier_images/`, `ml_detection/yolo_dataset/`, `ml_detection/runs/`, `ml_output/`, `*.pt`
- Backup files like `*.backup_*`

So only source code, config, docs, and small assets are pushed; large/generated data stays local.
