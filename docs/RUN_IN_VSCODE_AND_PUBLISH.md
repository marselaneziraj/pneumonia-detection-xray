# Running in VS Code, then publishing to GitHub

## 0. What you need

* VS Code with the **Python** extension (VS Code offers the recommended extensions when you open the folder).
* Python **3.10–3.13** (`python --version`).
* A Kaggle account (for the API token).
* For the full run: an NVIDIA GPU, **or** a free Colab T4 (`notebooks/colab_gpu_run.ipynb`). CPU works but is slow: about 10–15 min per DenseNet epoch on a laptop.

## 1. Open the project

1. Unzip `pneumonia-detection-xray.zip` somewhere **outside** OneDrive/Dropbox, e.g. `~/code/pneumonia-detection-xray`.
2. VS Code → **File → Open Folder…** → choose `pneumonia-detection-xray`.
3. Open a terminal with **Terminal → New Terminal** (Ctrl+\`).

## 2. Create the virtual environment and install

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,data]"
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,data]"
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**GPU users:** install the CUDA build of torch **before** `pip install -e .`, using the command for your CUDA version from <https://pytorch.org/get-started/locally/>, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`. **CPU-only machines** can save ~2 GB with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

Then press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick `./.venv`.

## 3. Check that the install works (no dataset needed, about 1 minute)

```bash
pytest
```

Expect every test to pass. The Testing tab (flask icon) shows the same tests.

```bash
ruff check src tests
```

## 4. Get the data

1. kaggle.com → avatar → **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Put the token where the Kaggle CLI looks for it:
   * macOS/Linux: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   * Windows: move it to `C:\Users\<you>\.kaggle\kaggle.json`
3. Download (about 1.2 GB):

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data --unzip
```

4. Record the provenance in README.md (the "Kaggle version" and "Retrieved" rows):

```bash
kaggle datasets metadata paultimothymooney/chest-xray-pneumonia -p data
```

## 5. Run the pipeline

You can type the commands in the terminal, or use **Terminal → Run Task… → "pneumonia: …"**. "pneumonia: FULL PIPELINE" runs them all in order. You can also use the Run and Debug panel (F5) to step through any command with breakpoints.

```bash
pneumonia prepare --config configs/densenet121.yaml
```

This prints the class and patient tables and writes `splits/` and `results/figures/class_balance.png`.

```bash
pneumonia train --config configs/densenet121.yaml
```

```bash
pneumonia train --config configs/resnet18.yaml
```

```bash
pneumonia train --config configs/resnet18_scratch.yaml
```

```bash
pneumonia evaluate
```

```bash
pneumonia gradcam --per-case 3
```

```bash
pneumonia report
```

On macOS/Linux, `bash scripts/run_all.sh` runs all of the above. On Windows, use `scripts\run_all.ps1`.

Quick end-to-end check before the long run (CPU, 1 minute, synthetic images, writes only `*_smoke/` folders):

```bash
pneumonia make-synthetic --out data_synthetic
```

```bash
pneumonia prepare --config configs/smoke.yaml
```

```bash
pneumonia train --config configs/smoke.yaml --limit-train-batches 3
```

```bash
pneumonia evaluate --configs configs/smoke.yaml
```

The same smoke mode works on the real data: add `--limit-train-batches 5` to any `pneumonia train` command.

**Running on Colab instead:** push the repo first (section 7, without the figures), open `notebooks/colab_gpu_run.ipynb` in Colab, and run all cells. Download `results_bundle.zip` at the end and unzip it over your local folder.

## 6. Do the human part (this is what makes the project credible)

1. Open `results/figures/gradcam_panel.png`.
2. Open `results/gradcam_audit.csv` and fill `manual_location` for every row with `lung`, `outside` or `mixed`. Note what you saw in `manual_notes` (e.g. "R marker", "collimation edge", "right lower lobe").
3. Run `pneumonia report` again so the counts land in README, report and model card.
4. Replace the "_fill in_" paragraphs in `README.md` ("What I see") and `report.md` (sections 3–5) with what the numbers and maps actually show, including the bad news.
5. Fill the Kaggle version and retrieval date in README.

## 7. Publish to GitHub

### 7a. Placeholders

Already filled in for `marselaneziraj` (README, pyproject.toml, LICENSE, Colab notebook). To show your full name instead, edit line 3 of `LICENSE` and `authors` in `pyproject.toml`.

### 7b. Create the empty remote

On github.com → **New repository**:

* Name: `pneumonia-detection-xray`, visibility **Public**.
* **Do NOT tick any of the three init boxes: "Add a README file", ".gitignore" and "Choose a license" must all stay unticked.** The repo already ships all three. Ticking any of them creates a commit your local clone doesn't have, and your first push then fails with `! [rejected] main -> main (fetch first)`.

Or with the GitHub CLI (creates it empty):

```bash
gh repo create pneumonia-detection-xray --public --description "Recall-tuned pneumonia detection on paediatric chest X-rays with patient-grouped splits and Grad-CAM audit"
```

### 7c. First commit and push

Run these from the project folder:

```bash
git init
```

```bash
git branch -M main
```

```bash
git add .
```

```bash
git add -f results/figures
```

```bash
git status
```

Check the `git status` output: there must be **no** `data/`, `.venv/`, `runs/` or `*.pt` files. `results/figures/*.png`, `results/results.csv`, `results/metrics.json` and `splits/*.csv` **must** be there.

```bash
python scripts/check_large_files.py --max-mb 50
```

```bash
git commit -m "Initial commit: recall-tuned pneumonia detection with patient-grouped splits and Grad-CAM audit"
```

```bash
git remote add origin https://github.com/marselaneziraj/pneumonia-detection-xray.git
```

```bash
git push -u origin main
```

If GitHub shows another default branch, set it to main: repo → **Settings → General → Default branch → main**, or:

```bash
gh repo edit marselaneziraj/pneumonia-detection-xray --default-branch main
```

If you ticked an init box by mistake and the push was rejected, merge GitHub's commit in and keep your files:

```bash
git pull origin main --allow-unrelated-histories --no-rebase -X ours
```

```bash
git push -u origin main
```

### 7d. If you upload files through the web interface instead

GitHub's web uploader commits into **whichever directory you are currently viewing**, so click into `results/figures` *before* choosing **Add file → Upload files**. If `results/figures` doesn't exist on the remote yet, create it first: **Add file → Create new file** and type `results/figures/.gitkeep` in the name box. Each `/` you type creates a directory. Commit, click into the new folder, then upload the PNGs.

### 7e. About box and topics

Repo page → ⚙ next to **About**:

* **Description (112 chars):** `Recall-tuned DenseNet121 for paediatric pneumonia X-rays: patient-grouped splits, baselines, Grad-CAM audit.`
* **Topics:** `medical-imaging`, `chest-xray`, `pneumonia-detection`, `pytorch`, `transfer-learning`, `grad-cam`, `explainable-ai`, `densenet`

After the push, the **Actions** tab should show CI going green within about 5 minutes.
