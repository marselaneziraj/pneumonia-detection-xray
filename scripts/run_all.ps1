# Full pipeline (Windows PowerShell). Assumes .venv is active and data\ is downloaded.
$ErrorActionPreference = "Stop"
pneumonia prepare --config configs/densenet121.yaml; if ($LASTEXITCODE) { exit 1 }
pneumonia train --config configs/densenet121.yaml; if ($LASTEXITCODE) { exit 1 }
pneumonia train --config configs/resnet18.yaml; if ($LASTEXITCODE) { exit 1 }
pneumonia train --config configs/resnet18_scratch.yaml; if ($LASTEXITCODE) { exit 1 }
pneumonia evaluate; if ($LASTEXITCODE) { exit 1 }
pneumonia gradcam --per-case 3; if ($LASTEXITCODE) { exit 1 }
pneumonia report
