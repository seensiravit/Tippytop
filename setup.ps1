# One-shot environment setup for autoresearch_lg (the LangGraph harness).
# Creates .venv, installs the package + langgraph-cli, seeds .env, and
# checks for the KuaiRand-Pure data -- same steps README.md documents by
# hand, just scripted. Safe to re-run (every step is idempotent).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$DataDir = ".\KuaiRand-Pure\data"
$DataFiles = @(
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
    "user_features_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv"
)

Write-Host "== autoresearch_lg setup =="

# ---- 1. Python --------------------------------------------------------------
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Host "ERROR: no python/python3 on PATH. Install Python 3.11+ first." -ForegroundColor Red
    exit 1
}
$pythonVersion = & $pythonCmd.Source --version
Write-Host "-- using $pythonVersion"

# ---- 2. Virtual environment ---------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "-- creating .venv"
    & $pythonCmd.Source -m venv .venv
} else {
    Write-Host "-- .venv already exists, reusing it"
}

$VenvPy = ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "ERROR: .venv exists but no python.exe inside it -- delete .venv and re-run." -ForegroundColor Red
    exit 1
}

# ---- 3. Install ----------------------------------------------------------------
Write-Host "-- installing autoresearch_lg + langgraph-cli (this can take a minute)"
& $VenvPy -m pip install -q --upgrade pip
& $VenvPy -m pip install -q -e . "langgraph-cli[inmem]"
Write-Host "-- installed"

# ---- 4. .env ---------------------------------------------------------------------
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "-- created .env from .env.example -- add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY before running anything"
} else {
    Write-Host "-- .env already exists, leaving it alone"
}

# ---- 5. KuaiRand-Pure data --------------------------------------------------------
$missing = $false
foreach ($f in $DataFiles) {
    if (-not (Test-Path (Join-Path $DataDir $f))) { $missing = $true }
}
if ($missing) {
    Write-Host ""
    Write-Host "-- KuaiRand-Pure data not found under $DataDir. Download it:"
    Write-Host "     Invoke-WebRequest https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz -OutFile KuaiRand-Pure.tar.gz"
    Write-Host "     tar xzf KuaiRand-Pure.tar.gz"
    Write-Host "   (run those from the repo root -- they produce .\KuaiRand-Pure\)"
} else {
    Write-Host "-- KuaiRand-Pure data found"
}

Write-Host ""
Write-Host "== done =="
Write-Host "Activate the venv, then:"
Write-Host "  .venv\Scripts\Activate.ps1"
$tag = Get-Date -Format "MMMdd" | ForEach-Object { $_.ToLower() }
Write-Host "  python -m autoresearch_lg.cli setup --tag $tag"
Write-Host "  python -m autoresearch_lg.cli run   --tag $tag"
