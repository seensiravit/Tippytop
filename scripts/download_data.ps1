# Download KuaiRand-Pure into ./KuaiRand-Pure/. Run from anywhere.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $root
try {
    $url = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
    if (Test-Path "KuaiRand-Pure\data\log_standard_4_08_to_4_21_pure.csv") {
        Write-Host "Data already present at $root\KuaiRand-Pure\data — nothing to do."
        return
    }
    if (-not (Test-Path "KuaiRand-Pure.tar.gz")) {
        Write-Host "Downloading (~46 MB)..."
        Invoke-WebRequest -Uri $url -OutFile "KuaiRand-Pure.tar.gz"
    }
    tar xzf "KuaiRand-Pure.tar.gz"   # -> ./KuaiRand-Pure/data/
    Write-Host "Done. Data in $root\KuaiRand-Pure\data"
} finally {
    Pop-Location
}
