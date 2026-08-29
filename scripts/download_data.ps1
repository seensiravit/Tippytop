# Download KuaiRand-Pure into the vendored kit dir. Run from repo root.
$ErrorActionPreference = "Stop"
$kit = Join-Path $PSScriptRoot "..\kuairand-starter-kit\kuairand-starter-kit"
Push-Location $kit
try {
    $url = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
    if (-not (Test-Path "KuaiRand-Pure.tar.gz")) {
        Invoke-WebRequest -Uri $url -OutFile "KuaiRand-Pure.tar.gz"
    }
    tar xzf "KuaiRand-Pure.tar.gz"   # -> ./KuaiRand-Pure/data/
    Write-Host "Done. Data in $kit\KuaiRand-Pure\data"
} finally {
    Pop-Location
}
