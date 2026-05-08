Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$VenvSitePackages = Join-Path $RepoRoot "venv\Lib\site-packages"
$Python = $null

if (Test-Path $VenvPython) {
    $PyvenvConfig = Join-Path $RepoRoot "venv\pyvenv.cfg"
    $BaseExecutableExists = $true
    if (Test-Path $PyvenvConfig) {
        $ExecutableLine = Get-Content $PyvenvConfig | Where-Object { $_ -like "executable = *" } | Select-Object -First 1
        if ($ExecutableLine) {
            $BaseExecutable = $ExecutableLine -replace "^executable = ", ""
            $BaseExecutableExists = Test-Path $BaseExecutable
        }
    }
    if ($BaseExecutableExists) {
        $OldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $VenvPython --version *> $null
        $VenvWorks = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $OldErrorActionPreference
        if ($VenvWorks) {
            $Python = $VenvPython
        }
    }
}

if (-not $Python) {
    if (-not (Test-Path $BundledPython)) {
        throw "No working Python found. Install Python 3.10+ or rebuild venv."
    }
    if (-not (Test-Path $VenvSitePackages)) {
        throw "venv packages not found. Run: python -m pip install -r requirements.txt"
    }
    $Python = $BundledPython
    $env:PYTHONPATH = (Resolve-Path $VenvSitePackages).Path
}

$env:STUDY_TUTOR_OFFLINE_MODE = "1"

Write-Host "Starting AI Study Tutor in offline mode..."
Write-Host "Open: http://localhost:8501"
Write-Host "Load profile: Sample Student"
& $Python -m streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
