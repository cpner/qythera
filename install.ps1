# Qythera PowerShell Installer
Write-Host "`n  Qythera - AI Superintelligence`n" -ForegroundColor Magenta

# Find Python
$python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $python) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install from: https://www.python.org/downloads/"
    exit 1
}

Write-Host "Python: $(& $python.Source --version)"

# Find project
if (Test-Path "core\__init__.py") {
    Write-Host "Found project in current directory"
} elseif (Test-Path "qythera\core\__init__.py") {
    Set-Location qythera
    Write-Host "Found project in .\qythera"
} else {
    Write-Host "Downloading Qythera..."
    git clone https://github.com/cpner/qythera.git
    Set-Location qythera
}

# Install numpy
Write-Host "Installing numpy..."
& $python.Source -m pip install numpy --quiet 2>$null

Write-Host "`nInstallation Complete!" -ForegroundColor Green
Write-Host "`nStart: python -m core.inference.server --port 8080"
Write-Host "Or open web/standalone.html in browser"
