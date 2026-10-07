# Build dist\MosaicDiff\MosaicDiff.exe and, when Inno Setup is installed,
# dist\MosaicDiffSetup.exe.
# Run this on the machine that already has the NVIDIA PyTorch environment.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python -m pip install --disable-pip-version-check pyinstaller
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path "dist\MosaicDiff") {
    Remove-Item -Recurse -Force "dist\MosaicDiff"
}

& $python -m PyInstaller --noconfirm --workpath pyi-build --distpath Shipped "build\MosaicDiff.spec"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $iscc) {
    Write-Host "MosaicDiff.exe is in dist\MosaicDiff."
    Write-Host "Install Inno Setup 6 to also build MosaicDiffSetup.exe, then run this script again."
    exit 0
}

& $iscc "build\installer.iss"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Installer: dist\MosaicDiffSetup.exe"
