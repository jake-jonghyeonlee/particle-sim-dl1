param(
    [string]$PythonExe = "",
    [string]$VenvPath = (Join-Path (Split-Path -Parent $PSScriptRoot) "psdl1-venv"),
    [switch]$ReuseVenv,
    [string]$SmokeOut = ""
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Fail([string]$Message) { throw $Message }
function Check-Exit([string]$Label) {
    if ($LASTEXITCODE -ne 0) { Fail "$Label failed (exit $LASTEXITCODE). Preserve logs and use a fresh path after fixing the cause." }
}
# Windows PowerShell 5.1: no PS7-only operators, RuntimeInformation, or UTF8-no-BOM dependency.
if ($env:OS -ne "Windows_NT" -or [IntPtr]::Size -ne 8 -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    Fail "Use native Windows x64 PowerShell 5.1+ (x86 and ARM64 are unsupported)."
}
# pip ignores external configuration and indexes. Runtime never uses uv.
# pip compares PIP_CONFIG_FILE with os.devnull case-sensitively (Windows: nul).
$env:PIP_CONFIG_FILE = "nul"
$env:PIP_NO_INDEX = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_CACHE_DIR = "1"
$env:PIP_REQUIRE_VIRTUALENV = "1"
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
if ($PythonExe) {
    $PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
} else {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { Fail "Preinstall CPython 3.12 x64. Supply -PythonExe 'C:\path\python.exe', or install py.exe Launcher." }
    $PythonExe = (& py -3.12 -I -c "import sys; print(sys.executable)")
    Check-Exit "Python Launcher"
    $PythonExe = $PythonExe.Trim()
}
$targetCheck = "import platform,struct,sys; assert sys.platform=='win32' and platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 and platform.machine().upper()=='AMD64', 'CPython 3.12 Windows x64 required'"
& $PythonExe -I -c $targetCheck
Check-Exit "Base interpreter architecture/version check"
$bundle = [IO.Path]::GetFullPath($PSScriptRoot)
$VenvPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($VenvPath)
if ($VenvPath.Equals($bundle, [StringComparison]::OrdinalIgnoreCase) -or $VenvPath.StartsWith($bundle + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    Fail "VenvPath must be outside the immutable bundle directory."
}
& $PythonExe -I (Join-Path $bundle "verify_windows_bundle.py") $bundle
Check-Exit "Bundle inventory/hash/target/closure check"
if (Test-Path -LiteralPath $VenvPath) {
    if (-not $ReuseVenv) { Fail "VenvPath already exists. It was preserved. Use a fresh path or explicit -ReuseVenv for exact-match validation without installing anything." }
    if (-not (Test-Path -LiteralPath (Join-Path $VenvPath "pyvenv.cfg"))) { Fail "ReuseVenv requires an existing isolated venv." }
} else {
    if ($ReuseVenv) { Fail "ReuseVenv path does not exist." }
    & $PythonExe -I -m venv $VenvPath
    Check-Exit "venv creation"
    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    & $venvPython -I -c $targetCheck
    Check-Exit "venv interpreter check"
    & $venvPython -I -m pip --isolated --disable-pip-version-check install --no-index --no-cache-dir --only-binary=:all: --require-hashes --find-links (Join-Path $bundle "wheelhouse") -r (Join-Path $bundle "requirements-windows.txt")
    Check-Exit "Offline wheel installation"
}
$venvPython = Join-Path $VenvPath "Scripts\python.exe"
& $venvPython -I -c $targetCheck
Check-Exit "venv interpreter check"
& $venvPython -I (Join-Path $bundle "verify_windows_bundle.py") $bundle --installed-prefix $VenvPath
Check-Exit "Installed versions/isolation/source check"
& $venvPython -I -m pip --isolated --disable-pip-version-check check
Check-Exit "pip check"
& $venvPython -I -c "import psdl1,numpy,torch,PIL; import psdl1.model,psdl1.train,psdl1.search; assert torch.version.cuda is None; print('CPU runtime imports OK')"
Check-Exit "CPU runtime imports (DLL failure: check Microsoft VC++ x64 runtime prerequisites)"
if (-not $SmokeOut) { $SmokeOut = Join-Path (Split-Path -Parent $bundle) ("psdl1-smoke-" + [guid]::NewGuid().ToString("N")) }
$SmokeOut = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SmokeOut)
if ($SmokeOut.Equals($bundle, [StringComparison]::OrdinalIgnoreCase) -or $SmokeOut.StartsWith($bundle + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { Fail "SmokeOut must be outside the immutable bundle." }
& $venvPython -I (Join-Path $bundle "smoke_windows_bundle.py") --out $SmokeOut
Check-Exit "Installed-wheel synthetic PNG/diagnostic/export/CPU train/search smoke"
Write-Output "Offline install and smoke passed. Evidence: $SmokeOut"
Write-Output "Run directly (no activation or uv): & '$venvPython' -I -m psdl1.import_png --help"
