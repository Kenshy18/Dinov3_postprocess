$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..\..")

function Import-RuntimeEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($RawLine in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $Line = $RawLine.Trim()
        if ($Line.Length -eq 0 -or $Line.StartsWith("#") -or -not $Line.Contains("=")) {
            continue
        }
        $Parts = $Line.Split("=", 2)
        $Name = $Parts[0].Trim()
        $Value = $Parts[1].Trim().Trim("'").Trim('"')
        if ($Name.Length -gt 0) {
            Set-Item -Path "Env:$Name" -Value $Value
        }
    }
}

$RuntimeEnv = if ($env:GUI_RUNTIME_ENV) { $env:GUI_RUNTIME_ENV } else { Join-Path $RootDir ".runtime\gui_runtime.env" }
$LegacyRuntimeEnv = Join-Path $RootDir "configs\gui_runtime.env"
if (Test-Path $RuntimeEnv) {
    Import-RuntimeEnv $RuntimeEnv
} elseif (Test-Path $LegacyRuntimeEnv) {
    Import-RuntimeEnv $LegacyRuntimeEnv
}

$RuntimeProfileDefault = Join-Path $RootDir ".runtime\runtime_profile.json"
if (-not (Test-Path $RuntimeProfileDefault) -and (Test-Path (Join-Path $RootDir "configs\runtime_profile.json"))) {
    $RuntimeProfileDefault = Join-Path $RootDir "configs\runtime_profile.json"
}
$BatchBenchmarkDefault = Join-Path $RootDir ".runtime\runtime_benchmark.json"
if (-not (Test-Path $BatchBenchmarkDefault) -and (Test-Path (Join-Path $RootDir "configs\runtime_benchmark.json"))) {
    $BatchBenchmarkDefault = Join-Path $RootDir "configs\runtime_benchmark.json"
}

if (-not $env:DINOV3_RUNTIME_PROFILE) {
    $env:DINOV3_RUNTIME_PROFILE = $RuntimeProfileDefault
}
if (-not $env:DINOV3_BATCH_BENCHMARK) {
    $env:DINOV3_BATCH_BENCHMARK = $BatchBenchmarkDefault
}
if (-not $env:DINOV3_TRT_BACKBONE_ENGINE) {
    $env:DINOV3_TRT_BACKBONE_ENGINE = Join-Path $RootDir "checkpoints\trt\dinov3_backbone_fp32_720x1280_dynamic_bf16_forced_b1_8_8.engine"
}

$PythonCandidates = @()
if ($env:GUI_RUNTIME_PYTHON) {
    $PythonCandidates += $env:GUI_RUNTIME_PYTHON
}
$PythonCandidates += Join-Path $RootDir ".venv_integrated\Scripts\python.exe"
$PythonCandidates += Join-Path $RootDir ".venv_integrated\bin\python"
$PythonCandidates += "python"

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -eq "python" -or (Test-Path $Candidate)) {
        $Python = $Candidate
        break
    }
}
if (-not $Python) {
    throw "Python executable was not found. Run tools/setup_gui_runtime.sh in WSL/Linux or set GUI_RUNTIME_PYTHON."
}

& $Python (Join-Path $RootDir "apps\qt_ui\app.py") @args
