[CmdletBinding()]
param(
    [string]$Config = "configs/config.ini",
    [string[]]$Steps = @(),
    [string]$PythonBin = "python",
    [string]$LogDir = "logs"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Get-UtcStamp {
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

function Normalize-Step([string]$Value) {
    $s = $Value.Trim().ToLowerInvariant()
    switch ($s) {
        "01" { return "01" }
        "1" { return "01" }
        "select" { return "01" }
        "02" { return "02" }
        "2" { return "02" }
        "dataset" { return "02" }
        "03" { return "03" }
        "3" { return "03" }
        "models" { return "03" }
        "model" { return "03" }
        "05" { return "05" }
        "5" { return "05" }
        "plots" { return "05" }
        "figures" { return "05" }
        "06" { return "06" }
        "6" { return "06" }
        "perendpoint" { return "06" }
        "endpoint" { return "06" }
        "baseline" { return "06" }
        default { throw ("Invalid step selector: {0}" -f $Value) }
    }
}

function Resolve-Steps([string[]]$Raw) {
    if ($null -eq $Raw -or $Raw.Count -eq 0 -or [string]::IsNullOrWhiteSpace(($Raw -join ""))) {
        return @("01", "02", "03", "05")
    }
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $resolved = [System.Collections.Generic.List[string]]::new()
    foreach ($rawItem in $Raw) {
        foreach ($tok in ($rawItem -split ',')) {
            foreach ($part in ($tok -split '\s+')) {
                if ([string]::IsNullOrWhiteSpace($part)) { continue }
                $step = Normalize-Step $part
                if ($seen.Add($step)) { $resolved.Add($step) }
            }
        }
    }
    return $resolved.ToArray()
}

function Get-StepName([string]$StepId) {
    switch ($StepId) {
        "01" { return "select_roofs" }
        "02" { return "make_ml_dataset" }
        "03" { return "run_models" }
        "05" { return "make_plots" }
        "06" { return "add_per_endpoint_baseline" }
        default { throw ("Unknown step id: {0}" -f $StepId) }
    }
}

function Get-StepScript([string]$StepId) {
    switch ($StepId) {
        "01" { return "scripts/select_roofs.py" }
        "02" { return "scripts/make_ml_dataset.py" }
        "03" { return "scripts/run_models.py" }
        "05" { return "scripts/make_plots.py" }
        "06" { return "scripts/add_per_endpoint_baseline.py" }
        default { throw ("Unknown step id: {0}" -f $StepId) }
    }
}

function Run-Step([string]$StepId, [string]$RunTs) {
    $name = Get-StepName $StepId
    $script = Get-StepScript $StepId
    $logFile = Join-Path $LogDir ("{0}_{1}_{2}.log" -f $RunTs, $StepId, $name)

    if (-not (Test-Path -LiteralPath $script -PathType Leaf)) {
        throw ("Missing script for step {0}: {1}" -f $StepId, $script)
    }

    Write-Host ("[{0}] START step {1} ({2})" -f (Get-UtcStamp), $StepId, $name)
    ("[{0}] CMD: {1} {2} --config {3}" -f (Get-UtcStamp), $PythonBin, $script, $Config) | Tee-Object -FilePath $logFile

    $stdoutFile = Join-Path $LogDir ("{0}_{1}_{2}.stdout.tmp" -f $RunTs, $StepId, $name)
    $stderrFile = Join-Path $LogDir ("{0}_{1}_{2}.stderr.tmp" -f $RunTs, $StepId, $name)

    if (Test-Path $stdoutFile) { Remove-Item $stdoutFile -Force }
    if (Test-Path $stderrFile) { Remove-Item $stderrFile -Force }

    $proc = Start-Process `
        -FilePath $PythonBin `
        -ArgumentList @($script, "--config", $Config) `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutFile `
        -RedirectStandardError $stderrFile

    if (Test-Path $stdoutFile) {
        Get-Content $stdoutFile | Tee-Object -FilePath $logFile -Append
        Remove-Item $stdoutFile -Force
    }
    if (Test-Path $stderrFile) {
        Get-Content $stderrFile | Tee-Object -FilePath $logFile -Append
        Remove-Item $stderrFile -Force
    }

    if ($proc.ExitCode -ne 0) {
        ("[{0}] FAIL step {1} ({2}), exit code {3}" -f (Get-UtcStamp), $StepId, $name, $proc.ExitCode) | Tee-Object -FilePath $logFile -Append
        exit $proc.ExitCode
    }
    ("[{0}] DONE step {1} ({2})" -f (Get-UtcStamp), $StepId, $name) | Tee-Object -FilePath $logFile -Append
}

if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw ("Config not found: {0}. Copy configs/config.ini.dist to configs/config.ini first." -f $Config)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$runTs = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$selected = if ($null -eq $Steps -or $Steps.Count -eq 0 -or [string]::IsNullOrWhiteSpace(($Steps -join ""))) {
    if ([string]::IsNullOrWhiteSpace($env:STEPS)) { @() } else { @($env:STEPS) }
} else {
    $Steps
}
$stepsToRun = Resolve-Steps $selected

Write-Host ("[{0}] Using config: {1}" -f (Get-UtcStamp), $Config)
Write-Host ("[{0}] Steps: {1}" -f (Get-UtcStamp), ($stepsToRun -join ' '))

foreach ($step in $stepsToRun) {
    Run-Step -StepId $step -RunTs $runTs
}

Write-Host ("[{0}] Pipeline finished successfully" -f (Get-UtcStamp))
