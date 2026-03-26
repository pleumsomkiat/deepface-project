$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    (Join-Path $projectDir "env_runtime\\Scripts\\python.exe"),
    (Join-Path $projectDir "env_face\\Scripts\\python.exe"),
    (Join-Path $projectDir "venv_face\\Scripts\\python.exe")
)

$python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    Write-Error "No main runtime found. Expected one of: env_runtime, env_face, venv_face"
    exit 1
}

Set-Location $projectDir
& $python (Join-Path $projectDir "main_checkin.py")
exit $LASTEXITCODE
