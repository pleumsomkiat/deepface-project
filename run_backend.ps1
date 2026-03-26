$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectDir "attendance_backend"

if (-not (Test-Path $backendDir)) {
    Write-Error "Missing attendance_backend folder. Expected: $backendDir"
    exit 1
}

$envFile = Join-Path $backendDir ".env"
if (-not $env:MONGO_URI -and (Test-Path $envFile)) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*MONGO_URI\s*=\s*(.+?)\s*$') {
            $value = $matches[1].Trim()
            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $env:MONGO_URI = $value
            break
        }
    }
}

if (-not $env:MONGO_URI) {
    Write-Error "Missing MONGO_URI. Set it in this shell or create attendance_backend\\.env with MONGO_URI=..."
    exit 1
}

Set-Location $backendDir
node server.js
exit $LASTEXITCODE
