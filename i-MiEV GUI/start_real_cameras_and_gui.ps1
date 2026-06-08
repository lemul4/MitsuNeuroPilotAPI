$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\lemul\miniconda3\envs\lead\python.exe"
$Config = Join-Path $Root "config\real_cameras.json"
$CameraOut = Join-Path $Root "camera_service.out.log"
$CameraErr = Join-Path $Root "camera_service.err.log"

Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*dual_camera_service.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Get-Process |
    Where-Object { $_.ProcessName -like "ffmpeg*" } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Remove-Item -LiteralPath $CameraOut, $CameraErr -ErrorAction SilentlyContinue

Start-Process `
    -WindowStyle Hidden `
    -FilePath $Python `
    -ArgumentList @("-u", "hardware\dual_camera_service.py", "--config", $Config) `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $CameraOut `
    -RedirectStandardError $CameraErr

Start-Sleep -Seconds 3

Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", "set MITSU_REAL_CAMERA_EXTERNAL=1&& `"$Python`" main.py") `
    -WorkingDirectory $Root

Write-Host "Camera service started with $Config"
Write-Host "GUI started. Logs:"
Write-Host "  $CameraOut"
Write-Host "  $CameraErr"
