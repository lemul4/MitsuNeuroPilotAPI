$ErrorActionPreference = "Stop"

$Cp1251 = [System.Text.Encoding]::GetEncoding(1251)
[Console]::OutputEncoding = $Cp1251
[Console]::InputEncoding = $Cp1251
$OutputEncoding = $Cp1251
chcp 1251 | Out-Null

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\lemul\miniconda3\envs\lead\python.exe"
$Config = Join-Path $Root "config\real_cameras.json"
$CameraOut = Join-Path $Root "camera_service.out.log"
$CameraErr = Join-Path $Root "camera_service.err.log"

Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*dual_camera_service.py*" -or $_.CommandLine -like "*main.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Get-Process |
    Where-Object { $_.ProcessName -like "ffmpeg*" } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Remove-Item -LiteralPath $CameraOut, $CameraErr -ErrorAction SilentlyContinue

$env:PYTHONUTF8 = "0"
$env:PYTHONIOENCODING = "cp1251"

Start-Process `
    -WindowStyle Hidden `
    -FilePath $Python `
    -ArgumentList "-u `"hardware\dual_camera_service.py`" --config `"$Config`"" `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $CameraOut `
    -RedirectStandardError $CameraErr

Start-Sleep -Seconds 3

$GuiEnv = @(
    "set PYTHONUTF8=0",
    "set PYTHONIOENCODING=cp1251",
    "set MITSU_REAL_CAMERA_BACKEND=zmq",
    "set MITSU_REAL_CAMERA_EXTERNAL=1",
    "set MITSU_REAL_POSE_MODE=nmea_serial",
    "set MITSU_GPS_COM_PORT=COM4",
    "set MITSU_GPS_BAUDRATE=9600",
    "set MITSU_AUTO_CONNECT_DEVICE=COM3",
    "set MITSU_AUTO_AI_PREVIEW=1",
    "set MITSU_REAL_ALLOW_ROUTE_PID_FALLBACK=1",
    "set MITSU_REAL_MODEL_PID_WAYPOINT_INDEX=2",
    "set MITSU_REAL_VISUALIZATION=0",
    "set MITSU_REAL_VISUALIZATION_FREQUENCY=5",
    "set MITSU_REAL_MODEL_DEEP_PROFILE=0",
    "set MITSU_REAL_MODEL_DEEP_PROFILE_PERIOD_SEC=1.0",
    "set MITSU_CAN_TX_DEBUG=0",
    "set MITSU_CAN_RX_DEBUG=0",
    "set MITSU_GPS_MIN_SATELLITES=6",
    "set MITSU_GPS_MAX_HDOP=3",
    "set MITSU_GPS_LOCK_SAMPLES=3",
    "set MITSU_GPS_MAX_JUMP_M=35",
    "set MITSU_GPS_MAX_ORIGIN_DISTANCE_M=2000",
    "set MITSU_REAL_POSE_STALE_MS=1500",
    "set MITSU_REAL_ENABLE_ACTUATION=1",
    "set MITSU_REAL_DRY_RUN=0",
    "set MITSU_GPS_LOG_RAW=0",
    "`"$Python`" main.py"
) -join "&&"

Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList @("/c", $GuiEnv) `
    -WorkingDirectory $Root

Write-Host "Camera service started with $Config"
Write-Host "GUI backend: zmq; camera service: external; pose: nmea_serial COM4@9600; auto device: COM3; AI Preview: on; model waypoint: 2; real visualization: off; model deep profile: off; CAN TX debug: off"
Write-Host "REAL ACTUATION ENABLED: MITSU_REAL_ENABLE_ACTUATION=1; MITSU_REAL_DRY_RUN=0"
Write-Host "GUI started. Logs:"
Write-Host "  $CameraOut"
Write-Host "  $CameraErr"
