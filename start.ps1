# LocalBrain 快速启动脚本
# 运行此脚本会自动启动后端和前端服务，并打开浏览器

param(
    [switch]$NoExit
)

$ErrorActionPreference = "Continue"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "正在启动 LocalBrain..." -ForegroundColor Green

# 检查端口是否被占用
$backendPort = 8000
$frontendPort = 5173

$backendInUse = Get-NetTCPConnection -LocalPort $backendPort -ErrorAction SilentlyContinue
$frontendInUse = Get-NetTCPConnection -LocalPort $frontendPort -ErrorAction SilentlyContinue

if ($backendInUse) {
    Write-Host "后端端口 $backendPort 已被占用，跳过后端启动" -ForegroundColor Yellow
} else {
    Write-Host "[1/4] 启动后端服务..." -ForegroundColor Yellow
    Start-Process powershell.exe -ArgumentList "-NoProfile", "-Command", "Set-Location '$ScriptDir\backend'; D:\Software\uv\envs\trae_cn\Scripts\Activate.ps1; D:\Software\uv\envs\trae_cn\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -WindowStyle Hidden
}

if ($frontendInUse) {
    Write-Host "前端端口 $frontendPort 已被占用，跳过前端启动" -ForegroundColor Yellow
} else {
    Write-Host "[2/4] 启动前端服务..." -ForegroundColor Yellow
    Start-Process powershell.exe -ArgumentList "-NoProfile", "-Command", "Set-Location '$ScriptDir\frontend'; npm run dev" -WindowStyle Hidden -WorkingDirectory "$ScriptDir\frontend"
}

# 等待服务启动
Write-Host "[3/4] 等待服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 6

# 打开浏览器
Write-Host "[4/4] 打开浏览器..." -ForegroundColor Yellow
Start-Process "http://localhost:5174"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " LocalBrain 已成功启动!" -ForegroundColor Green
Write-Host " 后端: http://localhost:8000" -ForegroundColor Cyan
Write-Host " 前端: http://localhost:5174" -ForegroundColor Cyan
Write-Host " API文档: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 如果不是 NoExit 模式，则自动关闭终端
if (-not $NoExit) {
    Write-Host "3秒后自动关闭终端..." -ForegroundColor Gray
    Start-Sleep -Seconds 3
    # 获取当前 PowerShell 窗口并关闭
    $host.UI.RawUI.WindowTitle = "LocalBrain 启动器 (将自动关闭)"
    Stop-Process -Id $PID -Force
}
