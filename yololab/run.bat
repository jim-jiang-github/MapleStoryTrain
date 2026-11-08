@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

where python >nul 2>nul
if errorlevel 1 (
  echo [x] 没找到 python，请先安装 Python 并加入 PATH
  pause
  exit /b 1
)

python "%~dp0app.py" %*
if errorlevel 1 pause
