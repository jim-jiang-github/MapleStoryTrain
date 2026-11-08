@echo off
chcp 65001 >nul
title WZ YOLO Training Monitor (RTX 4090)
cd /d "%~dp0"
"C:\Users\JimJiang\.workbuddy\binaries\python\versions\3.13.12\python.exe" src\live_view.py
pause
