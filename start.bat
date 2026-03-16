@echo off
set LOKY_MAX_CPU_COUNT=%NUMBER_OF_PROCESSORS%
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
