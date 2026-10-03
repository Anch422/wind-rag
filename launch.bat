@echo off
cd /d "%~dp0"
call conda activate wind-rag
if errorlevel 1 (
    echo Open an Anaconda Prompt and run: conda activate wind-rag
    pause
    exit /b 1
)
python app.py
if errorlevel 1 pause
