@echo off
echo [*] Creating virtual environment in .venv ...
python -m venv .venv

echo [*] Activating virtual environment ...
call .\.venv\Scripts\activate.bat

echo [*] Upgrading pip ...
python -m pip install --upgrade pip

echo [*] Installing dependencies from requirements.txt ...
pip install -r requirements.txt

echo [✓] Setup for GPU is complete. You can now run:
echo .venv\Scripts\python.exe train.py
echo .venv\Scripts\python.exe test.py test-dir result-dir --device cuda

pause

