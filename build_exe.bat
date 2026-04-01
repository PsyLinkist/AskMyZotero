@echo off
setlocal

echo ========================================
echo AskMyZotero EXE Build Script
echo Activate your Python env in this terminal first, then run this script.
echo ========================================
echo Using Python:
python -c "import sys; print(sys.executable)"
if %errorlevel% neq 0 (
  echo [ERROR] Python is not available in current environment.
  exit /b 1
)

if not exist "config.dist.yaml" (
  echo [ERROR] Missing config.dist.yaml - use as bundled config.yaml template without secrets.
  exit /b 1
)

echo.
echo [0/3] Cleaning old artifacts and staging bundled config...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "AskMyZotero.spec" del /f /q "AskMyZotero.spec"
if exist "build_bundle" rmdir /s /q "build_bundle"
mkdir "build_bundle"
copy /Y "config.dist.yaml" "build_bundle\config.yaml" >nul
if %errorlevel% neq 0 (
  echo [ERROR] Failed to copy config.dist.yaml to build_bundle\config.yaml
  exit /b 1
)

echo.
echo [1/3] Installing PyInstaller...
python -m pip install pyinstaller
if %errorlevel% neq 0 (
  echo [ERROR] Failed to install pyinstaller.
  exit /b 1
)

echo.
echo [2/3] Building AskMyZotero.exe ^(embedded config from config.dist.yaml -^> config.yaml in bundle^)...
python -m PyInstaller --noconfirm --clean --onefile --name AskMyZotero ^
  --collect-data "tiktoken" ^
  --collect-data "tiktoken_ext" ^
  --collect-data "certifi" ^
  --collect-submodules "tiktoken_ext" ^
  --hidden-import "tiktoken_ext.openai_public" ^
  --add-data "zotero_rag_ui.html;." ^
  --add-data "settings.html;." ^
  --add-data "build_bundle\config.yaml;." ^
  launcher.py
set "BUILD_ERR=%errorlevel%"

if exist "build_bundle" rmdir /s /q "build_bundle"

if %BUILD_ERR% neq 0 (
  echo [ERROR] Build failed.
  exit /b 1
)

echo.
echo [DONE] Build complete: dist\AskMyZotero.exe
if exist "dist\AskMyZotero.exe" (
  start "" "dist"
)
pause
