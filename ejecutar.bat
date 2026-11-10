@echo off
chcp 65001 >nul
title Sink Tree - Rutas optimas
cd /d "%~dp0"

set "PY="
py --version >nul 2>&1 && set "PY=py"
if not defined PY (python --version >nul 2>&1 && set "PY=python")
if not defined PY (
  echo.
  echo No se encontro Python en este computador.
  echo Instalelo desde https://www.python.org/downloads/
  echo y marque la casilla "Add python.exe to PATH". Luego vuelva a abrir este archivo.
  echo.
  pause
  exit /b 1
)

echo Instalando librerias (solo tarda la primera vez)...
%PY% -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo No se pudieron instalar las librerias. Revise su conexion a internet.
  pause
  exit /b 1
)

echo.
echo Abriendo el programa en el navegador: http://localhost:8501
echo Para cerrarlo, cierre esta ventana.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
%PY% -m streamlit run main.py --server.headless true --browser.gatherUsageStats false
pause
