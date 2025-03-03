@echo off
title Music Source Separation Web UI

REM Rastgele bir port belirle (1000-9000 arası)
set /a PORT=1000 + (%RANDOM% %% 8001)

REM Kullanıcıdan paylaşım yöntemini seçmesini iste
echo Available sharing methods: gradio, localtunnel, ngrok
set /p METHOD="Choose a sharing method (default: gradio): "
if "%METHOD%"=="" set METHOD=gradio

REM Ngrok token'ını iste (yalnızca ngrok seçildiyse)
set NGROK_TOKEN=
if /i "%METHOD%"=="ngrok" (
    set /p NGROK_TOKEN="Enter your Ngrok token (get it from ngrok.com): "
    if "%NGROK_TOKEN%"=="" (
        echo Ngrok token is required for ngrok method!
        pause
        exit /b 1
    )
)

REM Komutu çalıştır
if /i "%METHOD%"=="gradio" (
    python main.py --method gradio --port %PORT%
) else if /i "%METHOD%"=="localtunnel" (
    python main.py --method localtunnel --port %PORT%
) else if /i "%METHOD%"=="ngrok" (
    python main.py --method ngrok --port %PORT% --ngrok-token "%NGROK_TOKEN%"
) else (
    echo Invalid method! Use gradio, localtunnel, or ngrok.
)

REM Terminali açık tut
pause
