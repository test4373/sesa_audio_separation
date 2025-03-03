@echo off
title Music Source Separation Setup

REM Ortamı temizle
echo Cleaning environment...
if exist "Music-Source-Separation-Training" (
    rmdir /s /q "Music-Source-Separation-Training"
)

REM Depoyu klonla
echo Cloning repository...
git clone --progress https://github.com/ZFTurbo/Music-Source-Separation-Training.git
cd Music-Source-Separation-Training
git checkout 2a79da39c9ef86c070b44125a6d5d0e4dbaa4d9b

REM Bağımlılıkları kur
echo Installing dependencies...
curl -o requirements.txt https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/requirements.txt
python -m pip install -r requirements.txt
python -m pip install --upgrade numpy librosa

REM Dosyaları indir
echo Downloading necessary files...
curl -o WebUi2.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/WebUi2.py
curl -o inference.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/inference.py
curl -o utils.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/utils.py
curl -o ensemble.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/ensemble.py
curl -o download.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/download.py
curl -o gui.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/gui.py
curl -o helpers.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/helpers.py
curl -o main.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/main.py
curl -o models.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/models.py
curl -o processing.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/processing.py

echo Setup completed successfully!
pause
