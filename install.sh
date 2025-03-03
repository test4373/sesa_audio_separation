#!/bin/bash

# Ortamı temizle
echo "Cleaning environment..."
rm -rf Music-Source-Separation-Training

# Depoyu klonla
echo "Cloning repository..."
git clone --progress https://github.com/ZFTurbo/Music-Source-Separation-Training.git
cd Music-Source-Separation-Training
git checkout 2a79da39c9ef86c070b44125a6d5d0e4dbaa4d9b

# Bağımlılıkları kur
echo "Installing dependencies..."
wget -q -O requirements.txt https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/requirements.txt
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade numpy librosa

# Dosyaları indir
echo "Downloading necessary files..."
wget -q -O WebUi2.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/WebUi2.py
wget -q -O inference.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/inference.py
wget -q -O utils.py https://raw.githubusercontent.com/test4373/My-Colab/refs/heads/main/utils.py
wget -q -O ensemble.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/ensemble.py
wget -q -O download.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/download.py
wget -q -O gui.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/gui.py
wget -q -O helpers.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/helpers.py
wget -q -O main.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/main.py
wget -q -O models.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/models.py
wget -q -O processing.py https://raw.githubusercontent.com/test4373/Music-Source-Separation-Training/refs/heads/main/processing.py

echo "Setup completed successfully!"
