import os
import shutil
import glob
import re
import subprocess
import random
import yaml
from pathlib import Path
import torch
import yaml
import gradio as gr
import threading
import time
import librosa
import soundfile as sf
import numpy as np
import requests
import json
import locale
from datetime import datetime
import yt_dlp
import validators
from pytube import YouTube
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import math
import hashlib
import gc
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
from urllib.parse import urlparse, quote
import gdown
import argparse
from tqdm.auto import tqdm
import torch.nn as nn
from model import get_model_config, MODEL_CONFIGS

# Temel dizinler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # BASE_PATH yerine BASE_DIR
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OLD_OUTPUT_DIR = os.path.join(BASE_DIR, "old_output")
AUTO_ENSEMBLE_TEMP = os.path.join(BASE_DIR, "auto_ensemble_temp")
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_folder")
VIDEO_TEMP = os.path.join(BASE_DIR, "video_temp")
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")
INFERENCE_SCRIPT_PATH = os.path.join(BASE_DIR, "inference.py")

# Dizinleri oluştur
for directory in [BASE_DIR, INPUT_DIR, OUTPUT_DIR, OLD_OUTPUT_DIR, AUTO_ENSEMBLE_TEMP, AUTO_ENSEMBLE_OUTPUT, VIDEO_TEMP, ENSEMBLE_DIR]:
    os.makedirs(directory, exist_ok=True)

# YAML için özel sınıf ve yapılandırıcı
class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def tuple_constructor(loader, node):
    """Loads a tuple from a YAML sequence."""
    values = loader.construct_sequence(node)
    return tuple(values)

# PyYAML ile tuple constructor'ı kaydet
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def update_model_dropdown(category):
    """Kategoriye göre model dropdown'ını günceller."""
    return gr.Dropdown(choices=list(MODEL_CONFIGS[category].keys()), label="Model")

# Dosya yükleme işlevi (paylaşılan)
def handle_file_upload(uploaded_file, file_path, is_auto_ensemble=False):
    if uploaded_file:
        target = uploaded_file.name if hasattr(uploaded_file, 'name') else uploaded_file
        return target, target
    elif file_path and os.path.exists(file_path):
        return file_path, file_path
    return None, None

def clear_directory(directory):
    """Deletes all files in the given directory."""
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"{f} could not be deleted: {e}")

def clear_temp_folder(folder_path, exclude_items=None):
    """Safely clears contents of a directory while preserving specified items."""
    try:
        if not os.path.exists(folder_path):
            print(f"⚠️ Directory does not exist: {folder_path}")
            return False
        if not os.path.isdir(folder_path):
            print(f"⚠️ Path is not a directory: {folder_path}")
            return False
        exclude_items = exclude_items or []
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            if item_name in exclude_items:
                continue
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"⚠️ Error deleting {item_path}: {str(e)}")
        return True
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return False

def clear_old_output():
    old_output_folder = os.path.join(BASE_DIR, 'old_output')  # BASE_PATH -> BASE_DIR
    try:
        if not os.path.exists(old_output_folder):
            return "❌ Old output folder does not exist"
        shutil.rmtree(old_output_folder)
        os.makedirs(old_output_folder, exist_ok=True)
        return "✅ Old outputs successfully cleared!"
    except Exception as e:
        return f"🔥 Error: {str(e)}"

def shorten_filename(filename, max_length=30):
    """Shortens a filename to a specified maximum length."""
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    return base[:15] + "..." + base[-10:] + ext

def clean_filename(title):
    """Removes special characters from a filename."""
    return re.sub(r'[^\w\-_\. ]', '', title).strip()

def convert_to_wav(file_path):
    """Converts the audio file to WAV format."""
    original_filename = os.path.basename(file_path)
    filename, ext = os.path.splitext(original_filename)
    if ext.lower() == '.wav':
        return file_path
    wav_output = os.path.join(ENSEMBLE_DIR, f"{filename}.wav")
    try:
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-acodec', 'pcm_s16le', '-ar', '44100', wav_output
        ]
        subprocess.run(command, check=True, capture_output=True)
        return wav_output
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error ({e.returncode}): {e.stderr.decode()}")
        return None

def generate_random_port():
    """Generates a random port number."""
    return random.randint(1000, 9000)

def update_file_list():
    # OUTPUT_DIR ve OLD_OUTPUT_DIR'dan .wav dosyalarını al
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.wav"))  # BASE_DIR/output
    old_output_files = glob.glob(os.path.join(OLD_OUTPUT_DIR, "*.wav"))  # BASE_DIR/old_output
    
    # Dosya listesini birleştir
    files = output_files + old_output_files
    
    # Gradio Dropdown için seçenekleri döndür
    return gr.Dropdown(choices=files)

def save_uploaded_file(uploaded_file, is_input=False, target_dir=None):
    """Saves an uploaded file to the specified directory."""
    media_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.mp4']
    target_dir = target_dir or (INPUT_DIR if is_input else OUTPUT_DIR)
    timestamp_patterns = [
        r'_\d{8}_\d{6}_\d{6}$', r'_\d{14}$', r'_\d{10}$', r'_\d+$'
    ]

    if hasattr(uploaded_file, 'name'):
        original_filename = os.path.basename(uploaded_file.name)
    else:
        original_filename = os.path.basename(str(uploaded_file))

    if is_input:
        base_filename = original_filename
        for pattern in timestamp_patterns:
            base_filename = re.sub(pattern, '', base_filename)
        for ext in media_extensions:
            base_filename = base_filename.replace(ext, '')
        file_ext = next(
            (ext for ext in media_extensions if original_filename.lower().endswith(ext)),
            '.wav'
        )
        clean_filename = f"{base_filename.strip('_- ')}{file_ext}"
    else:
        clean_filename = original_filename

    target_path = os.path.join(target_dir, clean_filename)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path):
        os.remove(target_path)

    if hasattr(uploaded_file, 'read'):
        with open(target_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        shutil.copy(uploaded_file, target_path)

    print(f"File saved successfully: {os.path.basename(target_path)}")
    return target_path

def move_old_files(output_folder):
    """Moves old files to the old_output directory."""
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            new_filename = f"{os.path.splitext(filename)[0]}_old{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(OLD_OUTPUT_DIR, new_filename)
            shutil.move(file_path, new_file_path)

def conf_edit(config_path, chunk_size, overlap):
    """Edits the configuration file with chunk size and overlap."""
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap
    if data['inference']['batch_size'] == 1:
        data['inference']['batch_size'] = 2

    print(f"Using custom overlap and chunk_size: overlap={overlap}, chunk_size={chunk_size}")
    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper)
