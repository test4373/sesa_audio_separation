import os

os.chdir('/content/Music-Source-Separation-Training')
import torch
import yaml
import gradio as gr
import subprocess
import threading
import random
import time
import shutil
import librosa
import soundfile as sf
import numpy as np
import requests
import json
import locale
import shutil
from datetime import datetime
import glob
import yt_dlp
import validators
from pytube import YouTube
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import math
import hashlib
import re
import gc
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
from urllib.parse import urlparse
from urllib.parse import quote
import gdown


os.makedirs('/content/Music-Source-Separation-Training/input', exist_ok=True)
os.makedirs('/content/Music-Source-Separation-Training/output', exist_ok=True)
os.makedirs('/content/drive/MyDrive/output', exist_ok=True)
os.makedirs('/content/drive/MyDrive/ensemble_folder', exist_ok=True)
os.makedirs('/content/Music-Source-Separation-Training/old_output', exist_ok=True)
os.makedirs('/content/Music-Source-Separation-Training/auto_ensemble_temp', exist_ok=True)

def clear_old_output():
    old_output_folder = os.path.join(BASE_PATH, 'old_output')
    try:
        if not os.path.exists(old_output_folder):
            return "❌ Old output folder does not exist"
        
        # Tüm dosya ve alt klasörleri sil
        shutil.rmtree(old_output_folder)
        os.makedirs(old_output_folder, exist_ok=True)
        
        return "✅ Old outputs successfully cleared!"
    
    except Exception as e:
        error_msg = f"🔥 Error: {str(e)}"
        print(error_msg)
        return error_msg
    
    print("All files in old_output have been deleted.")

def shorten_filename(filename, max_length=30):
    """
    Shortens a filename to a specified maximum length
    
    Args:
        filename (str): The filename to be shortened
        max_length (int): Maximum allowed length for the filename
    
    Returns:
        str: Shortened filename
    """
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    
    # Take first 15 and last 10 characters
    shortened = base[:15] + "..." + base[-10:] + ext
    return shortened

def update_progress(progress=gr.Progress()):
    def track_progress(percent):
        progress(percent/100)
    return track_progress


def clear_input_folder():
    # Folder cleanup process
    input_path = "/content/Music-Source-Separation-Training/input"
    if os.path.exists(input_path):
        shutil.rmtree(input_path)
    os.makedirs(input_path, exist_ok=True)

# Özel karakterleri temizlemek için
def clean_filename(title):
    return re.sub(r'[^\w\-_\. ]', '', title).strip()

def download_callback(url, download_type='direct', cookie_file=None):
    try:
        # 1. TEMİZLİK VE KLASÖR HAZIRLIĞI
        BASE_PATH = "/content/Music-Source-Separation-Training"
        INPUT_DIR = os.path.join(BASE_PATH, "input")
        COOKIE_PATH = os.path.join(BASE_PATH, "cookies.txt")
        
        # Input klasörünü temizle ve yeniden oluştur
        clear_input_folder()
        os.makedirs(INPUT_DIR, exist_ok=True)

        # 2. URL DOĞRULAMA
        if not validators.url(url):
            return None, "❌ Invalid URL", None, None, None, None

        # 3. COOKIE YÖNETİMİ
        if cookie_file is not None:
            try:
                with open(cookie_file.name, "rb") as f:
                    cookie_content = f.read()
                with open(COOKIE_PATH, "wb") as f:
                    f.write(cookie_content)
                print("✅ Cookie file updated!")
            except Exception as e:
                print(f"⚠️ Cookie installation error: {str(e)}")

        wav_path = None
        download_success = False

        # 4. İNDİRME TÜRÜNE GÖRE İŞLEM
        if download_type == 'drive':
            # GOOGLE DRIVE İNDİRME
            try:
                file_id = re.search(r'/d/([^/]+)', url).group(1) if '/d/' in url else url.split('id=')[-1]
                original_filename = "drive_download.wav"
                
                # Gdown ile indirme
                output_path = os.path.join(INPUT_DIR, original_filename)
                gdown.download(
                    f'https://drive.google.com/uc?id={file_id}',
                    output_path,
                    quiet=True,
                    fuzzy=True
                )
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    wav_path = output_path
                    download_success = True
                    print(f"✅ Downloaded from Google Drive: {wav_path}")
                else:
                    raise Exception("File size zero or file not created")

            except Exception as e:
                error_msg = f"❌ Google Drive download error: {str(e)}"
                print(error_msg)
                return None, error_msg, None, None, None, None

        else:
            # YOUTUBE/DİREKT LİNK İNDİRME
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(INPUT_DIR, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0'
                }],
                'cookiefile': COOKIE_PATH if os.path.exists(COOKIE_PATH) else None,
                'nocheckcertificate': True,
                'ignoreerrors': True,
                'retries': 3
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
                    temp_path = ydl.prepare_filename(info_dict)
                    wav_path = os.path.splitext(temp_path)[0] + '.wav'
                    
                    if os.path.exists(wav_path):
                        download_success = True
                        print(f"✅ Downloaded successfully: {wav_path}")
                    else:
                        raise Exception("WAV conversion failed")

            except Exception as e:
                error_msg = f"❌ Download error: {str(e)}"
                print(error_msg)
                return None, error_msg, None, None, None, None

        # 5. SON KONTROLLER VE TEMİZLİK
        if download_success and wav_path:
            # Input klasöründeki gereksiz dosyaları temizle
            for f in os.listdir(INPUT_DIR):
                if f != os.path.basename(wav_path):
                    os.remove(os.path.join(INPUT_DIR, f))
            
            return (
                gr.File(value=wav_path),
                "🎉 Downloaded successfully!",
                gr.File(value=wav_path),
                gr.File(value=wav_path),
                gr.Audio(value=wav_path),
                gr.Audio(value=wav_path)
            )

        return None, "❌ Download failed", None, None, None, None

    except Exception as e:
        error_msg = f"🔥 Critical Error: {str(e)}"
        print(error_msg)
        return None, error_msg, None, None, None, None

        
# Hook function to track download progress
def download_progress_hook(d):
    if d['status'] == 'finished':
        print('Download complete, conversion in progress...')
    elif d['status'] == 'downloading':
        downloaded_bytes = d.get('downloaded_bytes', 0)
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        if total_bytes > 0:
            percent = downloaded_bytes * 100. / total_bytes
            print(f'Downloading: {percent:.1f}%')

# Define the global variable at the top
INPUT_DIR = "/content/Music-Source-Separation-Training/input"

def download_file(url):
    # Encode the URL to handle spaces and special characters
    encoded_url = quote(url, safe=':/')

    path = 'ckpts'
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return

    try:
        response = torch.hub.download_url_to_file(encoded_url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")
        


def generate_random_port():
    return random.randint(1000, 9000)

    clear_memory()

# Markdown annotations
markdown_intro = """
# Voice Parsing Tool

This tool is used to parse audio files.
"""

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
tuple_constructor)



def conf_edit(config_path, chunk_size, overlap):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # handle cases where 'use_amp' is missing from config:
    if 'use_amp' not in data.keys():
      data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap

    if data['inference']['batch_size'] == 1:
      data['inference']['batch_size'] = 2

    print("Using custom overlap and chunk_size values:")
    print(f"overlap = {data['inference']['num_overlap']}")
    print(f"chunk_size = {data['audio']['chunk_size']}")
    print(f"batch_size = {data['inference']['batch_size']}")


    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)
        
def save_uploaded_file(uploaded_file, is_input=False, target_dir=None):
    try:
        # Medya dosya uzantıları
        media_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.mp4']
        
        # Hedef dizini belirle
        if target_dir is None:
            target_dir = INPUT_DIR if is_input else VİDEO_TEMP
        
        # Zaman damgası pattern'leri
        timestamp_patterns = [
            r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
            r'_\d{14}$',             # _20231215123456
            r'_\d{10}$',             # _1702658400
            r'_\d+$'                 # Herhangi bir sayı
        ]
        
        # Dosya adını al
        if hasattr(uploaded_file, 'name'):
            original_filename = os.path.basename(uploaded_file.name)
        else:
            original_filename = os.path.basename(str(uploaded_file))
        
        # Dosya adını temizle (sadece input'lar için)
        if is_input:
            base_filename = original_filename
            # Zaman damgalarını sil
            for pattern in timestamp_patterns:
                base_filename = re.sub(pattern, '', base_filename)
            # Çoklu uzantıları sil
            for ext in media_extensions:
                base_filename = base_filename.replace(ext, '')
            
            # Dosya uzantısını belirle
            file_ext = next(
                (ext for ext in media_extensions if original_filename.lower().endswith(ext)),
                '.wav'
            )
            clean_filename = f"{base_filename.strip('_- ')}{file_ext}"
        else:
            clean_filename = original_filename

        # Hedef dizini belirle (DÜZELTME BURADA)
        target_directory = INPUT_DIR if is_input else OUTPUT_DIR
        target_path = os.path.join(target_dir, clean_filename)
        
        # Dizini oluştur (yoksa)
        os.makedirs(target_directory, exist_ok=True)
        
        # Dizindeki TÜM önceki dosyaları sil
        for filename in os.listdir(target_directory):
            file_path = os.path.join(target_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"{file_path} Not deleted: {e}")

        # Yeni dosyayı kaydet
        if hasattr(uploaded_file, 'read'):
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            shutil.copy(uploaded_file, target_path)
            
        print(f"File saved successfully: {os.path.basename(target_path)}")
        return target_path
    
    except Exception as e:
        print(f"File save error: {e}")
        return None

        clear_memory()

def save_uploaded_file(uploaded_file, is_input=False, target_dir=None):
    try:
        # Medya dosya uzantıları
        media_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.mp4']
        
        # Hedef dizini belirle
        if target_dir is None:
            target_dir = INPUT_DIR if is_input else OUTPUT_DIR
        
        # Zaman damgası pattern'leri
        timestamp_patterns = [
            r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
            r'_\d{14}$',             # _20231215123456
            r'_\d{10}$',             # _1702658400
            r'_\d+$'                 # Herhangi bir sayı
        ]
        
        # Dosya adını al
        if hasattr(uploaded_file, 'name'):
            original_filename = os.path.basename(uploaded_file.name)
        else:
            original_filename = os.path.basename(str(uploaded_file))
        
        # Dosya adını temizle (sadece input'lar için)
        if is_input:
            base_filename = original_filename
            # Zaman damgalarını sil
            for pattern in timestamp_patterns:
                base_filename = re.sub(pattern, '', base_filename)
            # Çoklu uzantıları sil
            for ext in media_extensions:
                base_filename = base_filename.replace(ext, '')
            
            # Dosya uzantısını belirle
            file_ext = next(
                (ext for ext in media_extensions if original_filename.lower().endswith(ext)),
                '.wav'
            )
            clean_filename = f"{base_filename.strip('_- ')}{file_ext}"
        else:
            clean_filename = original_filename

        # Hedef dizini belirle (DÜZELTME BURADA)
        target_directory = INPUT_DIR if is_input else OUTPUT_DIR
        target_path = os.path.join(target_directory, clean_filename)
        
        # Dizini oluştur (yoksa)
        os.makedirs(target_directory, exist_ok=True)
        
        # Dizindeki TÜM önceki dosyaları sil
        for filename in os.listdir(target_directory):
            file_path = os.path.join(target_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"{file_path} Not deleted: {e}")

        # Yeni dosyayı kaydet
        if hasattr(uploaded_file, 'read'):
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            shutil.copy(uploaded_file, target_path)
            
        print(f"File saved successfully: {os.path.basename(target_path)}")
        return target_path
    
    except Exception as e:
        print(f"File save error: {e}")
        return None

def handle_file_upload(file_obj, file_path_input, is_auto_ensemble=False):
    try:
        target_dir = INPUT_DIR if not is_auto_ensemble else VİDEO_TEMP
        
        # Dosya yolu girilmişse onu kullan
        if file_path_input and os.path.exists(file_path_input):
            saved_path = save_uploaded_file(file_path_input, is_input=True, target_dir=target_dir)
        # Dosya yüklenmişse onu kullan
        elif file_obj:
            saved_path = save_uploaded_file(file_obj, is_input=True, target_dir=target_dir)
        else:
            return [None, None]
            
        return [
            gr.File(value=saved_path),
            gr.Audio(value=saved_path)
        ]
    except Exception as e:
        print(f"Hata: {str(e)}")
        return [None, None]        

def move_old_files(output_folder):
    old_output_folder = os.path.join(BASE_PATH, 'old_output')
    os.makedirs(old_output_folder, exist_ok=True)

    # Eski dosyaları taşı ve adlarının sonuna "old" ekle
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            # Yeni dosya adını oluştur
            new_filename = f"{os.path.splitext(filename)[0]}_old{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(old_output_folder, new_filename)
            shutil.move(file_path, new_file_path) 

def move_wav_files2(auto_ensemble_temp):
    AUTO_ENSEMBLE_TEMP = os.path.join(AUTO_ENSEMBLE_TEMP, 'auto_ensemble_temp')
    os.makedirs(AUTO_ENSEMBLE_TEMP, exist_ok=True)

    # Eski dosyaları taşı ve adlarının sonuna "old" ekle
    for filename in os.listdir(VİDEO_TEMP):
        file_path = os.path.join(VİDEO_TEMP, filename)
        if os.path.isfile(file_path):
            # Yeni dosya adını oluştur
            new_filename = f"{os.path.splitext(filename)[0]}_old{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(VİDEO_TEMP, new_filename)
            shutil.move(file_path, new_file_path)             


def extract_model_name(full_model_string):
    """
    Function to clear model name
    """
    if not full_model_string:
        return ""

    cleaned = str(full_model_string)

    # Remove the description
    if ' - ' in cleaned:
        cleaned = cleaned.split(' - ')[0]

    # Remove emoji prefixes
    emoji_prefixes = ['✅ ', '👥 ', '🗣️ ', '🏛️ ', '🔇 ', '🔉 ', '🎬 ', '🎼 ', '✅(?) ']
    for prefix in emoji_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    return cleaned.strip()

COOKIE_PATH = '/content/'
BASE_PATH = '/content/Music-Source-Separation-Training'
INPUT_DIR = os.path.join(BASE_PATH, 'input')
AUTO_ENSEMBLE_TEMP = os.path.join(BASE_PATH, 'auto_ensemble_temp')
OUTPUT_DIR = '/content/drive/MyDrive/output'
OLD_OUTPUT_DIR = '/content/drive/MyDrive/old_output'
AUTO_ENSEMBLE_OUTPUT = '/content/drive/MyDrive/ensemble_folder'
INFERENCE_SCRIPT_PATH = '/content/Music-Source-Separation-Training/inference.py'
VİDEO_TEMP = '/content/Music-Source-Separation-Training/video_temp'
os.makedirs(VİDEO_TEMP, exist_ok=True)  # Klasörü oluşturduğundan emin ol

def clear_directory(directory):
    """Deletes all files in the given directory."""
    files = glob.glob(os.path.join(directory, '*'))  # Dizin içindeki tüm dosyaları al
    for f in files:
        try:
            os.remove(f)  # remove files
        except Exception as e:
            print(f"{f} could not be deleted: {e}")     

def create_directory(directory):
    """Creates the given directory (if it exists, if not)."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"{directory} directory created.")
    else:
        print(f"{directory} directory already exists.")     

def convert_to_wav(file_path):
    """Converts the audio file to WAV format and moves it to the input directory."""
    
    # Directory paths
    BASE_DIR = "/content/Music-Source-Separation-Training"
    WAV_FOLDER = os.path.join(BASE_DIR, "wav_folder")
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    
    # Create directories if they don't exist
    os.makedirs(WAV_FOLDER, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Get file information
    original_filename = os.path.basename(file_path)
    filename, ext = os.path.splitext(original_filename)
    
    # If already a WAV file, return its path directly
    if ext.lower() == '.wav':
        return file_path  # Return the original path if it's already a WAV file

    try:
        # 1. Copy the original file to the wav_folder
        temp_input = os.path.join(WAV_FOLDER, original_filename)
        shutil.copy(file_path, temp_input)
        
        # 2. Prepare for WAV conversion
        wav_output = os.path.join(WAV_FOLDER, f"{filename}.wav")
        
        # 3. Run FFmpeg command to convert to WAV
        command = [
            'ffmpeg', '-y', '-i', temp_input,
            '-acodec', 'pcm_s16le', '-ar', '44100', wav_output
        ]
        subprocess.run(command, check=True, capture_output=True)
        
        # 4. Move the converted WAV to the input directory
        final_output = os.path.join(INPUT_DIR, f"{filename}.wav")
        
        # Check if the final output is the same as the wav_output
        if final_output == wav_output:
            print(f"Warning: The source and destination are the same file: {final_output}")
            return final_output  # Return the path if they are the same

        shutil.move(wav_output, final_output)
        
        # 5. Clear the input directory after moving the WAV
        clear_directory(INPUT_DIR)
        
        return final_output
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg Error ({e.returncode}):\nInput: {e.stderr.decode()}"
        print(error_msg)
        return None
        
    finally:
        # Clean up temporary files in wav_folder
        for f in os.listdir(WAV_FOLDER):
            file_path = os.path.join(WAV_FOLDER, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Cleanup error: {file_path} - {str(e)}")
       

def process_audio(input_audio_file, model, chunk_size, overlap, export_format, use_tta, demud_phaseremix_inst, extract_instrumental, denoise, *args, **kwargs):
    # Determine the audio path
    if input_audio_file is not None:
        audio_path = input_audio_file.name
    else:
        print("No audio file provided.")
        return [None] * 12  # Error case

    clear_directory(INPUT_DIR)    

     # Eski dosyaları taşı
    move_old_files(OUTPUT_DIR)   

    # Convert to WAV if necessary
    wav_path = convert_to_wav(audio_path)

    # Check if wav_path is valid
    if wav_path is None:
        print("Failed to convert audio to WAV format.")
        return [None] * 12  # Error case

    # Model adı temizleme
    clean_model = extract_model_name(model)
    print(f"Processing audio from: {audio_path} using model: {clean_model}")

    # Gerekli dizinleri oluştur
    create_directory(INPUT_DIR)
    create_directory(OUTPUT_DIR)
    create_directory(OLD_OUTPUT_DIR)


    # Ses dosyasını kaydetme
    if input_audio_file is not None:
        dest_path = save_uploaded_file(input_audio_file, is_input=True)
    else:
        dest_path = audio_path  # Kullanıcının girdiği dosya yolunu kullan

    if not dest_path:
        print("Failed to save file")
        return [None] * 12

    # Model yapılandırması
    model_type, config_path, start_check_point = "", "", ""

    if clean_model == 'VOCALS-InstVocHQ':
            model_type = 'mdx23c'
            config_path = 'ckpts/config_vocals_mdx23c.yaml'
            start_check_point = 'ckpts/model_vocals_mdx23c_sdr_10.17.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt')

    elif clean_model == 'VOCALS-MelBand-Roformer (by KimberleyJSN)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
            start_check_point = 'ckpts/MelBandRoformer.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
            download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1297 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1296 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
            download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-RoformerLargev1 (by unwa)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/config_bsrofoL.yaml'
            start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
            download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
            download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Mel-Roformer big beta 4 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
            start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Melband-Roformer BigBeta5e (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/big_beta5e.yaml'
            start_check_point = 'ckpts/big_beta5e.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v1 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst.yaml'
            start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v2 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
            start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
            start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
            start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'KARAOKE-MelBand-Roformer (by aufr33 & viperx)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
            start_check_point = 'ckpts/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'OTHER-BS-Roformer_1053 (by viperx)':
            model_type = 'bs_roformer'
            config_path = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.yaml'
            start_check_point = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.ckpt'
            download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt')
            download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'CROWD-REMOVAL-MelBand-Roformer (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer _crowd.yaml'
            start_check_point = 'ckpts/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-VitLarge23 (by ZFTurbo)':
            model_type = 'segm_models'
            config_path = 'ckpts/config_vocals_segm_models.yaml'
            start_check_point = 'ckpts/model_vocals_segm_models_sdr_9.77.ckpt'
            download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt')

    elif clean_model == 'CINEMATIC-BandIt_Plus (by kwatcharasupat)':
            model_type = 'bandit'
            config_path = 'ckpts/config_dnr_bandit_bsrnn_multi_mus64.yaml'
            start_check_point = 'ckpts/model_bandit_plus_dnr_sdr_11.47.chpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt')

    elif clean_model == 'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)':
            model_type = 'mdx23c'
            config_path = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
            start_check_point = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
            download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt')
            download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml')

    elif clean_model == '4STEMS-SCNet_MUSDB18 (by starrytong)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet.yaml'
            start_check_point = 'ckpts/scnet_checkpoint_musdb18.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt')

    elif clean_model == 'DE-REVERB-MDX23C (by aufr33 & jarredou)':
            model_type = 'mdx23c'
            config_path = 'ckpts/config_dereverb_mdx23c.yaml'
            start_check_point = 'ckpts/dereverb_mdx23c_sdr_6.9096.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml')

    elif clean_model == 'DENOISE-MelBand-Roformer-1 (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
            start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DENOISE-MelBand-Roformer-2 (by aufr33)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
            start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt')
            download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT (by Unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
            start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
            download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
            download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_v1e (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_melbandroformer_inst.yaml'
            start_check_point = 'ckpts/inst_v1e.ckpt'
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'bleed_suppressor_v1 (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_bleed_suppressor_v1.yaml'
            start_check_point = 'ckpts/bleed_suppressor_v1.ckpt'
            download_file('https://shared.multimedia.workers.dev/download/1/other/bleed_suppressor_v1.ckpt')
            download_file('https://shared.multimedia.workers.dev/download/1/other/config_bleed_suppressor_v1.yaml')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer (by Becruily)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_instrumental_becruily.yaml'
            start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
            download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
            download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
            conf_edit(config_path, chunk_size, overlap)
    
    elif clean_model == 'INST-MelBand-Roformer (by Becruily)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_instrumental_becruily.yaml'
            start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
            download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
            download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet_xl.yaml'
            start_check_point = 'ckpts/model_scnet_ep_54_sdr_9.8051.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_Large (by starrytong)':
            model_type = 'scnet'
            config_path = 'ckpts/config_musdb18_scnet_large_starrytong.yaml'
            start_check_point = 'ckpts/SCNet-large_starrytong_fixed.ckpt'
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml')
            download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt')
            conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)':
           model_type = 'bs_roformer'
           config_path = 'ckpts/config_bs_roformer_384_8_2_485100.yaml'
           start_check_point = 'ckpts/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
           download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml')
           download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt')
           conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          conf_edit(config_path, chunk_size, overlap)
          
    elif clean_model == 'DE-REVERB-Echo-MelBand-Roformer (by Sucial)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_dereverb-echo_mel_band_roformer.yaml'
          start_check_point = 'ckpts/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'
          download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt')
          download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_less_aggressive_anvuew':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_anvuew':
          model_type = 'mel_band_roformer'
          config_path = 'dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
          conf_edit(config_path, chunk_size, overlap)  


    elif clean_model == 'inst_gabox (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxBV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxBv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'inst_gaboxBV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxBv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)     


    elif clean_model == 'inst_gaboxBFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'inst_gaboxFV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    
    elif clean_model == 'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)':
          model_type = 'bs_roformer'
          config_path = 'ckpts/config_chorus_male_female_bs_roformer.yaml'
          start_check_point = 'ckpts/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'
          download_file('https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt')
          download_file('https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml')
          conf_edit(config_path, chunk_size, overlap)


    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
          start_check_point = 'ckpts/kimmel_unwa_ft2.ckpt'
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt')
          conf_edit(config_path, chunk_size, overlap)      

    elif clean_model == 'voc_gaboxBSroformer (by Gabox)':
          model_type = 'bs_roformer'
          config_path = 'ckpts/voc_gaboxBSroformer.yaml'
          start_check_point = 'ckpts/voc_gaboxBSR.ckpt'
          download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml')
          download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformer (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformerFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'voc_gaboxMelReformerFV2 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_gaboxFv2.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_GaboxFv3 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv3.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'Intrumental_Gabox (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/intrumental_gabox.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_Fv4Noise (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_Fv4Noise.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_V5 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV5.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model2.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/model3.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
          start_check_point = 'ckpts/kimmel_unwa_ft2_bleedless.ckpt'
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
          download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxFV1 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/inst_gaboxFv1.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_gaboxFV6 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV6.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'denoisedebleed (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
          start_check_point = 'ckpts/denoisedebleed.ckpt'
          download_file('https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INSTV5N (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV5N.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'Voc_Fv3 (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/voc_gabox.yaml'
          start_check_point = 'ckpts/voc_Fv3.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'MelBandRoformer4StemFTLarge (SYH99999)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/config.yaml'
          start_check_point = 'ckpts/MelBandRoformer4StemFTLarge.ckpt'
          download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml')
          download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'dereverb_mel_band_roformer_mono (by anvuew)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
          start_check_point = 'ckpts/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
          download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt')
          conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INSTV6N (by Gabox)':
          model_type = 'mel_band_roformer'
          config_path = 'ckpts/inst_gabox.yaml'
          start_check_point = 'ckpts/INSTV6N.ckpt'
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
          download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt')
          conf_edit(config_path, chunk_size, overlap)









    # Other model options will be added here...
    # (All the elif blocks you gave in the previous code will go here)


    else:
        print(f"Unsupported model: {clean_model}")
        return [None] * 12  # Hata durumu

    result = run_command_and_process_files(model_type, config_path, start_check_point, INPUT_DIR, OUTPUT_DIR, wav_path, extract_instrumental, use_tta, demud_phaseremix_inst, clean_model)

    # İşlem tamamlandıktan sonra giriş dizinini temizle
    clear_directory(INPUT_DIR)

    
    return result
    


def clean_model_name(model):
    """
    Clean and standardize model names for filename
    """
    # Mapping of complex model names to simpler, filename-friendly versions
    model_name_mapping = {
        'VOCALS-InstVocHQ': 'InstVocHQ',
        'VOCALS-MelBand-Roformer (by KimberleyJSN)': 'KimberleyJSN',
        'VOCALS-BS-Roformer_1297 (by viperx)': 'VOCALS_BS_Roformer1297',
        'VOCALS-BS-Roformer_1296 (by viperx)': 'VOCALS-BS-Roformer_1296',
        'VOCALS-BS-RoformerLargev1 (by unwa)': 'UnwaLargeV1',
        'VOCALS-Mel-Roformer big beta 4 (by unwa)': 'UnwaBigBeta4',
        'VOCALS-Melband-Roformer BigBeta5e (by unwa)': 'UnwaBigBeta5e',
        'INST-Mel-Roformer v1 (by unwa)': 'UnwaInstV1',
        'INST-Mel-Roformer v2 (by unwa)': 'UnwaInstV2',
        'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)': 'UnwaDualityV1',
        'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)': 'UnwaDualityV2',
        'KARAOKE-MelBand-Roformer (by aufr33 & viperx)': 'KaraokeMelBandRoformer',
        'VOCALS-VitLarge23 (by ZFTurbo)': 'VitLarge23',
        'VOCALS-MelBand-Roformer (by Becruily)': 'BecruilyVocals',
        'INST-MelBand-Roformer (by Becruily)': 'BecruilyInst',
        'VOCALS-MelBand-Roformer Kim FT (by Unwa)': 'KimFT',
        'INST-MelBand-Roformer Kim FT (by Unwa)': 'KimFTInst',
        'OTHER-BS-Roformer_1053 (by viperx)': 'OtherViperx1053',
        'CROWD-REMOVAL-MelBand-Roformer (by aufr33)': 'CrowdRemovalRoformer',
        'CINEMATIC-BandIt_Plus (by kwatcharasupat)': 'CinematicBandItPlus',
        'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)': 'DrumSepMDX23C',
        '4STEMS-SCNet_MUSDB18 (by starrytong)': 'FourStemsSCNet',
        'DE-REVERB-MDX23C (by aufr33 & jarredou)': 'DeReverbMDX23C',
        'DENOISE-MelBand-Roformer-1 (by aufr33)': 'DenoiseMelBand1',
        'DENOISE-MelBand-Roformer-2 (by aufr33)': 'DenoiseMelBand2',
        'INST-MelBand-Roformer (by Becruily)': 'BecruilyInst',
        '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)': 'FourStemsSCNetXL',
        '4STEMS-SCNet_Large (by starrytong)': 'FourStemsSCNetLarge',
        '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)': 'FourStemsBSRoformer',
        'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)': 'DeReverbMelBandAggr',
        'DE-REVERB-Echo-MelBand-Roformer (by Sucial)': 'DeReverbEchoMelBand',
        'bleed_suppressor_v1 (by unwa)': 'BleedSuppressorV1',
        'inst_v1e (by unwa)': 'InstV1E',
        'inst_gabox ( by Gabox)': 'InstGabox',
        'inst_gaboxBV1 (by Gabox)': 'InstGaboxBV1',
        'inst_gaboxBV2 (by Gabox)': 'InstGaboxBV2',
        'inst_gaboxBFV1 (by Gabox)': 'InstGaboxBFV1',
        'inst_gaboxFV2 (by Gabox)': 'InstGaboxFV2',
        'inst_gaboxFV1 (by Gabox)': 'InstGaboxFV1',
        'dereverb_mel_band_roformer_less_aggressive_anvuew': 'DereverbMelBandRoformerLessAggressive',
        'dereverb_mel_band_roformer_anvuew': 'DereverbMelBandRoformer',
        'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)': 'MaleFemale-BS-RoFormer(by aufr33)',
        'VOCALS-MelBand-Roformer (by Becruily)': 'Vocals-MelBand-Roformer(by Becruily)',
        'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)': 'Vocals-MelBand-Roformer-KİM-FT-2(by Unwa)',
        'voc_gaboxMelRoformer (by Gabox)': 'voc_gaboxMelRoformer',
        'voc_gaboxBSroformer (by Gabox)': 'voc_gaboxBSroformer',
        'voc_gaboxMelRoformerFV1 (by Gabox)': 'voc_gaboxMelRoformerFV1',
        'voc_gaboxMelRoformerFV2 (by Gabox)': 'voc_gaboxMelRoformerFV2',
        'SYH99999/MelBandRoformerSYHFTB1 (by Amane)': 'MelBandRoformerSYHFTB1',
        'inst_V5 (by Gabox)': 'INSTV5 (by Gabox)',
        'inst_Fv4Noise (by Gabox)': 'Inst_Fv4Noise (by Gabox)',
        'Intrumental_Gabox (by Gabox)': 'Intrumental_Gabox (by Gabox)',
        'inst_GaboxFv3 (by Gabox)': 'INST_GaboxFv3 (by Gabox)',
        'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)': 'MelBandRoformerSYHFTB1_model1',
        'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)': 'MelBandRoformerSYHFTB1_model2',
        'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)': 'MelBandRoformerSYHFTB1_model3',
        'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)': 'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)',
        'inst_gaboxFV6 (by Gabox)': 'inst_gaboxFV6 (by Gabox)',
        'denoisedebleed (by Gabox)': 'denoisedebleed (by Gabox)',
        'INSTV5N (by Gabox)': 'INSTV5N (by Gabox)',
        'Voc_Fv3 (by Gabox)': 'Voc_Fv3 (by Gabox)',
        'MelBandRoformer4StemFTLarge (SYH99999)': 'MelBandRoformer4StemFTLarge (SYH99999)',
        'dereverb_mel_band_roformer_mono (by anvuew)': 'dereverb_mel_band_roformer_mono (by anvuew)',
        'INSTV6N (by Gabox)': 'INSTV6N (by Gabox)',

        
        # Add more mappings as needed
    }

    # Use mapping if exists, otherwise clean the model name
    if model in model_name_mapping:
        return model_name_mapping[model]
    
    # General cleaning if not in mapping
    cleaned = re.sub(r'\s*\(.*?\)', '', model)  # Remove parenthetical info
    cleaned = cleaned.replace('-', '_')
    cleaned = ''.join(char for char in cleaned if char.isalnum() or char == '_')
    
    return cleaned

def shorten_filename(filename, max_length=30):
    """
    Shortens a filename to a specified maximum length
    
    Args:
        filename (str): The filename to be shortened
        max_length (int): Maximum allowed length for the filename
    
    Returns:
        str: Shortened filename
    """
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    
    # Take first 15 and last 10 characters
    shortened = base[:15] + "..." + base[-10:] + ext
    return shortened

def clean_filename(filename):
    """
    Temizlenmiş dosya adını döndürür
    """
    # Zaman damgası ve gereksiz etiketleri temizleme desenleri
    cleanup_patterns = [
        r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
        r'_\d{14}$',              # _20231215123456
        r'_\d{10}$',              # _1702658400
        r'_\d+$'                  # Herhangi bir sayı
    ]
    
    # Dosya adını ve uzantısını ayır
    base, ext = os.path.splitext(filename)
    
    # Zaman damgalarını temizle
    for pattern in cleanup_patterns:
        base = re.sub(pattern, '', base)
    
    # Dosya türü etiketlerini temizle
    file_types = ['vocals', 'instrumental', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry', 'male', 'female']
    for type_keyword in file_types:
        base = base.replace(f'_{type_keyword}', '')
    
    # Dosya türünü tespit et
    detected_type = None
    for type_keyword in file_types:
        if type_keyword in base.lower():
            detected_type = type_keyword
            break
    
    # Zaman damgaları ve gereksiz etiketlerden temizlenmiş base
    clean_base = base.strip('_- ')
    
    return clean_base, detected_type, ext

def run_command_and_process_files(model_type, config_path, start_check_point, INPUT_DIR, OUTPUT_DIR, dest_path, extract_instrumental, use_tta, demud_phaseremix_inst, clean_model):
    try:
        # Komut parçalarını oluştur
        cmd_parts = [
            "python", "inference.py",
            "--model_type", model_type,
            "--config_path", config_path,
            "--start_check_point", start_check_point,
            "--input_folder", INPUT_DIR,
            "--store_dir", OUTPUT_DIR,
            "--audio_path", dest_path  # İşlenecek ses dosyasının yolu
        ]

        # Opsiyonel parametreleri ekle
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")

        if use_tta:
            cmd_parts.append("--use_tta")

        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        # Komutu çalıştır
        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Çıktıları gerçek zamanlı olarak yazdır
        for line in process.stdout:
            print(line.strip())

        for line in process.stderr:
            print(line.strip())

        process.wait()

        # Model adını temizle
        filename_model = clean_model_name(clean_model)

        # Çıktı dosyalarını al
        output_files = os.listdir(OUTPUT_DIR)

        # Dosya yeniden adlandırma fonksiyonu
        def rename_files_with_model(folder, filename_model):
            for filename in sorted(os.listdir(folder)):
                file_path = os.path.join(folder, filename)

                # Medya dosyası değilse atla
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue

                base, ext = os.path.splitext(filename)

                # Temiz base adı
                clean_base = base.strip('_- ')

                # Yeni dosya adını oluştur
                new_filename = f"{clean_base}_{filename_model}{ext}"

                new_file_path = os.path.join(folder, new_filename)
                os.rename(file_path, new_file_path)

        # Dosyaları yeniden adlandır
        rename_files_with_model(OUTPUT_DIR, filename_model)

        # Güncellenmiş dosya listesini al
        output_files = os.listdir(OUTPUT_DIR)

        # Dosya bulma fonksiyonu
        def find_file(keyword):
            matching_files = [
                os.path.join(OUTPUT_DIR, f) for f in output_files 
                if keyword in f.lower()
            ]
            return matching_files[0] if matching_files else None

        # Farklı dosya türlerini bul
        vocal_file = find_file('vocals')
        instrumental_file = find_file('instrumental')
        phaseremix_file = find_file('phaseremix')
        drum_file = find_file('drum')
        bass_file = find_file('bass')
        other_file = find_file('other')
        effects_file = find_file('effects')
        speech_file = find_file('speech')
        music_file = find_file('music')
        dry_file = find_file('dry')
        male_file = find_file('male')
        female_file = find_file('female')
        

        # Bulunan dosyaları döndür
        return (
            vocal_file or None,
            instrumental_file or None,
            phaseremix_file or None,
            drum_file or None,
            bass_file or None,
            other_file or None,
            effects_file or None,
            speech_file or None,
            music_file or None,
            dry_file or None,
            male_file or None,
            female_file or None
            
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return (None,) * 12

       

def create_interface():
    # Let's define the model options in advance
    model_choices = {
        "Vocal Separation": [
            'Voc_Fv3 (by Gabox)',
            'VOCALS-BS-Roformer_1297 (by viperx)',
            'VOCALS-BS-Roformer_1296 (by viperx)',
            '✅ VOCALS-Mel-Roformer big beta 4 (by unwa) - Melspectrogram based high performance',
            'VOCALS-BS-RoformerLargev1 (by unwa) - Comprehensive model',
            'VOCALS-InstVocHQ - General purpose model',
            'VOCALS-MelBand-Roformer (by KimberleyJSN) - Alternative model',
            'VOCALS-VitLarge23 (by ZFTurbo) - Transformer-based model',
            'VOCALS-MelBand-Roformer Kim FT (by Unwa)',
            'VOCALS-MelBand-Roformer (by Becruily)',
            '✅ VOCALS-Melband-Roformer BigBeta5e (by unwa)',
            'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)',
            'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)',
            'voc_gaboxMelRoforner (by Gabox)',
            'voc_gaboxBSroformer (by Gabox)',
            'voc_gaboxMelRoformerFV1 (by Gabox)',
            'voc_gaboxMelRoformerFV2 (by Gabox)',
            'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)'
        ],
        "Instrumental Separation": [
            'INSTV5N (by Gabox)',
            'inst_gaboxFV6 (by Gabox)',
            '✅ INST-Mel-Roformer v2 (by unwa) - Most recent instrumental separation model',
            '✅ inst_v1e (by unwa)',
            '✅ INST-Mel-Roformer v1 (by unwa) - Old instrumental separation model',
            'INST-MelBand-Roformer (by Becruily)',
            'inst_gaboxFV2 (by Gabox)',
            'inst_gaboxFV1 (by Gabox)',
            'inst_gaboxBV2 (by Gabox)',
            'inst_gaboxBV1 (by Gabox)',
            'inst_gabox (by Gabox)',
            '✅(?) inst_GaboxFv3 (by Gabox)',
            'Intrumental_Gabox (by Gabox)',
            '✅(?) inst_Fv4Noise (by Gabox)',
            '✅(?) inst_V5 (by Gabox)',
            'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa) - Latest version instrumental separation',
            'INST-VOC-Mel-Roformer a.k.a. duality (by unwa) - Previous version',
            'INST-Separator MDX23C (by aufr33) - Alternative instrumental separation',
            'INSTV6N (by Gabox)'
        ],
        "Karaoke & Accompaniment": [
            '✅ KARAOKE-MelBand-Roformer (by aufr33 & viperx) - Advanced karaoke separation'
        ],
        "Noise & Effect Removal": [
            '',
            '🔇 DENOISE-MelBand-Roformer-1 (by aufr33) - Basic noise reduction',
            '🔉 DENOISE-MelBand-Roformer-2 (by aufr33) - Advanced noise reduction',
            'bleed_suppressor_v1 (by unwa) - dont use it if you dont know what youre doing',
            'dereverb_mel_band_roformer_mono (by anvuew)',
            '👥 CROWD-REMOVAL-MelBand-Roformer (by aufr33) - Crowd noise removal',
            '🏛️ DE-REVERB-MDX23C (by aufr33 & jarredou) - Reverb reduction',
            '🏛️ DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)',
            '🗣️ DE-REVERB-Echo-MelBand-Roformer (by Sucial)',
            'dereverb_mel_band_roformer_less_aggressive_anvuew',
            'dereverb_mel_band_roformer_anvuew'


        ],
        "Drum Separation": [
            '✅ DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou) - Detailed drum separation'
        ],
        "Multi-Stem & Other Models": [
            'MelBandRoformer4StemFTLarge (SYH99999)',
            '🎬 4STEMS-SCNet_MUSDB18 (by starrytong) - Multi-stem separation',
            '🎼 CINEMATIC-BandIt_Plus (by kwatcharasupat) - Cinematic music analysis',
            'OTHER-BS-Roformer_1053 (by viperx) - Other special models',
            '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)',
            '4STEMS-SCNet_Large (by starrytong)',
            '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)',
            'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)',
            'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)',
            'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)'
        ],
    }


    def update_models(category):
        models = model_choices.get(category, [])
        return gr.Dropdown(
            label="Select Model",
            choices=models,
            value=models[0] if models else None
        )


    def ensemble_files(args):
        """
        Ensemble audio files using the external script
        
        Args:
            args (list): Command-line arguments for ensemble script
        """
        try:
            
            script_path = "/content/Music-Source-Separation-Training/ensemble.py"
            
            
            full_command = ["python", script_path] + args
            
            
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=True
            )
            
            print("Ensemble successful:")
            print(result.stdout)
            return result.stdout
        
        except subprocess.CalledProcessError as e:
            print(f"Ensemble error: {e}")
            print(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            print(f"Unexpected error during ensemble: {e}")
            raise      

    def refresh_audio_files(directory):
        """
        Refreshes and lists audio files in the specified directory and old_output directory.
        
        Args:
            directory (str): Path of the directory to be scanned.
        
        Returns:
            list: List of discovered audio files.
        """
        try:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
            audio_files = [
                f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and os.path.splitext(f)[1].lower() in audio_extensions
            ]
            
            # Eski dosyaları da kontrol et
            old_output_directory = os.path.join(BASE_PATH, 'old_output')
            old_audio_files = [
                f for f in os.listdir(old_output_directory)
                if os.path.isfile(os.path.join(old_output_directory, f))
                and os.path.splitext(f)[1].lower() in audio_extensions
            ]
            
            return sorted(audio_files + old_audio_files)
        except Exception as e:
            print(f"Audio file listing error: {e}")
            return []

    
    # Global değişken tanımlamaları
    BASE_PATH = '/content/Music-Source-Separation-Training'
    AUTO_ENSEMBLE_TEMP = os.path.join(BASE_PATH, 'auto_ensemble_temp')
    model_output_dir = os.path.join(BASE_PATH, 'auto_ensemble_temp')

    def auto_ensemble_process(audio_input, selected_models, chunk_size, overlap, export_format2, 
                         use_tta, extract_instrumental, ensemble_type, 
                         progress=gr.Progress(), *args, **kwargs):  # Removed audio_path parameter
        try:
            # 1. Input validation
            if audio_input is None:
                return None, "No audio file provided"
                
            audio_path = audio_input.name  # Get path directly from audio_input
            if not os.path.exists(audio_path):
                return None, "Input file not found"

            # 2. WAV'a dönüştürme
            wav_path = convert_to_wav(audio_path)
            if wav_path is None:
                return None, "Failed to convert audio to WAV format."

            print(f"Converted WAV file path: {wav_path}")

            # 3. Model işlemleri
            all_outputs = []
            total_models = len(selected_models)
            
            # Geçici klasörleri hazırla
            clear_directory(AUTO_ENSEMBLE_TEMP)
            clear_directory(AUTO_ENSEMBLE_OUTPUT)
            shutil.rmtree(AUTO_ENSEMBLE_TEMP, ignore_errors=True)
            os.makedirs(AUTO_ENSEMBLE_TEMP, exist_ok=True)
            os.makedirs(VİDEO_TEMP, exist_ok=True)
            os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True) 

            shutil.copy(wav_path, os.path.join(INPUT_DIR, os.path.basename(wav_path)))

            for idx, model in enumerate(selected_models):
                progress((idx + 1) / total_models, f"Processing {model}...")
                
                clean_model = extract_model_name(model)
                print(f"Processing using model: {clean_model}")      

                # Model çıktı klasörü
                model_output_dir = os.path.join(AUTO_ENSEMBLE_TEMP)
                os.makedirs(model_output_dir, exist_ok=True)

                model_type, config_path, start_check_point = "", "", ""

                if clean_model == 'VOCALS-InstVocHQ':
                        model_type = 'mdx23c'
                        config_path = 'ckpts/config_vocals_mdx23c.yaml'
                        start_check_point = 'ckpts/model_vocals_mdx23c_sdr_10.17.ckpt'
                        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt')

                elif clean_model == 'VOCALS-MelBand-Roformer (by KimberleyJSN)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
                        start_check_point = 'ckpts/MelBandRoformer.ckpt'
                        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
                        download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-BS-Roformer_1297 (by viperx)':
                        model_type = 'bs_roformer'
                        config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
                        start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
                        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
                        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-BS-Roformer_1296 (by viperx)':
                        model_type = 'bs_roformer'
                        config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
                        start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
                        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
                        download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-BS-RoformerLargev1 (by unwa)':
                        model_type = 'bs_roformer'
                        config_path = 'ckpts/config_bsrofoL.yaml'
                        start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
                        download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
                        download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-Mel-Roformer big beta 4 (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
                        start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-Melband-Roformer BigBeta5e (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/big_beta5e.yaml'
                        start_check_point = 'ckpts/big_beta5e.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INST-Mel-Roformer v1 (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_inst.yaml'
                        start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INST-Mel-Roformer v2 (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
                        start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
                        start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
                        start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'KARAOKE-MelBand-Roformer (by aufr33 & viperx)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
                        start_check_point = 'ckpts/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
                        download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
                        download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'OTHER-BS-Roformer_1053 (by viperx)':
                        model_type = 'bs_roformer'
                        config_path = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.yaml'
                        start_check_point = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.ckpt'
                        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt')
                        download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'CROWD-REMOVAL-MelBand-Roformer (by aufr33)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/model_mel_band_roformer _crowd.yaml'
                        start_check_point = 'ckpts/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-VitLarge23 (by ZFTurbo)':
                        model_type = 'segm_models'
                        config_path = 'ckpts/config_vocals_segm_models.yaml'
                        start_check_point = 'ckpts/model_vocals_segm_models_sdr_9.77.ckpt'
                        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt')

                elif clean_model == 'CINEMATIC-BandIt_Plus (by kwatcharasupat)':
                        model_type = 'bandit'
                        config_path = 'ckpts/config_dnr_bandit_bsrnn_multi_mus64.yaml'
                        start_check_point = 'ckpts/model_bandit_plus_dnr_sdr_11.47.chpt'
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt')

                elif clean_model == 'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)':
                        model_type = 'mdx23c'
                        config_path = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
                        start_check_point = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
                        download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt')
                        download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml')

                elif clean_model == '4STEMS-SCNet_MUSDB18 (by starrytong)':
                        model_type = 'scnet'
                        config_path = 'ckpts/config_musdb18_scnet.yaml'
                        start_check_point = 'ckpts/scnet_checkpoint_musdb18.ckpt'
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt')

                elif clean_model == 'DE-REVERB-MDX23C (by aufr33 & jarredou)':
                        model_type = 'mdx23c'
                        config_path = 'ckpts/config_dereverb_mdx23c.yaml'
                        start_check_point = 'ckpts/dereverb_mdx23c_sdr_6.9096.ckpt'
                        download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt')
                        download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml')

                elif clean_model == 'DENOISE-MelBand-Roformer-1 (by aufr33)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
                        start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
                        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt')
                        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'DENOISE-MelBand-Roformer-2 (by aufr33)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
                        start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'
                        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt')
                        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-MelBand-Roformer Kim FT (by Unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
                        start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
                        download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
                        download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_v1e (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_melbandroformer_inst.yaml'
                        start_check_point = 'ckpts/inst_v1e.ckpt'
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
                        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'bleed_suppressor_v1 (by unwa)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_bleed_suppressor_v1.yaml'
                        start_check_point = 'ckpts/bleed_suppressor_v1.ckpt'
                        download_file('https://shared.multimedia.workers.dev/download/1/other/bleed_suppressor_v1.ckpt')
                        download_file('https://shared.multimedia.workers.dev/download/1/other/config_bleed_suppressor_v1.yaml')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-MelBand-Roformer (by Becruily)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_instrumental_becruily.yaml'
                        start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
                        download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
                        download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
                        conf_edit(config_path, chunk_size, overlap)
                
                elif clean_model == 'INST-MelBand-Roformer (by Becruily)':
                        model_type = 'mel_band_roformer'
                        config_path = 'ckpts/config_instrumental_becruily.yaml'
                        start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
                        download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
                        download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)':
                        model_type = 'scnet'
                        config_path = 'ckpts/config_musdb18_scnet_xl.yaml'
                        start_check_point = 'ckpts/model_scnet_ep_54_sdr_9.8051.ckpt'
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == '4STEMS-SCNet_Large (by starrytong)':
                        model_type = 'scnet'
                        config_path = 'ckpts/config_musdb18_scnet_large_starrytong.yaml'
                        start_check_point = 'ckpts/SCNet-large_starrytong_fixed.ckpt'
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml')
                        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt')
                        conf_edit(config_path, chunk_size, overlap)

                elif clean_model == '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)':
                      model_type = 'bs_roformer'
                      config_path = 'ckpts/config_bs_roformer_384_8_2_485100.yaml'
                      start_check_point = 'ckpts/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
                      download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml')
                      download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
                      start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
                      conf_edit(config_path, chunk_size, overlap)
                      
                elif clean_model == 'DE-REVERB-Echo-MelBand-Roformer (by Sucial)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config_dereverb-echo_mel_band_roformer.yaml'
                      start_check_point = 'ckpts/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'
                      download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt')
                      download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'dereverb_mel_band_roformer_less_aggressive_anvuew':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
                      start_check_point = 'ckpts/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'dereverb_mel_band_roformer_anvuew':
                      model_type = 'mel_band_roformer'
                      config_path = 'dereverb_mel_band_roformer_anvuew.yaml'
                      start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
                      conf_edit(config_path, chunk_size, overlap)  


                elif clean_model == 'inst_gabox (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gabox.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_gaboxBV1 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gaboxBv1.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt')
                      conf_edit(config_path, chunk_size, overlap)


                elif clean_model == 'inst_gaboxBV2 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gaboxBv2.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt')
                      conf_edit(config_path, chunk_size, overlap)     


                elif clean_model == 'inst_gaboxBFV1 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/gaboxFv1.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
                      conf_edit(config_path, chunk_size, overlap)


                elif clean_model == 'inst_gaboxFV2 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gaboxFv2.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                
                elif clean_model == 'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)':
                      model_type = 'bs_roformer'
                      config_path = 'ckpts/config_chorus_male_female_bs_roformer.yaml'
                      start_check_point = 'ckpts/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'
                      download_file('https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt')
                      download_file('https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml')
                      conf_edit(config_path, chunk_size, overlap)


                elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
                      start_check_point = 'ckpts/kimmel_unwa_ft2.ckpt'
                      download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
                      download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt')
                      conf_edit(config_path, chunk_size, overlap)      

                elif clean_model == 'voc_gaboxBSroformer (by Gabox)':
                      model_type = 'bs_roformer'
                      config_path = 'ckpts/voc_gaboxBSroformer.yaml'
                      start_check_point = 'ckpts/voc_gaboxBSR.ckpt'
                      download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml')
                      download_file('https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'voc_gaboxMelReformer (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/voc_gabox.yaml'
                      start_check_point = 'ckpts/voc_gabox.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'voc_gaboxMelReformerFV1 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/voc_gabox.yaml'
                      start_check_point = 'ckpts/voc_gaboxFv1.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'voc_gaboxMelReformerFV2 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/voc_gabox.yaml'
                      start_check_point = 'ckpts/voc_gaboxFv2.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_GaboxFv3 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gaboxFv3.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'Intrumental_Gabox (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/intrumental_gabox.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_Fv4Noise (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_Fv4Noise.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_V5 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/INSTV5.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config.yaml'
                      start_check_point = 'ckpts/model.ckpt'
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config.yaml'
                      start_check_point = 'ckpts/model2.ckpt'
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config.yaml'
                      start_check_point = 'ckpts/model3.ckpt'
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml')
                      download_file('https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
                      start_check_point = 'ckpts/kimmel_unwa_ft2_bleedless.ckpt'
                      download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
                      download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_gaboxFV1 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/inst_gaboxFv1.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'inst_gaboxFV6 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/INSTV6.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'denoisedebleed (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
                      start_check_point = 'ckpts/denoisedebleed.ckpt'
                      download_file('https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INSTV5N (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/INSTV5N.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'Voc_Fv3 (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/voc_gabox.yaml'
                      start_check_point = 'ckpts/voc_Fv3.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'MelBandRoformer4StemFTLarge (SYH99999)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/config.yaml'
                      start_check_point = 'ckpts/MelBandRoformer4StemFTLarge.ckpt'
                      download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml')
                      download_file('https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'dereverb_mel_band_roformer_mono (by anvuew)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
                      start_check_point = 'ckpts/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
                      download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

                elif clean_model == 'INSTV6N (by Gabox)':
                      model_type = 'mel_band_roformer'
                      config_path = 'ckpts/inst_gabox.yaml'
                      start_check_point = 'ckpts/INSTV6N.ckpt'
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml')
                      download_file('https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt')
                      conf_edit(config_path, chunk_size, overlap)

              

                # Ana sekme komut yapısını kullan
                cmd = [
                    "python", 
                    "inference.py",
                    "--model_type", model_type,
                    "--config_path", config_path,
                    "--start_check_point", start_check_point,
                    "--input_folder", INPUT_DIR,
                    "--store_dir", model_output_dir,
                    "--audio_path", wav_path  # Doğrudan wav_path kullan
                ]

                if use_tta:
                    cmd.append("--use_tta")
                if extract_instrumental:
                    cmd.append("--extract_instrumental")

                print(f"Running command: {' '.join(cmd)}")

                # Hata yakalama ile çalıştırma
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    print(result.stdout)
                    if result.returncode != 0:
                        print(f"Error: {result.stderr}")
                        return None, f"Model {model} failed: {result.stderr}"
                except Exception as e:
                    return None, f"Critical error with {model}: {str(e)}"
                
                # Çıktıları topla
                model_outputs = glob.glob(os.path.join(model_output_dir, "*.wav"))
                all_outputs.extend(model_outputs)

            # 4. Ensemble işlemi
            if len(all_outputs) < 2:
                return None, "At least 2 models required for ensemble"

            ensemble_output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"ensemble_{int(time.time())}.wav")
            ensemble_cmd = [
                "python", "ensemble.py",
                "--files", *all_outputs,
                "--type", ensemble_type,
                "--output", ensemble_output_path
            ]

            # Hata ayıklama için komutu yazdır
            print("Running ensemble command:", " ".join(ensemble_cmd)) 

            # Komutu çalıştır ve çıktıyı yakala
            result = subprocess.run(ensemble_cmd, capture_output=True, text=True)
            print("Ensemble stdout:", result.stdout)
            print("Ensemble stderr:", result.stderr)

            if os.path.exists(ensemble_output_path):
                print(f"✅ Ensemble saved to: {ensemble_output_path}")
                return ensemble_output_path, "Success!"
            else:
                print(f"❌ Failed to save ensemble!")
                return None, "Ensemble failed: No output file."

        except Exception as e:
            print(f"🔥 Critical error: {str(e)}")
            return None, str(e)
        
        finally:
            # İşlem tamamlandıktan sonra giriş dizinini temizle
            shutil.rmtree('/content/Music-Source-Separation-Training/auto_ensemble_temp', ignore_errors=True)
            clear_directory(AUTO_ENSEMBLE_TEMP)
            clear_directory(VİDEO_TEMP)
            clear_directory(INPUT_DIR)
            gc.collect()
             

    main_input_key = "shared_audio_input"
    # Global components
    input_audio_file = gr.File(visible=True)
    auto_input_audio_file = gr.File(visible=True)
    original_audio = gr.Audio(visible=True)
    

    css = """
    .header {
        background: #1e1e2f; /* Lacivert renk */
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .header-title {
        color: transparent !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.4);
        font-size: 2.5rem !important;
        letter-spacing: -0.5px;
        background: linear-gradient(45deg, #FFD700 30%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: text-glow 3s infinite alternate;
    }
    @keyframes text-glow {
        0% {
            text-shadow: 0 0 5px rgba(255, 255, 255, 0), 0 0 10px rgba(255, 255, 255, 0), 0 0 15px rgba(255, 255, 255, 0);
        }
        50% {
            text-shadow: 0 0 5px rgba(255, 255, 255, 1), 0 0 10px rgba(255, 255, 255, 1), 0 0 15px rgba(255, 255, 255, 1);
        }
        100% {
            text-shadow: 0 0 5px rgba(255, 255, 255, 0), 0 0 10px rgba(255, 255, 255, 0), 0 0 15px rgba(255, 255, 255, 0);
        }
    }
    .header-subtitle {
        color: #e0e7ff !important;
        font-size: 1.4rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.5px;
    }
    .version-badge {
        background: rgba(255,255,255,0.15) !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 24px !important;
        font-size: 1rem !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.25);
        color: #f0abfc !important;
        margin-top: 1rem !important;
        transition: all 0.3s ease;
    }
    .version-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(79,70,229,0.4);
    }

    /* Button Effects */
    button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: linear-gradient(135deg, #333333 0%, #555555 100%) !important; /* Siyah ve gri tonları */
        border: none !important;
        color: white !important;
        border-radius: 8px !important; /* Buton köşe yuvarlama */
        padding: 8px 16px !important; /* Buton boyutları */
        position: relative;
        overflow: hidden !important;
        font-size: 0.9rem !important; /* Buton yazı boyutu */
    }
    button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.5) !important;
    }
    button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg , 
            transparent 20%, 
            rgba(255,255,255,0.5) 50%, 
            transparent 80%);
        animation: button-shine 3s infinite linear;
    }
    @keyframes button-shine {
        0% { transform: rotate(0deg) translateX(-50%); }
        100% { transform: rotate(360deg) translateX(-50%); }
    }

    /* Background Effects */
    body {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        min-height: 100vh;
        margin: 0;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
    }

    /* Light Animation */
    .header {
        background-color: #1e1e2f !important; /* Lacivert renk */
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="gray",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Poppins"), "Arial", "sans-serif"]
        ),
        css=css
    ) as demo:
        with gr.Column():
            gr.Markdown("""
            <div class="header">
                <div class="header-title">🌀 By Sir Joseph</div>
                <div class="header-subtitle">Source owner: ZFTurbo</div>
                <div class="version-badge">Version 3.0</div>
            </div>
            """)
        
        with gr.Tabs():
            # Audio Separation Tab
            with gr.Tab("Audio Separation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 Input Settings")
                        with gr.Group():
                            input_audio_file = gr.File(label="Upload file")
                            file_path_input = gr.Textbox(
                                label="Or enter file path",
                                placeholder="Enter full path to audio file",
                                interactive=True
                            )

                        # Noise reduction tip
                        gr.Markdown("""
                        <div style="
                            background: rgba(245,245,220,0.15);
                            padding: 1rem;
                            border-radius: 8px;
                            border-left: 3px solid #6c757d;
                            margin: 1rem 0 1.5rem 0;
                        ">
                            <b>🔈 Processing Tip:</b> For noisy results, use <code>bleed_suppressor_v1</code> 
                            or <code>denoisedebleed</code> models in the <i>"Denoise & Effect Removal"</i> 
                            category to clean the output
                        </div>
                        """)

                        model_category = gr.Dropdown(
                            label="Model Category",
                            choices=list(model_choices.keys()),
                            value="Vocal Separation"
                        )

                        model_dropdown = gr.Dropdown(
                            label="Select Model",
                            interactive=True
                        )

                        process_btn = gr.Button("🚀 Start Processing", variant="primary")
                        clear_old_output_btn = gr.Button("🧹 Clear Outputs")
                        clear_old_output_status = gr.Textbox(label="Status", interactive=False)
    


                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("Original Audio"):
                                original_audio = gr.Audio(
                                    label="Original",
                                    interactive=False,
                                    every=1,
                                    elem_id="original_audio_player"
                                )
                            with gr.Tab("Vocals"):
                                vocals_audio = gr.Audio(label="vocals", show_download_button=True)
                            with gr.Tab("Instrumental"):
                                instrumental_audio = gr.Audio(label="instrumental", show_download_button=True)
                            with gr.Tab("Advanced"):
                                phaseremix_audio = gr.Audio(label="Phase Remix")  
                                drum_audio = gr.Audio(label="Drums")
                                bass_audio = gr.Audio(label="Bass")
                                other_audio = gr.Audio(label="Other")
                                effects_audio = gr.Audio(label="Effects")
                                speech_audio = gr.Audio(label="Speech")
                                music_audio = gr.Audio(label="Music")
                                dry_audio = gr.Audio(label="Dry")
                                male_audio = gr.Audio(label="Male")
                                female_audio = gr.Audio(label="Female")

                        # Expert Settings Accordion
                        with gr.Accordion("⚙️ Expert Settings", open=False):
                            with gr.Group():
                                export_format = gr.Dropdown(
                                    label="Output Format",
                                    choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                    value='wav FLOAT'
                                )
                                
                                overlap = gr.Slider(
                                    label="Overlap",
                                    minimum=2,
                                    maximum=50,
                                    step=1,
                                    value=2,
                                    info="Recommended: 2-10 (Higher values increase quality but require more VRAM)"
                                )
                                
                                chunk_size = gr.Dropdown(
                                    label="Chunk Size",
                                    choices=[352800, 485100],
                                    value=352800,
                                    info="Don't change unless you have specific requirements"
                                )

                            # Advanced Settings Sub-Accordion
                            with gr.Accordion("🔧 Advanced Processing Options", open=False):
                                use_tta = gr.Checkbox(
                                    label="Use TTA (Test Time Augmentation)", 
                                    value=False,
                                    info="Improves quality but increases processing time"
                                )
                                
                                use_demud_phaseremix_inst = gr.Checkbox(
                                    label="Enable Demud Phase Remix",
                                    info="Advanced phase correction for instrumental tracks"
                                )
                                
                                extract_instrumental = gr.Checkbox(
                                    label="Extract Instrumental Version",
                                    info="Generate separate instrumental track"
                                )                   
                        



            # Oto Ensemble Sekmesi
            with gr.Tab("Auto Ensemble"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            auto_input_audio_file = gr.File(label="Upload file")
                            auto_file_path_input = gr.Textbox(
                                label="Or enter file path",
                                placeholder="Enter full path to audio file",
                                interactive=True
                            )

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Row():
                                auto_use_tta = gr.Checkbox(label="Use TTA", value=False)
                                auto_extract_instrumental = gr.Checkbox(label="Instrumental Only")
                            
                            with gr.Row():
                                auto_overlap = gr.Slider(
                                    label="Overlap",
                                    minimum=2,
                                    maximum=50,
                                    value=2,
                                    step=1
                                )
                                auto_chunk_size = gr.Dropdown(
                                    label="Chunk Size",
                                    choices=[352800, 485100],
                                    value=352800
                                )
                                export_format2 = gr.Dropdown(
                                    label="Output Format",
                                    choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                    value='wav FLOAT'
                                )

                        # Model Seçim Bölümü
                        with gr.Group():
                            gr.Markdown("### 🧠 Model Selection") 
                            with gr.Row():
                                auto_category_dropdown = gr.Dropdown(
                                label="Model Category",
                                choices=list(model_choices.keys()),
                                value="Vocal Separation"
                            )

                            # Model seçimi (tek seferde)
                            auto_model_dropdown = gr.Dropdown(
                                label="Select Models from Category",
                                choices=model_choices["Vocal Separation"],
                                multiselect=True,
                                max_choices=50,
                                interactive=True
                            )

                            # Seçilen modellerin listesi (ayrı kutucuk)
                            selected_models = gr.Dropdown(
                                label="Selected Models",
                                choices=[],
                                multiselect=True,
                                interactive=False  # Kullanıcı buraya direkt seçim yapamaz
                            )

                            
                            with gr.Row():
                                add_btn = gr.Button("➕ Add Selected", variant="secondary")
                                clear_btn = gr.Button("🗑️ Clear All", variant="stop")

                        # Ensemble Ayarları
                        with gr.Group():
                            gr.Markdown("### ⚡ Ensemble Settings")
                            with gr.Row():
                                auto_ensemble_type = gr.Dropdown(
                                    label="Method",
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                          'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value='avg_wave'
                                )
                            
                            gr.Markdown("**Recommendation:** avg_wave and max_fft best results")

                        auto_process_btn = gr.Button("🚀 Start Processing", variant="primary")

                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("🔊 Original Audio"):
                                original_audio2 = gr.Audio(
                                        label=" Original Audio",
                                        interactive=False,
                                        every=1,  # Her 1 saniyede bir güncelle
                                        elem_id="original_audio_player"
                                    )
                            with gr.Tab("🎚️ Ensemble Result"):
                                auto_output_audio = gr.Audio(
                                    label="Output Preview",
                                    show_download_button=True,
                                    interactive=False
                                )
                        
                        auto_status = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            placeholder="Waiting for processing...",
                            elem_classes="status-box"
                        )

                        gr.Markdown("""
                            <div style="
                                background: rgba(110, 142, 251, 0.1);
                                padding: 1.2rem;
                                border-radius: 12px;
                                border-left: 4px solid #6e8efb;
                                margin: 1rem 0;
                                backdrop-filter: blur(3px);
                                border: 1px solid rgba(255,255,255,0.2);
                            ">
                                <div style="display: flex; align-items: start; gap: 1rem;">
                                    <div style="
                                        font-size: 1.4em;
                                        color: #6e8efb;
                                        margin-top: -2px;
                                    ">⚠️</div>
                                    <div style="color: #2d3748;">
                                        <h4 style="
                                            margin: 0 0 0.8rem 0;
                                            color: #4a5568;
                                            font-weight: 600;
                                            font-size: 1.1rem;
                                        ">
                                            Model Selection Guidelines
                                        </h4>
                                        <ul style="
                                            margin: 0;
                                            padding-left: 1.2rem;
                                            color: #4a5568;
                                            line-height: 1.6;
                                        ">
                                            <li><strong>Avoid cross-category mixing:</strong> Combining vocal and instrumental models may create unwanted blends</li>
                                            <li><strong>Special model notes:</strong>
                                                <ul style="padding-left: 1.2rem; margin: 0.5rem 0;">
                                                    <li>Duality models (v1/v2) - Output both stems</li>
                                                    <li>MDX23C Separator - Hybrid results</li>
                                                </ul>
                                            </li>
                                            <li><strong>Best practice:</strong> Use 3-5 similar models from same category</li>
                                        </ul>
                                        <div style="
                                            margin-top: 1rem;
                                            padding: 0.8rem;
                                            background: rgba(167, 119, 227, 0.1);
                                            border-radius: 8px;
                                            color: #6e8efb;
                                            font-size: 0.9rem;
                                        ">
                                            💡 Pro Tip: Start with "VOCALS-MelBand-Roformer BigBeta5e" + "VOCALS-BS-Roformer_1297" combination
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """)

                # Kategori değişim fonksiyonunu güncelleyelim
                def update_models(category):
                    return gr.Dropdown(choices=model_choices[category])

                def add_models(new_models, existing_models):
                    updated = list(set(existing_models + new_models))
                    return gr.Dropdown(choices=updated, value=updated)

                def clear_models():
                    return gr.Dropdown(choices=[], value=[]) 

                # Etkileşimler
                def update_category(target):
                    category_map = {
                        "Only Vocals": "Vocal Separation",
                        "Only Instrumental": "Instrumental Separation"
                    }
                    return category_map.get(target, "Vocal Separation")

                # Otomatik yenileme için olayı bağla
                input_audio_file.upload(
                    fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=False),
                    inputs=[input_audio_file, file_path_input],
                    outputs=[input_audio_file, original_audio]
                )

                file_path_input.change(
                    fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=False),
                    inputs=[input_audio_file, file_path_input],
                    outputs=[input_audio_file, original_audio]
                )

                auto_input_audio_file.upload(
                    fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=True),
                    inputs=[auto_input_audio_file, auto_file_path_input],
                    outputs=[auto_input_audio_file, original_audio2]
                )

                auto_file_path_input.change(
                    fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=True),
                    inputs=[auto_input_audio_file, auto_file_path_input],
                    outputs=[auto_input_audio_file, original_audio2]
                )


                auto_category_dropdown.change(
                    fn=update_models,
                    inputs=auto_category_dropdown,
                    outputs=auto_model_dropdown
                )

                add_btn.click(
                     fn=add_models,
                     inputs=[auto_model_dropdown, selected_models],
                     outputs=selected_models
                )

                clear_btn.click(
                     fn=clear_models,
                     inputs=[],
                     outputs=selected_models
                )

                auto_process_btn.click(
                    fn=auto_ensemble_process,
                    inputs=[
                        auto_input_audio_file,
                        selected_models,
                        auto_chunk_size,
                        auto_overlap,
                        export_format2,
                        auto_use_tta,
                        auto_extract_instrumental,
                        auto_ensemble_type,
                        gr.State(None)
                    ],
                    outputs=[auto_output_audio, auto_status]
                )                        

            # İndirme Sekmesi
            with gr.Tab("Download Sources"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🗂️ Cloud Storage")
                        drive_url_input = gr.Textbox(label="Google Drive Shareable Link")
                        drive_download_btn = gr.Button("⬇️ Download from Drive", variant="secondary")
                        drive_download_status = gr.Textbox(label="Download Status")
                        drive_download_output = gr.File(label="Downloaded File", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 🌐 Direct Links")
                        direct_url_input = gr.Textbox(label="Audio File URL")
                        direct_download_btn = gr.Button("⬇️ Download from URL", variant="secondary")
                        direct_download_status = gr.Textbox(label="Download Status")
                        direct_download_output = gr.File(label="Downloaded File", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 🍪 Cookie Management")
                        cookie_file = gr.File(
                            label="Upload Cookies.txt",
                            file_types=[".txt"],
                            interactive=True,
                            elem_id="cookie_upload"
                        )
                        gr.Markdown("""
                        <div style="margin-left:15px; font-size:0.95em">
                        
                        **📌 Why Needed?**  
                        - Access age-restricted content  
                        - Download private/unlisted videos  
                        - Bypass regional restrictions  
                        - Avoid YouTube download limits  

                        **⚠️ Important Notes**  
                        - NEVER share your cookie files!  
                        - Refresh cookies when:  
                          • Getting "403 Forbidden" errors  
                          • Downloads suddenly stop  
                          • Seeing "Session expired" messages  

                        **🔄 Renewal Steps**  
                        1. Install this <a href="https://chromewebstore.google.com/detail/get-cookiestxt-clean/ahmnmhfbokciafffnknlekllgcnafnie" target="_blank">Chrome extension</a>  
                        2. Login to YouTube in Chrome  
                        3. Click extension icon → "Export"  
                        4. Upload the downloaded file here  

                        **⏳ Cookie Lifespan**  
                        - Normal sessions: 24 hours  
                        - Sensitive operations: 1 hour  
                        - Password changes: Immediate invalidation  

                        </div>
                        """)


                        # Event handlers
                        model_category.change(
                            fn=update_models,
                            inputs=model_category,
                            outputs=model_dropdown
                        )

                        clear_old_output_btn.click(
                            fn=clear_old_output,
                            outputs=clear_old_output_status
                        )

                        process_btn.click(
                            fn=process_audio,
                            inputs=[
                                input_audio_file,
                                model_dropdown,
                                chunk_size,
                                overlap,
                                export_format,
                                use_tta,
                                use_demud_phaseremix_inst,
                                extract_instrumental,
                                gr.State(None),
                                gr.State(None)
                            ],
                            outputs=[
                                vocals_audio, instrumental_audio, phaseremix_audio,
                                drum_audio, bass_audio, other_audio, effects_audio,
                                speech_audio, music_audio, dry_audio, male_audio, female_audio
                            ]
                        )

                        drive_download_btn.click(
                            fn=download_callback,
                            inputs=[drive_url_input, gr.State('drive')],
                            outputs=[
                                drive_download_output,  # 0. Dosya çıktısı
                                drive_download_status,  # 1. Durum mesajı
                                input_audio_file,       # 2. Ana ses dosyası girişi
                                auto_input_audio_file,  # 3. Oto ensemble girişi
                                original_audio,         # 4. Orijinal ses çıktısı
                                original_audio2 
                            ]
                        )

                        direct_download_btn.click(
                            fn=download_callback,
                            inputs=[direct_url_input, gr.State('direct'), cookie_file],
                            outputs=[
                                direct_download_output,  # 0. Dosya çıktısı
                                direct_download_status,  # 1. Durum mesajı
                                input_audio_file,        # 2. Ana ses dosyası girişi
                                auto_input_audio_file,   # 3. Oto ensemble girişi
                                original_audio,           # 4. Orijinal ses çıktısı
                                original_audio2 
                            ]
                        )

            
            with gr.Tab("Manuel Ensemble"):
                gr.Markdown("# 🎵 Manuel Ensemble Tool")
                
                with gr.Row():
                    with gr.Column():
                        refresh_btn = gr.Button("🔄 Refresh Audio Files")

                        ensemble_type = gr.Dropdown(
                            label="Ensemble Algorithm",
                            choices=[
                                'avg_wave',
                                'median_wave',
                                'min_wave',
                                'max_wave',
                                'avg_fft',
                                'median_fft',
                                'min_fft',
                                'max_fft'
                            ],
                            value='avg_wave'
                        )
                        
                        file_dropdowns = []
                        audio_files = refresh_audio_files('/content/drive/MyDrive/output')
                        
                        for i in range(10):
                            file_dropdown = gr.Dropdown(
                                label=f"Audio File {i+1}",
                                choices=['None'] + audio_files,
                                value='None'
                            )
                            file_dropdowns.append(file_dropdown)
                        
                        def update_audio_dropdowns():
                            updated_files = refresh_audio_files('/content/drive/MyDrive/output')
                            return [
                                gr.Dropdown(choices=['None'] + updated_files, value='None')
                                for _ in range(10)
                            ]
                        
                        refresh_btn.click(
                            fn=update_audio_dropdowns,
                            outputs=file_dropdowns
                        )
                        
                        weights_input = gr.Textbox(
                            label="Weights (comma-separated, optional)",
                            placeholder="e.g., 1.0, 1.2, 0.8"
                        )
                    
                    with gr.Column():
                        ensemble_output_audio = gr.Audio(label="Ensembled Audio")
                        ensemble_status = gr.Textbox(label="Status")

                        ensemble_process_btn = gr.Button("Ensemble Audio")
                    
                    def ensemble_audio_fn(file_1, file_2, file_3, file_4, file_5, 
                                          file_6, file_7, file_8, file_9, file_10, 
                                          ensemble_type, weights_input):
                        try:
                            file_dropdowns = [
                                file_1, file_2, file_3, file_4, file_5,
                                file_6, file_7, file_8, file_9, file_10
                            ]
                            
                            files = []
                            paths_to_check = [
                                '/content/drive/MyDrive/output',
                                '/content/Music-Source-Separation-Training/old_output'
                            ]
        
                            for f in file_dropdowns:
                                if f != 'None':
                                    for path in paths_to_check:
                                        full_path = os.path.join(path, f)
                                        if os.path.exists(full_path):
                                            files.append(full_path)
                                            break
        
                            if len(files) < 2:
                                return None, "Select at least 2 files for ensemble"
        
                            if weights_input and weights_input.strip():
                                weights = [float(w.strip()) for w in weights_input.split(',')]
                                if len(weights) != len(files):
                                    return None, "Weights must match number of selected files"
                            else:
                                weights = None
                            
                            output_path = "/content/drive/MyDrive/ensemble_folder/ensembled_audio.wav"
                            
                            ensemble_args = [
                                "--files"] + files + [
                                "--type", ensemble_type,
                                "--output", output_path
                            ]
                            
                            if weights:
                                ensemble_args.extend(["--weights"] + [str(w) for w in weights])
                            
                            ensemble_files(ensemble_args)
                            
                            return output_path, "Ensemble successful!"
                        
                        except Exception as e:
                            return None, f"Error: {str(e)}"
                    
                    ensemble_process_btn.click(
                        fn=ensemble_audio_fn,
                        inputs=file_dropdowns + [ensemble_type, weights_input],
                        outputs=[ensemble_output_audio, ensemble_status]
                    )

    return demo

def launch_with_share():
    try:
        port = generate_random_port()
        demo = create_interface()
        
        share_link = demo.launch(
            share=True,
            server_port=port,
            server_name='0.0.0.0',
            inline=False,
            allowed_paths=[
                '/content',
                '/content/drive/MyDrive/output',
                '/tmp'
                '/model_output_dir',
                'model_output_dir'
            ]
        )
        
        print(f"🌐 Gradio Share Link: {share_link}")
        print(f"🔌 Local Server Port: {port}")
        
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("🛑 Server stopped by user.")
    except Exception as e:
        print(f"❌ Error during server launch: {e}")
    finally:
        try:
            demo.close()
        except:
            pass

if __name__ == "__main__":
    launch_with_share()
