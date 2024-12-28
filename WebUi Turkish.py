#@markdown ## (sana bir link vericek onu aç örnek olarak: https://RastgeleSayılar.gradio.live)
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
import subprocess
import yt_dlp
import validators
from pytube import YouTube
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import math
import hashlib
import requests
import re
import gc
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
import requests
from urllib.parse import urlparse
from urllib.parse import quote

os.makedirs('/content/Music-Source-Separation-Training/input', exist_ok=True)
os.makedirs('/content/Music-Source-Separation-Training/output', exist_ok=True)


def update_progress(progress=gr.Progress()):
    def track_progress(percent):
        progress(percent/100)
    return track_progress


def clear_input_folder():
    # Klasörü temizleme işlemi
    input_path = "/content/Music-Source-Separation-Training/input"
    if os.path.exists(input_path):
        shutil.rmtree(input_path)
    os.makedirs(input_path, exist_ok=True)


import os
import shutil
import time
import validators
import yt_dlp
import gdown

def download_callback(url, download_type='direct'):
    try:
        # Klasörü temizle
        clear_input_folder()

        # Hedef klasör
        input_path = "/content/Music-Source-Separation-Training/input"
        os.makedirs(input_path, exist_ok=True)

        # URL kontrolü
        if not validators.url(url):
            return None, "Geçersiz URL", None, None

        # İndirme türüne göre farklı işlemler
        if download_type == 'direct':
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(input_path, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0',
                }],
                'max_filesize': 10 * 1024 * 1024 * 1024,  # 10 GB sınırı
                'nooverwrites': True,
                'no_color': True,
                'progress_hooks': [download_progress_hook]
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                wav_path = ydl.prepare_filename(info_dict).replace(f".{info_dict['ext']}", ".wav")

        elif download_type == 'drive':
            # Google Drive'dan büyük dosya indirmek için gdown kullan
            file_id = url.split("/")[-2] if "/file/d/" in url else url.split("=")[-1]
            output_path = os.path.join(input_path, "downloaded_file.wav")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

            # İndirme sonrası dosya kontrolü
            if os.path.exists(output_path):
                wav_path = output_path
            else:
                return None, "Dosya indirilemedi", None, None

        else:
            return None, "Geçersiz indirme türü", None, None

        # Dosya kontrolleri
        if wav_path and os.path.exists(wav_path):
            filename = os.path.basename(wav_path)
            input_file_path = os.path.join(input_path, filename)

            # Aynı isimde dosya varsa timestamp ekle
            if os.path.exists(input_file_path):
                base, ext = os.path.splitext(filename)
                timestamp = int(time.time())
                filename = f"{base}_{timestamp}{ext}"
                input_file_path = os.path.join(input_path, filename)

            # Dosyayı taşı
            shutil.move(wav_path, input_file_path)

            return (
                gr.File(value=input_file_path),  # İndirilen dosya
                f"Başarıyla indirildi: {filename}",  # Mesaj
                gr.File(value=input_file_path),  # input_audio güncellemesi
                gr.Audio(value=input_file_path)  # original_audio için ses
            )

        return None, "Dosya indirilemedi", None, None

    except Exception as e:
        print(f"Download hatası: {e}")
        return None, str(e), None, None

# İndirme ilerlemesini takip etmek için hook fonksiyonu
def download_progress_hook(d):
    if d['status'] == 'finished':
        print('İndirme tamamlandı, dönüştürme yapılıyor...')
    elif d['status'] == 'downloading':
        downloaded_bytes = d.get('downloaded_bytes', 0)
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        if total_bytes > 0:
            percent = downloaded_bytes * 100. / total_bytes
            print(f'İndiriliyor: {percent:.1f}%')

# Global değişkeni en üstte tanımlayın
INPUT_DIR = "/content/Music-Source-Separation-Training/input"

def download_file(url, directory='ckpts'):
    """
    Belirtilen URL'den dosya indirir

    Args:
        url (str): İndirilecek dosyanın URL'si
        directory (str, optional): İndirilen dosyanın kaydedileceği dizin. Varsayılan 'ckpts'

    Returns:
        str: İndirilen dosyanın tam yolu
    """
    # Dizini oluştur
    os.makedirs(directory, exist_ok=True)

    # Dosya adını URL'den çıkar
    filename = os.path.basename(url)
    filepath = os.path.join(directory, filename)

    # Eğer dosya zaten varsa indirme
    if os.path.exists(filepath):
        print(f"{filename} zaten mevcut.")
        return filepath

    try:
        # İndirme işlemi
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP hata kontrolü

        # Dosyayı kaydet
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"{filename} başarıyla indirildi.")
        return filepath

    except Exception as e:
        print(f"{filename} indirilemedi: {e}")
        return None

        clear_memory()


def generate_random_port():
    return random.randint(1000, 9000)

    clear_memory()

# Markdown açıklamaları
markdown_intro = """
# Ses Ayrıştırma Aracı

Bu araç, ses dosyalarını ayrıştırmak için kullanılır.
"""

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def tuple_constructor(loader, node):
    values = loader.construct_sequence(node)
    return tuple(values)

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def conf_edit(config_path, chunk_size, overlap):
    print("Using custom overlap/chunk_size values")
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=IndentDumper, allow_unicode=True)


def save_uploaded_file(uploaded_file, is_input=False):
    """
    Saves the uploaded file in the specified directory, 
    removing existing timestamps and multiple extensions
    """
    try:
        # Media file extensions
        media_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
        
        # Timestamp patterns
        timestamp_patterns = [
            r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
            r'_\d{14}$',              # _20231215123456
            r'_\d{10}$',              # _1702658400
        ]
        
        # Safely get the filename
        if hasattr(uploaded_file, 'name'):
            original_filename = os.path.basename(uploaded_file.name)
        else:
            original_filename = os.path.basename(str(uploaded_file))
        
        # Remove timestamps and multiple extensions from filename
        base_filename = original_filename
        
        # Clear timestamps
        for pattern in timestamp_patterns:
            base_filename = re.sub(pattern, '', base_filename)
        
        # Clear multiple extensions
        for ext in media_extensions:
            base_filename = base_filename.replace(ext, '')
        
        # Determine file extension
        file_ext = next((ext for ext in media_extensions if original_filename.lower().endswith(ext)), '.wav')
        
        # Create clean filename
        clean_filename = base_filename.strip('_- ') + file_ext
        
        # Determine target directory
        target_directory = INPUT_DIR if is_input else OUTPUT_DIR
        
        # Create full target path
        target_path = os.path.join(target_directory, clean_filename)
        
        # If a file with the same name exists, create a unique name
        counter = 1
        original_target_path = target_path
        while os.path.exists(target_path):
            base, ext = os.path.splitext(original_target_path)
            target_path = f"{base}_{counter}{ext}"
            counter += 1
        
        # Save the file
        if hasattr(uploaded_file, 'read'):
            # Gradio file object
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            # If it's already a file path
            shutil.copy(uploaded_file, target_path)
        
        print(f"Dosya başarıyla kaydedildi: {os.path.basename(target_path)}")
        return target_path
    
    except Exception as e:
        print(f"Dosya kaydetme hatası: {e}")
        return None

        clear_memory()

        clear_memory()

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
    emoji_prefixes = ['✅ ', '👥 ', '🗣️ ', '🏛️ ', '🔇 ', '🔉 ', '🎬 ', '🎼 ']
    for prefix in emoji_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    return cleaned.strip()

BASE_PATH = '/content/Music-Source-Separation-Training'
INPUT_DIR = os.path.join(BASE_PATH, 'input')
OUTPUT_DIR = '/content/drive/MyDrive/output'

def clear_directory(directory):
    """Verilen dizindeki tüm dosyaları siler."""
    files = glob.glob(os.path.join(directory, '*'))  # Dizin içindeki tüm dosyaları al
    for f in files:
        try:
            os.remove(f)  # Dosyayı sil
        except Exception as e:
            print(f"{f} silinemedi: {e}")

def create_directory(directory):
    """Verilen dizini oluşturur (varsa, yoksa)."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"{directory} dizini oluşturuldu.")
    else:
        print(f"{directory} dizini zaten mevcut.")

def process_audio(input_audio, model, chunk_size, overlap, export_format, use_tta, extract_instrumental, *args, **kwargs):
    # Create input and output directories
    create_directory(INPUT_DIR)
    create_directory(OUTPUT_DIR)

    # Delete existing files
    clear_directory(INPUT_DIR)

    # Clear model name
    clean_model = extract_model_name(model)
    print(f"Temizlenmiş Model Adı: {clean_model}")

    # File control
    if input_audio is None:
        print("Dosya yüklenmedi")
        return [None] * 9

    # Save file
    dest_path = save_uploaded_file(input_audio, is_input=True)

    if not dest_path:
        print("Dosya kaydedilemedi")
        return [None] * 9

    # Export format parsing
    if export_format == 'wav FLOAT':
        flac_file = False
        pcm_type = 'FLOAT'
        file_ext = 'wav'
    else:
        flac_file = True
        pcm_type = export_format.split(' ')[1]
        file_ext = 'flac'

    # Define input_folder and output_folder
    input_folder = INPUT_DIR
    output_folder = OUTPUT_DIR

    # Model seçimi ve ilgili parametreleri belirle
    model_type, config_path, start_check_point = "", "", ""

    if clean_model == 'VOCALS-InstVocHQ':
        model_type = 'mdx23c'
        config_path = 'ckpts/config_vocals_mdx23c.yaml'
        start_check_point = 'ckpts/model_vocals_mdx23c_sdr_10.17.ckpt'
        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt')

    elif clean_model == 'VOCALS-MelBand-Roformer (Yapımcı: KimberleyJSN)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_vocals_mel_band_roformer_kj.yaml'
        start_check_point = 'ckpts/MelBandRoformer.ckpt'
        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml')
        download_file('https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1297 (Yapımcı: viperx)':
        model_type = 'bs_roformer'
        config_path = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml'
        start_check_point = 'ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml')
        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-Roformer_1296 (Yapımcı: viperx)':
        model_type = 'bs_roformer'
        config_path = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.yaml'
        start_check_point = 'ckpts/model_bs_roformer_ep_368_sdr_12.9628.ckpt'
        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt')
        download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-BS-RoformerLargev1 (Yapımcı: unwa)':
        model_type = 'bs_roformer'
        config_path = 'ckpts/config_bsrofoL.yaml'
        start_check_point = 'ckpts/BS-Roformer_LargeV1.ckpt'
        download_file('https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt')
        download_file('https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Mel-Roformer big beta 4 (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_big_beta4.yaml'
        start_check_point = 'ckpts/melband_roformer_big_beta4.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-Melband-Roformer BigBeta5e (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/big_beta5e.yaml'
        start_check_point = 'ckpts/big_beta5e.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v1 (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_inst.yaml'
        start_check_point = 'ckpts/melband_roformer_inst_v1.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-Mel-Roformer v2 (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_inst_v2.yaml'
        start_check_point = 'ckpts/melband_roformer_inst_v2.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
        start_check_point = 'ckpts/melband_roformer_instvoc_duality_v1.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-VOC-Mel-Roformer a.k.a. duality v2 (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_instvoc_duality.yaml'
        start_check_point = 'ckpts/melband_roformer_instvox_duality_v2.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'KARAOKE-MelBand-Roformer (Yapımcı: aufr33 & viperx)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_mel_band_roformer_karaoke.yaml'
        start_check_point = 'ckpts/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'
        download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt')
        download_file('https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'OTHER-BS-Roformer_1053 (Yapımcı: viperx)':
        model_type = 'bs_roformer'
        config_path = 'ckpts/model_bs_roformer_ep_937_sdr_10.5309.yaml'
        start_check_point = ' ckpts/model_bs_roformer_ep_937_sdr_10.5309.ckpt'
        download_file('https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt')
        download_file('https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'CROWD-REMOVAL-MelBand-Roformer (Yapımcı: aufr33)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/model_mel_band_roformer_crowd.yaml'
        start_check_point = 'ckpts/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-VitLarge23 (Yapımcı: ZFTurbo)':
        model_type = 'segm_models'
        config_path = 'ckpts/config_vocals_segm_models.yaml'
        start_check_point = 'ckpts/model_vocals_segm_models_sdr_9.77.ckpt'
        download_file('https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt')

    elif clean_model == 'CINEMATIC-BandIt_Plus (Yapımcı: kwatcharasupat)':
        model_type = 'bandit'
        config_path = 'ckpts/config_dnr_bandit_bsrnn_multi_mus64.yaml'
        start_check_point = 'ckpts/model_bandit_plus_dnr_sdr_11.47.chpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt')

    elif clean_model == 'DRUMSEP-MDX23C_DrumSep_6stem (Yapımcı: aufr33 & jarredou)':
        model_type = 'mdx23c'
        config_path = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
        start_check_point = 'ckpts/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'
        download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt')
        download_file('https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml')

    elif clean_model == '4STEMS-SCNet_MUSDB18 (Yapımcı: starrytong)':
        model_type = 'scnet'
        config_path = 'ckpts/config_musdb18_scnet.yaml'
        start_check_point = 'ckpts/scnet_checkpoint_musdb18.ckpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt')

    elif clean_model == 'DE-REVERB-MDX23C (Yapımcı: aufr33 & jarredou)':
        model_type = 'mdx23c'
        config_path = 'ckpts/config_dereverb_mdx23c.yaml'
        start_check_point = 'ckpts/dereverb_mdx23c_sdr_6.9096.ckpt'
        download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt')
        download_file('https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml')

    elif clean_model == 'DENOISE-MelBand-Roformer-1 (Yapımcı: aufr33)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
        start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'
        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt')
        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DENOISE-MelBand-Roformer-2 (Yapımcı: aufr33)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/model_mel_band_roformer_denoise.yaml'
        start_check_point = 'ckpts/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'
        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt')
        download_file('https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer Kim FT (Yapımcı: Unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
        start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
        download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
        download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'inst_v1e (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_melbandroformer_inst.yaml'
        start_check_point = 'ckpts/inst_v1e.ckpt'
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt')
        download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'bleed_suppressor_v1 (Yapımcı: unwa)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_bleed_suppressor_v1.yaml'
        start_check_point = 'ckpts/bleed_suppressor_v1.ckpt'
        download_file('https://shared.multimedia.workers.dev/download/1/other/bleed_suppressor_v1.ckpt')
        download_file('https://shared.multimedia.workers.dev/download/1/other/config_bleed_suppressor_v1.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'VOCALS-MelBand-Roformer (Yapımcı: Becruily)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_instrumental_becruily.yaml'
        start_check_point = 'ckpts/mel_band_roformer_vocals_becruily.ckpt'
        download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml')
        download_file('https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'INST-MelBand-Roformer (Yapımcı: Becruily)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_instrumental_becruily.yaml'
        start_check_point = 'ckpts/mel_band_roformer_instrumental_becruily.ckpt'
        download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml')
        download_file('https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_XL_MUSDB18 (Yapımcı: ZFTurbo)':
        model_type = 'scnet'
        config_path = 'ckpts/config_musdb18_scnet_xl.yaml'
        start_check_point = 'ckpts/model_scnet_ep_54_sdr_9.8051.ckpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-SCNet_Large (Yapımcı: starrytong)':
        model_type = 'scnet'
        config_path = 'ckpts/config_musdb18_scnet_large_starrytong.yaml'
        start_check_point = 'ckpts/SCNet-large_starrytong_fixed.ckpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == '4STEMS-BS-Roformer_MUSDB18 (Yapımcı: ZFTurbo)':
        model_type = 'bs_roformer'
        config_path = 'ckpts/config_bs_roformer_384_8_2_485100.yaml'
        start_check_point = 'ckpts/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml')
        download_file('https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (Yapımcı: anvuew)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/dereverb_mel_band_roformer_anvuew.yaml'
        start_check_point = 'ckpts/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
        download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt')
        download_file('https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml')
        conf_edit(config_path, chunk_size, overlap)

    elif clean_model == 'DE-REVERB-Echo-Mel Band-Roformer (Yapımcı: Sucial)':
        model_type = 'mel_band_roformer'
        config_path = 'ckpts/config_dereverb-echo_mel_band_roformer.yaml'
        start_check_point = 'ckpts/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'
        download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt')
        download_file('https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml')
        conf_edit(config_path, chunk_size, overlap)


    # Other model options will be added here...
    # (All the elif blocks you gave in the previous code will go here)


    else:
        print(f"Desteklenmeyen model: {clean_model}")
        return None, None, None, None, None, None, None, None, None


    cmd_parts = [
        "python", "inference.py",
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--input_folder", INPUT_DIR,
        "--store_dir", OUTPUT_DIR
    ]

    # Add optional parameters
    if extract_instrumental:
        cmd_parts.append("--extract_instrumental")

    # FLAC and PCM settings
    if flac_file:
        cmd_parts.append("--flac_file")
        cmd_parts.extend(["--pcm_type", pcm_type])
    elif pcm_type != 'FLOAT':
        cmd_parts.extend(["--pcm_type", pcm_type])

    if use_tta:
        cmd_parts.append("--use_tta")

    # Run command and process files
    return run_command_and_process_files(cmd_parts, BASE_PATH, output_folder, clean_model)

def clean_model_name(model):
    """
    Clean and standardize model names for filename
    """
    # Mapping of complex model names to simpler, filename-friendly versions
    model_name_mapping = {
        'VOCALS-InstVocHQ': 'InstVocHQ',
        'VOCALS-MelBand-Roformer (Yapımcı: KimberleyJSN)': 'KimberleyJSN',
        'VOCALS-BS-Roformer_1297 (Yapımcı: viperx)': 'Viperx1297',
        'VOCALS-BS-Roformer_1296 (Yapımcı: viperx)': 'Viperx1296',
        'VOCALS-BS-RoformerLargev1 (Yapımcı: unwa)': 'UnwaLargeV1',
        'VOCALS-Mel-Roformer big beta 4 (Yapımcı: unwa)': 'UnwaBigBeta4',
        'VOCALS-Melband-Roformer BigBeta5e (Yapımcı: unwa)': 'UnwaBigBeta5e',
        'INST-Mel-Roformer v1 (Yapımcı: unwa)': 'UnwaInstV1',
        'INST-Mel-Roformer v2 (Yapımcı: unwa)': 'UnwaInstV2',
        'INST-VOC-Mel-Roformer a.k.a. duality (Yapımcı: unwa)': 'UnwaDualityV1',
        'INST-VOC-Mel-Roformer a.k.a. duality v2 (Yapımcı: unwa)': 'UnwaDualityV2',
        'KARAOKE-MelBand-Roformer (Yapımcı: aufr33 & viperx)': 'KaraokeRoformer',
        'VOCALS-VitLarge23 (Yapımcı: ZFTurbo)': 'VitLarge23',
        'VOCALS-MelBand-Roformer (Yapımcı: Becruily)': 'BecruilyVocals',
        'INST-MelBand-Roformer (Yapımcı: Becruily)': 'BecruilyInst',
        # Gerekirse daha fazla eşleştirme ekleyin
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
    file_types = ['vocals', 'instrumental', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry']
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

def run_command_and_process_files(cmd_parts, BASE_PATH, output_folder, clean_model):
    try:
        # Run subprocess
        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Print outputs in real-time
        for line in process.stdout:
            print(line.strip())

        for line in process.stderr:
            print(line.strip())

        process.wait()

        # Clean the model name for filename
        filename_model = clean_model_name(clean_model)

        # Get updated file list
        output_files = os.listdir(output_folder)

        # File renaming function
        def rename_files_with_model(folder, filename_model):
            # Dictionary to track first occurrence of each file type
            processed_types = {}

            # Sort files to ensure consistent processing
            for filename in sorted(output_files):
                # Full path of the file
                file_path = os.path.join(folder, filename)

                # Skip if not a media file
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue

                # Dosya adını ve uzantısını ayır
                base, ext = os.path.splitext(filename)
                
                # Zaman damgası ve gereksiz etiketleri temizleme desenleri
                cleanup_patterns = [
                    r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
                    r'_\d{14}$',              # _20231215123456
                    r'_\d{10}$',              # _1702658400
                    r'_\d+$'                  # Herhangi bir sayı
                ]
                
                # Zaman damgalarını temizle
                for pattern in cleanup_patterns:
                    base = re.sub(pattern, '', base)
                
                # Dosya türü etiketlerini tespit et
                file_types = ['vocals', 'instrumental', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry']
                detected_type = None
                
                for type_keyword in file_types:
                    if type_keyword in base.lower():
                        detected_type = type_keyword
                        break
                
                # Dosya türü etiketlerini temizle
                for type_keyword in file_types:
                    base = base.replace(f'_{type_keyword}', '')
                
                # Temiz base adı
                clean_base = base.strip('_- ')

                # If file type is found
                if detected_type:
                    # If this is the first file of this type, rename it
                    if detected_type not in processed_types:
                        # Yeni dosya adını oluştur
                        new_filename = f"{clean_base}_{detected_type}_{filename_model}{ext}"
                        new_file_path = os.path.join(folder, new_filename)
                        
                        # Rename the file
                        os.rename(file_path, new_file_path)
                        processed_types[detected_type] = new_file_path

        # Rename files
        rename_files_with_model(output_folder, filename_model)

        # Get updated file list after renaming
        output_files = os.listdir(output_folder)

        # File finding function
        def find_file(keyword):
            # Find files with the keyword
            matching_files = [
                os.path.join(output_folder, f) for f in output_files 
                if keyword in f.lower()
            ]
            
            return matching_files[0] if matching_files else None

        # Find different types of files
        vocal_file = find_file('vocals')
        instrumental_file = find_file('instrumental')
        drum_file = find_file('drum')
        bass_file = find_file('bass')
        other_file = find_file('other')
        effects_file = find_file('effects')
        speech_file = find_file('speech')
        music_file = find_file('music')
        dry_file = find_file('dry')

        # Return found files
        return (
            vocal_file or None,
            instrumental_file or None,
            drum_file or None,
            bass_file or None,
            other_file or None,
            effects_file or None,
            speech_file or None,
            music_file or None,
            dry_file or None
        )

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return (None,) * 9


def create_interface():
    # Model seçeneklerini önceden tanımlayalım
    model_choices = {
        "Vokal Ayırma": [
            'VOCALS-BS-Roformer_1297 (Yapımcı: viperx)',
            '✅ VOCALS-Mel-Roformer big beta 4 (Yapımcı: unwa) - Melspectrogram tabanlı yüksek performans',
            '✅ VOCALS-BS-RoformerLargev1 (Yapımcı: unwa) - Geniş kapsamlı model',
            'VOCALS-InstVocHQ - Genel amaçlı model',
            'VOCALS-MelBand-Roformer (Yapımcı: KimberleyJSN) - Alternatif model',
            'VOCALS-VitLarge23 (Yapımcı: ZFTurbo) - Transformer tabanlı model',
            'VOCALS-MelBand-Roformer Kim FT (Yapımcı: Unwa)',
            'VOCALS-BS-Roformer_1296 (Yapımcı: viperx)',
            'VOCALS-MelBand-Roformer (Yapımcı: Becruily)',
            'VOCALS-Melband-Roformer BigBeta5e (Yapımcı: unwa)'
        ],
        "Enstrüman Ayırma": [
            '✅ INST-VOC-Mel-Roformer a.k.a. duality v2 (Yapımcı: unwa) - Son versiyon enstrüman ayrıştırma',
            '✅ INST-VOC-Mel-Roformer a.k.a. duality (Yapımcı: unwa) - Önceki versiyon',
            'INST-Separator MDX23C (Yapımcı: aufr33) - Alternatif enstrüman ayrıştırma',
            '✅ INST-Mel-Roformer v2 (Yapımcı: unwa) - En güncel enstrumental ayrıştırma modeli',
            '✅ inst_v1e (Yapımcı: unwa)',
            '✅ INST-Mel-Roformer v1 (Yapımcı: unwa) - Eski enstrumental ayrıştırma modeli',
            'INST-MelBand-Roformer (Yapımcı: Becruily)'
        ],
        "Karaoke & Benzerleri": [
            '✅ KARAOKE-MelBand-Roformer (Yapımcı: aufr33 & viperx) - Gelişmiş karaoke ayrıştırma'
        ],
        "Gürültü & Efekt Silme": [
            '👥 CROWD-REMOVAL-MelBand-Roformer (Yapımcı: aufr33) - Kalabalık gürültüsü temizleme',
            '🏛️ DE-REVERB-MDX23C (Yapımcı: aufr33 & jarredou) - Yankı azaltma',
            '🏛️ DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (Yapımcı: anvuew)',
            '🗣️ DE-REVERB-Echo-MelBand-Roformer (Yapımcı: Sucial)',
            '🔇 DENOISE-MelBand-Roformer-1 (Yapımcı: aufr33) - Temel gürültü azaltma',
            '🔉 DENOISE-MelBand-Roformer-2 (Yapımcı: aufr33) - Gelişmiş gürültü azaltma'
        ],
        "Davul Ayırma": [
            '✅ DRUMSEP-MDX23C_DrumSep_6stem (Yapımcı: aufr33 & jarredou) - Detaylı davul ayrıştırma'
        ],
        "Çoklu Ayırma & Diğer Modeller": [
            '🎬 4STEMS-SCNet_MUSDB18 (Yapımcı: starrytong) - Çok katmanlı ayrıştırma',
            '🎼 CINEMATIC-BandIt_Plus (Yapımcı: kwatcharasupat) - Sinematik müzik analizi',
            'OTHER-BS-Roformer_1053 (Yapımcı: viperx) - Diğer özel modeller',
            '4STEMS-SCNet_XL_MUSDB18 (Yapımcı: ZFTurbo)',
            '4STEMS-SCNet_Large (Yapımcı: starrytong)',
            '4STEMS-BS-Roformer_MUSDB18 (Yapımcı: ZFTurbo)'
        ]
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
            
            print("Topluluk başarılı:")
            print(result.stdout)
            return result.stdout
        
        except subprocess.CalledProcessError as e:
            print(f"Topluluk hatası: {e}")
            print(f"Hata çıktısı: {e.stderr}")
            raise
        except Exception as e:
            print(f"Toplama sırasında beklenmeyen hata: {e}")
            raise

    def listeyi_yenile(directory):
        """
        Refreshes and lists audio files in the specified directory
    
        Args:
            directory (str): Path of the directory to be scanned
    
        Returns:
            list: List of discovered audio files
        """
        try:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
            audio_files = [
                f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and os.path.splitext(f)[1].lower() in audio_extensions
            ]
            return sorted(audio_files)
        except Exception as e:
            print(f"Ses dosyası listeleme hatası: {e}")
            return []

    with gr.Blocks() as demo:
        gr.Markdown("# 🎵 Müzik Kaynak Ayırma Aracı")

        with gr.Tabs():
            with gr.Tab("Ses Ayırma"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_audio = gr.File(label="Ses Dosyası Seç", type="filepath")

                        model_category = gr.Dropdown(
                            label="Model Kategorisi",
                            choices=list(model_choices.keys())
                        )

                        model_dropdown = gr.Dropdown(label="Model Seç")

                        overlap = gr.Slider(
                        label="Overlap",
                        info="bu genelde 5 ila 2 arası olmalı farklı bişiler istersen değiştir.",
                        minimum=2,
                        maximum=50,
                        step=1,
                        value=2
                    )

                    model_category.change(
                        fn=update_models,
                        inputs=model_category,
                        outputs=model_dropdown
                    )

                    with gr.Column(scale=1):
                        chunk_size = gr.Dropdown(
                            label="Chunk Size",
                            info="buna dokunma.",
                            choices=[352800, 485100],
                            value=352800
                        )

                        use_tta = gr.Checkbox(
                            label="TTA Kullan",
                            info="Test Time Augmentation: Modelin tahmin performansını artırır. Aynı zamanda işlem süresini de uzatır."
                        )

                        # Extract Instrumental Checkbox
                        extract_instrumental = gr.Checkbox(
                            label="Extract Instrumental",
                            info="bunu aktifleştirirseniz çıktı sesin tam tersini verir instrumental vericekse bunun tersi vocal olur ve 2 çıktıyı da verir.",
                            value=False
                        )

                        ses_çıktısı = gr.Dropdown(
                            label="Export Format",
                            choices=[
                                'wav FLOAT',
                                'flac PCM_16',
                                'flac PCM_24'
                            ],
                            value='wav FLOAT'
                        )

                        process_btn = gr.Button("Sesi İşle")

                        with gr.Column():
                            original_audio = gr.Audio(label="Orijinal Ses")
                            vocals_audio = gr.Audio(label="Vocals")
                            instrumental_audio = gr.Audio(label="Instrumental")
                            drum_audio = gr.Audio(label="Drum")
                            bass_audio = gr.Audio(label="Bass")
                            other_audio = gr.Audio(label="Other")
                            effects_audio = gr.Audio(label="effects")
                            speech_audio = gr.Audio(label="speech")
                            music_audio = gr.Audio(label="music")
                            dry_audio = gr.Audio(label="dry")

                input_audio.upload(
                    fn=lambda x: x,
                    inputs=input_audio,
                    outputs=original_audio
                )

                process_btn.click(
                    fn=process_audio,
                    inputs=[
                        input_audio,
                        model_dropdown,
                        chunk_size,
                        overlap,
                        ses_çıktısı,
                        use_tta,
                        extract_instrumental,
                        gr.State(None),
                        gr.State(None)
                    ],
                    outputs=[
                        vocals_audio,
                        instrumental_audio,
                        drum_audio,
                        bass_audio,
                        other_audio,
                        effects_audio,
                        speech_audio,
                        music_audio,
                        dry_audio
                    ]
                )
            with gr.Tab("İndirme"):
                gr.Markdown("## 🔗 Ses Dosyası İndirme")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📂 Google Drive İndirme")
                        drive_url_input = gr.Textbox(label="Google Drive Linki")
                        drive_download_btn = gr.Button("İndir")
                        drive_download_status = gr.Textbox(label="Durum")
                        drive_download_output = gr.File(label="İndirilen Dosya")

                    with gr.Column():
                        gr.Markdown("### 🌐 Direkt URL İndirme")
                        direct_url_input = gr.Textbox(label="Direkt URL")
                        direct_download_btn = gr.Button("İndir")
                        direct_download_status = gr.Textbox(label="Durum")
                        direct_download_output = gr.File(label="İndirilen Dosya")

                drive_download_btn.click(
                    fn =download_callback,
                    inputs=[drive_url_input, gr.State('drive')],
                    outputs=[drive_download_output, drive_download_status, input_audio, original_audio]
                )

                direct_download_btn.click(
                    fn=download_callback,
                    inputs=[direct_url_input, gr.State('direct')],
                    outputs=[direct_download_output, direct_download_status, input_audio, original_audio]
                )
                
            with gr.Tab("Ses Birleştirme"):
                gr.Markdown("# 🎵 Ses Birleştirme Aracı")
            
                with gr.Row():
                    with gr.Column():
                    
                        refresh_btn = gr.Button("🔄 listeyi_yenile")

                    
                        ensemble_type = gr.Dropdown(
                            label="Birleştirme Algoritması",
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
                        drive_audio_files = listeyi_yenile('/content/drive/MyDrive/output')
                        
                        for i in range(10):
                            file_dropdown = gr.Dropdown(
                                label=f"Ses Dosyası {i+1}",
                                choices=['Yok'] + drive_audio_files,
                                value='Yok'
                            )
                            file_dropdowns.append(file_dropdown)
                        
                        def update_audio_dropdowns():
                            updated_files = listeyi_yenile('/content/drive/MyDrive/output')
                            return [
                                gr.Dropdown(choices=['Yok'] + updated_files, value='Yok')
                                for _ in range(10)
                            ]
                        
                        refresh_btn.click(
                            fn=update_audio_dropdowns,
                            outputs=file_dropdowns
                        )
                        
                        weights_input = gr.Textbox(
                            label="Ağırlıklar (virgülle ayrılmış, isteğe bağlı)",
                            placeholder="örn., 1.0, 1.2, 0.8"
                        )
                    
                    with gr.Column():
                        ensemble_output_audio = gr.Audio(label="Birleştirilmiş Ses", type="filepath")
                        ensemble_status = gr.Textbox(label="Durum")
                        
                        ensemble_process_btn = gr.Button("Sesleri Birleştir")
                
                def ensemble_audio_fn(file_1, file_2, file_3, file_4, file_5, 
                                      file_6, file_7, file_8, file_9, file_10, 
                                      ensemble_type, weights_input):
                    try:
                        file_dropdowns = [
                            file_1, file_2, file_3, file_4, file_5,
                            file_6, file_7, file_8, file_9, file_10
                        ]
                        
                        files = [
                            os.path.join('/content/drive/MyDrive/output', f)
                            for f in file_dropdowns 
                            if f != 'Yok'
                        ]
                        
                        if len(files) < 2:
                            return None, "Birleştirme için en az 2 dosya seçin"
                        
                        if weights_input and weights_input.strip():
                            weights = [float(w.strip()) for w in weights_input.split(',')]
                            if len(weights) != len(files):
                                return None, "Ağırlıklar seçilen dosya sayısıyla eşleşmelidir"
                        else:
                            weights = None
                        
                        output_path = "/tmp/birlestirilmis_ses.wav"
                        
                        ensemble_args = [
                            "--files"] + files + [
                            "--type", ensemble_type,
                            "--output", output_path
                        ]
                        
                        if weights:
                            ensemble_args.extend(["--weights"] + [str(w) for w in weights])
                        
                        ensemble_files(ensemble_args)
                        
                        return output_path, "Birleştirme başarılı!"
                    
                    except Exception as e:
                        return None, f"Hata: {str(e)}"
                
                ensemble_process_btn.click(
                    fn=ensemble_audio_fn,
                    inputs=file_dropdowns + [ensemble_type, weights_input],
                    outputs=[ensemble_output_audio, ensemble_status]
                )

        return demo

def launch_with_share():
    port = generate_random_port()
    demo = create_interface()
    share_link = demo.launch(
        share=True,
        server_port=port,
        server_name='0.0.0.0',
        inline=False,
        allowed_paths=['/content/drive/MyDrive/output']
    )
    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Döngü durduruldu.")

if __name__ == "__main__":
    launch_with_share()
