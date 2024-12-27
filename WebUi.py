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

os.makedirs('/content/Music-Source-Separation-Training/input', exist_ok=True)
os.makedirs('/content/Music-Source-Separation-Training/output', exist_ok=True)

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


import os
import shutil
import time
import validators
import yt_dlp
import gdown

def download_callback(url, download_type='direct'):
    try:
        # Clear folder
        clear_input_folder()

        # Target folder
        input_path = "/content/Music-Source-Separation-Training/input"
        os.makedirs(input_path, exist_ok=True)

        # URL control
        if not validators.url(url):
            return None, "Invalid URL", None, None

        # Different operations depending on the type of download
        if download_type == 'direct':
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(input_path, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '0',
                }],
                'max_filesize': 10 * 1024 * 1024 * 1024,  # 10 GB limit
                'nooverwrites': True,
                'no_color': True,
                'progress_hooks': [download_progress_hook]
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                wav_path = ydl.prepare_filename(info_dict).replace(f".{info_dict['ext']}", ".wav")

        elif download_type == 'drive':
            # Use gdown to download large files from Google Drive
            file_id = url.split("/")[-2] if "/file/d/" in url else url.split("=")[-1]
            output_path = os.path.join(input_path, "downloaded_file.wav")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=False)

            # File check after download
            if os.path.exists(output_path):
                wav_path = output_path
            else:
                return None, "Failed to download file", None, None

        else:
            return None, "Invalid download type", None, None

        # File checks
        if wav_path and os.path.exists(wav_path):
            filename = os.path.basename(wav_path)
            input_file_path = os.path.join(input_path, filename)

            # Add timestamp if there is a file with the same name
            if os.path.exists(input_file_path):
                base, ext = os.path.splitext(filename)
                timestamp = int(time.time())
                filename = f"{base}_{timestamp}{ext}"
                input_file_path = os.path.join(input_path, filename)

            # Move file
            shutil.move(wav_path, input_file_path)

            return (
                gr.File(value=input_file_path),  # Downloaded file
                f"successfully downloaded: {filename}",  # Message
                gr.File(value=input_file_path),  # input_audio update
                gr.Audio(value=input_file_path)  # audio for original_audio
            )

        return None, "Failed to download file", None, None

    except Exception as e:
        print(f"Download error: {e}")
        return None, str(e), None, None

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

def download_file(url, directory='ckpts'):
    """
    Downloads file from specified URL

    Args:
        url (str): URL of the file to download
        directory (str, optional): Directory to save the downloaded file. Default 'ckpts'

    Returns:
        str: Full path to the downloaded file
    """
    # Create index
    os.makedirs(directory, exist_ok=True)

    # Extract filename from URL
    filename = os.path.basename(url)
    filepath = os.path.join(directory, filename)

    # Download if the file already exists
    if os.path.exists(filepath):
        print(f"{filename} Already exists.")
        return filepath

    try:
        # Download process
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP hata kontrolü

        # Save File
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"{filename} downloaded successfully.")
        return filepath

    except Exception as e:
        print(f"{filename} failed to download: {e}")
        return None

        clear_memory()


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
    """Saves the uploaded file in the specified directory."""
    try:
        # Get file name safely
        if hasattr(uploaded_file, 'name'):
            file_name = os.path.basename(uploaded_file.name)
        else:
            file_name = os.path.basename(str(uploaded_file))

        # Specify input directory
        target_directory = INPUT_DIR if is_input else OUTPUT_DIR

        # Create destination file path
        base, ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime("%H-%M-%S")
        unique_filename = f"{base}_{timestamp}{ext}"
        target_path = os.path.join(target_directory, unique_filename)

        # Save File
        if hasattr(uploaded_file, 'read'):
            # Gradio file object
            with open(target_path, "wb") as f:
                f.write(uploaded_file.read())
        else:
            # If it is already a file path
            shutil.copy(uploaded_file, target_path)

        print(f"{unique_filename} saved successfully: {target_path}")
        return target_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

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
    emoji_prefixes = ['✅ ', '👥 ', '🏛️ ', '🔇 ', '🔉 ', '🎬 ', '🎼 ']
    for prefix in emoji_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]

    return cleaned.strip()

BASE_PATH = '/content/Music-Source-Separation-Training'
INPUT_DIR = os.path.join(BASE_PATH, 'input')
OUTPUT_DIR = '/content/drive/MyDrive/output'

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

def process_audio(input_audio, model, chunk_size, overlap, flac_file, use_tta, pcm_type, extract_instrumental):
    # Create input and output directories
    create_directory(INPUT_DIR)
    create_directory(OUTPUT_DIR)

    # Delete existing files
    clear_directory(INPUT_DIR)

    # Clear model name
    clean_model = extract_model_name(model)
    print(f"Cleaned Model Name: {clean_model}")

    # File control
    if input_audio is None:
        print("File not uploaded")
        return None, None, None, None, None, None, None, None, None

    # Save file
    dest_path = save_uploaded_file(input_audio, is_input=True)

    if not dest_path:
        print("Failed to save file")
        return None, None, None, None, None, None, None, None, None

    # define input_folder and output_folder
    input_folder = INPUT_DIR
    output_folder = "/content/drive/MyDrive/output"

    # Model selection and specify relevant parameters
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

    elif clean_model == 'big beta 5 (by unwa)':
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

    elif clean_model == 'kimmel_unwa_ft (by unwa)':
            model_type = 'mel_band_roformer'
            config_path = 'ckpts/config_kimmel_unwa_ft.yaml'
            start_check_point = 'ckpts/kimmel_unwa_ft.ckpt'
            download_file('https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt')
            download_file('https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml')
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


    # Other model options will be added here...
    # (All the elif blocks you gave in the previous code will go here)


    else:
        print(f"Unsupported model: {clean_model}")
        return None, None, None, None, None, None, None, None, None


    # Prepare command parameters
    cmd_parts = [
        "python", "inference.py",
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--input_folder", input_folder,
        "--store_dir", output_folder
    ]

    # Add optional parameters
    if extract_instrumental:
        cmd_parts.append("--extract_instrumental")

    # Check if the flac_file option is on
    if flac_file:
        cmd_parts.append("--flac_file")

        # Add the selected pcm_type when flac_file is open
        if pcm_type and pcm_type.strip() in ["PCM_16", "PCM_24"]:
            cmd_parts.extend(["--pcm_type", pcm_type])  # value from drop-down list
        else:
            print(f"Invalid pcm_type: {pcm_type}. Defaulting to 'PCM_24'.")
            cmd_parts.extend(["--pcm_type", "PCM_24"])  # default value
    else:
        # Add default pcm_type when flac_file is off
        cmd_parts.extend(["--pcm_type", "PCM_24"])  # default value

    if use_tta:
        cmd_parts.append("--use_tta")

     # Run the command
    try:
        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Read and print outputs
        for line in process.stdout:
            print(line.strip())

        for line in process.stderr:
            print(line.strip())

        process.wait()

        # Find output files
        output_files = os.listdir(output_folder)

        # Find specific stem files
        vocal_file = next((os.path.join(output_folder, f) for f in output_files if 'vocals' in f.lower()), None)
        instrumental_file = next((os.path.join(output_folder, f) for f in output_files if 'instrumental' in f.lower()), None)
        drum_file = next((os.path.join(output_folder, f) for f in output_files if 'drum' in f.lower()), None)
        bass_file = next((os.path.join(output_folder, f) for f in output_files if 'bass' in f.lower()), None)
        other_file = next((os.path.join(output_folder, f) for f in output_files if 'other' in f.lower()), None)
        effects_file = next((os.path.join(output_folder, f) for f in output_files if 'effects' in f.lower()), None)
        speech_file = next((os.path.join(output_folder, f) for f in output_files if 'speech' in f.lower()), None)
        music_file = next((os.path.join(output_folder, f) for f in output_files if 'music' in f.lower()), None)
        dry_file = next((os.path.join(output_folder, f) for f in output_files if 'dry' in f.lower()), None)

        # Return found files
        return (
            vocal_file,
            instrumental_file,
            drum_file,
            bass_file,
            other_file,
            effects_file,
            speech_file,
            music_file,
            dry_file
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None, None

def create_interface():
    # Let's define the model options in advance
    model_choices = {
        "New Additions": [
            'big beta 5 (by unwa)'

        ],
        "Vocal Separation": [
            'VOCALS-BS-Roformer_1297 (by viperx)',
            '✅ VOCALS-Mel-Roformer big beta 4 (by unwa) - Melspectrogram based high performance',
            '✅ VOCALS-BS-RoformerLargev1 (by unwa) - Comprehensive model',
            'VOCALS-InstVocHQ - General purpose model',
            'VOCALS-MelBand-Roformer (by KimberleyJSN) - Alternative model',
            'VOCALS-VitLarge23 (by ZFTurbo) - Transformer-based model'
        ],
        "Instrumental Separation": [
            'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa) - Latest version instrumental separation',
            'INST-VOC-Mel-Roformer a.k.a. duality (by unwa) - Previous version',
            'INST-Separator MDX23C (by aufr33) - Alternative instrumental separation',
            '✅ INST-Mel-Roformer v2 (by unwa) - Most recent instrumental separation model',
            '✅ inst_v1e (by unwa)',
            '✅ INST-Mel-Roformer v1 (by unwa) - Old instrumental separation model',
            'kimmel_unwa_ft (by unwa)'
        ],
        "Karaoke & Accompaniment": [
            '✅ KARAOKE-MelBand-Roformer (by aufr33 & viperx) - Advanced karaoke separation'
        ],
        "Noise & Effect Removal": [
            '👥 CROWD-REMOVAL-MelBand-Roformer (by aufr33) - Crowd noise removal',
            '🏛️ DE-REVERB-MDX23C (by aufr33 & jarredou) - Reverb reduction',
            '🔇 DENOISE-MelBand-Roformer-1 (by aufr33) - Basic noise reduction',
            '🔉 DENOISE-MelBand-Roformer-2 (by aufr33) - Advanced noise reduction'
        ],
        "Drum Separation": [
            '✅ DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou) - Detailed drum separation'
        ],
        "Multi-Stem & Other Models": [
            '🎬 4STEMS-SCNet_MUSDB18 (by starrytong) - Multi-stem separation',
            '🎼 CINEMATIC-BandIt_Plus (by kwatcharasupat) - Cinematic music analysis',
            'OTHER-BS-Roformer_1053 (by viperx) - Other special models',
            'bleed_suppressor_v1 (by unwa) - dont use it if you dont know what youre doing'
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
            print(f"Audio file listing error: {e}")
            return []

    with gr.Blocks() as demo:
        gr.Markdown("# 🎵 Music Source Separation Tool")

        with gr.Tabs():
            with gr.Tab("Audio Separation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_audio = gr.File(label="Select Audio File", type="filepath")

                        model_category = gr.Dropdown(
                            label="Model Category",
                            choices=list(model_choices.keys())
                        )

                        model_dropdown = gr.Dropdown(label="Select Model")

                        overlap = gr.Slider(
                            label="Overlap",
                            info="It's usually between 5 and 2. Change it if you want something different.",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=4
                        )

                    model_category.change(
                        fn=update_models,
                        inputs=model_category,
                        outputs=model_dropdown
                    )

                    with gr.Column(scale=1):
                        chunk_size = gr.Dropdown(
                            label="Chunk Size",
                            info="don't touch this.",
                            choices=[352800, 485100],
                            value=352800
                        )

                        flac_file = gr.Checkbox(
                            label="FLAC File",
                            info="Open it if you want flac output (beta)"
                        )

                        use_tta = gr.Checkbox(
                            label="Use TTA",
                            info="Test Time Augmentation:It improves the prediction performance of the model. It also increases the processing time."
                        )

                        extract_instrumental = gr.Checkbox(
                            label="Extract Instrumental",
                            info="If you turn it off, it will give 1 of vocal or instrumental.",
                            value=False
                        )

                        pcm_type = gr.Dropdown(
                            label="PCM Type",
                            choices=[
                                'PCM_16',
                                'PCM_24'
                            ]
                        )

                        process_btn = gr.Button("Process Audio")

                        with gr.Column():
                            original_audio = gr.Audio(label="Original Audio")
                            vocals_audio = gr.Audio(label="Vocals")
                            instrumental_audio = gr.Audio(label="Instrumental")
                            drum_audio = gr.Audio(label="Drum")
                            bass_audio = gr.Audio(label="Bass")
                            other_audio = gr.Audio(label="Other")
                            effects_audio = gr.Audio(label="Effects")
                            speech_audio = gr.Audio(label="Speech")
                            music_audio = gr.Audio(label="Music")
                            dry_audio = gr.Audio(label="Dry")

                input_audio.upload(
                    fn=lambda x: x,
                    inputs=input_audio,
                    outputs=original_audio
                )

                process_btn.click(
                    fn=process_audio,
                    inputs=[
                        input_audio, model_dropdown, chunk_size, overlap,
                        flac_file, use_tta, pcm_type, extract_instrumental
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

            
            with gr.Tab("Audio, File Download"):
                gr.Markdown("## 🔗 Audio File Download")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📂 Google Drive Download")
                        drive_url_input = gr.Textbox(label="Google Drive Link")
                        drive_download_btn = gr.Button("Download")
                        drive_download_status = gr.Textbox(label="Status")
                        drive_download_output = gr.File(label="Downloaded File")

                    with gr.Column():
                        gr.Markdown("### 🌐 Direct URL Download")
                        direct_url_input = gr.Textbox(label="Direct URL")
                        direct_download_btn = gr.Button("Download")
                        direct_download_status = gr.Textbox(label="Status")
                        direct_download_output = gr.File(label="Downloaded File")

                drive_download_btn.click(
                    fn=download_callback,
                    inputs=[drive_url_input, gr.State('drive')],
                    outputs=[drive_download_output, drive_download_status, input_audio, original_audio]
                )

                direct_download_btn.click(
                    fn=download_callback,
                    inputs=[direct_url_input, gr.State('direct')],
                    outputs=[direct_download_output, direct_download_status, input_audio, original_audio]
                )

            
            with gr.Tab("Audio Ensemble"):
                gr.Markdown("# 🎵 Audio Ensemble Tool")
            
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
                        drive_audio_files = refresh_audio_files('/content/drive/MyDrive/output')
                        
                        for i in range(10):
                            file_dropdown = gr.Dropdown(
                                label=f"Audio File {i+1}",
                                choices=['None'] + drive_audio_files,
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
                            
                            files = [
                                os.path.join('/content/drive/MyDrive/output', f)
                                for f in file_dropdowns 
                                if f != 'None'
                            ]
                            
                            if len(files) < 2:
                                return None, "Select at least 2 files for ensemble"
                            
                            
                            if weights_input and weights_input.strip():
                                weights = [float(w.strip()) for w in weights_input.split(',')]
                                if len(weights) != len(files):
                                    return None, "Weights must match number of selected files"
                            else:
                                weights = None
                            
                            
                            output_path = "/tmp/ensembled_audio.wav"
                            
                            
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
