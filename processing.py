import os
import glob
import subprocess
import time
import gc
import shutil
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from datetime import datetime
from helpers import INPUT_DIR, OLD_OUTPUT_DIR, ENSEMBLE_DIR, AUTO_ENSEMBLE_TEMP, move_old_files, clear_directory, BASE_DIR
from model import get_model_config
import torch
import yaml
import gradio as gr
import threading
import random
import librosa
import soundfile as sf
import numpy as np
import requests
import json
import locale
import re
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
from urllib.parse import urlparse, quote
import gdown
from clean_model import clean_model_name, shorten_filename, clean_filename

import warnings
warnings.filterwarnings("ignore")

# BASE_DIR'i dinamik olarak güncel dizine ayarla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # processing.py'nin bulunduğu dizin
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")  # inference.py'nin tam yolu
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Çıkış dizini BASE_DIR/output olarak güncellendi
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_output")  # Ensemble çıkış dizini

def extract_model_name(full_model_string):
    """Extracts the clean model name from a string."""
    if not full_model_string:
        return ""
    cleaned = str(full_model_string)
    if ' - ' in cleaned:
        cleaned = cleaned.split(' - ')[0]
    emoji_prefixes = ['✅ ', '👥 ', '🗣️ ', '🏛️ ', '🔇 ', '🔉 ', '🎬 ', '🎼 ', '✅(?) ']
    for prefix in emoji_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.strip()

def run_command_and_process_files(model_type, config_path, start_check_point, INPUT_DIR, OUTPUT_DIR, extract_instrumental, use_tta, demud_phaseremix_inst, clean_model):
    try:
        # inference.py'nin tam yolunu kullan
        cmd_parts = [
            "python", INFERENCE_PATH,
            "--model_type", model_type,
            "--config_path", config_path,
            "--start_check_point", start_check_point,
            "--input_folder", INPUT_DIR,
            "--store_dir", OUTPUT_DIR,
        ]
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")
        if use_tta:
            cmd_parts.append("--use_tta")
        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_DIR,  # Çalışma dizini olarak BASE_DIR kullan
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line.strip())
        for line in process.stderr:
            print(line.strip())

        process.wait()

        filename_model = clean_model_name(clean_model)

        def rename_files_with_model(folder, filename_model):
            for filename in sorted(os.listdir(folder)):
                file_path = os.path.join(folder, filename)
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue
                base, ext = os.path.splitext(filename)
                clean_base = base.strip('_- ')
                new_filename = f"{clean_base}_{filename_model}{ext}"
                new_file_path = os.path.join(folder, new_filename)
                os.rename(file_path, new_file_path)

        rename_files_with_model(OUTPUT_DIR, filename_model)

        output_files = os.listdir(OUTPUT_DIR)

        def find_file(keyword):
            matching_files = [
                os.path.join(OUTPUT_DIR, f) for f in output_files 
                if keyword in f.lower()
            ]
            return matching_files[0] if matching_files else None

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
        bleed_file = find_file('bleed')
        karaoke_file = find_file('karaoke')

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
            female_file or None,
            bleed_file or None,
            karaoke_file or None
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        return (None,) * 14

def process_audio(input_audio_file, model, chunk_size, overlap, export_format, use_tta, demud_phaseremix_inst, extract_instrumental, clean_model, *args, **kwargs):
    """Processes audio using the specified model and returns separated stems."""
    if input_audio_file is not None:
        audio_path = input_audio_file.name
    else:
        existing_files = os.listdir(INPUT_DIR)
        if existing_files:
            audio_path = os.path.join(INPUT_DIR, existing_files[0])
        else:
            print("No audio file provided and no existing file in input directory.")
            return [None] * 14

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
    move_old_files(OUTPUT_DIR)

    clean_model_name_full = extract_model_name(model)
    print(f"Processing audio from: {audio_path} using model: {clean_model_name_full}")

    model_type, config_path, start_check_point = get_model_config(clean_model_name_full, chunk_size, overlap)

    outputs = run_command_and_process_files(
        model_type=model_type,
        config_path=config_path,
        start_check_point=start_check_point,
        INPUT_DIR=INPUT_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        extract_instrumental=extract_instrumental,
        use_tta=use_tta,
        demud_phaseremix_inst=demud_phaseremix_inst,
        clean_model=clean_model_name_full
    )

    return outputs

def ensemble_audio_fn(files, method, weights):
    try:
        if len(files) < 2:
            return None, "⚠️ Minimum 2 files required"
        
        valid_files = [f for f in files if os.path.exists(f)]
        if len(valid_files) < 2:
            return None, "❌ Valid files not found"
        
        output_dir = os.path.join(BASE_DIR, "ensembles")  # BASE_DIR üzerinden dinamik
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/ensemble_{timestamp}.wav"
        
        ensemble_args = [
            "--files", *valid_files,
            "--type", method.lower().replace(' ', '_'),
            "--output", output_path
        ]
        
        if weights and weights.strip():
            weights_list = [str(w) for w in map(float, weights.split(','))]
            ensemble_args += ["--weights", *weights_list]
        
        result = subprocess.run(
            ["python", "ensemble.py"] + ensemble_args,
            capture_output=True,
            text=True
        )
        
        log = f"✅ Success!\n{result.stdout}" if not result.stderr else f"❌ Error!\n{result.stderr}"
        return output_path, log

    except Exception as e:
        return None, f"⛔ Critical Error: {str(e)}"

def auto_ensemble_process(input_audio_file, selected_models, chunk_size, overlap, export_format, use_tta, extract_instrumental, ensemble_type, _state, *args, **kwargs):
    """Processes audio with multiple models and performs ensemble."""
    try:
        if not selected_models or len(selected_models) < 1:
            return None, "❌ No models selected"

        if input_audio_file is None:
            existing_files = os.listdir(INPUT_DIR)
            if not existing_files:
                return None, "❌ No input audio provided"
            audio_path = os.path.join(INPUT_DIR, existing_files[0])
        else:
            audio_path = input_audio_file.name

        # AUTO_ENSEMBLE_TEMP'i de BASE_DIR üzerinden tanımla
        auto_ensemble_temp = os.path.join(BASE_DIR, "auto_ensemble_temp")
        os.makedirs(auto_ensemble_temp, exist_ok=True)
        os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)
        clear_directory(auto_ensemble_temp)

        all_outputs = []
        for model in selected_models:
            clean_model = extract_model_name(model)
            model_output_dir = os.path.join(auto_ensemble_temp, clean_model)
            os.makedirs(model_output_dir, exist_ok=True)

            model_type, config_path, start_check_point = get_model_config(clean_model, chunk_size, overlap)

            cmd = [
                "python", INFERENCE_PATH,
                "--model_type", model_type,
                "--config_path", config_path,
                "--start_check_point", start_check_point,
                "--input_folder", INPUT_DIR,
                "--store_dir", model_output_dir,
            ]
            if use_tta:
                cmd.append("--use_tta")
            if extract_instrumental:
                cmd.append("--extract_instrumental")

            print(f"Running command: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(f"Error: {result.stderr}")
                    return None, f"Model {model} failed: {result.stderr}"
            except Exception as e:
                return None, f"Critical error with {model}: {str(e)}"

            model_outputs = glob.glob(os.path.join(model_output_dir, "*.wav"))
            if not model_outputs:
                raise FileNotFoundError(f"{model} failed to produce output")
            all_outputs.extend(model_outputs)

        def wait_for_files(files, timeout=300):
            start = time.time()
            while time.time() - start < timeout:
                missing = [f for f in files if not os.path.exists(f)]
                if not missing:
                    return True
                time.sleep(5)
            raise TimeoutError(f"Missing files: {missing[:3]}...")

        wait_for_files(all_outputs)

        quoted_files = [f'"{f}"' for f in all_outputs]
        timestamp = str(int(time.time()))
        output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"ensemble_{timestamp}.wav")
        
        ensemble_cmd = [
            "python", "ensemble.py",
            "--files", *quoted_files,
            "--type", ensemble_type,
            "--output", f'"{output_path}"'
        ]

        result = subprocess.run(
            " ".join(ensemble_cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        if not os.path.exists(output_path):
            raise RuntimeError("Ensemble dosyası oluşturulamadı")
        
        return output_path, "✅ Success!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
    finally:
        shutil.rmtree(auto_ensemble_temp, ignore_errors=True)
        gc.collect()
