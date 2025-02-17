# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from datetime import datetime

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint, load_lora_weights

import warnings
warnings.filterwarnings("ignore")


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

def get_soundfile_subtype(pcm_type, is_float=False):
    """
    PCM türüne göre uygun soundfile subtypei belirle
    
    Args:
        pcm_type (str): PCM türü ('PCM_16', 'PCM_24', 'FLOAT')
        is_float (bool): Float formatı kullanılıp kullanılmayacağı
    
    Returns:
        str: Soundfile subtype
    """
    if is_float:
        return 'FLOAT'
    
    subtype_map = {
        'PCM_16': 'PCM_16',
        'PCM_24': 'PCM_24',
        'FLOAT': 'FLOAT'
    }
    return subtype_map.get(pcm_type, 'FLOAT')

def run_folder(model, args, config, device, verbose: bool = False):
    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Dosya sayısını ve progress için değişkenler
    total_files = len(mixture_paths)
    current_file = 0

    # Progress tracking
    for path in mixture_paths:
        try:
            # Dosya işleme başlangıcı
            current_file += 1
            print(f"Processing file {current_file}/{total_files}")
            
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {path}')
            print(f'Error message: {str(e)}')
            continue

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            # Dosya formatı ve PCM türü belirleme
            is_float = getattr(args, 'export_format', '').startswith('wav FLOAT')
            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            
            # Subtype belirleme
            if codec == 'flac':
                subtype = get_soundfile_subtype(args.pcm_type, is_float)
            else:
                subtype = get_soundfile_subtype('FLOAT', is_float)

            shortened_filename = shorten_filename(os.path.basename(path))
            output_filename = f"{shortened_filename}_{instr}.{codec}"
            output_path = os.path.join(args.store_dir, output_filename)
        
            sf.write(output_path, estimates.T, sr, subtype=subtype)

        # Progress yüzdesi hesaplama
        progress_percent = int((current_file / total_files) * 100)
        print(f"Progress: {progress_percent}%")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', 
                        help="Model type (bandit, bs_roformer, mdx23c, etc.)")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', 
                        help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="Folder with mixtures to process")
    parser.add_argument("--audio_path", type=str, help="Path to a single audio file to process")  # Yeni argüman
    parser.add_argument("--store_dir", default="", type=str, help="Path to store results")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, 
                        help='List of GPU IDs')
    parser.add_argument("--extract_instrumental", action='store_true', 
                        help="Invert vocals to get instrumental if provided")
    parser.add_argument("--force_cpu", action='store_true', 
                        help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', 
                        help="Output flac file instead of wav")
    parser.add_argument("--export_format", type=str, 
                        choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'], 
                        default='flac PCM_24', 
                        help="Export format and PCM type")
    parser.add_argument("--pcm_type", type=str, 
                        choices=['PCM_16', 'PCM_24'], 
                        default='PCM_24', 
                        help="PCM type for FLAC files")
    parser.add_argument("--use_tta", action='store_true', 
                        help="Enable test time augmentation")
    parser.add_argument("--lora_checkpoint", type=str, default='', 
                        help="Initial checkpoint to LoRA weights")

    # Argümanları ayrıştır
    parsed_args = parser.parse_args(args)

    # Burada parsed_args.audio_path ile ses dosyası yolunu kullanabilirsiniz
    print(f"Audio path provided: {parsed_args.audio_path}")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Cihaz seçimi
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if type(args.device_ids) == list else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
         device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # Çoklu CUDA GPU kullanımı
    if type(args.device_ids) == list and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
