
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings
warnings.filterwarnings("ignore")


def shorten_filename(filename, max_length=30):
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    shortened = base[:15] + "..." + base[-10:]
    return shortened + ext

def get_soundfile_subtype(pcm_type, is_float=False):
    if is_float:
        return 'FLOAT'
    subtype_map = {
        'PCM_16': 'PCM_16',
        'PCM_24': 'PCM_24',
        'FLOAT': 'FLOAT'
    }
    return subtype_map.get(pcm_type, 'FLOAT')

def run_folder(model, args, config, device, ckpt_name, verbose: bool = False):
    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    total_files = len(mixture_paths)
    current_file = 0

    for path in mixture_paths:
        current_file += 1
        try:
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

        if args.demud_phaseremix_inst:
            print(f"Applying Demud phase remix (instrumental) to: {path}")
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            instruments.append('instrumental_phaseremix')
            
            if 'instrumental' not in instruments and 'Instrumental' not in instruments:
                mix_modified = mix_orig - 2 * waveforms_orig[instr]
                waveforms_modified = demix(config, model, mix_modified, device, model_type=args.model_type)
                waveforms_orig['instrumental_phaseremix'] = mix_orig + waveforms_modified[instr]
            else:
                mix_modified = 2 * waveforms_orig[instr] - mix_orig
                waveforms_modified = demix(config, model, mix_modified, device, model_type=args.model_type)
                waveforms_orig['instrumental_phaseremix'] = mix_orig + mix_modified - waveforms_modified[instr]

        file_name = os.path.splitext(os.path.basename(path))[0]
        
        for instr in instruments:
              estimates = waveforms_orig[instr]
              if 'normalize' in config.inference:
                  if config.inference['normalize'] is True:
                      estimates = denormalize_audio(estimates, norm_params)

              short_name = shorten_filename(file_name)
              output_filename = f"{short_name}_{ckpt_name}{instr}"
    
              codec = 'flac' if args.flac_file else 'wav'
              output_path = os.path.join(args.store_dir, f"{output_filename}.{codec}")
    
              is_float = args.export_format.startswith('wav FLOAT')
              subtype = get_soundfile_subtype(args.pcm_type, is_float) if codec == 'flac' else 'FLOAT'

              sf.write(output_path, estimates.T, sr, subtype=subtype)

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
    parser.add_argument("--audio_path", type=str, help="Path to a single audio file to process")
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
    parser.add_argument("--demud_phaseremix_inst", action='store_true',
                        help="Enable Demud phase remix for instrumental separation")

    args = parser.parse_args()

    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
         device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    # Sadece checkpoint adını al
    ckpt_name = ""
    if args.start_check_point:
        ckpt_name = os.path.splitext(os.path.basename(args.start_check_point))[0] + "_"

    run_folder(model, args, config, device, ckpt_name, verbose=True)

if __name__ == "__main__":
    proc_folder(None)
