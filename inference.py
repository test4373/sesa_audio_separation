# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from typing import Dict, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio, draw_spectrogram
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings
warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose: bool = False):
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
        print(f"Processing track {current_file}/{total_files}: {path}")
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
        output_dir = os.path.join(args.store_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = os.path.join(output_dir, f"{instr}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)

        # Yüzde hesaplama
        percentage = (current_file / total_files) * 100
        print(f"Progress: {percentage:.2f}% - Elapsed time: {time.time() - start_time:.2f} seconds.")

    print(f"Total elapsed time: {time.time() - start_time:.2f} seconds.")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--demud_phaseremix_inst", action='store_true',
                        help="Enable Demud phase remix for instrumental separation")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                             " While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def proc_folder(dict_args):
    args = parse_args(dict_args)
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

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
