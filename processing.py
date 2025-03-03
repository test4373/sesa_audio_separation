import os
import glob
import subprocess
import time
import gc
import shutil
from datetime import datetime
from helpers import INPUT_DIR, OUTPUT_DIR, OLD_OUTPUT_DIR, ENSEMBLE_DIR, AUTO_ENSEMBLE_TEMP, AUTO_ENSEMBLE_OUTPUT, move_old_files
from models import get_model_config

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

def process_audio(input_audio_file, model, chunk_size, overlap, export_format, use_tta, demud_phaseremix_inst, extract_instrumental, clean_model):
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

    clean_model = extract_model_name(model)
    print(f"Processing audio from: {audio_path} using model: {clean_model}")

    model_type, config_path, start_check_point = get_model_config(clean_model, chunk_size, overlap)

    cmd = [
        "python", "inference.py",
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", start_check_point,
        "--input_folder", INPUT_DIR,
        "--store_dir", OUTPUT_DIR,
    ]
    if use_tta:
        cmd.append("--use_tta")
    if extract_instrumental:
        cmd.append("--extract_instrumental")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return [None] * 14
    except Exception as e:
        print(f"Critical error with {model}: {str(e)}")
        return [None] * 14

    outputs = []
    for stem in ["vocals", "instrumental", "drums", "bass", "other", "effects", "speech", "bleed", "music", "karaoke", "phaseremix", "dry", "male", "female"]:
        file = glob.glob(os.path.join(OUTPUT_DIR, f"*{stem}*.wav"))
        outputs.append(file[0] if file else None)
    return outputs

def ensemble_audio_fn(files, method, weights):
    """Performs audio ensemble processing."""
    try:
        if len(files) < 2:
            return None, "⚠️ Minimum 2 files required"
        valid_files = [f for f in files if os.path.exists(f)]
        if len(valid_files) < 2:
            return None, "❌ Valid files not found"
        
        output_dir = os.path.join(BASE_PATH, "ensembles")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"ensemble_{timestamp}.wav")

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

def auto_ensemble_process(input_audio_file, selected_models, chunk_size, overlap, export_format, use_tta, extract_instrumental, ensemble_type, _state):
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

        os.makedirs(AUTO_ENSEMBLE_TEMP, exist_ok=True)
        os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)
        clear_directory(AUTO_ENSEMBLE_TEMP)

        all_outputs = []
        for model in selected_models:
            clean_model = extract_model_name(model)
            model_output_dir = os.path.join(AUTO_ENSEMBLE_TEMP, clean_model)
            os.makedirs(model_output_dir, exist_ok=True)

            model_type, config_path, start_check_point = get_model_config(clean_model, chunk_size, overlap)

            cmd = [
                "python", "inference.py",
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
        shutil.rmtree(AUTO_ENSEMBLE_TEMP, ignore_errors=True)
        gc.collect()
