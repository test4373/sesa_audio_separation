import os
import re
import validators
import yt_dlp
import torch
import gdown
from urllib.parse import quote
from helpers import INPUT_DIR, COOKIE_PATH, clear_directory, clear_temp_folder, BASE_DIR
import yaml

# BASE_DIR ve ilgili dizinler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")

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

def download_callback(url, download_type='direct', cookie_file=None):
    clear_temp_folder("/tmp", exclude_items=["gradio", "config.json"])
    clear_directory(INPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)

    if not validators.url(url):
        return None, "❌ Invalid URL", None, None, None, None

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

    if download_type == 'drive':
        try:
            file_id = re.search(r'/d/([^/]+)', url).group(1) if '/d/' in url else url.split('id=')[-1]
            output_path = os.path.join(INPUT_DIR, "drive_download.wav")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                wav_path = output_path
                download_success = True
            else:
                raise Exception("File size zero or file not created")
        except Exception as e:
            error_msg = f"❌ Google Drive download error: {str(e)}"
            print(error_msg)
            return None, error_msg, None, None, None, None
    else:
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
                else:
                    raise Exception("WAV conversion failed")
        except Exception as e:
            error_msg = f"❌ Download error: {str(e)}"
            print(error_msg)
            return None, error_msg, None, None, None, None

    if download_success and wav_path:
        for f in os.listdir(INPUT_DIR):
            if f != os.path.basename(wav_path):
                os.remove(os.path.join(INPUT_DIR, f))
        return (
            wav_path,
            "🎉 Downloaded successfully!",
            wav_path,
            wav_path,
            wav_path,
            wav_path
        )
    return None, "❌ Download failed", None, None, None, None

def download_file(url):
    """Downloads a file from a URL."""
    encoded_url = quote(url, safe=':/')
    path = os.path.join(BASE_DIR, 'ckpts')
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return
    try:
        torch.hub.download_url_to_file(encoded_url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

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
