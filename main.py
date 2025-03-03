import os
import threading
import urllib.request
import time
import sys
import random
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
import numpy as np
import librosa
import shutil
from gui import create_interface
from pyngrok import ngrok

import warnings
warnings.filterwarnings("ignore")

def generate_random_port():
    """Generates a random port between 1000 and 9000."""
    return random.randint(1000, 9000)

def start_gradio(port, share=False):
    """Starts the Gradio interface with optional sharing."""
    demo = create_interface()
    demo.launch(
        server_port=port,
        server_name='0.0.0.0',
        share=share,
        allowed_paths=[os.path.join(os.path.expanduser("~"), "Music-Source-Separation", "input"), "/tmp", "/content"],
        inline=False
    )

def start_localtunnel(port):
    """Starts the Gradio interface with localtunnel sharing."""
    print(f"Starting Localtunnel on port {port}...")
    os.system('npm install -g localtunnel &>/dev/null')
    
    with open('url.txt', 'w') as file:
        file.write('')
    os.system(f'lt --port {port} >> url.txt 2>&1 &')
    time.sleep(2)
    
    endpoint_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
    with open('url.txt', 'r') as file:
        tunnel_url = file.read().replace("your url is: ", "").strip()

    print(f"Share Link: {tunnel_url}")
    print(f"Password IP: {endpoint_ip}")
    
    start_gradio(port, share=False)

def start_ngrok(port, ngrok_token):
    """Starts the Gradio interface with ngrok sharing."""
    print(f"Starting Ngrok on port {port}...")
    try:
        ngrok.set_auth_token(ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(port)
        print(f"Ngrok URL: {tunnel.public_url}")
        
        start_gradio(port, share=False)
    except Exception as e:
        print(f"Error starting ngrok: {e}")
        sys.exit(1)

def main(method="gradio", port=None, ngrok_token=""):
    """Main entry point for the application."""
    # Portu otomatik belirle veya kullanıcıdan geleni kullan
    port = port or generate_random_port()
    print(f"Selected port: {port}")

    # Paylaşım yöntemine göre işlem yap
    if method == "gradio":
        print("Starting Gradio with built-in sharing...")
        start_gradio(port, share=True)
    elif method == "localtunnel":
        start_localtunnel(port)
    elif method == "ngrok":
        if not ngrok_token:
            print("Error: Ngrok token is required for ngrok method!")
            sys.exit(1)
        start_ngrok(port, ngrok_token)
    else:
        print("Error: Invalid method! Use 'gradio', 'localtunnel', or 'ngrok'.")
        sys.exit(1)

    # Sürekli çalışır durumda tut (gerekirse)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Process stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Music Source Separation Web UI")
    parser.add_argument("--method", type=str, default="gradio", choices=["gradio", "localtunnel", "ngrok"], help="Sharing method (default: gradio)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: random between 1000-9000)")
    parser.add_argument("--ngrok-token", type=str, default="", help="Ngrok authentication token (required for ngrok)")
    args = parser.parse_args()
    
    main(method=args.method, port=args.port, ngrok_token=args.ngrok_token)
