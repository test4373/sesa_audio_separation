#!/bin/bash

# Rastgele bir port belirle (1000-9000 arası)
PORT=$((1000 + RANDOM % 8001))

# Kullanıcıdan paylaşım yöntemini seçmesini iste
echo "Available sharing methods: gradio, localtunnel, ngrok"
read -p "Choose a sharing method (default: gradio): " METHOD
METHOD=${METHOD:-gradio}

# Ngrok token'ını iste (yalnızca ngrok seçildiyse)
NGROK_TOKEN=""
if [ "$METHOD" = "ngrok" ]; then
    read -p "Enter your Ngrok token (get it from ngrok.com): " NGROK_TOKEN
    if [ -z "$NGROK_TOKEN" ]; then
        echo "Ngrok token is required for ngrok method!"
        exit 1
    fi
fi

# Komutu çalıştır
case $METHOD in
    "gradio")
        python3 main.py --method gradio --port $PORT
        ;;
    "localtunnel")
        python3 main.py --method localtunnel --port $PORT
        ;;
    "ngrok")
        python3 main.py --method ngrok --port $PORT --ngrok-token "$NGROK_TOKEN"
        ;;
    *)
        echo "Invalid method! Use gradio, localtunnel, or ngrok."
        ;;
esac

# Terminali açık tut (isteğe bağlı)
read -p "Press Enter to exit..."
