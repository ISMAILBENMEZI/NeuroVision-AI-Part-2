#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p mlartifacts

uv run mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --allowed-hosts "localhost:*,127.0.0.1:*,rid-annotated-configuration-jpg.trycloudflare.com" \
    --cors-allowed-origins "https://rid-annotated-configuration-jpg.trycloudflare.com" \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts
