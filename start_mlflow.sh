#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p mlartifacts

uv run mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --allowed-hosts "*.trycloudflare.com,localhost" \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts
