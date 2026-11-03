#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python3 scripts/setup_stage3.py
docker compose config --quiet
if [ "${1:-}" = --build ]; then
  docker compose build api
  docker compose build worker
fi
docker compose up -d
printf '%s\n' '网页：http://127.0.0.1:8080；首次启动请等待模型工作端加载。'
printf '%s\n' '查看进度：docker compose logs -f worker'
