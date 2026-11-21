#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "$project_dir" \
  "$project_dir/mujoco_simulator_python/mujoco_simulator_python.py" \
  --config b2w_piper.yaml \
  --port "${MUJOCO_SIM_PORT:-8085}" \
  "$@"
