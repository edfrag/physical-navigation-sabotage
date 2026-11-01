#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t aopp-turtlebot4 -f "$project_dir/Dockerfile" "$project_dir"
