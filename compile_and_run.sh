#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCALE_DIR="$ROOT_DIR/scale/scale-letkf/scale"
TEST_DIR="$ROOT_DIR/test/SC23"

cd "$SCALE_DIR"
make

cd "$TEST_DIR"
./exec_timed.sh

cd "$ROOT_DIR"
