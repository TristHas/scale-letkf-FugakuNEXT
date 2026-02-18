#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCALE_DIR="$ROOT_DIR/scale/scale-letkf/scale"
TEST_DIR="$ROOT_DIR/test/SC23"

cd "$ROOT_DIR/scale/scale-rm/src"
make -j

cd "$SCALE_DIR"
make

cd "$ROOT_DIR"
cp scale/scale-letkf/scale/letkf/letkf test/SC23/bin/letkf

cd "$TEST_DIR"
./exec_timed.sh


cd "$ROOT_DIR"
