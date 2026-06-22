#!/usr/bin/env bash
# Master script: run examples 07→10 sequentially, generate GIFs.
# Each run gets its own timestamped output dir.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

run_example() {
    local dir="$1"
    local name="$(basename $dir)"
    echo "======================================================================"
    echo "[$name] Starting at $(date)"
    echo "======================================================================"
    cd "$dir"

    # Generate observation data
    python data/gen_obs.py

    # Launch training
    falcon launch -o output/run

    # Generate GIF
    python make_gif.py output/run && echo "[$name] GIF done."

    echo "[$name] Finished at $(date)"
    cd "$SCRIPT_DIR"
}

run_example "$SCRIPT_DIR/07_letters"
run_example "$SCRIPT_DIR/08_letters_noise"
run_example "$SCRIPT_DIR/09_two_words"
run_example "$SCRIPT_DIR/10_scene"

echo "======================================================================"
echo "All examples complete at $(date)"
echo "======================================================================"
