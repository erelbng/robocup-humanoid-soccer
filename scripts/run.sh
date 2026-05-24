#!/usr/bin/env bash
# ╔═══════════════════════════════════════════════════════════════════╗
# ║  RoboCup Humanoid Soccer RL - Setup & Run Script                 ║
# ╚═══════════════════════════════════════════════════════════════════╝
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

# ── Colors ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${GREEN}[ROBOCUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Venv management ──────────────────────────────────────────────
ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment at .venv ..."
        python3 -m venv "$VENV_DIR"
        log "Virtual environment created."
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    log "Activated venv: $(python3 --version) at $(which python3)"
}

install_deps() {
    ensure_venv
    log "Installing project in editable mode ..."
    pip install --upgrade pip --quiet
    pip install -e "." --quiet
    log "Core dependencies installed."

    # Genesis is optional (needs GPU + specific platform)
    if pip install -e ".[genesis]" --quiet 2>/dev/null; then
        log "Genesis simulator installed."
    else
        warn "Genesis install skipped (may need GPU / specific platform)."
    fi

    log "All dependencies ready."
}

# ── Robot assets ─────────────────────────────────────────────────
setup_robot_assets() {
    log "Setting up Booster K1 robot assets..."
    mkdir -p models/robot
    if [ ! -d "models/robot/K1" ]; then
        if command -v git &> /dev/null; then
            git clone --depth 1 https://github.com/BoosterRobotics/booster_assets.git /tmp/booster_assets 2>/dev/null || true
            if [ -d "/tmp/booster_assets/robots/K1" ]; then
                cp -r /tmp/booster_assets/robots/K1 models/robot/K1
                log "K1 robot assets copied to models/robot/K1"
            else
                warn "Could not clone booster_assets. Creating placeholder."
                mkdir -p models/robot/K1
                echo "<!-- Placeholder: download K1_22dof.urdf from https://github.com/BoosterRobotics/booster_assets/tree/main/robots/K1 -->" > models/robot/K1/K1_22dof.urdf
            fi
            rm -rf /tmp/booster_assets
        else
            warn "git not available. Please manually download K1 robot model."
            mkdir -p models/robot/K1
        fi
    else
        log "K1 robot assets already present."
    fi
}

# ── Field generation ─────────────────────────────────────────────
generate_field() {
    ensure_venv
    log "Generating field models..."
    mkdir -p models/field
    python3 models/field_generator.py configs/field_hsl_2026.json -o models/field
    log "Field models generated in models/field/"
}

# ── Activate venv for run commands ───────────────────────────────
activate_or_die() {
    if [ ! -d "$VENV_DIR" ]; then
        err "No .venv found. Run './scripts/run.sh setup' first."
    fi
    source "$VENV_DIR/bin/activate"
}

# ── Commands ─────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        log "Setting up RoboCup Humanoid Soccer RL project..."
        install_deps
        setup_robot_assets
        generate_field
        mkdir -p checkpoints logs videos
        log "Setup complete! Run: ./scripts/run.sh train-phase1"
        ;;
    train)
        shift
        activate_or_die
        log "Starting training..."
        python3 -m training.train "$@"
        ;;
    train-phase1)
        shift
        activate_or_die
        log "Starting Phase 1 training (single-robot skills)..."
        python3 -m training.train --phase 1 "$@"
        ;;
    train-phase2)
        shift
        activate_or_die
        log "Starting Phase 2 training (match)..."
        python3 -m training.train --phase 2 "$@"
        ;;
    eval|evaluate)
        shift
        activate_or_die
        if [ $# -lt 1 ]; then
            err "Usage: ./scripts/run.sh eval <checkpoint_path> [--phase phase1|phase2]"
        fi
        log "Running MuJoCo evaluation..."
        python3 -m evaluation.evaluate "$@"
        ;;
    generate-field)
        generate_field
        ;;
    debug-genesis)
        shift
        activate_or_die
        log "Running Genesis spawn debugger..."
        python3 -m scripts.debug_genesis_spawn "$@"
        ;;
    debug-mujoco)
        shift
        activate_or_die
        log "Running MuJoCo scene debugger..."
        python3 -m scripts.debug_mujoco_scene "$@"
        ;;
    make-ball-texture)
        shift
        activate_or_die
        python3 -m models.textures.make_ball_texture "$@"
        ;;
    shell)
        activate_or_die
        log "Dropping into venv shell. Type 'exit' to leave."
        exec "$SHELL"
        ;;
    help|*)
        echo "Usage: $0 {setup|train|train-phase1|train-phase2|eval|generate-field|debug-genesis|debug-mujoco|make-ball-texture|shell}"
        echo ""
        echo "Commands:"
        echo "  setup             - Create venv, install deps, download robot assets, generate field"
        echo "  train             - Run full training pipeline (Phase 1 + 2)"
        echo "  train-phase1      - Train single-robot skills only"
        echo "  train-phase2      - Train multi-robot match (needs Phase 1 checkpoint)"
        echo "  eval              - Evaluate checkpoint in MuJoCo"
        echo "  generate-field    - Regenerate field models from JSON"
        echo "  debug-genesis     - Spawn debugger for Genesis (--render --screenshot=PATH)"
        echo "  debug-mujoco      - Spawn debugger for MuJoCo (--screenshot=PATH)"
        echo "  make-ball-texture - Re-render the soccer ball texture"
        echo "  shell             - Open a shell with the venv activated"
        exit 1
        ;;
esac
