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

# ── Skill-library helpers ───────────────────────────────────────

VALID_SKILLS=(standup walk dribble shoot)

is_valid_skill() {
    local s="$1"
    for v in "${VALID_SKILLS[@]}"; do
        [ "$v" = "$s" ] && return 0
    done
    return 1
}

# ── Commands ─────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        log "Setting up RoboCup Humanoid Soccer RL project..."
        install_deps
        setup_robot_assets
        generate_field
        mkdir -p checkpoints logs videos
        log "Setup complete!"
        log "Next: train each skill, then the orchestrator."
        log "  ./scripts/run.sh train-skill walk"
        log "  ./scripts/run.sh train-skill standup"
        log "  ./scripts/run.sh train-skill dribble"
        log "  ./scripts/run.sh train-skill shoot"
        log "  ./scripts/run.sh train-orchestrator"
        ;;
    train-skill)
        shift
        activate_or_die
        if [ $# -lt 1 ]; then
            err "Usage: ./scripts/run.sh train-skill <walk|standup|dribble|shoot> [extra-args...]"
        fi
        SKILL="$1"; shift
        if ! is_valid_skill "$SKILL"; then
            err "Unknown skill '$SKILL'. Valid: ${VALID_SKILLS[*]}"
        fi
        log "Training skill: $SKILL"
        python3 -m training.train_skill --skill "$SKILL" "$@"
        ;;
    train-skills-all)
        # Train every skill sequentially in its OWN python process — so
        # Genesis releases its GPU allocation between skills and we
        # don't hit the cross-stage OOM that drove this refactor.
        shift
        activate_or_die
        for SKILL in "${VALID_SKILLS[@]}"; do
            log "── Training skill: $SKILL ──"
            python3 -m training.train_skill --skill "$SKILL" "$@" || \
                err "Skill $SKILL training failed; halting sequence."
        done
        log "All skills trained."
        ;;
    train-orchestrator)
        shift
        activate_or_die
        log "Starting Phase-2 orchestrator training (4v4 self-play)..."
        python3 -m training.train_orchestrator "$@"
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
        cat <<EOF
Usage: $0 <command> [args...]

Setup:
  setup                     Create venv, install deps, fetch K1 assets, generate field.

Skill-library training (Phase 1 — each skill in its own process):
  train-skill <name>        Train a single skill: walk | standup | dribble | shoot.
  train-skills-all          Train every skill sequentially.

Orchestrator (Phase 2 — discrete-skill controller, 4v4 self-play):
  train-orchestrator        Train the orchestrator over frozen skill checkpoints.

Evaluation & utilities:
  eval <ckpt>               Evaluate a checkpoint in MuJoCo.
  generate-field            Regenerate field models from configs/field_hsl_2026.json.
  debug-genesis             Genesis spawn debugger (--render --screenshot=PATH).
  debug-mujoco              MuJoCo scene debugger (--screenshot=PATH).
  make-ball-texture         Re-render the soccer ball texture.
  shell                     Open a shell inside the project's venv.

Training options (forwarded to train-skill / train-orchestrator):
  --algorithm ppo|flashsac  Select RL algorithm (skills only; default ppo).
  --vec-num-envs N          Parallel Genesis env count.
  --device {auto,cpu,gpu}   Hardware preset. 'auto' detects CUDA.
  --total-timesteps N       Override default timestep budget.
  --resume PATH             Resume from checkpoint (full state load).
  --init-from PATH          PPO-only: warm-start policy weights from another ckpt.
  --wandb                   Also log to Weights & Biases (TensorBoard always on).

Examples:
  ./scripts/run.sh train-skill walk --device gpu --vec-num-envs 1024 --wandb
  ./scripts/run.sh train-skill dribble --init-from checkpoints/skill_walk/skill_walk_step50000000.pt
  ./scripts/run.sh train-orchestrator --num-envs 256 --wandb
EOF
        exit 1
        ;;
esac
