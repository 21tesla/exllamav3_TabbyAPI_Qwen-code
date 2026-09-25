#!/usr/bin/env bash
#
# install.sh -- bring up the whole TabbyAPI + Qwen Code stack on one machine.
#
# This is the executable form of README.md. Where the README describes a step,
# this script performs it, and the two are meant to be read together: README.md
# explains the architecture and the DSML parsing, this file is the checklist.
#
# Steps, in order (run a subset by naming it):
#
#   exllama   build the ExLlamaV3 fork (venv + CUDA extension) into $LLAMA_DIR
#   model     download the DeepSeek-V4-Flash EXL3 pack into $MODELS_DIR
#   serving   write the TabbyAPI config + DeepSeek sampler preset (never overwrites)
#   tabby     pull the image and (re)create the TabbyAPI container
#   proxy     install and start the tabby_proxy user instance, enable lingering
#   verify    end-to-end check: docker -> upstream :5000 -> proxy :8081
#
#   ./install.sh              # every step
#   ./install.sh tabby proxy  # just those two
#
# Everything is idempotent: re-running is the intended way to repair a stack
# that came back wrong after a reboot. Nothing here needs sudo except the
# one-time removal of a legacy system unit, and that is skipped if sudo is not
# available without a password.
#
# Override any path or name by exporting it before running, e.g.
#   MODELS_DIR=/mnt/models ./install.sh

set -Eeuo pipefail

# ---------------------------------------------------------------- configuration

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The fork that actually serves the pack. The published TabbyAPI image bundles
# *upstream* ExLlamaV3, so the host-side scripts (conversion, eval, examples)
# come from here rather than from the container.
LLAMA_DIR="${LLAMA_DIR:-$HOME/software/exllamav3-anemone}"
VENV_DIR="${VENV_DIR:-$LLAMA_DIR/venv}"

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-0731-exl3-2.32bpw}"
MODEL_HF_REPO="${MODEL_HF_REPO:-anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw}"

# Mounted into the container at /app/config. Holds config.yml, api_tokens.yml and
# the sampler_overrides/ copy TabbyAPI resolves relative to its working directory.
CONFIG_DIR="${CONFIG_DIR:-$HOME/software/tabbyapi-config}"

TABBY_IMAGE="${TABBY_IMAGE:-ghcr.io/theroyallab/tabbyapi:cu13}"
TABBY_CONTAINER="${TABBY_CONTAINER:-tabbyapi}"
UPSTREAM_PORT="${UPSTREAM_PORT:-5000}"
PROXY_PORT="${PROXY_PORT:-8081}"

# A single-arch "120a" build breaks the DSA decode graph path, so the main
# extension is built for plain 12.0. The optional FP4 prefill kernel targets
# sm_120a separately, from exllamav3/anemone_fp4/.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

FORCE_REBUILD="${FORCE_REBUILD:-0}"

# ------------------------------------------------------------------- utilities

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# true if sudo is usable without an interactive prompt. Never prompts: an agent
# or a remote shell cannot answer one.
sudo_ready() {
    have sudo && sudo -n true 2>/dev/null
}

# TabbyAPI answers 401 to an unauthenticated request once it is up, and 503
# while it loads a model. Both mean the server is alive; 000 means it is not.
probe() {
    local out
    out="$(curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-5}" "$1" 2>/dev/null)" || true
    printf '%s' "${out:-000}"
}

wait_for() {
    local url="$1" label="$2" tries="${3:-60}" code=000
    for _ in $(seq 1 "$tries"); do
        code="$(probe "$url")"
        case "$code" in 200|401) printf '%s: HTTP %s\n' "$label" "$code"; return 0 ;; esac
        sleep 2
    done
    printf '%s: HTTP %s (never came up)\n' "$label" "$code"
    return 1
}

# ---------------------------------------------------------------------- exllama

step_exllama() {
    say "ExLlamaV3 fork"
    [ -d "$LLAMA_DIR" ] || die "$LLAMA_DIR not found. Clone the fork first:
  git clone https://github.com/anoane/exllamav3-anemone $LLAMA_DIR"

    local built=( "$LLAMA_DIR"/exllamav3_ext.cpython-*.so )
    if [ -e "${built[0]}" ] && [ "$FORCE_REBUILD" != 1 ]; then
        echo "CUDA extension already built in $LLAMA_DIR (FORCE_REBUILD=1 to rebuild)"
    else
        [ -d "$VENV_DIR" ] || { echo "creating venv"; python3 -m venv "$VENV_DIR"; }
        "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
        "$VENV_DIR/bin/pip" install -r "$LLAMA_DIR/requirements.txt"
        # Must be the same command line: setup.py reads the variable at build
        # time, via torch's cpp_extension.
        ( cd "$LLAMA_DIR" && TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
            "$VENV_DIR/bin/pip" install -e . --no-build-isolation )
        echo "built with TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    fi

    [ -x "$VENV_DIR/bin/python" ] || die "no python in $VENV_DIR"
    echo "venv: $("$VENV_DIR/bin/python" --version)"
}

# ------------------------------------------------------------------------ model

step_model() {
    say "Model pack"
    local dest="$MODELS_DIR/$MODEL_NAME"
    if [ -f "$dest/quantization_config.json" ]; then
        echo "already present: $dest ($(du -sh "$dest" 2>/dev/null | cut -f1))"
    else
        mkdir -p "$MODELS_DIR"
        have hf || die "the 'hf' CLI (huggingface_hub) is required to download the pack"
        echo "downloading $MODEL_HF_REPO"
        hf download "$MODEL_HF_REPO" --local-dir "$dest"
    fi

    # The pack ships the DeepSeek-specified chat template and sampler preset
    # under serve/. If a model_dir is given in config.yml, TabbyAPI reads the
    # template from the model directory root, so both copies must agree.
    if [ -f "$dest/serve/chat_template.jinja" ]; then
        if ! cmp -s "$dest/serve/chat_template.jinja" "$dest/chat_template.jinja"; then
            warn "$dest/chat_template.jinja differs from the pack's serve/ copy."
            warn "The serve/ copy is the one ANEMONE.md section 4 says to use:"
            warn "  cp '$dest/serve/chat_template.jinja' '$dest/chat_template.jinja'"
        fi
    fi
}

# ---------------------------------------------------------------------- serving

step_serving() {
    say "TabbyAPI configuration"
    mkdir -p "$CONFIG_DIR"
    local cfg="$CONFIG_DIR/config.yml" tokens="$CONFIG_DIR/api_tokens.yml"

    # config.yml. Written once; never overwritten, because it is the file a user
    # is most likely to have tuned.
    if [ -f "$cfg" ]; then
        echo "keeping existing $cfg"
        grep -qxE '[[:space:]]*override_preset:[[:space:]]*deepseek_v4.*' "$cfg" \
            || warn "$cfg does not select override_preset: deepseek_v4"
    else
        cat > "$cfg" <<'YAML'
# TabbyAPI configuration. Every value here has an application default; this file
# only lists what is worth pinning.
network:
  host: 127.0.0.1
  port: 5000

model:
  # Resolved inside the container, where the pack is mounted at /app/models.
  model_dir: /app/models

# The DeepSeek sampler recommendation, not TabbyAPI's safe_defaults. safe_defaults
# would fill temperature 0.8 / min_p 0.05 into requests that omit samplers, and
# low-temperature or truncating samplers are what drive this model into
# repetition loops during long reasoning. See ANEMONE.md section 4.
sampling:
  override_preset: deepseek_v4
YAML
        echo "wrote $cfg"
    fi

    # sampler_overrides/deepseek_v4.yml. TabbyAPI resolves this path relative to
    # its working directory (/app), so a copy inside the config mount is not
    # found on its own; the proxy step copies it to /app/sampler_overrides.
    local preset_src="$MODELS_DIR/$MODEL_NAME/serve/sampler_overrides/deepseek_v4.yml"
    if [ -f "$CONFIG_DIR/sampler_overrides/deepseek_v4.yml" ]; then
        echo "keeping existing $CONFIG_DIR/sampler_overrides/deepseek_v4.yml"
    elif [ -f "$preset_src" ]; then
        install -Dm644 "$preset_src" "$CONFIG_DIR/sampler_overrides/deepseek_v4.yml"
        echo "wrote $CONFIG_DIR/sampler_overrides/deepseek_v4.yml"
    else
        warn "preset not found at $preset_src; TabbyAPI will refuse to start while"
        warn "config.yml names a preset it cannot resolve. Run the model step first,"
        warn "or copy the preset in by hand."
    fi

    # api_tokens.yml pins the key so a container rebuild cannot rotate it and
    # silently invalidate ~/.qwen/settings.json. TabbyAPI generates a fresh
    # token_hex(16) whenever this file is missing.
    if [ -f "$tokens" ]; then
        echo "keeping existing $tokens"
    elif [ -n "${TABBY_API_KEY:-}" ]; then
        cat > "$tokens" <<YAML
# Keys for this TabbyAPI instance. TabbyAPI reloads this file on change, so keys
# can be added or revoked without a restart.
#
# api_key accepts a single key or a list of keys.
api_key: $TABBY_API_KEY
admin_key: $TABBY_API_KEY
YAML
        chmod 600 "$tokens"
        echo "wrote $tokens (from TABBY_API_KEY, mode 600)"
    else
        warn "TABBY_API_KEY is not exported and $tokens does not exist."
        warn "TabbyAPI will generate its own key, which will not match"
        warn "~/.qwen/settings.json. Export the key from settings.json first."
    fi
}

# ------------------------------------------------------------------------ tabby

step_tabby() {
    say "TabbyAPI container"
    have docker || die "docker is not installed"
    docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running?)"

    docker image inspect "$TABBY_IMAGE" >/dev/null 2>&1 || {
        echo "pulling $TABBY_IMAGE"
        docker pull "$TABBY_IMAGE"
    }

    # The published image ships upstream ExLlamaV3, which is sufficient for this
    # pack (its quantization_config.json carries no per-expert K table). A pack
    # that did carry one would need an image built on the fork instead.
    if docker container inspect "$TABBY_CONTAINER" >/dev/null 2>&1; then
        local running
        running="$(docker inspect -f '{{.State.Running}}' "$TABBY_CONTAINER")"
        if [ "$running" = "true" ]; then
            echo "$TABBY_CONTAINER already running"
        else
            echo "starting existing $TABBY_CONTAINER"
            docker start "$TABBY_CONTAINER"
        fi
    else
        echo "creating $TABBY_CONTAINER"
        # --shm-size is not optional: ExLlamaV3 keeps tensor-parallel and CPU MoE
        # handoff buffers in /dev/shm, where Docker's 64 MiB default fails.
        # --restart unless-stopped is what survives a reboot; a container run
        # without it is gone after the host comes back.
        docker run --gpus all --shm-size=8g --name "$TABBY_CONTAINER" \
            -d \
            -p "127.0.0.1:$UPSTREAM_PORT:5000" \
            -v "$MODELS_DIR:/app/models" \
            -v "$CONFIG_DIR:/app/config" \
            --ulimit memlock=-1 --ulimit nofile=1048576 \
            --restart unless-stopped \
            "$TABBY_IMAGE"
    fi

    # TabbyAPI looks for sampler_overrides relative to /app, not /app/config, so
    # the preset is copied into the container even though config.yml is mounted.
    if [ -f "$CONFIG_DIR/sampler_overrides/deepseek_v4.yml" ]; then
        docker exec "$TABBY_CONTAINER" mkdir -p /app/sampler_overrides 2>/dev/null || true
        docker cp "$CONFIG_DIR/sampler_overrides/deepseek_v4.yml" \
            "$TABBY_CONTAINER:/app/sampler_overrides/deepseek_v4.yml" 2>/dev/null \
            && echo "installed deepseek_v4 sampler preset in the container" \
            || warn "could not copy the sampler preset into the container"
    fi

    echo "waiting for TabbyAPI to load $MODEL_NAME"
    wait_for "http://127.0.0.1:$UPSTREAM_PORT/v1/models" "upstream :$UPSTREAM_PORT" 150 \
        || warn "TabbyAPI has not answered yet; see: docker logs -f $TABBY_CONTAINER"
}

# ------------------------------------------------------------------------ proxy

step_proxy() {
    say "Schema guard proxy"
    have systemctl || die "systemd is required"

    local unit_dir="$HOME/.config/systemd/user"
    # The unit file is installed from this repo alone. It needs no edit when
    # LLAMA_DIR sits at ~/software/exllamav3-anemone, which is what its %h paths
    # already assume.
    local unit_src="$REPO_DIR/tabby-proxy@.service"
    [ -f "$unit_src" ] || die "tabby-proxy@.service not found in $REPO_DIR"

    # The unit's ExecStart points at the proxy inside LLAMA_DIR, so the two
    # copies must agree. An identical file is left untouched: this repo is not
    # the place to rewrite a checkout the user owns.
    local proxy_src="$REPO_DIR/tabby_proxy.py"
    if [ -f "$LLAMA_DIR/tabby_proxy.py" ] && ! cmp -s "$proxy_src" "$LLAMA_DIR/tabby_proxy.py"; then
        warn "$LLAMA_DIR/tabby_proxy.py differs from this repo's copy, and the unit"
        warn "runs the one in $LLAMA_DIR. Bring it in step deliberately:"
        warn "  diff -u '$LLAMA_DIR/tabby_proxy.py' '$proxy_src'"
        warn "  install -m755 '$proxy_src' '$LLAMA_DIR/tabby_proxy.py'"
    fi

    install -Dm644 "$unit_src" "$unit_dir/tabby-proxy@.service"
    echo "installed $unit_dir/tabby-proxy@.service"

    # The old system unit used %h, which in a system unit expands to /root, and
    # died at status=203/EXEC. Both would bind :8081, so only one may exist.
    if [ -f /etc/systemd/system/tabby-proxy.service ]; then
        if sudo_ready; then
            echo "removing the legacy system unit"
            sudo -n systemctl disable --now tabby-proxy.service 2>/dev/null || true
            sudo -n rm -f /etc/systemd/system/tabby-proxy.service
            sudo -n systemctl daemon-reload || true
        else
            warn "/etc/systemd/system/tabby-proxy.service still exists and needs root to"
            warn "remove. It binds the same port as the user instance, so one will"
            warn "crash-loop until it is gone:"
            warn "  sudo systemctl disable --now tabby-proxy.service"
            warn "  sudo rm /etc/systemd/system/tabby-proxy.service"
        fi
    fi

    # A user service stops at logout unless lingering is on, which is what makes
    # it start at boot without anyone logging in.
    loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable lingering"
    systemctl --user daemon-reload

    local instance
    instance="tabby-proxy@$(systemd-escape --path "$HOME").service"
    systemctl --user enable --now "$instance"
    echo "enabled and started $instance"

    # Self-test: fixtures for every DSML dialect the checkpoint emits.
    if "$VENV_DIR/bin/python" "$LLAMA_DIR/tabby_proxy.py" --selftest >/dev/null 2>&1; then
        echo "proxy self-test: pass"
    else
        warn "proxy self-test failed; run it to see why:"
        warn "  $VENV_DIR/bin/python $LLAMA_DIR/tabby_proxy.py --selftest"
    fi
}

# ----------------------------------------------------------------------- verify

step_verify() {
    say "End-to-end verification"

    local ok=0
    docker inspect -f 'tabbyapi: {{.State.Status}} (restart={{.HostConfig.RestartPolicy.Name}})' \
        "$TABBY_CONTAINER" 2>/dev/null || { warn "container $TABBY_CONTAINER not found"; ok=1; }

    wait_for "http://127.0.0.1:$UPSTREAM_PORT/v1/models" "upstream :$UPSTREAM_PORT" 20 || ok=1
    wait_for "http://127.0.0.1:$PROXY_PORT/v1/models"    "via proxy :$PROXY_PORT"  10 || ok=1

    local instance
    instance="tabby-proxy@$(systemd-escape --path "$HOME").service"
    printf '%s is-active=%s is-enabled=%s linger=%s\n' \
        "$instance" \
        "$(systemctl --user is-active "$instance" 2>&1)" \
        "$(systemctl --user is-enabled "$instance" 2>&1)" \
        "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo unknown)"

    # With a key, the proxy must relay a real completion. A key that does not
    # match TabbyAPI's is the failure this catches.
    if [ -n "${TABBY_API_KEY:-}" ]; then
        local code
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 \
            "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
            -H "x-api-key: $TABBY_API_KEY" -H 'content-type: application/json' \
            -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"stream\":false}" 2>/dev/null || echo 000)"
        printf 'authenticated completion: HTTP %s\n' "$code"
        [ "$code" = 200 ] || { warn "expected 200; 401 means the key does not match TabbyAPI"; ok=1; }
    else
        warn "TABBY_API_KEY not set; skipping the authenticated completion check"
    fi

    return "$ok"
}

# -------------------------------------------------------------------- dispatcher

ALL_STEPS=(exllama model serving tabby proxy verify)
run_step() {
    case "$1" in
        exllama) step_exllama ;;
        model)   step_model ;;
        serving) step_serving ;;
        tabby)   step_tabby ;;
        proxy)   step_proxy ;;
        verify)  step_verify ;;
        *)       die "unknown step '$1' (expected one of: ${ALL_STEPS[*]})" ;;
    esac
}

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    printf '\nEnvironment overrides:\n'
    printf '  LLAMA_DIR=%s\n  MODELS_DIR=%s\n  CONFIG_DIR=%s\n' "$LLAMA_DIR" "$MODELS_DIR" "$CONFIG_DIR"
    printf '  MODEL_NAME=%s\n  TABBY_IMAGE=%s\n  TABBY_API_KEY=<from settings.json>\n' "$MODEL_NAME" "$TABBY_IMAGE"
    printf '  FORCE_REBUILD=1 to rebuild the CUDA extension\n'
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ "$#" -eq 0 ]; then
    for step in "${ALL_STEPS[@]}"; do run_step "$step"; done
else
    for step in "$@"; do run_step "$step"; done
fi

say "done"
