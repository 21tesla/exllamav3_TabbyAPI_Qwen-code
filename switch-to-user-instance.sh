#!/usr/bin/env bash
# Switch the schema-guard proxy from the system unit to a `systemctl --user`
# instance.
#
# Only step 1 needs your sudo password. It runs through your own terminal
# because an assistant's shells usually have no usable sudo cache (sudo's
# credential cache is keyed per-TTY, and each tool invocation gets its own
# pty, so `sudo -v` typed elsewhere does not help them).
#
#   ~/software/exllamav3-anemone/switch-to-user-instance.sh
#
# Rollback is at the bottom of this script's output.
set -euo pipefail

# Paging would stop this script at a `less` prompt when stdout is a terminal.
export SYSTEMD_PAGER=cat
export PAGER=cat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$HOME/.config/systemd/user/tabby-proxy.service"
REPO_TEMPLATE="$SCRIPT_DIR/tabby-proxy@.service"
SYSTEM_UNIT=tabby-proxy.service
INSTANCE="tabby-proxy@$(systemd-escape --path "$HOME").service"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

listener_pids() {
    ss -ltnp 2>/dev/null | grep '127\.0\.0\.1:8081' \
        | grep -oP 'pid=\K[0-9]+' | sort -u | tr '\n' ' '
}

# --- preflight --------------------------------------------------------------
[[ -f "$TEMPLATE" ]] || die "user template missing: $TEMPLATE"
if [[ -f "$REPO_TEMPLATE" ]] && ! cmp -s "$REPO_TEMPLATE" "$TEMPLATE"; then
    echo "note: $TEMPLATE differs from the repo copy; run install-tabby-proxy-service.sh"
    echo "      with the --user unit path, or copy it by hand, if that is unintended."
fi

if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
    echo "warning: lingering is off, so the instance would stop at logout."
    echo "         enable it with: loginctl enable-linger $USER"
fi

echo "== switch: $SYSTEM_UNIT  ->  $INSTANCE =="
echo "== state BEFORE =="
printf '  system unit  : %s / %s\n' \
    "$(systemctl is-active "$SYSTEM_UNIT" 2>/dev/null || true)" \
    "$(systemctl is-enabled "$SYSTEM_UNIT" 2>/dev/null || true)"
printf '  user instance: %s / %s\n' \
    "$(systemctl --user is-active "$INSTANCE" 2>/dev/null || true)" \
    "$(systemctl --user is-enabled "$INSTANCE" 2>/dev/null || true)"
printf '  listeners    : %s\n' "$(listener_pids)"

echo
read -r -p "Stop and disable $SYSTEM_UNIT, then start $INSTANCE? [y/N] " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "aborted; nothing changed"
    exit 0
fi

# --- 1. the one privileged step --------------------------------------------
echo
echo "== 1. stop + disable the system unit (frees 127.0.0.1:8081) =="
sudo systemctl disable --now "$SYSTEM_UNIT"

# --- 2. the unprivileged half ----------------------------------------------
echo
echo "== 2. enable + start the user instance =="
systemctl --user daemon-reload
systemctl --user enable --now "$INSTANCE"

# Give it a moment to bind before judging it.
for _ in 1 2 3 4 5 6 7 8; do
    [[ "$(systemctl --user is-active "$INSTANCE" || true)" == active ]] && break
    sleep 1
done

# --- verification ----------------------------------------------------------
echo
echo "== state AFTER =="
printf '  system unit  : %s / %s\n' \
    "$(systemctl is-active "$SYSTEM_UNIT" 2>/dev/null || true)" \
    "$(systemctl is-enabled "$SYSTEM_UNIT" 2>/dev/null || true)"
printf '  user instance: %s / %s\n' \
    "$(systemctl --user is-active "$INSTANCE" || true)" \
    "$(systemctl --user is-enabled "$INSTANCE" || true)"
printf '  MainPID      : %s\n' \
    "$(systemctl --user show -p MainPID --value "$INSTANCE")"
printf '  WorkingDirectory: %s\n' \
    "$(systemctl --user show -p WorkingDirectory --value "$INSTANCE")"

echo "  ExecStart    :"
systemctl --user show --no-pager -p ExecStart "$INSTANCE" | sed 's/^/    /'

count="$(listener_pids | wc -w)"
printf '  listeners on 8081: %s (%s)\n' "$count" "$(listener_pids)"

if [[ "$(systemctl --user is-active "$INSTANCE" || true)" != active || "$count" -ne 1 ]]; then
    printf '\n!! the switch did not settle cleanly.\n'
    printf '!! last journal lines:\n'
    journalctl --user -u "$INSTANCE" -n 20 --no-pager | sed 's/^/!!   /'
    printf '!! roll back with:\n'
    printf '!!   systemctl --user disable --now %s\n' "$INSTANCE"
    printf '!!   sudo systemctl enable --now %s\n' "$SYSTEM_UNIT"
    exit 1
fi

echo
echo "Switched. The proxy is now the user instance and will start at boot via lingering."
echo
echo "Rollback if ever needed:"
printf '  systemctl --user disable --now %s\n' "$INSTANCE"
printf '  sudo systemctl enable --now %s\n' "$SYSTEM_UNIT"
