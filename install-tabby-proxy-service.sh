#!/usr/bin/env bash
# Install the TabbyAPI schema-guard proxy as a system service.
#
# Run as your normal user. The one privileged step asks for your own sudo
# password, which is why this is a script you run rather than something an
# assistant executes on your behalf.
#
#   ~/software/exllamav3-anemone/install-tabby-proxy-service.sh
#
set -euo pipefail

# systemctl pages its output when stdout is a terminal, which would stop this
# script at a `less` prompt; --no-pager is added per call below as well.
export SYSTEMD_PAGER=cat
export PAGER=cat

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/tabby-proxy.service"
UNIT=/etc/systemd/system/tabby-proxy.service
SERVICE=tabby-proxy.service
BACKUP="$HOME/tabby-proxy.service.bak"

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

if [[ ! -f "$SRC" ]]; then
    die "source unit not found: $SRC"
fi

# In a *system* unit, %h resolves to the system manager's home (/root), not to
# the User='s home, so a %h-based ExecStart dies with status=203/EXEC. Refuse to
# install such a unit. (%h is only "your home" in a `systemctl --user` unit.)
if grep -qE '^(WorkingDirectory|ExecStart|EnvironmentFile)=-?[^=]*%h' "$SRC"; then
    die "$SRC uses %h in an active directive; that resolves to /root and cannot start"
fi

echo "== candidate unit: $SRC =="
cat "$SRC"

echo
echo "== diff against the deployed unit =="
if [[ -f "$UNIT" ]]; then
    diff -u "$UNIT" "$SRC" || true
else
    echo "(nothing deployed yet)"
fi

echo
read -r -p "Install this unit and restart $SERVICE? [y/N] " reply
if [[ ! "$reply" =~ ^[Yy]$ ]]; then
    echo "aborted; nothing changed"
    exit 0
fi

echo
if [[ -f "$UNIT" ]]; then
    if cp "$UNIT" "$BACKUP" 2>/dev/null; then
        echo "backed up the deployed unit to $BACKUP"
    else
        echo "warning: could not back up $UNIT (continuing)"
    fi
else
    echo "(no deployed unit to back up)"
fi

echo "== installing $UNIT =="
sudo install -m 644 "$SRC" "$UNIT"

echo "== systemctl daemon-reload =="
sudo systemctl daemon-reload

# systemd-analyze verify cannot catch a bad path here: specifiers are expanded
# at start time, so the only reliable check is what systemd itself resolved.
echo "== resolved ExecStart =="
systemctl show --no-pager -p ExecStart "$SERVICE"
if systemctl show --no-pager -p ExecStart "$SERVICE" | grep -q '/root'; then
    printf '\n!! systemd resolved this unit against /root, so it would fail to start.\n'
    printf '!! Not restarting. Restore with:\n'
    printf '!!   sudo install -m 644 %s %s && sudo systemctl daemon-reload\n' "$BACKUP" "$UNIT"
    exit 1
fi

echo "== enabling and restarting =="
sudo systemctl enable "$SERVICE" >/dev/null 2>&1 || true
sudo systemctl restart "$SERVICE"
sleep 4

echo
echo "== verification =="
printf 'md5 deployed     : %s\n' "$(md5sum "$UNIT" | cut -d' ' -f1)"
printf 'md5 source       : %s\n' "$(md5sum "$SRC" | cut -d' ' -f1)"
printf 'state            : %s\n' "$(systemctl is-active "$SERVICE")"
printf 'MainPID          : %s\n' "$(systemctl show -p MainPID --value "$SERVICE")"
printf 'WorkingDirectory : %s\n' "$(systemctl show -p WorkingDirectory --value "$SERVICE")"

if ss -ltn 2>/dev/null | grep -q '127\.0\.0\.1:8081'; then
    printf 'listener         : 127.0.0.1:8081 OK\n'
else
    printf 'listener         : MISSING on 127.0.0.1:8081\n'
    echo
    echo "-- last journal lines --"
    journalctl -u "$SERVICE" -n 15 --no-pager
    exit 1
fi

echo
echo "Service is up and listening."
echo "Rollback if ever needed:"
printf '  sudo install -m 644 %s %s && sudo systemctl daemon-reload && sudo systemctl restart %s\n' \
    "$BACKUP" "$UNIT" "$SERVICE"
