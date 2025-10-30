#!/usr/bin/env bash
set -euo pipefail

: "${DISPLAY:=:99}"
: "${WEB_PORT:=6080}"
: "${VNC_PORT:=5900}"
: "${RESOLUTION:=1920x1080x24}"

export DISPLAY
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export QT_X11_NO_MITSHM=${QT_X11_NO_MITSHM:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# --- clean any stale X locks for this display ---
num="${DISPLAY#:}"  # strip leading colon, e.g. :99 -> 99
rm -f "/tmp/.X${num}-lock" "/tmp/.X11-unix/X${num}" || true
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

# 1) Start virtual X
Xvfb "$DISPLAY" -screen 0 "$RESOLUTION" -ac +iglx >/tmp/xvfb.log 2>&1 &

# 2) Minimal WM so Tk/GLUT behave
fluxbox >/tmp/fluxbox.log 2>&1 &

# 3) VNC + noVNC
x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -forever -shared -nopw \
  -quiet >/tmp/x11vnc.log 2>&1 &

# Auto-redirect "/" to the client with autoconnect
# (noVNC serves files from /usr/share/novnc by default)
printf '%s\n' '<meta http-equiv="refresh" content="0;url=vnc_lite.html?autoconnect=1&resize=scale">' \
  > /usr/share/novnc/index.html

/usr/share/novnc/utils/novnc_proxy --vnc "localhost:${VNC_PORT}" \
  --listen "0.0.0.0:${WEB_PORT}" >/tmp/novnc.log 2>&1 &

echo "noVNC on http://localhost:${WEB_PORT} (DISPLAY=$DISPLAY, RES=$RESOLUTION)"

# --- wait until the X server is actually ready ---
timeout 10 bash -c 'until xdpyinfo >/dev/null 2>&1; do sleep 0.2; done' || true

# --- clean shutdown of all background procs on Ctrl-C/stop ---
trap 'echo "[starter] shutting down..."; kill 0; wait' INT TERM

# 5) Run your app (ensure this points at the right config!)
exec python /app/UI.py
