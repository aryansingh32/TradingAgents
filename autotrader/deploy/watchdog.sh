#!/bin/sh
# Restart the bot when its heartbeat is older than MAX_AGE seconds (a hung
# process that systemd still sees as running). The bot rewrites the
# heartbeat file every risk loop.
HEARTBEAT="${1:-/opt/autotrader/autotrader_data/btc/heartbeat}"
MAX_AGE="${2:-300}"
SERVICE="${3:-autotrader}"
if [ ! -f "$HEARTBEAT" ]; then exit 0; fi
age=$(( $(date +%s) - $(stat -c %Y "$HEARTBEAT") ))
if [ "$age" -gt "$MAX_AGE" ]; then
  logger -t autotrader-watchdog "heartbeat ${age}s old, restarting $SERVICE"
  systemctl restart "$SERVICE"
fi
