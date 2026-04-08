#!/bin/bash
# Shows Mininet iptables commands in real time
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/mininet_commands.log"

echo "=========================================="
echo "  MININET CLI — LIVE IPTABLES COMMANDS"
echo "=========================================="
echo ""
tail -f "$LOG_FILE"
