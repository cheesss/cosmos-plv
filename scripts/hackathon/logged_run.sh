#!/usr/bin/env bash
# Run a command and append {name, command, start, end, exit_code} to
# results/hackathon/commands_log.jsonl.  Usage: logged_run.sh <name> <cmd...>
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
name="$1"; shift
start=$(date -u +%FT%TZ)
"$@"; rc=$?
end=$(date -u +%FT%TZ)
python3 - "$name" "$start" "$end" "$rc" "$*" >> "$REPO/results/hackathon/commands_log.jsonl" <<'PY'
import json, sys
n, s, e, rc, cmd = sys.argv[1:6]
print(json.dumps({"name": n, "command": cmd, "start_utc": s, "end_utc": e, "exit_code": int(rc)}))
PY
exit $rc
