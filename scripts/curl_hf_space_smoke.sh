#!/usr/bin/env bash
# Smoke-test a deployed Qubit-Medic OpenEnv Space (curl only).
# Usage: BASE_URL=https://ronitraj-quantumscribe.hf.space ./scripts/curl_hf_space_smoke.sh
set -euo pipefail
BASE_URL="${BASE_URL:-https://ronitraj-quantumscribe.hf.space}"
BASE_URL="${BASE_URL%/}"
pass=0
fail=0
_http() { curl -sS -o /dev/null -w "%{http_code}" "$@"; }
check() {
  local name="$1" code="$2" want="$3"
  if [[ "$code" == "$want" ]]; then
    echo "OK  $name (HTTP $code)"; pass=$((pass + 1))
  else
    echo "FAIL $name (HTTP $code, want $want)"; fail=$((fail + 1))
  fi
}
echo "=== Qubit-Medic Space: $BASE_URL ==="
c=$(_http "$BASE_URL/") && check "GET /" "$c" 200
c=$(_http "$BASE_URL/healthz") && check "GET /healthz" "$c" 200
c=$(_http "$BASE_URL/health") && check "GET /health" "$c" 200
c=$(_http "$BASE_URL/metadata") && check "GET /metadata" "$c" 200
c=$(_http "$BASE_URL/schema") && check "GET /schema" "$c" 200
c=$(_http "$BASE_URL/docs") && check "GET /docs" "$c" 200
c=$(_http "$BASE_URL/openapi.json") && check "GET /openapi.json" "$c" 200
c=$(_http "$BASE_URL/state") && check "GET /state" "$c" 200
c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/state") && check "POST /state" "$c" 200
R=$(curl -sS -X POST "$BASE_URL/reset?forced_level=L1_warmup" -H "Content-Type: application/json" -d '{"seed": 42}')
EP=$(echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('observation',d)['episode_id'])")
S=$(curl -sS -X POST "$BASE_URL/step" -H "Content-Type: application/json" -d "{\"action\": {\"raw_response\": \"X_ERRORS=[] Z_ERRORS=[]\", \"episode_id\": $EP}}")
RWD=$(echo "$S" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('reward', 'x'))")
if [[ "$RWD" != "x" ]]; then echo "OK  POST /reset + POST /step (reward=$RWD)"; pass=$((pass+1)); else echo "FAIL /step"; fail=$((fail+1)); fi
c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/close") && check "POST /close" "$c" 200
c=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/decode" -H "Content-Type: application/json" -d '{"syndrome": [0,0,0,0,0,0,0,0], "level": "L1_warmup"}') && check "POST /decode" "$c" 200
c=$(_http "$BASE_URL/mcp") && check "GET /mcp (POST-only => 405)" "$c" 405
echo "=== $pass passed, $fail failed ==="
[[ "$fail" -eq 0 ]]
