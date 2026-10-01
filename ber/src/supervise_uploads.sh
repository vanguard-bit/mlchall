#!/bin/bash
# Keep v3, v4, and v5 submission files on track until 06:00 IST.
# One score at a time. Restart a stage only when its process is gone and its file is short.

ROOT=/home/loki/projects/mlchall
PY=$ROOT/ber/.venv/bin/python
export PYTHONPATH=$ROOT/ber/src
LOG=$ROOT/ber/data/supervise.log
ROWS=1732545
DEADLINE=$(date -d '2026-09-27 06:05:00' +%s)

cd "$ROOT" || exit 1

log() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

complete() {
  local f=$1
  [[ -f $f ]] || return 1
  local n
  n=$(wc -l < "$f")
  [[ "$n" -eq $ROWS ]]
}

python_running() {
  local script=$1
  local pid comm
  for pid in $(pgrep -f "ber/src/$script" || true); do
    comm=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case $comm in
      *bin/python*"ber/src/$script"*) return 0 ;;
    esac
  done
  return 1
}

log "supervisor up deadline=$(date -d "@$DEADLINE" '+%F %H:%M:%S')"

while [[ $(date +%s) -lt $DEADLINE ]]; do
  if complete output/v3/matching_results.tsv \
    && complete output/v4/matching_results.tsv \
    && complete output/v5/matching_results.tsv; then
    log "all three files have $ROWS lines"
    exit 0
  fi

  if ! complete output/v3/matching_results.tsv; then
    if ! python_running score_v3.py; then
      log "restart v3 score"
      "$PY" -u ber/src/score_v3.py > ber/data/score_v3.log 2>&1 || log "v3 score exited $?"
    fi
  elif ! complete output/v4/matching_results.tsv; then
    if ! python_running score_v4.py && ! python_running score_v3.py; then
      log "start v4 score"
      "$PY" -u ber/src/score_v4.py > ber/data/score_v4.log 2>&1 || log "v4 score exited $?"
    fi
  elif ! complete output/v5/matching_results.tsv; then
    if ! python_running score_v5.py && ! python_running train_v5.py && ! python_running score_v4.py; then
      if [[ ! -f ber/data/scoreboard/lgbm_v5.txt ]]; then
        log "train v5"
        "$PY" -u ber/src/train_v5.py > ber/data/train_v5.log 2>&1 || log "v5 train exited $?"
      fi
      if [[ -f ber/data/scoreboard/lgbm_v5.txt ]]; then
        log "start v5 score"
        "$PY" -u ber/src/score_v5.py > ber/data/score_v5.log 2>&1 || log "v5 score exited $?"
      else
        log "v5 model missing, will retry"
      fi
    fi
  fi
  sleep 45
done

log "deadline reached"
complete output/v3/matching_results.tsv && log "v3 ready" || log "v3 NOT ready"
complete output/v4/matching_results.tsv && log "v4 ready" || log "v4 NOT ready"
complete output/v5/matching_results.tsv && log "v5 ready" || log "v5 NOT ready"
