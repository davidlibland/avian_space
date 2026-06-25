#!/usr/bin/env bash
# Autonomous "teach miners to FILL, then hand off to RL" daemon (run_4).
#
# Diagnosis (2026-06-24): the RL policy DID learn the mine->sell loop but lands
# ~half-full (mean cargo_frac 0.49 vs expert 0.97) — premature sell trips. The
# new superlinear knob cargo_sold_fill_exponent makes full holds pay
# disproportionately more, to teach "fill before selling". Traders prove RL can
# learn this loop, so the end-state is RL-driven with light BC.
#
# Strategy:
#   CLIMB   : raise cargo_sold_fill_exponent until miner mean fill-at-landing
#             clears FILL_TARGET (measured from live [DIAG] landing logs).
#   HANDOFF : once fill is healthy, ramp bc_coeff back DOWN toward a low floor so
#             REWARDS (not imitation) drive the policy, verifying fill persists;
#             if fill regresses, step bc back up one (= min bc that holds fill).
# Requires the training to run with DIAG_LANDINGS=1 (this daemon exports it so
# its own restarts keep landing instrumentation on). Logs every decision to md.
# Launch detached:  nohup scripts/miner_fill.sh >/dev/null 2>&1 & disown
set -uo pipefail
cd /home/dlibland/dev/avian_space
export DIAG_LANDINGS=1
export DIAG_COOP=1   # also emit [COOP] cooperation-behavior samples
PY=.venv/bin/python
CFG=training_config.yaml
LOG=experiments/run_4/miner_fill.md
TRAINLOG=logs/rl_minersell_2026-06-22.log
export LD_LIBRARY_PATH="$PWD/target/release/deps:$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"

FILL_TARGET=0.75       # miner mean fill-at-landing we want (expert=0.97, was 0.49)
EXP_STEP=0.5; EXP_CAP=4.0
BC_STEP=0.10; BC_FLOOR=0.10
SETTLE_CYCLES=30
INTERVAL=1500          # ~25 min between decisions
GRACE=4               # settled checks below target before raising the exponent

log(){ printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
getcfg(){ grep -E "^  $1:" "$CFG" | awk '{print $2}'; }
setcfg(){ sed -i "s/^  $1: .*/  $1: $2/" "$CFG"; }
gt(){ awk "BEGIN{exit !($1>$2)}"; }

restart(){
  pkill -9 -x avian_space 2>/dev/null; sleep 3
  nohup ./target/release/avian_space --multiworld-train 8 > "$TRAINLOG" 2>&1 &
  disown
  sleep 18
}

stats(){  # echoes: meanfill sellrate n cargo_sold nav  (miner landings from recent DIAG log + TB)
  "$PY" - "$TRAINLOG" <<'PYEOF'
import sys, re
import numpy as np
sold_fills=[]; sold=0; n=0
try:
    lines=open(sys.argv[1], errors='ignore').read().splitlines()
    for l in reversed(lines):
        m=re.search(r'pers=miner ship=\w+ cargo_frac=([\d.]+) credits_earned=([\d.]+)', l)
        if m:
            # fill measured on SELL landings only (credits>0). Averaging over ALL
            # landings is confounded by empty repair/incidental stops (~47% land
            # empty) which cap the mean ~0.55 regardless of how full the sells are.
            if float(m.group(2)) > 0:
                sold_fills.append(float(m.group(1))); sold += 1
            n+=1
            if n>=400: break
except Exception: pass
meanfill = sum(sold_fills)/len(sold_fills) if sold_fills else float('nan')
sellrate = sold/n if n else float('nan')
try:
    from tbparse import SummaryReader
    df=SummaryReader("experiments/run_4/tb").scalars
    def L(t,k=10):
        v=df[df['tag']==t]['value'].values; return float(np.mean(v[-k:])) if len(v) else float('nan')
    cs=L('reward_per_step/miner/cargo_sold'); nav=L('policy/bc_agreement/nav')
except Exception: cs=nav=float('nan')
print(f"{meanfill:.3f} {sellrate:.3f} {n} {cs:.6f} {nav:.3f}")
PYEOF
}

PHASE=CLIMB; waits=0
log ""
log "## miner FILL daemon started (pid $$)"
log "exponent=$(getcfg cargo_sold_fill_exponent) bc=$(getcfg bc_coeff) drop_scale=$(getcfg asteroid_drop_scale) | FILL_TARGET=$FILL_TARGET (expert 0.97, was 0.49)"
log "(watching live training; expects DIAG_LANDINGS=1)"

while true; do
  sleep "$INTERVAL"
  if ! pgrep -x avian_space >/dev/null; then log "process DEAD -> restarting"; restart; fi
  read -r fill sellrate n cs nav <<< "$(stats)"
  cyc=$(grep -c '\[ppo\] cycle' "$TRAINLOG" 2>/dev/null || echo 0)
  exp=$(getcfg cargo_sold_fill_exponent); bc=$(getcfg bc_coeff)
  log "check phase=$PHASE exp=$exp bc=$bc cyc=$cyc | miner fill=$fill sellrate=$sellrate n=$n cargo_sold=$cs nav=$nav"

  if [ "$fill" = "nan" ]; then log "  no fill data yet -> skip"; continue; fi
  if [ "${cyc:-0}" -lt "$SETTLE_CYCLES" ]; then log "  not settled ($cyc<$SETTLE_CYCLES) -> wait"; continue; fi

  if [ "$PHASE" = "CLIMB" ]; then
    if gt "$fill" "$FILL_TARGET"; then
      log "  >>> FILL TARGET reached (fill=$fill >= $FILL_TARGET). Switch to HANDOFF: ramp bc DOWN, let RL drive."
      PHASE=HANDOFF; waits=0
    else
      waits=$((waits+1))
      if [ "$waits" -lt "$GRACE" ]; then
        log "  fill below target (learn-grace $waits/$GRACE) -> hold, let it learn"
      elif gt "$EXP_CAP" "$exp"; then
        newexp=$(awk "BEGIN{e=$exp+$EXP_STEP; if(e>$EXP_CAP)e=$EXP_CAP; printf \"%.1f\",e}")
        log "  fill still low -> raise cargo_sold_fill_exponent $exp -> $newexp + restart"
        setcfg cargo_sold_fill_exponent "$newexp"; restart; waits=0
      else
        log "  !! exponent at cap $EXP_CAP and fill still $fill -> FLAG (may need other lever), holding"
      fi
    fi
    continue
  fi

  # PHASE = HANDOFF: lower bc while fill persists
  if ! gt "$fill" "$FILL_TARGET"; then
    newbc=$(awk "BEGIN{printf \"%.2f\", $bc+$BC_STEP}")
    log "  fill regressed during handoff (fill=$fill) -> raise bc back to $newbc (min bc that holds fill) + restart -> DONE"
    setcfg bc_coeff "$newbc"; restart
    log "## DONE: fill sustained; minimum bc_coeff=$newbc at exponent=$exp (fill was $fill)"
    break
  fi
  if ! gt "$bc" "$BC_FLOOR"; then
    log "## DONE: fill sustained down to bc floor $BC_FLOOR — RL drives it. exponent=$exp fill=$fill"
    break
  fi
  newbc=$(awk "BEGIN{b=$bc-$BC_STEP; if(b<$BC_FLOOR)b=$BC_FLOOR; printf \"%.2f\",b}")
  log "  fill healthy ($fill) -> lower bc_coeff $bc -> $newbc + restart (hand toward RL)"
  setcfg bc_coeff "$newbc"; restart
done
log "daemon exiting (pid $$)"
