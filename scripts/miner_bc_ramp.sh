#!/usr/bin/env bash
# Autonomous miner mine->sell curriculum (run_4), no human in the loop.
#
# The BC expert now routes a full-hold miner to a planet to sell (choose_target_slot),
# so a STRONG BC anchor forces the policy to imitate selling. Strategy:
#   PHASE UP   : ramp bc_coeff UP (+0.15, cap 0.90) until miners learn to sell
#                (miner cargo_sold/step > SELL_THRESH), restarting run_4 each step.
#   PHASE DOWN : once selling is learned, ramp bc_coeff back DOWN (-0.10, floor 0.20),
#                verifying selling persists; if it collapses, step bc back up one
#                (= minimum bc that sustains selling) and finish.
#
# asteroid_hit_miner (3.5) and cargo_sold_miner (1.5) are held fixed; bc_coeff is
# the only knob. Every decision + numbers is appended to the md log below.
# Launch detached:  nohup scripts/miner_bc_ramp.sh >/dev/null 2>&1 & disown
set -uo pipefail
cd /home/dlibland/dev/avian_space
PY=.venv/bin/python
CFG=training_config.yaml
LOG=experiments/run_4/miner_bc_ramp.md
TRAINLOG=logs/rl_minersell_2026-06-22.log
STDLIB="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"
export LD_LIBRARY_PATH="$PWD/target/release/deps:$STDLIB"

SELL_THRESH=0.0002      # miner cargo_sold/step above this == "selling learned"
BC_UP_STEP=0.15;  BC_CAP=0.90
BC_DOWN_STEP=0.10; BC_FLOOR=0.20
SETTLE_CYCLES=30        # cycles since last restart before a decision is allowed
INTERVAL=1500           # ~25 min between decisions

log(){ printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
getcfg(){ grep -E "^  $1:" "$CFG" | awk '{print $2}'; }
setcfg(){ sed -i "s/^  $1: .*/  $1: $2/" "$CFG"; }
gt(){ awk "BEGIN{exit !($1>$2)}"; }   # gt A B  -> true if A>B (handles floats; nan->false)

metrics(){  # echoes: cargo_sold asteroid_hit total nav ev burnin landing pickup  (or 8x nan)
  "$PY" - <<'PYEOF'
from tbparse import SummaryReader
import numpy as np
try:
    df=SummaryReader("experiments/run_4/tb").scalars
    def L(t,k=10):
        v=df[df["tag"]==t]["value"].values
        return float(np.mean(v[-k:])) if len(v) else float('nan')
    W={"landing":1,"cargo_sold":1,"ship_hit":1,"asteroid_hit":1,"pickup":1,"health_gated":1,"health_raw":0,"damage":1}
    tot=sum(L(f"reward_per_step/miner/{t}")*w for t,w in W.items())
    print(f"{L('reward_per_step/miner/cargo_sold'):.6f} {L('reward_per_step/miner/asteroid_hit'):.6f} {tot:.6f} "
          f"{L('policy/bc_agreement/nav'):.3f} {L('train/explained_variance'):.3f} {L('train/value_burnin'):.3f} "
          f"{L('reward_per_step/miner/landing'):.6f} {L('reward_per_step/miner/pickup'):.6f}")
except Exception as e:
    print("nan nan nan nan nan nan nan nan")
PYEOF
}

restart(){
  pkill -9 -x avian_space 2>/dev/null; sleep 3
  nohup ./target/release/avian_space --multiworld-train 8 > "$TRAINLOG" 2>&1 &
  disown
  sleep 18
}

PHASE=UP
log ""
log "## miner BC-ramp daemon started (pid $$)"
log "fixed: asteroid_hit_miner=$(getcfg asteroid_hit_miner) cargo_sold_miner=$(getcfg cargo_sold_miner) | start bc_coeff=$(getcfg bc_coeff) | SELL_THRESH=$SELL_THRESH"

while true; do
  sleep "$INTERVAL"
  if ! pgrep -x avian_space >/dev/null; then log "process DEAD -> restarting"; restart; fi
  read -r cs ah tot nav ev bi ld pk <<< "$(metrics)"
  cyc=$(grep -c '\[ppo\] cycle' "$TRAINLOG" 2>/dev/null || echo 0)
  bc=$(getcfg bc_coeff)
  log "check phase=$PHASE bc=$bc cyc=$cyc | miner: cargo_sold=$cs landing=$ld asteroid_hit=$ah pickup=$pk total=$tot | nav=$nav EV=$ev burnin=$bi"

  if [ "$cs" = "nan" ]; then log "  tb read failed (nan) -> skip"; continue; fi
  if [ "${cyc:-0}" -lt "$SETTLE_CYCLES" ]; then log "  not settled ($cyc<$SETTLE_CYCLES) -> wait"; continue; fi

  selling=0; gt "$cs" "$SELL_THRESH" && selling=1
  shooting_dead=0; gt 0.0008 "$ah" && shooting_dead=1   # ah<0.0008

  if [ "$PHASE" = "UP" ]; then
    if [ "$selling" = "1" ]; then
      log "  >>> SELLING LEARNED (cargo_sold=$cs > $SELL_THRESH) at bc=$bc -> switch PHASE=DOWN"
      PHASE=DOWN
    elif gt "$bc" 0.899; then
      log "  bc at cap $BC_CAP, still not selling -> HOLD at cap, keep watching (BC alone insufficient; flag for review)"
    else
      newbc=$(awk "BEGIN{b=$bc+$BC_UP_STEP; if(b>$BC_CAP)b=$BC_CAP; printf \"%.2f\",b}")
      log "  not selling -> raise bc $bc -> $newbc + restart"
      setcfg bc_coeff "$newbc"; restart
    fi
  else  # PHASE=DOWN
    if [ "$selling" = "0" ] || [ "$shooting_dead" = "1" ]; then
      newbc=$(awk "BEGIN{printf \"%.2f\", $bc+$BC_DOWN_STEP}")
      log "  selling/shooting regressed at bc=$bc -> raise back to $newbc (min bc that sustains selling) + restart -> DONE"
      setcfg bc_coeff "$newbc"; restart
      log "## DONE: miners sell; minimum sustaining bc_coeff=$newbc (cargo_sold was $cs)"
      break
    elif ! gt "$bc" "$BC_FLOOR"; then
      log "## DONE: selling sustained all the way down to bc floor $BC_FLOOR (cargo_sold=$cs)"
      break
    else
      newbc=$(awk "BEGIN{b=$bc-$BC_DOWN_STEP; if(b<$BC_FLOOR)b=$BC_FLOOR; printf \"%.2f\",b}")
      log "  selling sustained -> lower bc $bc -> $newbc + restart"
      setcfg bc_coeff "$newbc"; restart
    fi
  fi
done
log "daemon exiting (pid $$)"
