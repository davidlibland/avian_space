#!/usr/bin/env bash
# Autonomous asteroid_drop_scale titration (run_4), no human in the loop.
#
# Context: added the small-hold `prospector` miner (cargo 6) + made
# asteroid_drop_scale a runtime config (1..=(size*scale) ore per shattered
# asteroid). Started HIGH (4.0) so a prospector fills in ~1 hit and learns to
# land & sell immediately. This daemon then titrates the drop scale BACK DOWN
# toward a reasonable floor while verifying selling persists — settling at the
# smallest drop scale that still sustains the mine->land->sell loop.
#
# Strategy:
#   CONFIRM : at scale 4.0, wait for miner cargo_sold/step > SELL_THRESH (selling).
#   TITRATE : step asteroid_drop_scale down (-0.5) toward FLOOR (0.8); at each
#             settled step verify selling persists. If it collapses, step the
#             scale back up one (= min scale that sustains selling) and finish.
# Holds the economic weights fixed (asteroid_hit_miner=1.0, pickup=1.2,
# cargo_sold=1.5, bc=0.05). Logs every decision+numbers to the md below.
# Launch detached:  nohup scripts/miner_dropscale.sh >/dev/null 2>&1 & disown
set -uo pipefail
cd /home/dlibland/dev/avian_space
PY=.venv/bin/python
CFG=training_config.yaml
LOG=experiments/run_4/miner_dropscale.md
TRAINLOG=logs/rl_minersell_2026-06-22.log
STDLIB="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"
export LD_LIBRARY_PATH="$PWD/target/release/deps:$STDLIB"

SELL_THRESH=0.0003      # miner cargo_sold/step above this == "selling established"
SCALE_STEP=0.5
SCALE_FLOOR=0.8         # reasonable end value (4x the old 0.2); don't go below
SETTLE_CYCLES=30
INTERVAL=1500           # ~25 min between decisions
# CONFIRM phase: first give miners a reward-driven grace period at the low bc;
# if selling still hasn't emerged, lean on the (verified-correct) BC sell-route
# signal by ramping bc_coeff up — at drop_scale 4.0 the expert routes a FULL
# prospector to land&sell, so imitation teaches the whole loop.
GRACE_CHECKS=5          # settled non-selling checks at low bc before bumping bc
BC_STEP=0.10
BC_CAP=0.50

log(){ printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
getcfg(){ grep -E "^  $1:" "$CFG" | awk '{print $2}'; }
setcfg(){ sed -i "s/^  $1: .*/  $1: $2/" "$CFG"; }
gt(){ awk "BEGIN{exit !($1>$2)}"; }

metrics(){  # cargo_sold pickup asteroid_hit landing total nav ev burnin
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
    print(f"{L('reward_per_step/miner/cargo_sold'):.6f} {L('reward_per_step/miner/pickup'):.6f} "
          f"{L('reward_per_step/miner/asteroid_hit'):.6f} {L('reward_per_step/miner/landing'):.6f} {tot:.6f} "
          f"{L('policy/bc_agreement/nav'):.3f} {L('train/explained_variance'):.3f} {L('train/value_burnin'):.3f}")
except Exception:
    print("nan nan nan nan nan nan nan nan")
PYEOF
}

restart(){
  pkill -9 -x avian_space 2>/dev/null; sleep 3
  nohup ./target/release/avian_space --multiworld-train 8 > "$TRAINLOG" 2>&1 &
  disown
  sleep 18
}

PHASE=CONFIRM
confirm_waits=0
log ""
log "## asteroid_drop_scale titration daemon started (pid $$)"
log "start scale=$(getcfg asteroid_drop_scale) floor=$SCALE_FLOOR | econ weights: asteroid_hit_miner=$(getcfg asteroid_hit_miner) pickup=$(getcfg pickup_reward_miner) cargo_sold=$(getcfg cargo_sold_miner) bc=$(getcfg bc_coeff) | SELL_THRESH=$SELL_THRESH | grace=$GRACE_CHECKS then bc-ramp to $BC_CAP"
log "(training already running with correct config — watching the live process, no startup restart)"

while true; do
  sleep "$INTERVAL"
  if ! pgrep -x avian_space >/dev/null; then log "process DEAD -> restarting"; restart; fi
  read -r cs pk ah ld tot nav ev bi <<< "$(metrics)"
  cyc=$(grep -c '\[ppo\] cycle' "$TRAINLOG" 2>/dev/null || echo 0)
  scale=$(getcfg asteroid_drop_scale)
  log "check phase=$PHASE scale=$scale cyc=$cyc | miner: cargo_sold=$cs landing=$ld pickup=$pk asteroid_hit=$ah total=$tot | nav=$nav EV=$ev burnin=$bi"

  if [ "$cs" = "nan" ]; then log "  tb read failed (nan) -> skip"; continue; fi
  if [ "${cyc:-0}" -lt "$SETTLE_CYCLES" ]; then log "  not settled ($cyc<$SETTLE_CYCLES) -> wait"; continue; fi

  selling=0; gt "$cs" "$SELL_THRESH" && selling=1

  if [ "$PHASE" = "CONFIRM" ]; then
    if [ "$selling" = "1" ]; then
      log "  >>> SELLING ESTABLISHED at scale=$scale bc=$(getcfg bc_coeff) (cargo_sold=$cs). Begin titrating drop scale DOWN."
      PHASE=TITRATE
    else
      confirm_waits=$((confirm_waits+1))
      bc=$(getcfg bc_coeff)
      if [ "$confirm_waits" -lt "$GRACE_CHECKS" ]; then
        log "  not selling yet (grace $confirm_waits/$GRACE_CHECKS at bc=$bc) -> let REWARDS drive, keep watching (pickup=$pk landing=$ld)"
      elif gt "$BC_CAP" "$bc"; then
        newbc=$(awk "BEGIN{b=$bc+$BC_STEP; if(b>$BC_CAP)b=$BC_CAP; printf \"%.2f\",b}")
        log "  grace elapsed, still not selling -> lean on the (verified-correct) BC sell-route: raise bc_coeff $bc -> $newbc + restart. At scale 4.0 the expert routes a FULL prospector to land&sell, so imitation teaches the loop."
        setcfg bc_coeff "$newbc"; restart
      else
        log "  !! bc at cap $BC_CAP and STILL not selling (cargo_sold=$cs) -> FLAG: deeper issue (routing/landing/planet reachability). Holding."
      fi
    fi
    continue
  fi

  # PHASE = TITRATE
  if [ "$selling" = "0" ]; then
    newscale=$(awk "BEGIN{printf \"%.2f\", $scale+$SCALE_STEP}")
    log "  selling COLLAPSED at scale=$scale (cargo_sold=$cs) -> step back up to $newscale (min scale that sustains selling) + restart -> DONE"
    setcfg asteroid_drop_scale "$newscale"; restart
    log "## DONE: minimum asteroid_drop_scale that sustains selling = $newscale (cargo_sold was $cs at $scale)"
    break
  fi
  if ! gt "$scale" "$SCALE_FLOOR"; then
    log "## DONE: selling sustained all the way down to floor scale=$SCALE_FLOOR (cargo_sold=$cs)"
    break
  fi
  newscale=$(awk "BEGIN{s=$scale-$SCALE_STEP; if(s<$SCALE_FLOOR)s=$SCALE_FLOOR; printf \"%.2f\",s}")
  log "  selling sustained (cargo_sold=$cs) -> lower asteroid_drop_scale $scale -> $newscale + restart"
  setcfg asteroid_drop_scale "$newscale"; restart
done
log "daemon exiting (pid $$)"
