#!/usr/bin/env bash
# Autonomous miner profit-loop curriculum v2 (run_4), no human in the loop.
#
# DIAGNOSIS that motivates this (see experiments/run_4/miner_bc_ramp.md): ramping
# bc_coeff to its 0.90 cap did NOT induce selling. Miners learned to SHOOT asteroids
# (asteroid_hit 17x up) but COLLECT less (pickup down) and never sold (cargo_sold~0),
# because asteroid_hit_miner=3.5 >> pickup_reward_miner=0.8 made shoot-for-its-own-
# sake optimal -> holds never filled -> sell-route (cargo_frac>=0.8) never fired.
#
# FIX applied before launch: make the ORE the reward, not the shot.
#   asteroid_hit_miner 3.5->0.5 (shot is just the means to release ore)
#   pickup_reward_miner 0.8->1.2 (collecting ore pays)
#   cargo_sold_miner 1.5 (the sale is the payoff)
#   bc_coeff 0.90->0.05 (SMALL anchor: rewards drive behavior; BC only faintly nudges
#     toward the expert's sell-when-full route. High BC made the policy clone the
#     shoot-happy expert and fought the reward rebalance.)
#
# This daemon HOLDS those weights and watches the loop emerge:
#   collecting (pickup up) -> holds fill -> selling (cargo_sold up).
# Escalate the ECONOMIC levers (not BC) if selling still lags; back off only if
# mining stops entirely. Every decision + numbers -> the md log below.
# Launch detached:  nohup scripts/miner_profit_v2.sh >/dev/null 2>&1 & disown
set -uo pipefail
cd /home/dlibland/dev/avian_space
PY=.venv/bin/python
CFG=training_config.yaml
LOG=experiments/run_4/miner_profit_v2.md
TRAINLOG=logs/rl_minersell_2026-06-22.log
STDLIB="$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"
export LD_LIBRARY_PATH="$PWD/target/release/deps:$STDLIB"

SELL_THRESH=0.0002      # miner cargo_sold/step above this == "selling established"
COLLECT_THRESH=0.0006   # miner pickup/step above this == "collecting well"
MINE_DEAD=0.00015       # asteroid_hit AND pickup both below this == "stopped mining"
CARGO_SOLD_CAP=3.0
PICKUP_CAP=2.5
SETTLE_CYCLES=30
INTERVAL=1500           # ~25 min between decisions
STALL_LIMIT=3           # settled checks with no selling before escalating an economic lever

log(){ printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
getcfg(){ grep -E "^  $1:" "$CFG" | awk '{print $2}'; }
setcfg(){ sed -i "s/^  $1: .*/  $1: $2/" "$CFG"; }
gt(){ awk "BEGIN{exit !($1>$2)}"; }

metrics(){  # cargo_sold asteroid_hit pickup landing total nav ev burnin
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
    print(f"{L('reward_per_step/miner/cargo_sold'):.6f} {L('reward_per_step/miner/asteroid_hit'):.6f} "
          f"{L('reward_per_step/miner/pickup'):.6f} {L('reward_per_step/miner/landing'):.6f} {tot:.6f} "
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

stall=0
log ""
log "## miner profit-loop v2 daemon started (pid $$)"
log "weights: asteroid_hit_miner=$(getcfg asteroid_hit_miner) pickup_reward_miner=$(getcfg pickup_reward_miner) cargo_sold_miner=$(getcfg cargo_sold_miner) bc_coeff=$(getcfg bc_coeff)"
log "targets: cargo_sold>$SELL_THRESH (selling), pickup>$COLLECT_THRESH (collecting). Apply the fix restart now."
restart

while true; do
  sleep "$INTERVAL"
  if ! pgrep -x avian_space >/dev/null; then log "process DEAD -> restarting"; restart; fi
  read -r cs ah pk ld tot nav ev bi <<< "$(metrics)"
  cyc=$(grep -c '\[ppo\] cycle' "$TRAINLOG" 2>/dev/null || echo 0)
  log "check bc=$(getcfg bc_coeff) ah_w=$(getcfg asteroid_hit_miner) pk_w=$(getcfg pickup_reward_miner) cs_w=$(getcfg cargo_sold_miner) cyc=$cyc stall=$stall | miner: cargo_sold=$cs pickup=$pk asteroid_hit=$ah landing=$ld total=$tot | nav=$nav EV=$ev burnin=$bi"

  if [ "$cs" = "nan" ]; then log "  tb read failed (nan) -> skip"; continue; fi
  if [ "${cyc:-0}" -lt "$SETTLE_CYCLES" ]; then log "  not settled ($cyc<$SETTLE_CYCLES) -> wait"; continue; fi

  # back-off: mining collapsed entirely (not shooting AND not collecting)
  if ! gt "$ah" "$MINE_DEAD" && ! gt "$pk" "$MINE_DEAD"; then
    ah_w=$(getcfg asteroid_hit_miner); newah=$(awk "BEGIN{printf \"%.2f\", $ah_w+0.5}")
    log "  MINING STALLED (asteroid_hit=$ah & pickup=$pk both < $MINE_DEAD) -> raise asteroid_hit_miner $ah_w -> $newah + restart"
    setcfg asteroid_hit_miner "$newah"; restart; continue
  fi

  # success: selling established. NOTE: bc_coeff is small (0.05) so rewards drive
  # behavior; we do NOT require high nav-agreement — the policy SHOULD diverge from
  # the (shoot-happy) BC expert toward the reward-optimal collect->sell loop, so a
  # falling nav-agreement is a sign the rewards are reshaping behavior, not a failure.
  if gt "$cs" "$SELL_THRESH"; then
    log "  >>> SELLING ESTABLISHED (cargo_sold=$cs > $SELL_THRESH, pickup=$pk, nav=$nav) at these weights."
    log "## DONE: miner mine->collect->sell loop online. cargo_sold=$cs pickup=$pk asteroid_hit=$ah landing=$ld nav=$nav"
    break
  fi

  # not selling yet
  stall=$((stall+1))
  if [ "$stall" -lt "$STALL_LIMIT" ]; then
    log "  not selling yet (stall $stall/$STALL_LIMIT) -> hold weights, keep watching (collecting pickup=$pk vs target $COLLECT_THRESH)"
    continue
  fi
  stall=0
  # escalate an ECONOMIC lever: if collecting is weak, raise pickup; else raise cargo_sold (holds fill but sale not worth a trip)
  if ! gt "$pk" "$COLLECT_THRESH"; then
    pk_w=$(getcfg pickup_reward_miner)
    if gt "$pk_w" "$PICKUP_CAP"; then log "  pickup_reward at cap & still not collecting -> FLAG: likely structural (fill threshold / collection range), holding"; continue; fi
    newpk=$(awk "BEGIN{b=$pk_w+0.5; if(b>$PICKUP_CAP)b=$PICKUP_CAP; printf \"%.2f\",b}")
    log "  collecting weak (pickup=$pk<$COLLECT_THRESH) -> raise pickup_reward_miner $pk_w -> $newpk + restart"
    setcfg pickup_reward_miner "$newpk"; restart
  else
    cs_w=$(getcfg cargo_sold_miner)
    if gt "$cs_w" "$CARGO_SOLD_CAP"; then log "  cargo_sold at cap, collecting OK but not selling -> FLAG: likely fill-threshold too high (rebuild needed), holding"; continue; fi
    newcs=$(awk "BEGIN{b=$cs_w+0.5; if(b>$CARGO_SOLD_CAP)b=$CARGO_SOLD_CAP; printf \"%.2f\",b}")
    log "  collecting OK (pickup=$pk) but not selling -> raise cargo_sold_miner $cs_w -> $newcs + restart"
    setcfg cargo_sold_miner "$newcs"; restart
  fi
done
log "daemon exiting (pid $$)"
