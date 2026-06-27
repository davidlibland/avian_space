#!/usr/bin/env bash
# Teach-then-handoff for the cooperation behaviors (run_4), mirroring the
# economic-loop curriculum (scripts/miner_fill.sh).
#
# Rationale: self-defense (#1) lives in the BC EXPERT, so we bumped bc_coeff to
# 0.30 to teach it (cold-start). But BC only pins the policy to a crude rule-based
# "face & fire"; once merchants reliably fight back, RL — which directly targets
# the rewards — should optimize PAST the expert. So:
#   TEACH   : hold bc=0.30 until merchants meaningfully fight back
#             (miner+trader ship_hit clearly positive = self-defense learned).
#   HANDOFF : ramp bc DOWN (-0.05/check, floor 0.10), verifying the learned
#             behaviors PERSIST (and ideally improve — RL beating BC). If defense
#             regresses, step bc back up one (= min bc that holds it).
# The assist reward (#2) is reward-driven and needs no bc, so it keeps shaping
# fighters throughout. Logs every decision to the md below.
# Launch detached:  nohup scripts/coop_handoff.sh >/dev/null 2>&1 & disown
set -uo pipefail
cd /home/dlibland/dev/avian_space
export DIAG_LANDINGS=1
export DIAG_COOP=1
PY=.venv/bin/python
CFG=training_config.yaml
LOG=experiments/run_4/coop_handoff.md
TRAINLOG=logs/rl_minersell_2026-06-22.log
export LD_LIBRARY_PATH="$PWD/target/release/deps:$(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"

DEF_THRESH=0.0002       # combined miner+trader ship_hit above this == merchants fight back (vs ~0 baseline)
BC_STEP=0.05; BC_FLOOR=0.10
SETTLE_CYCLES=30
INTERVAL=1500           # ~25 min between decisions
TEACH_GRACE=3           # settled checks with defense established before HANDOFF
TEACH_CAP=12            # if still not established after this many settled TEACH checks (~5h) -> FLAG, keep teaching

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

metrics(){  # echoes: merchant_ship_hit focus_fire threat_resp nav
  "$PY" - <<'PYEOF'
import re, statistics as st
import numpy as np
try:
    from tbparse import SummaryReader
    df=SummaryReader("experiments/run_4/tb").scalars
    def L(t,k=10):
        v=df[df['tag']==t]['value'].values; return float(np.mean(v[-k:])) if len(v) else float('nan')
    deff=L('reward_per_step/miner/ship_hit')+L('reward_per_step/trader/ship_hit')
    nav=L('policy/bc_agreement/nav')
except Exception:
    deff=nav=float('nan')
ff=[]; tr=[]
try:
    for l in open("logs/rl_minersell_2026-06-22.log", errors='ignore'):
        if '[COOP] sys=escort' not in l: continue
        m=re.search(r'focus_fire=([\d.naN]+).*threat_resp=([\d.naN]+) merch_def', l)
        if m:
            for s,arr in ((m.group(1),ff),(m.group(2),tr)):
                try:
                    v=float(s)
                    if v==v: arr.append(v)
                except Exception: pass
    ff=ff[-300:]; tr=tr[-300:]
except Exception: pass
print(f"{deff:.6f} {st.mean(ff) if ff else float('nan'):.3f} {st.mean(tr) if tr else float('nan'):.3f} {nav:.3f}")
PYEOF
}

PHASE=TEACH; teach_ok=0; teach_total=0
log ""
log "## coop teach-then-handoff daemon started (pid $$)"
log "bc=$(getcfg bc_coeff) assist=$(getcfg cooperative_assist_bonus) | TEACH until merchant ship_hit(miner+trader) > $DEF_THRESH (self-defense learned), then HANDOFF ramp bc -> $BC_FLOOR verifying persistence (RL should beat BC)."

while true; do
  sleep "$INTERVAL"
  if ! pgrep -x avian_space >/dev/null; then log "process DEAD -> restart"; restart; fi
  read -r def ff tr nav <<< "$(metrics)"
  cyc=$(grep -c '\[ppo\] cycle' "$TRAINLOG" 2>/dev/null || echo 0)
  bc=$(getcfg bc_coeff)
  log "check phase=$PHASE bc=$bc cyc=$cyc | merchant_ship_hit=$def focus_fire=$ff threat_resp=$tr nav=$nav"
  [ "$def" = "nan" ] && { log "  no data -> skip"; continue; }
  [ "${cyc:-0}" -lt "$SETTLE_CYCLES" ] && { log "  not settled ($cyc<$SETTLE_CYCLES) -> wait"; continue; }

  established=0; gt "$def" "$DEF_THRESH" && established=1

  if [ "$PHASE" = "TEACH" ]; then
    teach_total=$((teach_total+1))
    if [ "$established" = "1" ]; then
      teach_ok=$((teach_ok+1))
      if [ "$teach_ok" -ge "$TEACH_GRACE" ]; then
        log "  >>> SELF-DEFENSE ESTABLISHED (merchant ship_hit=$def > $DEF_THRESH for $teach_ok checks). Begin HANDOFF: ramp bc DOWN so RL optimizes past the expert."
        PHASE=HANDOFF
      else
        log "  defense emerging (established $teach_ok/$TEACH_GRACE, ship_hit=$def) -> hold bc"
      fi
    else
      teach_ok=0
      if [ "$teach_total" -ge "$TEACH_CAP" ]; then
        log "  !! merchants still not fighting back after $teach_total checks (ship_hit=$def) -> FLAG: may need higher bc, a stronger self-defense rule, or more escort/mining worlds. Holding bc=$bc, keep watching."
      else
        log "  not fighting back yet (ship_hit=$def <= $DEF_THRESH, $teach_total/$TEACH_CAP) -> hold bc=$bc, keep teaching"
      fi
    fi
    continue
  fi

  # PHASE = HANDOFF: ramp bc down while defense persists
  if ! gt "$def" "$(awk "BEGIN{print $DEF_THRESH*0.6}")"; then
    newbc=$(awk "BEGIN{printf \"%.2f\", $bc+$BC_STEP}")
    log "  defense REGRESSED during handoff (ship_hit=$def) -> raise bc back to $newbc (min bc that holds it) + restart -> DONE"
    setcfg bc_coeff "$newbc"; restart
    log "## DONE: min bc_coeff that holds self-defense = $newbc (merchant ship_hit=$def, focus_fire=$ff, threat_resp=$tr)"
    break
  fi
  if ! gt "$bc" "$BC_FLOOR"; then
    log "## DONE: cooperation+defense sustained down to bc floor $BC_FLOOR — RL drives it (ship_hit=$def focus_fire=$ff threat_resp=$tr)"
    break
  fi
  newbc=$(awk "BEGIN{b=$bc-$BC_STEP; if(b<$BC_FLOOR)b=$BC_FLOOR; printf \"%.2f\",b}")
  log "  defense holding (ship_hit=$def, focus_fire=$ff, threat_resp=$tr) -> lower bc $bc -> $newbc + restart (let RL optimize past BC)"
  setcfg bc_coeff "$newbc"; restart
done
log "daemon exiting (pid $$)"
