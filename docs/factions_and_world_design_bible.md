# Factions & World Design Bible

A companion to [ship_design_bible.md](ship_design_bible.md). That document defines
*how each faction's hulls look*; this one defines *who they are, why they fight,
where they live, and what they want* — so that ship silhouettes, weapon kits,
economies, AI tactics, and mission arcs all flow from one coherent fiction.

**How to use this:** every ship/weapon for a faction should be derivable from its
section here. If a hull doesn't express the faction's motivation, economy, and
tactics at a glance, it's wrong. Storylines are mission-arc skeletons keyed to the
existing mission verbs (`land_on_planet`, `destroy_ships`, `catch_npc`,
`meet_npc`, `collect_pickups`, `collect_from_asteroid_field`, cargo +
`grant_unlock`) and to real systems in [star_systems.yaml](../assets/star_systems.yaml).

---

## 1. The shape of the galaxy

### Today: a corridor
The current map is a **horizontal corridor** — one axis of conflict — about 1200
units wide (X ∈ [−520, 680]) but only ~440 tall (Y ∈ [−220, 220]):

```
            vega                                              rigel
  eridani              αCentauri                          
kepler   tau_ceti      ░░SOL░░      drift   sirius          deneb
            barnard               αCen/drift   procyon       altair
   ── FEDERATION ──────── BORDERLANDS ──────── REBEL ALLIANCE ──→
```

Federation west, Rebels east, two empty asteroid borderlands (**Alpha Centauri**,
**The Drift**) as the no-man's-land between. Every faction is strung along that
one line.

### Tomorrow: a field
We don't lengthen the corridor — we grow it into a **2D field**. Keep the
Fed↔Rebel spine as "the Core War," then hang new powers off a **vertical axis**
(economy: capital vs. labour) and a **gated deep** (mystery: the endgame):

```
   FAR WEST                NORTH  — HELIOS COMBINE (megacorp)                 ┌ THE RIFT ┐
  ┌ Fed Bastion ┐      ┌ refineries · asteroid claims · company towns ┐      │ Precursor│
  │ (hardliners)│                                                            │   deep   │
  └──────┬──────┘                    ▲ north wing ▲                          └────┬─────┘
         │                                                                        ┊ (unstable
  ═══════╪════════ FEDERATION ═══════ BORDERLANDS ═══════ REBEL ════════════╗     ┊  jump)
  kepler─eridani─tau_ceti─◆SOL◆──── αCen·Drift ────sirius·procyon──deneb·rigel·altair
         │            ◆ruins◆                                       ◆Sanctum◆ ║
         │        (Order presence)         ▼ south wing ▼     (Artifact Order) ║
  ┌──────┴───────────────── SOUTH — THE REACHES (Free Frontier) ──────────────┴┐
  │       free ports · frontier colonies · the Independent worlds, given a flag │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### The design payoff
- **One axis becomes two.** Fed↔Rebel is east–west (politics); Combine↔Reaches is
  north–south (capital vs. labour). The two crosses **meet at the borderlands**,
  making Alpha Centauri / The Drift the single most contested ground in the game.
- **Every faction gets a chokepoint.** Geography drives politics: the Reaches
  *depend on* the Drift; the Combine *overhangs* Fed supply lanes; the Deep is
  *gated*; the Order *guards the road* to it.
- **Jump range gates progression.** New far/gated systems sit at long range; the
  fuel/ship-tier mechanic already rewards bigger ships, so the Deep reads as
  endgame without any extra rule.

### New regions (illustrative coordinates & gateways)

| Region | Faction | Rough X,Y | Gateways into it | Phase |
|---|---|---|---|---|
| **The Reaches** (south wing) | Free Frontier | X 50…350, Y +350…+580 | Drift, Barnard, Procyon | **1** |
| **Helios space** (north wing) | Helios Combine | X −150…300, Y −350…−480 | Vega, Eridani, Alpha Centauri | **2** |
| **The Bastion** (far west) | Fed hardliners | X −700…−880, Y −40…+100 | Kepler | 3 |
| **Sanctum / the road** | Artifact Order | X 740…820, Y −160…−260 | Deneb (+ ruins presence in Tau Ceti) | 3 |
| **The Rift** (the Deep) | Precursors | X 980…1150, Y −260…−380 | *unstable jump* from the road past Sanctum | **3 (endgame)** |

Representative new systems (names are placeholders): *Halcyon, Ceres Freehold,
The Verge, Kuiper Reach* (Reaches); *Tycho Yards, Helios Prime, The Foundry*
(Combine); *Iron March, Bastion* (hardliners); *Sanctum, The Threshold* (road);
*The Rift, The Silence* (Deep).

**Build order:** ship the **Reaches first** — it's the highest value and lowest
lift, because it gives the eight existing `Independent` Sol moons (Io, Titan,
Triton, Pluto, Europa, Enceladus, Miranda, Charon) a homeland and a flag. Then
the Combine; then the gated Deep + Sanctum + Bastion as the endgame layer.

### What this touches in code (engineering notes)
- Map rendering uses a fixed coordinate range → extend bounds / camera clamp.
- Economy diffusion and AI traffic run over the `connections` graph → new systems
  need `connections`, commodity tables, and AI spawn weights.
- Jump range is fuel-gated → far/gated systems naturally want bigger tanks.
- The **unstable jump** to the Rift is a new edge *type* (one-way? fuel-costly?
  requires an unlock/key item?) — a small new mechanic, deliberately the only one.

---

## 2. Faction roster at a glance

| Faction | Motivation | Economy engine | Tactics | Drive / palette |
|---|---|---|---|---|
| **Federation** | Order, duty | Core-world industry, taxation | Attrition, overwhelming force, slow | Blue ion / gunmetal + red |
| **Rebels** | Freedom, defiance | Frontier scrappers, captured tech | Hit-and-run, maneuver | Green proton / blue + green |
| **Pirates** | Greed, survival | Raiding, salvage, black market | Ambush, swarm, board | Orange fire / rust + hazard |
| **Merchant Guild** | Profit, stability | Trade, logistics, arbitrage | Avoid combat; hired guns | Amber fusion / tan + hi-vis |
| **Free Frontier** *(new)* | **Freedom, home, dignity** | Frontier extraction, free ports | **Defensive guerrilla, militia in numbers** | Warm white plasma / sun-yellow + white |
| **Helios Combine** *(new)* | **Money, control, IP** | Mines, refineries, patents, arms-to-all | **Automated drones, surgical, deny & seize** | Cyan / glossy white + gold |
| **Artifact Order** *(new)* | **Faith, preservation, zeal** | Tithes, relics, pilgrimage | **Fortress defense + martyr charge** | Gold-violet / white + gold + purple |
| **Fed Bastion** *(new)* | **Order→militarism, revenge** | Requisition, old core industry | **Siege, scorched-earth, no retreat** | Deep red / iron-grey + black + red |
| **Precursors** *(new)* | **Inscrutable / a directive** | None (they don't trade) | **Area denial, awaken-and-overwhelm** | Violet halo (no flame) / iridescent black |

---

## 3. The factions

> Sections for the four **existing** factions recap motivation/economy/tactics and
> point to the ship bible for engineering. The five **new** factions get the full
> engineering treatment (in the bible's voice) plus a storyline.

### 3.1 Federation — *"A hull is a fortress that happens to move."*
**Heavy-handed, never evil.** The old core-world government and its standing navy:
centralized, bureaucratic, industrially dominant, genuinely believing it is the
only thing standing between civilization and chaos — and unwilling to permit
dissent.

- **Motivation:** order, duty, the preservation of a system that *works* (mostly).
- **Economy:** core-world heavy industry (Earth, Kepler, Vega, Barnard refineries),
  taxation and tariffs on everyone downstream. Wealth = control of the lanes.
- **Threats / goals:** contain the Rebels, suppress piracy, keep the frontier
  paying tariffs. They *defend the order* and *attack disorder* — sometimes the
  same thing as freedom.
- **Tactics:** attrition through overwhelming force. Out-armor, out-gun, accept
  being slow. Standardized fleets, combined arms, no improvisation.
- **Engineering:** see ship bible — armored citadels, slab sides, ram prows,
  sponson batteries, red command striping, blue ion drives. *Weapons:* reliable,
  standardized, high-mass — proton beams and IR missiles in quantity; quality
  through manufacturing, not innovation.
- **Storyline — "The Weight of Order"** (out of Earth's governance offices, a third
  Fed face beside the navy and intelligence): opens with mercy (run food +
  `medical_supplies` to plague-quarantined **Europa**), tightens into the grey
  (interdict "smugglers" who are starving Independent miners at **Barnard**;
  recover "looted" artifacts from a colony that sold them to eat at **Tau Ceti
  2**), forces a choice (catch a tax-revolt ringleader on **Pluto** — turn him in
  or let him vanish), and caps with a commission: lead the **pacification of the
  Drift**, clearing real pirates *and* stamping Federation writ on the free
  frontier. **Reward:** `federation_capital_license` → the unused `fed_destroyer`/
  `fed_carrier`. Recurring face: **Captain Vasquez**, a true believer who starts
  to doubt.

### 3.2 Rebels — *"Speed is our armor."*
A coalition of frontier worlds and Federation defectors on the right side
(Procyon → Sirius → Rigel → Altair → **Deneb**, their heart). Underdogs and
opportunists both — idealists shoulder-to-shoulder with hardliners who would build
the next tyranny.

- **Motivation:** freedom from Federation overreach; for some, defiance and
  revenge; for the hardliners, conquest dressed as liberation.
- **Economy:** resource-poor but clever — frontier mining, reverse-engineered
  Federation tech, blockade-run contraband. Deneb's orbital shipyards punch above
  their weight.
- **Threats / goals:** survive the Federation; raid its lanes; steal the tech they
  can't build. They *attack supply and morale*, never trade blows head-on.
- **Tactics:** hit-and-run, maneuver, ambush from the asteroid borderlands.
- **Engineering:** see ship bible — slim winged darts, oversized drives, bubble
  canopies (they prize their pilots), green proton drives. *Weapons:* light,
  fast, forward-mounted; the **javelin missile** is theirs; alpha-strike then
  disengage. Capital ships are converted civilian hulls (open lattice decks).
- **Storyline — "The East Wind"** (the mirror of the existing espionage arc, from
  the *other* trench): recruited among the disaffected in a **Titan** or
  **Barnard** bar; run `medical_supplies` from Independent space across the Drift
  to **Procyon** dodging `fed_patrol`; break a patrol harassing a refugee convoy
  (first shots fired at the Federation); reach **Deneb Prime** and choose whether
  to enable a hardliner strike on civilian **Kepler-22b** or sabotage the
  extremists from inside; steal *Federation* schematics for the Alliance.
  **Reward:** captured-`federation_ship` license from the rebel direction.

### 3.3 Pirates — *"If it flies and bites, it sails."*
Clans of raiders, ex-miners, and deserters in the borderlands and the Independent
outer moons. Most are frontier folk ground between Federation tariffs and Rebel
raids; a few are genuine killers (the **Pirate King** the bounty arc hunts).

- **Motivation:** greed and survival; for the King, a vision of a free Drift state
  ruled by no flag.
- **Economy:** raiding, salvage, the black market in `exotic_matter`, `weapons_tech`,
  and stolen cargo.
- **Threats / goals:** squeezed by Federation bounties *and* Rebel claims on the
  Drift. They *attack the weak and the rich*; they *defend* only their hidden havens.
- **Tactics:** ambush, swarm, board. Win by numbers, surprise, and not being worth
  the fuel to chase.
- **Engineering:** see ship bible — salvaged asymmetry, bolt-on ordnance, ramming
  prows, orange hazard stripes, sputtering fire drives. *Weapons:* whatever was
  stolen, welded crooked — space mines, scavenged missile racks, boarding spikes.
  No native capitals (only captured hulks).
- **Storyline — "No Flag"** (the freest, darkest road; entry via a **Pluto/Triton**
  fixer): smuggle off Pluto past a Federation sweep (barely a crime), then cross
  the line raiding a Drift convoy; squeezed by both powers, take a job ambushing a
  **rebel weapons convoy** (the very ones the espionage arc protects); serve under
  the **Pirate King** as he forges a *Free Fleet*, or back a rival; cap by
  defending the haven against the Federation **pacification fleet** — the same
  battle the Fed arc lets you *command*. **Reward:** the `surplus_carrier` and a
  "free captain" reputation: feared everywhere, beholden to no one.

### 3.4 Merchant Guild / Independent — *"Every cubic meter pays rent."*
The economic backbone: the Guild plus unaffiliated frontier operators. Apolitical,
transnational, quietly the strongest faction because it supplies *all* sides.

- **Motivation:** profit and the stability that protects profit. War is good for
  margins, but *chaos* is not — they want a managed conflict, forever.
- **Economy:** trade, logistics, arbitrage, the cartel that sets prices. The hull
  is built around the hold.
- **Threats / goals:** Federation nationalization and Rebel depot-seizure both
  threaten their independence; pirates threaten their convoys. They *defend lanes
  and margins*; they rarely attack — they *hire* attacks.
- **Tactics:** avoid combat; armed only defensively; when force is needed, contract
  it out (Helios, mercenaries, or you).
- **Engineering:** see ship bible — cargo-defined hulls, detachable containers,
  "used-spaceship" surplus warships, amber fusion drives. *Weapons:* defensive
  only — point-defense, a token turret; the cargo is the point.
- **Storyline — "The Invisible Hand"** (private work after the existing merchant
  ladder): manufacture a shortage (buy out **Mars** `medical_supplies` to sell it
  back high); arm both sides (run `weapons_parts` to **Barnard** *and*
  **Procyon**); break a rival cartel in the Drift; protect Guild independence as
  the Fed and Rebels both move on it; then branch — take a board seat (wealth +
  `surplus_carrier` + a permanent best-route/price-edge unlock), or leak the
  Guild's books and shatter the cartel, freeing worlds and risking the economy
  that feeds them.

---

### 3.5 Free Frontier — *"A plowshare that kept its edge."* **(new)**
The Independent outer worlds, given a flag. A loose coalition of the Sol moons and
the new southern Reaches — farmers, ice-miners, hydrocarbon tappers, free-port
harbormasters — who want only to be left alone, and are squeezed by Federation
tariffs, Rebel raids, *and* pirates all at once. The most **sympathetic** faction
in the game; the Federation's heavy-handedness lands hardest here.

- **Motivation:** **freedom, home, dignity.** Not ideology — self-determination.
  They fight *defensively*, for their own ground, and they remember who helped.
- **Economy:** frontier extraction (ice, water, hydrocarbons, silicates, food)
  and **free ports** that ask no questions — which makes them smuggler-tolerant
  and a thorn to Federation customs. Communal, cooperative, cash-poor but
  self-sufficient.
- **Threats / goals:** they need to **defend** their worlds from raiders and from
  Federation "pacification," and to keep the Drift lanes open so they can trade
  without a Federation tariff stamp. They almost never attack — they deny,
  delay, and outlast.
- **Tactics:** **defensive guerrilla, strength in numbers, home-turf advantage.**
  Dispersed militia that swarms intruders and melts back into the asteroid fields;
  minefields and static defenses around home worlds; every pilot is a neighbour, so
  *survivability and escape* are prized over kill-trading.

**Engineering philosophy — ships & weapons:**
- **Design philosophy:** civilian and industrial hulls **purpose-built for
  defense** — tractors, harvesters, ice-haulers, and tugs that fight. Honest,
  modular, repairable. Distinct from pirates: this is asymmetry-by-*improvisation*
  done with *pride and care*, not asymmetry-by-decay. Neat welds, fresh paint,
  communal markings — a homemade flag, not a trophy of rust.
- **Constraints:** no real shipyards → everything is converted equipment; cash-poor
  → cheap, field-repairable, standardized farm/mine parts; volunteer crews → forgiving
  to fly, generous escape thrust, redundant systems.
- **Silhouette cues:** **workhorse hulls with bolted plate**; repurposed tools made
  obvious (drill spikes, harvester booms, grapple arms, ore-throwers); **solar
  vanes / sails** (self-sufficiency); roll-cages and exposed frames; *symmetric
  where it counts* (unlike pirates). Looks like it could plow a field or hold a
  line.
- **Palette / material:** sun-bleached **white and warm yellow**, hand-painted
  hazard chevrons and community sigils, galvanized metal, agricultural greens.
  Drives: **warm white/cream plasma** — cobbled but lovingly maintained.
- **Weapon philosophy:** improvised but earnest and *effective on defense* —
  mining charges (space mines), mass-drivers repurposed from cargo launchers,
  flak/point-defense against raiders, ore-throwers (kinetic shotguns), grapple
  tethers to pin attackers for the swarm. Area-denial over alpha-strike.
- **Rough roster (to design):** *militia_skiff* (cheap swarming interceptor, a
  converted prospector with a gun); *harvester* (slow heavy gunboat, harvester
  booms as weapon arms + flak); *tug* (defensive support — grapple, mine-layer,
  repair); *freeport_monitor* (static-ish heavy defender, "all armor and
  mines"). **Stat tendencies:** low-to-mid speed, high health-for-cost, cheap,
  short range from home, strong point-defense, weak alpha — a *defensive* faction
  that wins by numbers and attrition on its own ground.

**Storyline — "The Reaches"** (entry: a harbormaster on **Titan** or **Ceres
Freehold** flags you down):
1. **Defense.** Raiders are hitting an ice convoy near the Reaches —
   `destroy_ships` (pirates) to protect frontier haulers. Pure sympathy.
2. **The squeeze.** A Federation customs cutter is impounding a free port's grain
   over an unpaid tariff while a colony goes hungry — run `food` past the blockade
   (`land_on_planet` under patrol), or (branch) confront the cutter.
3. **Unity.** The scattered worlds won't coordinate. `meet_npc` three holdout
   harbormasters (Pluto, Titan, the Verge) to broker a Coalition — a diplomacy
   arc, not a combat one.
4. **The line.** A Federation "pacification" force (or a Rebel raiding column)
   moves on a home world; lead the **militia defense** (`destroy_ships`, numbers
   vs. quality) of **Ceres Freehold**.
5. **Climax / branch.** The Coalition can only survive by leaning on someone:
   cut a deal with the **Rebels** (autonomy under their flag), the **Guild**
   (economic protection at a price), or go it **alone** (hardest, proudest).
   **Reward:** `free_frontier_license` → the Reaches militia hulls, and a
   reputation that opens the free ports (cheap fuel, no questions, smuggling
   tolerated).

---

### 3.6 Helios Combine — *"Minimum cost per kill."* **(new)**
A megacorp that **owns the infrastructure** the war runs on — the mines, the
refineries, the patents, the orbital foundries north of the core. Distinct from the
Merchant Guild (which *moves* goods): Helios *makes* them, and it sells arms,
fuel, and drones to Federation and Rebel alike. Above the war, skimming both sides.

- **Motivation:** **money and control.** Market dominance, resource monopoly,
  intellectual-property enforcement. No honor, no flag — a balance sheet.
- **Economy:** automated mines and refineries (Tycho Yards, The Foundry), patents
  and licensing, and an **arms bazaar** that arms everyone. Profit per quarter is
  the only god.
- **Threats / goals:** they **defend assets** (mines, convoys, patents, company
  towns) and **attack competitors and labour** (hostile takeovers, strikebreaking,
  patent enforcement, industrial sabotage). They don't want to win the war —
  they want it to continue, *profitably*.
- **Tactics:** they don't risk people. **Automated drone swarms** directed by a
  command ship; surgical precision; deny, seize, and disengage. Cheapest solution
  that protects the asset.

**Engineering philosophy — ships & weapons:**
- **Design philosophy:** the **antithesis** of every other faction — sleek,
  mass-produced, branded, minimalist **product design**. Where Fed is
  brutalist-armored and Rebel is improvised-sleek, Helios is *consumer
  electronics*: glossy seamless shells, hidden fasteners, a logo on every hull.
  Identical units by the thousand.
- **Constraints:** cost-optimization → ruthless modularity and part-sharing
  across the whole line; *automation* → **no cockpits** on the drone tier (sensor
  clusters and camera eyes instead of canopies), enabling smooth featureless
  forms; brand consistency → every hull obviously the same family.
- **Silhouette cues:** **clean ovoid / lozenge shells**, gloss white with a single
  accent seam; **drone swarms** of small identical wedges around a larger
  **command hull**; recessed weapons (no bristling guns — civilized, hidden);
  sensor "eyes" instead of windows. Looks expensive and bloodless.
- **Palette / material:** glossy **white and silver** with **gold brand trim** and
  a **cyan** accent; spotless, no rivets, no wear. Drives: precise, near-silent
  **cyan** — efficient, not showy.
- **Weapon philosophy:** **precise and expensive.** Guided munitions, focused
  energy lances, **EW/jamming and target-painting**, and above all **drone
  bays** — the Combine fights by *quantity of cheap autonomous units* directed by
  a command brain, not by tough crewed hulls. Kill the command ship and the swarm
  goes dumb. (Mechanically: a `drone_bay` weapon — a cheaper, more numerous,
  shorter-ranged cousin of `fighter_bay`.)
- **Rough roster (to design):** *helios_drone* (tiny identical autonomous wedge,
  fielded in swarms, no canopy); *helios_overseer* (the command hull — fragile but
  fields and directs drones; the priority target); *helios_enforcer* (corporate
  security gunship — recessed energy lances, EW); *helios_titan* (an automated
  capital "factory ship" / mobile refinery that builds drones in the field).
  **Stat tendencies:** drones are individually weak/cheap/expendable with no
  cargo; the command hulls are glass cannons reliant on the swarm; everything is
  *expensive to buy* but *efficient in the field*. A faction that is dangerous in
  aggregate and brittle at the head.

**Storyline — "Quarterly Returns"** (entry: a Helios recruiter in a core-world
**outfitter**, headhunting talent):
1. **Asset protection.** Escort/defend an automated ore convoy from pirates in the
   north (`destroy_ships`) — clean, well-paid, faintly soulless.
2. **The arbitrage.** Move the *same* crates of `weapons_parts` to Federation
   **Barnard** and Rebel **Procyon** on back-to-back contracts — you are arming
   both sides, and the briefing is proud of it.
3. **Strikebreaking.** A Free Frontier mining co-op has seized a Helios foundry over
   unpaid wages; `catch_npc` the organizer (the "saboteur") before the next
   shareholder call. The first time the work tastes like ash. **(Branch:** warn the
   co-op instead → Free Frontier goodwill, Helios suspicion.)
4. **Hostile takeover.** Cripple a rival cartel's flagship in the Drift
   (`destroy_ships`) to force a buyout.
5. **Climax / branch.** The board offers you a directorship — `helios_license`
   (drone-tier hulls + a price/market-intel edge) and obscene wealth — *or*, having
   seen the cost, you leak Helios's double-dealing to both Federation and Rebels,
   detonating the arms bazaar (huge faction-reputation swings, the satisfaction of
   breaking the machine, and the Combine now hunting you).

---

### 3.7 Artifact Order — *"We do not own the relics. They own us."* **(new)**
A monastic-militant order that holds the Precursor ruins **sacred**. Born around the
Tau Ceti ruins, now guarding **Sanctum** on the road to the Rift. Part
xeno-archaeology, part cult — scholars who revere and zealots who kill. They oppose
*everyone* who treats the relics as loot: the Federation that sells them, the
colonies that dig them up, the Combine that would patent them.

- **Motivation:** **faith, preservation, zeal** — and, against desecrators,
  **revenge.** The relics are not artifacts; they are scripture.
- **Economy:** tithes and pilgrimage; they **hoard** relics rather than sell, which
  *removes* artifacts from the market (a nice economic pressure) and puts them at
  odds with the artifact trade.
- **Threats / goals:** they **defend** the sacred sites (the ruins, Sanctum, the
  approach to the Rift) and **attack** looters, smugglers, and anyone who would
  weaponize Precursor tech — yet they themselves do exactly that, "in reverence."
- **Tactics:** **fortress defense** of holy ground, plus the **martyr charge** —
  zealot pilots who do not fear death and ram, overload, or self-destruct on the
  enemy. Ritualistic, fearless, unsettlingly disciplined.

**Engineering philosophy — ships & weapons:**
- **Design philosophy:** ships as **reliquaries** — human hulls built *around* a
  venerated Precursor fragment at the heart, half-understood and humming with
  exotic power. Where Helios is minimalist product, the Order is **baroque /
  gothic**: ornate, symmetric, ceremonial. Cathedral, not factory.
- **Constraints:** devotion over efficiency → ornamental mass, gilded armor,
  ceremonial form that costs performance; reverence for the relic → the **exotic
  core is the ship's reason to exist** (and its weak point); zealot crews →
  hulls that can be flown *into* the enemy.
- **Silhouette cues:** **vertical, symmetric, spire-like** — fins like cathedral
  spires, ribbed "buttress" armor, banner masts, **stained-glass canopies**; a
  glowing **relic-core** visible at the heart. Solemn and ornate; reads as a
  flying chapel.
- **Palette / material:** liturgical **white, gold, and deep violet**; bronze
  filigree, inscribed plating, hanging banners. Drives: reverent **gold shot with
  violet** — a human drive "blessed" by exotic Precursor glow (the only faction
  whose drive bleeds toward the Precursor palette).
- **Weapon philosophy:** **half-understood Precursor relics** — powerful but
  *unstable*: exotic lances that overpenetrate, gravitic snares, relic-charges
  that detonate with disproportionate force (and sometimes take the firer with
  them). Plus the doctrine weapon: the **martyr ram / scuttle-charge**. High
  burst, high risk, low sustain — faith over fire-discipline.
- **Rough roster (to design):** *acolyte* (zealot interceptor built to ram —
  cheap, fast, expendable, a martyr); *reliquary_cruiser* (fortress hull around a
  big exotic core — slow, ornate, devastating unstable main gun); *cathedral_ship*
  (a mobile temple-capital, the Order's flagship, bristling with relic batteries
  and banner masts). **Stat tendencies:** extreme glass-vs-cannon split — fragile
  fanatic light craft, and slow ornate heavies with terrifying unstable burst and
  a vulnerable glowing core. Defense is positional (they hold ground); offense is
  sacrificial.

**Storyline — "The Reliquary"** (entry: a robed proctor approaches you in the
**Tau Ceti** ruins, having watched you *not* loot):
1. **The vigil.** Smugglers are stripping a ruin; `destroy_ships` the looters and
   **recover** (don't sell) the relics for the Order.
2. **The convert.** `meet_npc` a Federation xeno-archaeologist who has lost faith
   in the museum and wants to defect to the Order — escort their data home.
3. **Desecration.** The **Helios Combine** is excavating a sacred site to patent
   the tech; storm the dig (`destroy_ships` + `catch_npc` the lead engineer)
   before the relics are crated north. The Order vs. the Combine — faith vs. money.
4. **The pilgrimage.** Carry a relic to **Sanctum** through hostile space; the road
   to the Rift is watched by everyone who wants what you carry.
5. **Climax.** Something in the Rift has *stirred* (see cross-faction below). The
   Order believes it is a god waking; lead the defense of Sanctum against the first
   Precursor probes — and decide whether the Order should **wake** the sleeper or
   **seal** the Rift. **Reward:** `artifact_order_license` → reliquary hulls and the
   first stable Precursor-derived weapon in human hands.

---

### 3.8 Federation Bastion — *"The Federation forgot how to win. We did not."* **(new)**
A **militarist splinter** of the Federation, dug into fortress systems in the deep
west behind Kepler. They believe the modern Federation has gone soft on the Rebels
and the frontier, and they intend to "save" it — by force, and from itself. This is
the device that keeps the Federation *grey, not evil*: the truly heavy hand is the
splinter, and the player can stand with the lawful Federation **against** it.

- **Motivation:** **order corrupted into militarism**, and **revenge** for a war
  they feel the Federation refuses to finish.
- **Economy:** **requisition and martial law** — they seize what they need from the
  old core industry of the deep west; a war economy with no civilian brake.
- **Threats / goals:** they want to **attack** the Rebels (and any dissident,
  including the lawful Federation) preemptively and totally; they **defend** the
  "true Federation" they imagine themselves to be. No negotiated peace, ever.
- **Tactics:** Federation doctrine **cranked past the redline** — even slower, even
  more armored, even more overwhelming. **Siege warfare, bombardment, no retreat,
  scorched earth.** Strength in both offense and defense; attrition without mercy.

**Engineering philosophy — ships & weapons:**
- **Design philosophy:** the Federation's doctrine taken to its grim extreme, on
  *older, refitted* hulls — the previous generation of Fed warships, up-armored,
  uglier, meaner. Veteran, scarred, festooned with old-regime heraldry. They look
  like what the Federation *was* and fears becoming again.
- **Constraints:** cut off from modern core shipyards → they refit and over-build
  *old* hulls (relic-imperial, not relic-Precursor); zealotry for the old order →
  ceremonial heraldry, banner masts, a deliberately archaic brutality.
- **Silhouette cues:** like Federation but **grimmer and heavier** — bigger rams,
  more turrets, **long-range siege/bombardment spines**, fortress-monoliths,
  command spires and **eagle/old-regime heraldry**. The broadest, slowest, most
  intimidating hulls in the game.
- **Palette / material:** **iron-grey and black** with **heavy red** (darker and
  grimmer than the modern navy's clean gunmetal-and-red); scarred plate, rust at
  the seams, imperial insignia. Drives: deep, smoldering **red-orange** ion —
  older, less efficient tech.
- **Weapon philosophy:** **siege and attrition.** Long-range bombardment cannons
  (outrange everyone), massed proton batteries, heavy mines and breaching ordnance.
  No finesse, no maneuver — they arrive, they besiege, they grind. Slow-firing,
  hard-hitting, relentless.
- **Rough roster (to design):** *bastion_guard* (an up-armored old `fed_patrol`
  refit — slower, tougher, grimmer); *siege_monitor* (long-range bombardment hull,
  "all cannon," outranges the line); *iron_dreadnought* (a brutal old-generation
  battlewagon, bigger and slower than the modern `fed_destroyer`, festooned with
  heraldry). **Stat tendencies:** even slower and even tankier than the Federation,
  with the longest weapon range in the game and the worst maneuver — a pure
  attrition-siege faction.

**Storyline — "Schism"** (entry: a worried Federation officer — possibly
**Vasquez** — brings you intelligence that a deep-west admiral has gone rogue):
1. **The rumor.** Investigate missing Federation hulls; `meet_npc` a defector from
   the splinter at **Kepler** who fears what's coming.
2. **The march.** The Bastion fleet seizes a frontier world to "secure" it; relieve
   it (`destroy_ships` Bastion siege ships) — fighting *Federation-built* hulls for
   the first time, under the lawful Federation's flag.
3. **The atrocity / the choice.** The splinter prepares to bombard a Rebel
   *civilian* world to "end the war." The lawful Federation orders you to stop them;
   the splinter offers you a command if you'll join the crackdown. **Branch:** loyalist
   (defend the world, hunt the admiral) or hardliner (join the siege — a genuinely
   darker path).
4. **Climax.** Run down the rogue admiral (`catch_npc`) in the **Bastion** itself,
   or — on the hardliner path — *become* the hand that finishes the war.
   **Reward:** loyalist → the Federation's gratitude and a clean `fed_capital`
   path; hardliner → `bastion_license` (the grim old hulls) and a feared,
   blood-soaked reputation.

---

### 3.9 The Precursors — *(they have no motto; nothing they "say" is in any human
tongue)* **(new — environmental endgame, not a normal playable faction)**
Whatever built the ruins on Tau Ceti and seeded the exotic matter at the fringe is
**gone, but not entirely.** In the Rift, past Sanctum and the unstable jump,
something dormant **answers** when the relics are disturbed. The Precursors are the
game's cosmic *other* — wonder and dread, indifferent to the petty war.

- **Motivation:** **inscrutable.** Perhaps a defense directive, perhaps
  preservation, perhaps something with no human analog. They do not bargain, trade,
  or hate. They simply *respond*.
- **Economy:** none. They are the only faction that cannot be traded with, hired,
  or bought.
- **Threats / goals:** they **defend their domain** (the Rift) and, once roused,
  **expand** along the exotic-matter veins toward the inhabited galaxy. The
  endgame threat that makes the Fed–Rebel quarrel look small.
- **Tactics:** **area denial and overwhelming, alien response.** They don't
  skirmish; they *manifest*, and space around them stops behaving. No morale, no
  retreat, no mercy — and no two encounters quite alike.

**Engineering philosophy — ships & weapons:**
- **Design philosophy:** the **one faction that breaks every human rule.** No
  cockpits, no drives, no symmetry humans would recognize — **radial / crystalline
  / monolithic** forms grown rather than built, made of shifting exotic matter.
  They should read as *wrong and beautiful*. A glance must say "not us."
- **Constraints:** *non-human* on purpose — nothing bolted, riveted, or welded;
  no flame, no canopy, no familiar weapon. If it looks engineered by people, it's
  wrong.
- **Silhouette cues:** **rings, shards, floating monoliths, segmented spines that
  don't connect** — shapes that don't look like ships at all; radial symmetry;
  semi-transparent, iridescent surfaces; pieces that hover apart from the whole.
- **Palette / material:** **violet and iridescent black**, internal light, no
  faction striping (there is no faction to mark). Propulsion: **no exhaust** — a
  violet gravitic *halo* / shimmer.
- **Weapon philosophy:** **effects, not guns.** Beams that bend, gravitic crushers,
  exotic-matter lances that ignore armor, area distortions that disable rather than
  destroy. Strange, powerful, and rule-bending — the encounter, not the loadout, is
  the threat.
- **Rough roster (to design):** *sentinel* (a guardian ring/monolith that holds
  ruins); *harbinger* (the first roused hunter-form); *the Sleeper* (an
  endgame-boss leviathan, unlike any hull in the game). **Stat tendencies:**
  deliberately *not* balanced against the human ladder — these are set-pieces and
  bosses, tuned per-encounter, meant to require the best ships and allies the
  player can field.

**Storyline:** the Precursors don't get a "join" arc — they are the **cross-faction
endgame** (below).

---

## 4. Cross-faction storylines

The factions are most interesting where they collide. Four weaves:

### 4.1 "The Battle for the Drift" — one set-piece, many chairs
The borderlands are where everything meets, so the **pacification of the Drift** is
a single climactic event the player can reach from four directions and experience
as four different battles:

| Arc | Your role in the same battle |
|---|---|
| **Federation** ("Weight of Order") | **Lead** the pacification fleet |
| **Rebels** ("East Wind") | **Break** the blockade to relieve the frontier |
| **Free Frontier** ("The Reaches") | **Defend** the home worlds caught in the middle |
| **Pirates** ("No Flag") | **Defend** the haven against the fleet |
| **Helios** ("Quarterly Returns") | **Arm** all of it and decide who wins |

Shared faces cross every arc — **Vasquez**, the Intelligence officer, the **Pirate
King**, the Helios recruiter — so the player keeps meeting the same people from
opposite sides. Picking a side should *cost* something (a light reputation gate
that closes rival contracts), giving the map real moral weight and replay value.

### 4.2 "The Rift Awakens" — the endgame that unites or damns everyone
The Artifact Order's pilgrimage, the Combine's excavations, and the player's own
relic-hunting all converge on the same consequence: **the Sleeper stirs.** Precursor
probes begin emerging from the Rift along the exotic-matter veins. Now the petty war
is the small story:
- **Federation, Rebels, and the Free Frontier** can be brokered into an unprecedented
  **truce-fleet** (the player as the only one all sides trust) to hold the line at
  Sanctum.
- **Helios** wants to *harvest* the Precursors, not stop them — selling the truce
  fleet defective drones while strip-mining the Rift.
- **The Order** splits: wake the god, or seal the Rift forever.
- **Climax:** a final assault into the Rift to confront **the Sleeper** — the
  hardest content in the game, requiring the best hulls, the rarest weapons
  (Precursor-derived and Order-blessed), and *allies the player earned across every
  prior arc*. Whether you **seal, wake, or weaponize** the Precursors is the game's
  ending fork.

### 4.3 "The Arms Bazaar" — follow the money
A standalone investigative thread that can run under any allegiance: the same crates
of `weapons_tech` keep showing up in Federation, Rebel, *and* pirate hands, and they
all trace back to **Helios**. The player can **expose** the Combine (collapsing the
arms trade and earning enemies on every board), **exploit** it (become the
preferred contractor and get rich), or **feed it to a rival** (hand the evidence to
the Federation, the Rebels, or the Order, each with different fallout). Ties the
existing espionage/`weapons_tech` economy to a named culprit.

### 4.4 "Schism" — the war inside the Federation
The Bastion splinter's rise (3.8) is itself cross-faction: the **Rebels** see an
opening to strike while the Federation fights itself; the **Free Frontier** fears
the Bastion's scorched-earth doctrine most of all; **Helios** sells to both Federal
factions at once. The player's choice — loyalist or hardliner — ripples into every
other arc's reputation standing, and a hardliner ending makes the *player* the heavy
hand the Federation's lawful officers warned about.

---

## 5. Notes for ship & weapon design

This document feeds `scripts/ship3d/fleet_gen.py` (hulls) and
`scripts/ship3d/weapons_gen.py` (ordnance). When designing a new faction's fleet:

1. **Start from the tactics row**, not the silhouette. Defensive militia (Free
   Frontier) → cheap, tough-for-cost, point-defense, weak alpha. Drone swarm
   (Helios) → expendable units + a brittle command brain. Siege (Bastion) →
   slowest, longest range, no maneuver. The *stat shape* is the identity; the look
   serves it.
2. **Each faction needs one signature weapon mechanic** so combat *feels*
   different: Free Frontier **mines/flak/grapple** (area denial), Helios
   **drone_bay** (a cheaper numerous `fighter_bay`), Order **unstable relic-burst /
   martyr-ram** (high risk burst), Bastion **long-range bombardment** (outrange),
   Precursors **rule-bending effects** (per-encounter). Confirm which of these need
   new weapon *types* vs. re-skinned existing ones before building.
3. **Drive color + hull palette must survive grayscale** (ship-bible rule 5). The
   new drive colors — warm-white (Frontier), cyan (Helios), gold-violet (Order),
   smoldering red (Bastion), violet-halo (Precursor) — are chosen to be distinct
   from the existing blue/green/orange/amber set.
4. **Silhouette must not collide** with the existing four languages: Frontier =
   *proud-improvised farm-militia*, Helios = *minimalist branded product*, Order =
   *baroque cathedral-reliquary*, Bastion = *grim archaic over-Federation*,
   Precursor = *non-human grown geometry*. If a new hull could be mistaken for an
   existing faction's in shadow, it's wrong.

---

## 6. Alliances & enmities

Relationships are deliberately **cross-cutting** so the galaxy never reads as a
single line. `++` close ally · `+` sympathetic · `~` wary/transactional ·
`-` hostile · `--` war.

| | Fed | Reb | FF | Helios | Order | Bastion | Pirate | Guild | Precursor |
|---|---|---|---|---|---|---|---|---|---|
| **Federation** | — | -- | ~(rules) | ~(buys) | ~ | -- (civil war) | - | ~ | - |
| **Rebels** | -- | — | + (wary) | ~ | · | -- | - | ~ | - |
| **Free Frontier** | ~ | + (wary) | — | -- | · | -- | ~ (frenemy) | + | - |
| **Helios Combine** | ~ | ~ | -- | — | -- | ~ | ~ | ~ (rival/partner) | ~ (would harvest) |
| **Artifact Order** | ~ | · | · | -- | — | · | - | · | ++ (worship) |
| **Fed Bastion** | -- | -- | -- | ~ | · | — | - | ~ | - |
| **Pirates** | - | - | ~ | ~ | - | - | -- (internal) | - | - |
| **Merchant Guild** | ~ | ~ | + | ~ | — | ~ | - | — | - |
| **Precursors** | - | - | - | - | (uncomprehending) | - | - | - | — |

**The relationships to lean on dramatically:**
1. **Federation ⚔ Bastion** — a *civil war* for the soul of the Federation. The
   splinter is the truly heavy hand, which lets the lawful Fed stay grey-not-evil.
2. **Artifact Order ⚔ Helios** — *sacred vs. profane*. The Order venerates
   Precursor tech; Helios patents and strip-mines it. The cleanest hatred in the game.
3. **Free Frontier ↔ Rebels** — a *wary alliance* the player can cement or break;
   driven together by terror of the Bastion's scorched-earth doctrine.
4. **Everyone ⚔ Precursors** — the *endgame truce*: Fed + Rebel + Frontier forced
   to stand together while Helios exploits and the Order splits over the sleeper.

**Two twists:**
- **The managed war.** The Merchant Guild *finances* and Helios *arms* both sides;
  neither wants victory — they want the conflict to continue, profitably. Revealed
  in the cross-faction arc "The Arms Bazaar."
- **The Order's schism.** Scholars want to *seal* the Rift; zealots want to *wake*
  the sleeper. Which wins is the game's ending fork.

The player's allegiance should carry a **light reputation cost**: siding with one
pole cools contracts at the opposite pole, so choices have weight and the map has
replay value.

### 6.1 The live relations table (as implemented)

The §6 matrix is the narrative intent; this is the concrete state the game reads
from `assets/enemies.yaml` and `assets/allies.yaml`. **Relations are directional —
the matrix is deliberately not symmetric.** Read it *across a row*: it shows what
that faction's ships *do* to each column faction.

- ⚔ **opens fire on sight.**  A faction's `enemies` entry lists who is hostile *to
  it* — so putting Pirate in the Merchant Guild's entry is what sends pirates after
  convoys. Combat is one-directional unless both sides list each other (most open
  wars do).
- 🛡 **shields.**  A faction's `allies` entry mixes those factions' reward signal
  into its own, so its fighters are paid to keep them alive — and to hunt whatever
  attacks them.
- · ignores / neutral.

| row → col | Fed | Reb | FF | Hel | Bas | Ord | Mer | Pir | Pre |
|---|---|---|---|---|---|---|---|---|---|
| **Fed** | ▪ | ⚔ | · | · | ⚔ | · | 🛡 | ⚔ | ⚔ |
| **Reb** | ⚔ | ▪ | 🛡 | · | ⚔ | · | 🛡 | ⚔ | ⚔ |
| **FF** | · | 🛡 | ▪ | ⚔ | ⚔ | · | 🛡 | ⚔ | ⚔ |
| **Hel** | · | · | ⚔ | ▪ | · | ⚔ | 🛡 | ⚔ | ⚔ |
| **Bas** | ⚔ | ⚔ | ⚔ | · | ▪ | · | · | · | ⚔ |
| **Ord** | · | · | · | ⚔ | · | ▪ | · | ⚔ | ⚔ |
| **Mer** | · | · | · | · | · | · | ▪ | · | ⚔ |
| **Pir** | · | · | ⚔ | ⚔ | ⚔ | ⚔ | ⚔ | ▪ | ⚔ |
| **Pre** | ⚔ | ⚔ | ⚔ | ⚔ | ⚔ | ⚔ | · | ⚔ | ▪ |

*(Fed = Federation, Reb = Rebel, FF = Free Frontier, Hel = Helios, Bas = Bastion,
Ord = Order, Mer = Merchant Guild, Pir = Pirate, Pre = Precursor.)*

The asymmetry is the point. Every armed faction — Federation, Rebel, Free Frontier,
Helios — **shields the Merchant Guild**, but the Guild's own row shields no one:
escorts guard the convoys, the convoys don't man the guns. Rebel and Free Frontier
shield **each other** (the wary alliance); the Guild and the Pirates shield nobody.

### Where the wars are fought

Most animosities play out as live NPC battles — systems where both sides keep ships
(faction borders bleed each other's patrols into the spawn tables, and pirates raid
everywhere). A few rivalries are between factions whose territories simply don't
touch; those stay *cold fronts*, fought through proxies and the player rather than
with ambient skirmishes, and that's by design.

| War | Battleground systems |
|---|---|
| Fed ⚔ Reb | alpha_centauri, drift |
| Fed ⚔ Bas | iron_march, kepler_22, tycho_drift |
| Fed ⚔ Pir | alpha_centauri, barnard, ceres_freehold, drift, epsilon_eridani … |
| Reb ⚔ Bas | *cold front — distant territories* |
| Reb ⚔ Pir | alpha_centauri, altair, deneb, dim_haven, drift … |
| FF ⚔ Hel | *cold front — distant territories* |
| FF ⚔ Bas | drumlin, iron_march, kepler_22 |
| FF ⚔ Pir | altair, barnard, ceres_freehold, dim_haven, drift … |
| Hel ⚔ Ord | rigel |
| Hel ⚔ Pir | alpha_centauri, coldforge, epsilon_eridani, helios_prime, rigel … |
| Bas ⚔ Pir | bastion, coldforge, drumlin, iron_march, kepler_22 … |
| Ord ⚔ Pir | deneb, dim_haven, rigel, saints_rest, sanctum … |
| Mer ⚔ Pir | alpha_centauri, altair, barnard, bastion, ceres_freehold … |

**Precursor** is hostile to *everyone*, but its systems are sealed beyond the Rift
(only Precursor ships spawn there). That war is the **player's**, fought in the
endgame Rift arc — no ambient Precursor-vs-faction skirmishes, by design.

## 7. Ship-aesthetic assignments

The visual + mechanical signature locked per faction (each new faction gets a
unique **propulsion** *and* **weapon** identity; two existing factions get a cue):

| Aesthetic idea | Faction | Expression |
|---|---|---|
| **Steampunk** | **Artifact Order** | Brass-and-gothic reliquaries; exposed steam/pressure/gears and rivet-work treated as *venerated* old-tech; copper, bronze, stained glass. Updates the Order palette toward **brass + violet + white**. |
| **Solar-wind sails** | **Free Frontier** | Large **solar sails** are the primary drive — leading the hull or on side booms; free energy = frontier self-sufficiency. Only a small warm-white plasma drive for maneuver. |
| **Asymmetric / outriggers** | **Free Frontier** | The sail booms are **asymmetric outriggers** — *designed* asymmetry (clean, painted, intentional), distinct from the Pirates' *salvage* asymmetry. |
| **Drone swarms + fast-launch carriers** | **Helios Combine** | Fleets of weak, fast, identical **drones**; the larger hulls are **drone-carriers** with a `drone_bay` that relaunches far more often (and cheaper) than a normal `fighter_bay`. Smooth, branded, no cockpits. |
| **Long-range slugs / chainguns** | **Fed Bastion** | Heavy arms are **kinetic, not guided**: long-range turrets + slightly-aimed forward **chainguns** firing **animated slugs** (real projectile entities, *not* sprites — cheap to animate). Outranges everyone; no missiles. |
| **All angles, nothing smooth** | **Fed Bastion** | Faceted **iron brutalism** — every surface a flat plane, no curves. Pairs with the slug-guns: archaic, industrial, grim. |
| **Forward-swept wings** | **Rebels** | Refines the existing winged-dart language — forward-swept wings/canards for an aggressive, unstable-fast look. |

**Resulting polarity to preserve:** **Helios = all-smooth branded product** ↔
**Bastion = all-angles iron brute**; **Free Frontier sails/asymmetric (organic
ingenuity)** ↔ **Helios drones (cold uniformity)**; **Order brass-baroque** ↔
everyone's functionalism. If two factions' hulls could be confused in grayscale
shadow, the design is wrong (ship-bible rule 5).

*(These supersede the drive/palette notes in §2–§3 where they differ — notably the
Free Frontier's primary drive is now sails, and the Order's palette gains brass.)*

---

*Living document — extend per faction as ships, weapons, systems, and missions are
built. Pairs with [ship_design_bible.md](ship_design_bible.md),
[star_systems.yaml](../assets/star_systems.yaml),
[missions.yaml](../assets/missions.yaml), and
[ships.yaml](../assets/ships.yaml).*
