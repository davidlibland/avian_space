# BIAB brief pack 1 — faction space themes (7 tracks)

## How to produce each track (~10 min per song)

The `paste/` folder holds one `.txt` per song containing ONLY the chord
chart — BIAB's text-paste reliably handles chords, but its title/tempo
metadata format is undocumented, so set those by hand (three clicks):

1. **File | New** in Band-in-a-Box.
2. Set the song basics on the main screen using the table below:
   click the **title** area and type the name, click the **key** and
   pick it, click the **tempo** and type it.
3. Open the song's `.txt` from `docs/biab_briefs/paste/`, select all,
   copy.
4. **Edit | Paste Special – from Clipboard text to Song(s)**. The
   chords land in the grid; the `a)` / `b)` marks become part markers
   (verse/chorus feel changes).
5. Open the **StylePicker** and pick a style per the song's "Style"
   row below. The named styles are suggestions — filter by the genre
   and trust your ear.
6. Song form: set **3 choruses** (Song Settings), chorus = bars 1–32,
   and **uncheck "Generate 2 bar ending"** so the render doesn't
   resolve/decay — I need a steady middle chorus to cut a loop from.
7. **Generate**, listen, regenerate or swap RealTracks until it feels
   right. Small chord edits are welcome — the chart is a starting
   point, not scripture.
8. **Export as WAV, 44.1 kHz**, named exactly as the "File" row (e.g.
   `space_federation.wav`). Drop the WAVs anywhere and tell me where —
   I trim the middle chorus to a seamless loop, normalize, encode ogg,
   and swap each one in for its placeholder in `assets/music/`.

Chart notation (all charts use only this): `|` bar line, chords by
name (slash = bass note, e.g. `D/F#`), two chords in one bar separated
by a space, `a)` / `b)` section markers. I deliberately left out
BIAB's fancier marks — `^C` (a "push": hit the chord an eighth early)
and `C.` / `C..` / `C...` (rest / shot / held chord) — they're legal
BIAB text syntax but paste inconsistently, and the style's groove
covers that feel anyway. Every form is 32 bars so the loop point lands
on a clean phrase boundary.

The current placeholder loops in `assets/music/` sketch the vibe each
slot is aiming at — worth a listen before each session.

---

## 1. The Fleet
| | |
|---|---|
| **File** | `space_federation.wav` |
| **Key / Tempo** | D major / 84 |
| **Style** | Orchestral / film-score (try `_SOUNDTRK.STY` or any "Film Score" RealStyle with French horn or brass) |
| **Mood** | Noble, unhurried — a navy patrolling calm space. Confident, never aggressive. |

## 2. Embers
| | |
|---|---|
| **File** | `space_rebel.wav` |
| **Key / Tempo** | E minor / 112 |
| **Style** | Driving folk-rock / celtic rock (try `_IRISHRK.STY` or a 16th-note acoustic-rock RealStyle) |
| **Mood** | Gritty, urgent, low strings and driving percussion — a cause worth losing for. Momentum without triumph. |

## 3. The Long Haul
| | |
|---|---|
| **File** | `space_freefrontier.wav` |
| **Key / Tempo** | G major / 76 |
| **Style** | Slow country/americana ballad with pedal steel or dobro (try `_COUNTRYB.STY` or a "Campfire" style) |
| **Mood** | Sparse americana against the void — loneliness that's chosen, not suffered. |

## 4. Cold Light
| | |
|---|---|
| **File** | `space_helios.wav` |
| **Key / Tempo** | A minor / 100 |
| **Style** | Minimal synth-pop / electronica (try `_ELECPOP.STY` or any "EuroDance/Synth" RealStyle) |
| **Mood** | Clean, automated, machine-precise with a faint melancholy. Repetition is the point. |

## 5. Anvil
| | |
|---|---|
| **File** | `space_bastion.wav` |
| **Key / Tempo** | C minor / 92 |
| **Style** | Heavy rock / metal (try `_HVYROCK.STY` or a "Heavy Rock 8ths" RealStyle) |
| **Mood** | Industrial, forge-rhythm, power chords like hammer blows. The faction that builds hulls hears its own foundries. |

## 6. Vespers
| | |
|---|---|
| **File** | `space_order.wav` |
| **Key / Tempo** | F major / 66 |
| **Style** | New-age / slow gospel-choir pad style (try `_NEWAGE.STY` or any slow style with a choir/organ RealTrack) |
| **Mood** | Choral, liturgical, open fifths — a monastery in vacuum. More stillness than motion. |

## 7. No Colors
| | |
|---|---|
| **File** | `space_pirate.wav` |
| **Key / Tempo** | B minor / 96 |
| **Style** | Minor-key tango / gypsy-jazz / spy groove (try `_GYPSYJZ.STY` or a "Tango"/"Spy" flavored RealStyle) |
| **Mood** | Tense, muted, coiled — an ostinato that never resolves. For pirate space AND contested systems: menace, not swashbuckling. |

---

Next packs (later): surface biome themes (5), the bar's jazz set, and
the title theme.
