#!/usr/bin/env python3
"""Generate MusicXML lead sheets for the BIAB brief-pack songs.

Band-in-a-Box (2020+) opens MusicXML with title, key, tempo, and chord
symbols intact — everything the clipboard text-paste route drops. One
file per song lands in docs/biab_briefs/musicxml/; in BIAB use
File | Open (file type: MusicXML) then pick a style and generate.

Charts are the same 32-bar forms as the paste/*.txt files (kept in sync
by hand — this script embeds them as the single source).

Run:  python3 scripts/gen_biab_musicxml.py
"""
import os
import re

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "biab_briefs", "musicxml")

# title, key-fifths, tempo, chart (bars separated by |, sections a)/b) ignored
SONGS = {
    "space_federation": ("The Fleet", 1, 84, """
        Em | Em | C | G | Am | Em | B7 | B7 | Em | Em | C | G | Am | B7 | Em | Em |
        G | G | D | D | C | C | B7 | B7 | Em | D | C | G | Am | B7 | Em | Em"""),
    "space_rebel": ("Embers", 1, 112, """
        Em | Em | C | C | G | D | Em | Em | Em | Em | C | C | Am | B7 | Em | Em |
        C | G | D | Em | C | G | B7 | B7 | Am | Em | C | G | Am | B7 | Em | Em"""),
    "space_freefrontier": ("The Long Haul", 1, 76, """
        G | G | C/G | G | Em | C | D | D | G | G/B | C | Am | G/D | D7 | G | G |
        C | C | G | G | Em | Em | D | D | C | G/B | Am | C/D | G | C/G | G | G"""),
    "space_helios": ("Cold Light", 0, 100, """
        Am | Am | F | F | C | C | G | G | Am | Am | F | F | C | G/B | Am | Am |
        F | F | Am | Am | Dm | Dm | E7 | E7 | F | G | Am | F | Dm | E7 | Am | Am"""),
    "space_bastion": ("Anvil", -3, 92, """
        Cm | Cm | Ab | Ab | Bb | Bb | Cm | Cm | Cm | Cm | Ab | Ab | G | G | Cm | Cm |
        Fm | Fm | Cm | Cm | Ab | Eb | G | G | Fm | Ab | Cm | Bb | Ab | G | Cm | Cm"""),
    "space_order": ("Vespers", -1, 66, """
        F | F | C/F | F | Bb/F | F | C | C | F | F | Gm | Dm | Bb | C | F | F |
        Dm | Dm | Bb | Bb | F/A | F/A | C | C | Dm | Bb | F/A | Gm | Bb/C | C | F | F"""),
    "space_pirate": ("No Colors", 2, 96, """
        Bm | Bm | Bm/A | Bm/A | G | G | F#7 | F#7 | Bm | Bm | Bm/A | Bm/A | G | F#7 | Bm | Bm |
        Em | Em | Bm | Bm | G | Em | F#7 | F#7 | Bm | D | G | Em | C#m7b5 | F#7 | Bm | Bm"""),
    "surface_garden": ("Greenfields", -1, 92, """
        F | F | Bb/F | F | Dm | Bb | C | C | F | F/A | Bb | Gm | F/C | C7 | F | F |
        Bb | Bb | F/A | F/A | Gm | Am | Bb | C | Dm | Bb | F/A | Gm7 | Bb/C | C | F | F"""),
    "surface_ice": ("Glasslands", 0, 63, """
        Am | Am | Em/G | Em/G | F | F | Esus E | E | Am | Am/G | F | F | Dm | Em | Am | Am |
        F | C/E | Dm | Am/C | Bm7b5 | E | Am | Am | F | C/E | Dm | E | Am | Em | Am | Am"""),
    "surface_rocky": ("Scree", -1, 80, """
        Dm | Dm | Bb | Bb | Dm | Dm | C | C | Dm | Dm | Bb | Gm | A | A | Dm | Dm |
        Gm | Gm | Dm | Dm | Bb | C | A | A | Gm | Bb | Dm | C | Bb | A | Dm | Dm"""),
    "surface_desert": ("Heat Shimmer", 4, 68, """
        E | E | E | E | D/E | D/E | E | E | E | E | G | G | D | D | E | E |
        A | A | E | E | G | D | E | E | A | G | E | D | E | E | E | E"""),
    "surface_interior": ("Recycled Air", -1, 84, """
        Dm7 | Dm7 | Gm7 | Gm7 | Bbmaj7 | Bbmaj7 | Am7 | Am7 | Dm7 | Dm7 | Gm7 | Gm7 | Bbmaj7 | Am7 | Dm7 | Dm7 |
        Fmaj7 | Fmaj7 | Em7b5 | A7 | Dm7 | Dm7 | Gm7 | C7 | Fmaj7 | Bbmaj7 | Em7b5 | A7 | Dm7 | Gm7 | Dm7 | Dm7"""),
    "bar": ("Last Call at the Port", -1, 126, """
        Fmaj7 | Gm7 C7 | Fmaj7 | Cm7 F7 | Bbmaj7 | Bbm7 Eb7 | Fmaj7 | Gm7 C7 |
        Fmaj7 | Gm7 C7 | Fmaj7 | Cm7 F7 | Bbmaj7 | Bbm7 Eb7 | Fmaj7 | Fmaj7 |
        Am7 | D7 | Gm7 | C7 | Am7 D7 | Gm7 C7 | Fmaj7 | D7 |
        Gm7 | C7 | Am7 D7 | Gm7 C7 | Fmaj7 | Gm7 C7 | Fmaj7 | Fmaj7"""),
    "menu": ("Wide Open", 0, 74, """
        C | C | G/B | G/B | Am | Am | F | F | C | C | G | G | F | G | C | C |
        Am | F | C | G | Am | F | C/E | G | F | F | C/E | Am | Dm7 | G | C | C"""),
}

# Optional composed melody per song: list of 32 bars, each a list of
# (pitch, quarters) with pitch like "F#4" (None = rest). BIAB loads these
# onto the Melody track and the band accompanies the tune.
MELODIES = {
    # The Fleet — E minor march, two motifs:
    #   A-motif (the designer's): held B (the fifth) resolving onto E with a
    #   16th snap pickup. Calls at bars 1/5/9/15; every response varies.
    #   B-motif: rising dotted figure (long-short-up-up) — striving where
    #   the A-motif commands. A-motif never appears in B; the loop back to
    #   bar 1 restates it.
    "space_federation": [
        [("B4", 2), ("E4", 1), ("E4", 0.25), ("F#4", 0.25), ("G4", 0.5)],
        [("B4", 1), ("A4", 1), ("G4", 1), ("F#4", 1)],
        [("E4", 2), ("G4", 1), ("A4", 1)],
        [("B4", 2), ("D5", 1), ("B4", 1)],
        [("B4", 2), ("E4", 1), ("E4", 0.25), ("F#4", 0.25), ("G4", 0.5)],
        [("G4", 1), ("F#4", 1), ("G4", 1), ("A4", 1)],
        [("F#4", 2), ("D#4", 1), ("F#4", 1)],
        [("B3", 1), ("D#4", 1), ("F#4", 1), ("A4", 1)],
        [("B4", 2), ("E4", 1), ("E4", 0.25), ("F#4", 0.25), ("G4", 0.5)],
        [("E5", 1), ("D5", 1), ("B4", 1), ("G4", 1)],
        [("C5", 2), ("B4", 1), ("A4", 1)],
        [("B4", 2), ("G4", 1), ("D4", 1)],
        [("C5", 1), ("B4", 1), ("A4", 1), ("E4", 1)],
        [("D#4", 2), ("F#4", 1), ("A4", 1)],
        [("B4", 2), ("E4", 1), ("E4", 0.25), ("F#4", 0.25), ("G4", 0.5)],
        [("E4", 2), ("B3", 1), ("E4", 1)],
        [("G4", 1.5), ("A4", 0.5), ("B4", 1), ("D5", 1)],
        [("D5", 2), ("B4", 2)],
        [("F#4", 1.5), ("G4", 0.5), ("A4", 1), ("D5", 1)],
        [("D5", 2), ("A4", 2)],
        [("C5", 1.5), ("D5", 0.5), ("E5", 1), ("G5", 1)],
        [("G5", 2), ("E5", 2)],
        [("D#5", 1.5), ("E5", 0.5), ("F#5", 1), ("B4", 1)],
        [("B4", 2), ("F#4", 2)],
        [("E5", 1.5), ("D5", 0.5), ("B4", 1), ("G4", 1)],
        [("F#4", 1.5), ("G4", 0.5), ("A4", 1), ("D5", 1)],
        [("E5", 1.5), ("D5", 0.5), ("C5", 1), ("G4", 1)],
        [("B4", 1.5), ("A4", 0.5), ("G4", 1), ("D4", 1)],
        [("A4", 1.5), ("B4", 0.5), ("C5", 1), ("E5", 1)],
        [("D#5", 2), ("F#5", 1), ("B4", 1)],
        [("B4", 2), ("A4", 1), ("F#4", 1)],
        [("E4", 4)],
    ],
    # Embers — modal folk grit, eighth-note runs, momentum without triumph.
    "space_rebel": [
        [("E4", 1), ("G4", 0.5), ("A4", 0.5), ("B4", 2)],
        [("B4", 0.5), ("A4", 0.5), ("G4", 0.5), ("A4", 0.5), ("B4", 2)],
        [("C5", 1), ("B4", 0.5), ("A4", 0.5), ("G4", 2)],
        [("A4", 1.5), ("G4", 0.5), ("E4", 2)],
        [("G4", 1), ("B4", 1), ("D5", 1.5), ("B4", 0.5)],
        [("A4", 1), ("F#4", 1), ("D4", 2)],
        [("E4", 1), ("G4", 1), ("B4", 1), ("E5", 1)],
        [("E5", 2), ("B4", 2)],
        [("E4", 1), ("G4", 0.5), ("A4", 0.5), ("B4", 2)],
        [("D5", 1), ("B4", 0.5), ("A4", 0.5), ("G4", 2)],
        [("C5", 1.5), ("B4", 0.5), ("A4", 1), ("G4", 1)],
        [("A4", 2), ("G4", 2)],
        [("A4", 1), ("C5", 1), ("E5", 1.5), ("E5", 0.5)],
        [("D#5", 1), ("B4", 1), ("F#4", 2)],
        [("G4", 1), ("F#4", 1), ("E4", 2)],
        [("E4", 4)],
        [("G4", 1), ("A4", 0.5), ("B4", 0.5), ("C5", 2)],
        [("B4", 1), ("D5", 1), ("G4", 2)],
        [("F#4", 1), ("A4", 1), ("D5", 2)],
        [("B4", 1.5), ("A4", 0.5), ("G4", 1), ("E4", 1)],
        [("G4", 1), ("A4", 0.5), ("B4", 0.5), ("C5", 2)],
        [("D5", 1), ("B4", 1), ("G4", 2)],
        [("B4", 1.5), ("A4", 0.5), ("F#4", 2)],
        [("D#5", 2), ("B4", 2)],
        [("C5", 1), ("B4", 0.5), ("A4", 0.5), ("E5", 2)],
        [("B4", 1), ("G4", 1), ("E4", 2)],
        [("A4", 1), ("G4", 0.5), ("A4", 0.5), ("C5", 2)],
        [("B4", 1.5), ("A4", 0.5), ("G4", 2)],
        [("A4", 1), ("C5", 1), ("E5", 1), ("D5", 1)],
        [("B4", 1.5), ("F#4", 0.5), ("B4", 2)],
        [("G4", 1), ("F#4", 1), ("E4", 2)],
        [("E4", 4)],
    ],
    # The Long Haul — lazy pentatonic, long tones, low register.
    "space_freefrontier": [
        [("G4", 2), ("B4", 1), ("A4", 1)],
        [("G4", 3), ("E4", 1)],
        [("E4", 1), ("G4", 1), ("A4", 1.5), ("G4", 0.5)],
        [("G4", 4)],
        [("E4", 1), ("G4", 1), ("B4", 2)],
        [("C5", 1.5), ("A4", 0.5), ("G4", 2)],
        [("A4", 2), ("F#4", 1), ("D4", 1)],
        [("D4", 3), ("D4", 1)],
        [("G4", 2), ("B4", 1), ("A4", 1)],
        [("B4", 1.5), ("A4", 0.5), ("G4", 2)],
        [("E4", 1), ("G4", 1), ("A4", 2)],
        [("C5", 1), ("A4", 1), ("E4", 2)],
        [("D4", 1), ("G4", 1), ("B4", 1.5), ("A4", 0.5)],
        [("A4", 2), ("F#4", 2)],
        [("G4", 3), ("E4", 1)],
        [("G4", 4)],
        [("E4", 1), ("G4", 1), ("C5", 2)],
        [("C5", 1.5), ("A4", 0.5), ("G4", 2)],
        [("B4", 1), ("A4", 1), ("G4", 2)],
        [("D4", 1), ("G4", 1), ("B4", 2)],
        [("E4", 2), ("G4", 1), ("B4", 1)],
        [("B4", 1.5), ("G4", 0.5), ("E4", 2)],
        [("A4", 2), ("F#4", 1), ("E4", 1)],
        [("D4", 3), ("D4", 1)],
        [("E4", 1), ("G4", 1), ("A4", 1), ("C5", 1)],
        [("B4", 1.5), ("G4", 0.5), ("D4", 2)],
        [("C5", 1), ("A4", 1), ("E4", 2)],
        [("D4", 1), ("F#4", 1), ("A4", 2)],
        [("G4", 2), ("B4", 1), ("A4", 1)],
        [("G4", 1.5), ("E4", 0.5), ("C4", 2)],
        [("G4", 3), ("D4", 1)],
        [("G4", 4)],
    ],
    # Cold Light — machine cells: even eighths, small transposed patterns.
    "space_helios": [
        [("A4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("A4", 0.5), ("C5", 0.5), ("E5", 1)],
        [("A4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("B4", 1), ("A4", 1)],
        [("F4", 0.5), ("A4", 0.5), ("C5", 0.5), ("A4", 0.5), ("F4", 0.5), ("A4", 0.5), ("C5", 1)],
        [("C5", 1), ("A4", 1), ("F4", 2)],
        [("G4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("G4", 0.5), ("C5", 0.5), ("E5", 1)],
        [("E5", 1), ("C5", 1), ("G4", 2)],
        [("G4", 0.5), ("B4", 0.5), ("D5", 0.5), ("B4", 0.5), ("G4", 1), ("B4", 1)],
        [("D5", 2), ("B4", 2)],
        [("A4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("A4", 0.5), ("C5", 0.5), ("E5", 1)],
        [("A4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("B4", 1), ("A4", 1)],
        [("F4", 0.5), ("A4", 0.5), ("C5", 0.5), ("A4", 0.5), ("F4", 0.5), ("A4", 0.5), ("C5", 1)],
        [("C5", 1), ("A4", 1), ("F4", 2)],
        [("G4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("G4", 1), ("E5", 1)],
        [("D5", 0.5), ("B4", 0.5), ("G4", 0.5), ("B4", 0.5), ("D5", 2)],
        [("A4", 0.5), ("C5", 0.5), ("E5", 0.5), ("C5", 0.5), ("A4", 2)],
        [("A4", 4)],
        [("A4", 0.5), ("C5", 0.5), ("F5", 0.5), ("C5", 0.5), ("A4", 0.5), ("C5", 0.5), ("F5", 1)],
        [("F5", 1), ("C5", 1), ("A4", 2)],
        [("E5", 0.5), ("C5", 0.5), ("A4", 0.5), ("C5", 0.5), ("E5", 2)],
        [("A4", 1), ("B4", 1), ("C5", 2)],
        [("D5", 0.5), ("F5", 0.5), ("D5", 0.5), ("F5", 0.5), ("A4", 2)],
        [("F5", 1), ("D5", 1), ("A4", 2)],
        [("B4", 0.5), ("D5", 0.5), ("G#4", 1), ("B4", 2)],
        [("G#4", 1), ("B4", 1), ("E5", 2)],
        [("A4", 0.5), ("C5", 0.5), ("F5", 1.5), ("C5", 0.5), ("A4", 1)],
        [("B4", 0.5), ("D5", 0.5), ("G5", 1.5), ("D5", 0.5), ("B4", 1)],
        [("C5", 0.5), ("E5", 0.5), ("A5", 1), ("E5", 1), ("C5", 1)],
        [("C5", 1), ("A4", 1), ("F4", 2)],
        [("F4", 0.5), ("A4", 0.5), ("D5", 1), ("F5", 2)],
        [("E5", 1), ("D5", 1), ("B4", 1), ("G#4", 1)],
        [("A4", 2), ("E5", 2)],
        [("A4", 4)],
    ],
    # Anvil — hammering low repeats, minor thirds, forge blows.
    "space_bastion": [
        [("C4", 1), ("C4", 0.5), ("C4", 0.5), ("Eb4", 2)],
        [("G4", 1.5), ("Eb4", 0.5), ("C4", 2)],
        [("Eb4", 1), ("Eb4", 0.5), ("Eb4", 0.5), ("Ab4", 2)],
        [("G4", 1.5), ("Eb4", 0.5), ("C4", 2)],
        [("D4", 1), ("D4", 0.5), ("D4", 0.5), ("F4", 2)],
        [("Bb4", 1.5), ("F4", 0.5), ("D4", 2)],
        [("C4", 1), ("Eb4", 1), ("G4", 2)],
        [("G4", 1), ("G4", 0.5), ("G4", 0.5), ("C4", 2)],
        [("C4", 1), ("C4", 0.5), ("C4", 0.5), ("Eb4", 2)],
        [("G4", 1.5), ("Eb4", 0.5), ("C4", 2)],
        [("Eb4", 1), ("Eb4", 0.5), ("Eb4", 0.5), ("Ab4", 2)],
        [("G4", 1.5), ("Eb4", 0.5), ("C4", 2)],
        [("D4", 1), ("D4", 0.5), ("D4", 0.5), ("B3", 2)],
        [("G4", 1.5), ("D4", 0.5), ("B3", 2)],
        [("C4", 1), ("C4", 1), ("Eb4", 1), ("G4", 1)],
        [("C4", 4)],
        [("F4", 1), ("F4", 0.5), ("F4", 0.5), ("Ab4", 2)],
        [("C5", 1.5), ("Ab4", 0.5), ("F4", 2)],
        [("G4", 1), ("G4", 0.5), ("G4", 0.5), ("Eb4", 2)],
        [("C4", 1), ("Eb4", 1), ("G4", 2)],
        [("Ab4", 1), ("Ab4", 0.5), ("Ab4", 0.5), ("C5", 2)],
        [("Bb4", 1.5), ("G4", 0.5), ("Eb4", 2)],
        [("D4", 1), ("D4", 0.5), ("D4", 0.5), ("G4", 1), ("B4", 1)],
        [("B4", 2), ("D5", 2)],
        [("C5", 1), ("Ab4", 1), ("F4", 2)],
        [("Ab4", 1), ("C5", 1), ("Eb5", 2)],
        [("G4", 1.5), ("Eb4", 0.5), ("C4", 2)],
        [("D4", 1), ("F4", 1), ("Bb4", 2)],
        [("C5", 1), ("C5", 0.5), ("C5", 0.5), ("Ab4", 2)],
        [("B4", 1.5), ("G4", 0.5), ("D4", 2)],
        [("C4", 1), ("C4", 0.5), ("C4", 0.5), ("Eb4", 1), ("G4", 1)],
        [("C4", 4)],
    ],
    # Vespers — chant: long stepwise tones, one high candle in the B section.
    "space_order": [
        [("F4", 2), ("G4", 2)],
        [("A4", 4)],
        [("G4", 2), ("E4", 2)],
        [("F4", 4)],
        [("Bb4", 2), ("A4", 2)],
        [("A4", 2), ("F4", 2)],
        [("G4", 2), ("E4", 2)],
        [("E4", 2), ("G4", 2)],
        [("F4", 2), ("A4", 2)],
        [("C5", 4)],
        [("Bb4", 2), ("G4", 2)],
        [("A4", 2), ("F4", 2)],
        [("F4", 2), ("D4", 2)],
        [("E4", 2), ("G4", 2)],
        [("F4", 4)],
        [("F4", 4)],
        [("D5", 2), ("C5", 2)],
        [("A4", 4)],
        [("Bb4", 2), ("D5", 2)],
        [("F5", 2), ("D5", 2)],
        [("C5", 2), ("A4", 2)],
        [("A4", 4)],
        [("G4", 2), ("E5", 2)],
        [("G4", 2), ("E4", 2)],
        [("D4", 2), ("F4", 2)],
        [("D5", 2), ("Bb4", 2)],
        [("C5", 2), ("A4", 2)],
        [("Bb4", 2), ("G4", 2)],
        [("F4", 2), ("D4", 2)],
        [("E4", 2), ("G4", 2)],
        [("F4", 4)],
        [("F4", 4)],
    ],
    # No Colors — coiled ostinato, tight range, never resolves early.
    "space_pirate": [
        [("B3", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("B3", 1), ("F#4", 1)],
        [("B3", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("F#4", 2)],
        [("E4", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("B3", 2)],
        [("D4", 1), ("C#4", 1), ("B3", 2)],
        [("G4", 0.5), ("F#4", 0.5), ("E4", 0.5), ("F#4", 0.5), ("D4", 2)],
        [("B4", 1.5), ("G4", 0.5), ("E4", 2)],
        [("E4", 0.5), ("F#4", 0.5), ("A#4", 1), ("C#5", 2)],
        [("A#4", 1), ("F#4", 1), ("C#4", 2)],
        [("B3", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("B3", 1), ("F#4", 1)],
        [("B3", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("F#4", 2)],
        [("E4", 0.5), ("D4", 0.5), ("C#4", 0.5), ("D4", 0.5), ("B3", 2)],
        [("D4", 1), ("C#4", 1), ("B3", 2)],
        [("B4", 0.5), ("A4", 0.5), ("G4", 0.5), ("A4", 0.5), ("B4", 2)],
        [("A#4", 1.5), ("F#4", 0.5), ("E4", 2)],
        [("D4", 0.5), ("C#4", 0.5), ("B3", 0.5), ("C#4", 0.5), ("D4", 1), ("F#4", 1)],
        [("B3", 3), ("F#4", 1)],
        [("E4", 0.5), ("G4", 0.5), ("F#4", 0.5), ("G4", 0.5), ("E4", 1), ("B4", 1)],
        [("G4", 1.5), ("E4", 0.5), ("B3", 2)],
        [("D4", 0.5), ("F#4", 0.5), ("B4", 1), ("F#4", 2)],
        [("D5", 1), ("B4", 1), ("F#4", 2)],
        [("G4", 1), ("B4", 1), ("D5", 1.5), ("B4", 0.5)],
        [("E5", 1), ("B4", 1), ("G4", 2)],
        [("A#4", 0.5), ("C#5", 0.5), ("E5", 1), ("C#5", 2)],
        [("A#4", 1), ("C#5", 1), ("F#4", 2)],
        [("B4", 0.5), ("D5", 0.5), ("C#5", 0.5), ("D5", 0.5), ("B4", 2)],
        [("A4", 1), ("F#4", 1), ("D4", 2)],
        [("G4", 0.5), ("A4", 0.5), ("B4", 1), ("D5", 2)],
        [("E4", 1), ("G4", 1), ("B4", 2)],
        [("G4", 1), ("E4", 1), ("C#5", 2)],
        [("A#4", 1.5), ("C#5", 0.5), ("F#4", 2)],
        [("D4", 0.5), ("C#4", 0.5), ("B3", 0.5), ("C#4", 0.5), ("D4", 2)],
        [("B3", 4)],
    ],
    # Wide Open — hopeful title anthem: rises, opens, comes home.
    "menu": [
        [("G4", 1), ("C5", 1.5), ("D5", 0.5), ("E5", 1)],
        [("E5", 2), ("D5", 1), ("C5", 1)],
        [("D5", 1.5), ("B4", 0.5), ("G4", 2)],
        [("B4", 2), ("D5", 2)],
        [("A4", 1), ("C5", 1.5), ("D5", 0.5), ("E5", 1)],
        [("E5", 2), ("C5", 2)],
        [("A4", 1.5), ("G4", 0.5), ("F4", 2)],
        [("A4", 2), ("C5", 2)],
        [("G4", 1), ("C5", 1.5), ("D5", 0.5), ("E5", 1)],
        [("G5", 2), ("E5", 2)],
        [("D5", 1.5), ("B4", 0.5), ("G4", 1), ("B4", 1)],
        [("D5", 2), ("B4", 2)],
        [("F4", 1), ("A4", 1), ("C5", 2)],
        [("D5", 1.5), ("B4", 0.5), ("G4", 2)],
        [("C5", 3), ("G4", 1)],
        [("C5", 4)],
        [("E5", 1), ("C5", 1), ("A4", 2)],
        [("A4", 1), ("C5", 1), ("F5", 2)],
        [("E5", 1.5), ("C5", 0.5), ("G4", 2)],
        [("B4", 1), ("D5", 1), ("G4", 2)],
        [("E5", 1), ("C5", 1), ("A4", 2)],
        [("A4", 1), ("C5", 1), ("F5", 2)],
        [("G5", 1.5), ("E5", 0.5), ("C5", 2)],
        [("D5", 2), ("B4", 2)],
        [("F5", 1.5), ("C5", 0.5), ("A4", 2)],
        [("C5", 1), ("A4", 1), ("F4", 2)],
        [("G4", 1), ("C5", 1), ("E5", 2)],
        [("E5", 1), ("C5", 1), ("A4", 2)],
        [("D5", 1.5), ("C5", 0.5), ("A4", 1), ("F4", 1)],
        [("G4", 1), ("B4", 1), ("D5", 2)],
        [("E5", 3), ("D5", 1)],
        [("C5", 4)],
    ],
}

KINDS = [  # ordered: longest suffix first
    ("maj7", "major-seventh"),
    ("m7b5", "half-diminished"),
    ("m7", "minor-seventh"),
    ("sus", "suspended-fourth"),
    ("m", "minor"),
    ("7", "dominant"),
    ("", "major"),
]


def parse_note(s):
    step = s[0]
    alter = s[1:].count("#") - s[1:].count("b")
    return step, alter


def parse_chord(sym):
    bass = None
    if "/" in sym:
        sym, b = sym.split("/")
        bass = parse_note(b)
    m = re.match(r"^([A-G][#b]?)(.*)$", sym)
    root, quality = m.group(1), m.group(2)
    for suffix, kind in KINDS:
        if quality == suffix:
            return parse_note(root), kind, bass
    raise ValueError(f"unknown chord quality {sym!r}")


def harmony_xml(sym):
    (rs, ra), kind, bass = parse_chord(sym)
    x = "      <harmony>\n"
    x += f"        <root><root-step>{rs}</root-step>"
    if ra:
        x += f"<root-alter>{ra}</root-alter>"
    x += "</root>\n"
    x += f"        <kind>{kind}</kind>\n"
    if bass:
        bs, ba = bass
        x += f"        <bass><bass-step>{bs}</bass-step>"
        if ba:
            x += f"<bass-alter>{ba}</bass-alter>"
        x += "</bass>\n"
    x += "      </harmony>\n"
    return x


def measure_xml(n, chords, fifths, tempo, melody=None):
    x = f'    <measure number="{n}">\n'
    if n == 1:
        x += (
            "      <attributes>\n"
            "        <divisions>4</divisions>\n"
            f"        <key><fifths>{fifths}</fifths></key>\n"
            "        <time><beats>4</beats><beat-type>4</beat-type></time>\n"
            "        <clef><sign>G</sign><line>2</line></clef>\n"
            "      </attributes>\n"
            f'      <direction><sound tempo="{tempo}"/></direction>\n'
        )
    # Section markers as rehearsal marks (BIAB maps these to its A/B part
    # markers on import — its own MusicXML export writes them this way).
    # All brief-pack forms are 16 + 16.
    if n in (1, 17):
        mark = "A" if n == 1 else "B"
        x += (
            "      <direction placement=\"above\">\n"
            "        <direction-type><rehearsal>"
            f"{mark}</rehearsal></direction-type>\n"
            "      </direction>\n"
        )
    # One chord: whole-bar duration under it. Two: half-bar (beats 1 and 3).
    durs = {1: [16], 2: [8, 8]}[len(chords)]
    # Harmony elements attach at the current musical position: emit chord 1,
    # notes filling its half, then chord 2, then the rest of the melody bar.
    if melody is None:
        for sym, d in zip(chords, durs):
            x += harmony_xml(sym)
            x += (
                "      <note><rest/>"
                f"<duration>{d}</duration><voice>1</voice></note>\n"
            )
    else:
        assert sum(q for _, q in melody) == 4, f"bar {n} melody != 4 beats"
        notes = [(p, int(q * 4)) for p, q in melody]  # divisions=4
        pos = 0
        chord_at = {0: chords[0]} if len(chords) == 1 else {0: chords[0], 8: chords[1]}
        for pitch, d in notes:
            if pos in chord_at:
                x += harmony_xml(chord_at.pop(pos))
            if pitch is None:
                x += f"      <note><rest/><duration>{d}</duration><voice>1</voice></note>\n"
            else:
                m2 = re.match(r"^([A-G])([#b]?)(\d)$", pitch)
                step, acc, octave = m2.group(1), m2.group(2), m2.group(3)
                alter = {"#": 1, "b": -1}.get(acc, 0)
                x += "      <note><pitch>"
                x += f"<step>{step}</step>"
                if alter:
                    x += f"<alter>{alter}</alter>"
                x += f"<octave>{octave}</octave></pitch>"
                x += f"<duration>{d}</duration><voice>1</voice></note>\n"
            pos += d
        assert not chord_at, f"bar {n}: 2nd chord lands mid-note (beat-3 chord needs a note boundary there)"
    x += "    </measure>\n"
    return x


def song_xml(title, fifths, tempo, chart, melody=None):
    bars = [b.strip() for b in chart.replace("\n", " ").split("|") if b.strip()]
    assert len(bars) == 32, f"{title}: {len(bars)} bars"
    if melody is not None:
        assert len(melody) == 32, f"{title}: melody has {len(melody)} bars"
    x = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 '
        'Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        '<score-partwise version="4.0">\n'
        f"  <work><work-title>{title}</work-title></work>\n"
        "  <part-list>\n"
        '    <score-part id="P1"><part-name>Melody</part-name></score-part>\n'
        "  </part-list>\n"
        '  <part id="P1">\n'
    )
    for i, bar in enumerate(bars, 1):
        chords = bar.split()
        assert 1 <= len(chords) <= 2, f"{title} bar {i}: {bar!r}"
        mbar = melody[i - 1] if melody is not None else None
        x += measure_xml(i, chords, fifths, tempo, mbar)
    x += "  </part>\n</score-partwise>\n"
    return x


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for stem, (title, fifths, tempo, chart) in SONGS.items():
        path = os.path.join(OUT, f"{stem}.musicxml")
        melody = MELODIES.get(stem)
        with open(path, "w") as f:
            f.write(song_xml(title, fifths, tempo, chart, melody))
        tag = " + melody" if melody else ""
        print(f"  {stem}.musicxml  ({title}, tempo {tempo}{tag})")
    print(f"Done → {os.path.relpath(OUT)}")
