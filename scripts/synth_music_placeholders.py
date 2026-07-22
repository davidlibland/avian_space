#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "imageio-ffmpeg",
# ]
# ///
"""Synthesize placeholder music loops for every MusicDirector slot.

One seamless 32-second ambient loop per track key (assets/music/<key>.ogg),
each with a distinct musical identity so contexts are tellable apart in
playtests. These are stand-ins: finished Band-in-a-Box renders replace them
file-for-file (same names) — see docs/music_and_ambience_plan.md and
docs/biab_briefs/.

Run:  uv run scripts/synth_music_placeholders.py
"""
import numpy as np, wave, subprocess, os
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
LOOP = 32.0  # seconds
rng = np.random.default_rng(0xA0D10)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "music")


def t(d):
    return np.linspace(0, d, int(SR * d), endpoint=False)


def sine(f, d, ph=0.0):
    return np.sin(2 * np.pi * f * t(d) + ph)


def saw(f, d):
    return 2 * ((f * t(d)) % 1.0) - 1


def noise(d):
    return rng.uniform(-1, 1, int(SR * d))


def fft_lowpass(x, cutoff):
    """Gentle spectral rolloff above `cutoff` Hz."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    gain = 1.0 / (1.0 + (freqs / max(cutoff, 1.0)) ** 2)
    return np.fft.irfft(spec * gain, n=len(x))


def fft_bandboost(x, center, width, amount):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    gain = 1.0 + amount * np.exp(-((freqs - center) ** 2) / (2 * width**2))
    return np.fft.irfft(spec * gain, n=len(x))


def note_at(x, start, tone):
    """Overlap-add `tone` into `x` at `start` seconds (wrapping past the end,
    which keeps event tails loop-seamless)."""
    i = int(start * SR) % len(x)
    n = len(tone)
    end = i + n
    if end <= len(x):
        x[i:end] += tone
    else:
        k = len(x) - i
        x[i:] += tone[:k]
        x[: end - len(x)] += tone[k:]
    return x


def env_tone(f, d, attack, tau, partials=((1, 1.0),)):
    """A decaying tone with attack, harmonic partials [(ratio, amp)]."""
    out = sum(a * sine(f * r, d) for r, a in partials)
    e = np.minimum(t(d) / max(attack, 1e-3), 1.0) * np.exp(-t(d) / tau)
    return out * e


def loopify(x, xfade=2.0):
    """Wrap-crossfade the tail into the head → seamless loop of length LOOP."""
    n = int(SR * xfade)
    body = x[: int(SR * LOOP)].copy()
    tail = x[int(SR * LOOP) : int(SR * LOOP) + n]
    ramp = np.linspace(0, 1, n)
    body[:n] = body[:n] * ramp + tail * (1 - ramp)
    return body


def norm(x, peak=0.5):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def drone(freqs, d, detune_hz=0.22, lowpass=None, shape=sine):
    """Detuned unison pairs per pitch — slow beating, thick pad."""
    x = np.zeros(int(SR * d))
    for f in freqs:
        x += shape(f - detune_hz, d) + shape(f + detune_hz, d)
    if lowpass:
        x = fft_lowpass(x, lowpass)
    return x


def lfo(rate_hz, d, depth=1.0, base=1.0):
    return base + depth * 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t(d) - np.pi / 2))


D = LOOP + 3.0  # generate with a tail for the wrap-crossfade

# ── Track generators ──────────────────────────────────────────────────────
# Pitches by name for readability.
def hz(name):
    names = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "Gb": 6,
             "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
    return 440.0 * 2 ** ((names[name[:-1]] - 9) / 12 + (int(name[-1]) - 4))


def menu():  # big hopeful Cmaj9 swell
    pad = drone([hz("C3"), hz("G3"), hz("E4"), hz("B4"), hz("D5")], D, lowpass=1800)
    pad *= lfo(1 / 16, D, depth=0.5, base=0.6)
    spark = np.zeros(int(SR * D))
    for i, n in enumerate(["C5", "D5", "E5", "G5", "B5", "G5", "E5", "D5"]):
        note_at(spark, i * 4.0 + 1.0, 0.10 * env_tone(hz(n), 3.0, 0.4, 1.2))
    return pad + spark


def space_federation():  # noble D-major brass pad, steady
    pad = drone([hz("D2"), hz("A2"), hz("D3"), hz("Gb3"), hz("A3")], D,
                lowpass=1200, shape=saw)
    pad *= lfo(1 / 8, D, depth=0.25, base=0.8)
    horn = np.zeros(int(SR * D))
    for i, n in enumerate(["D4", "A3", "D4", "E4"]):
        note_at(horn, i * 8.0, 0.35 * env_tone(hz(n), 5.0, 1.2, 2.5,
                                               partials=((1, 1), (2, 0.4), (3, 0.15))))
    return pad + fft_lowpass(horn, 1600)


def space_rebel():  # gritty E-minor drone + war-drum pulse
    pad = drone([hz("E2"), hz("G2"), hz("B2")], D, lowpass=700, shape=saw)
    drums = np.zeros(int(SR * D))
    thump = fft_lowpass(noise(0.22), 160) * np.exp(-t(0.22) / 0.05)
    for i in range(int(LOOP / 0.8)):
        acc = 1.0 if i % 4 == 0 else 0.5
        note_at(drums, i * 0.8, acc * thump)
    return pad * 0.7 + norm(drums, 0.9)


def space_freefrontier():  # open-fifth americana, sparse pentatonic guitar
    pad = drone([hz("G2"), hz("D3"), hz("G3")], D, lowpass=1400)
    pad *= lfo(1 / 16, D, depth=0.3, base=0.7)
    gtr = np.zeros(int(SR * D))
    lick = ["G3", "A3", "B3", "D4", "E4", "D4", "B3", "A3", "G3", "E3"]
    for i, n in enumerate(lick):
        note_at(gtr, i * 3.2 + 0.6, 0.30 * env_tone(hz(n), 2.4, 0.02, 0.7,
                                                    partials=((1, 1), (2, 0.5), (3, 0.2))))
    return pad + gtr


def space_helios():  # cold minimal electronica: 16th-note sine sequencer
    pad = drone([hz("A2"), hz("E3")], D, lowpass=900)
    seq = np.zeros(int(SR * D))
    steps = ["A4", "C5", "E5", "G5", "E5", "C5"]
    for i in range(int(LOOP / 0.25)):
        n = steps[i % len(steps)]
        amp = 0.16 if i % 4 == 0 else 0.08
        note_at(seq, i * 0.25, amp * env_tone(hz(n), 0.22, 0.005, 0.06))
    return pad * 0.6 + seq


def space_bastion():  # industrial: sub drone, anvil clank, forge thump
    pad = drone([hz("C1"), hz("G1"), hz("C2")], D, lowpass=320, shape=saw)
    hits = np.zeros(int(SR * D))
    clank = sum(a * sine(f, 1.2) for f, a in
                [(523 * 1.0, 1.0), (523 * 2.76, 0.6), (523 * 5.40, 0.35)])
    clank *= np.exp(-t(1.2) / 0.25)
    thump = sine(49, 0.4) * np.exp(-t(0.4) / 0.09)
    for i in range(int(LOOP / 2.0)):
        note_at(hits, i * 2.0, 0.5 * thump)
        if i % 2 == 1:
            note_at(hits, i * 2.0 + 1.0, 0.22 * clank)
    return pad * 0.8 + hits


def space_order():  # choral open fifths + temple bell
    pad = drone([hz("F2"), hz("C3"), hz("F3"), hz("C4")], D, lowpass=2000)
    pad = fft_bandboost(pad, 800, 250, 1.4)  # vowel-ish formant
    pad = fft_bandboost(pad, 1150, 200, 0.8)
    pad *= lfo(1 / 10.7, D, depth=0.35, base=0.65)
    bells = np.zeros(int(SR * D))
    bell = sum(a * sine(349 * r, 6.0) for r, a in
               [(1, 1.0), (2.0, 0.5), (2.98, 0.25), (4.2, 0.12)])
    bell *= np.exp(-t(6.0) / 1.8)
    for s in [0.0, 8.0, 16.0, 24.0]:
        note_at(bells, s, 0.18 * bell)
    return pad + bells


def space_pirate():  # tense minor-second drone, tremolo, sparse low hits
    pad = drone([hz("B2"), hz("C3")], D, lowpass=900) * lfo(6.0, D, 0.4, 0.6)
    pad += drone([hz("E2")], D, lowpass=500)
    hits = np.zeros(int(SR * D))
    hit = fft_lowpass(noise(0.3), 220) * np.exp(-t(0.3) / 0.06)
    for s in [0.0, 3.4, 5.0, 11.2, 14.6, 19.0, 24.8, 27.0]:
        note_at(hits, s, 0.55 * hit)
    return pad * 0.8 + hits


def surface_garden():  # pastoral major pad + high sparkle
    pad = drone([hz("C3"), hz("E3"), hz("G3"), hz("C4")], D, lowpass=1600)
    pad *= lfo(1 / 12, D, depth=0.3, base=0.7)
    spark = np.zeros(int(SR * D))
    for _ in range(24):
        n = rng.choice([hz("E5"), hz("G5"), hz("A5"), hz("C6"), hz("D6")])
        note_at(spark, rng.uniform(0, LOOP), 0.06 * env_tone(n, 1.5, 0.02, 0.4))
    return pad + spark


def surface_ice():  # glassy detuned highs over a cold floor
    pad = drone([hz("A3"), hz("E4"), hz("A4"), hz("B4")], D, detune_hz=0.6,
                lowpass=4000)
    pad *= lfo(1 / 16, D, depth=0.4, base=0.55)
    floor = drone([hz("A1")], D, lowpass=200)
    return pad * 0.6 + floor * 0.5


def surface_rocky():  # dark low fifth, slow-breathing filter
    x = drone([hz("D2"), hz("A2"), hz("D3")], D, lowpass=500, shape=saw)
    return x * lfo(1 / 13, D, depth=0.5, base=0.5)


def surface_desert():  # sparse bent drone + far whistle
    bend = 1.0 + 0.012 * np.sin(2 * np.pi * t(D) / 16.0)  # slow quarter-tone bow
    base = np.sin(2 * np.pi * hz("E2") * np.cumsum(bend) / SR)
    base += np.sin(2 * np.pi * hz("B2") * np.cumsum(bend) / SR) * 0.7
    whistle = np.zeros(int(SR * D))
    for s in [5.0, 19.0]:
        note_at(whistle, s, 0.08 * env_tone(hz("E5"), 4.0, 1.5, 1.5))
    return fft_lowpass(base, 800) + whistle


def surface_interior():  # warm station hum, vent wash
    hum = drone([110.0, 220.0, 165.0], D, detune_hz=0.35, lowpass=600)
    vent = fft_lowpass(noise(D), 350) * lfo(1 / 9, D, depth=0.5, base=0.5)
    return hum * 0.7 + vent * 0.35


def bar():  # jazz placeholder: swung ride, walking bass, soft stabs (Gm7-C7-F)
    x = np.zeros(int(SR * D))
    beat = 0.5  # 120 BPM
    walk = ["G2", "Bb2", "C3", "E3", "F3", "A3", "G3", "E3",
            "F3", "C3", "A2", "C3", "D3", "Gb3", "A3", "C4"]
    for i in range(int(LOOP / beat)):
        # ride: swung tick pair
        tick = fft_bandboost(noise(0.05), 6000, 2000, 3.0) * np.exp(-t(0.05) / 0.012)
        note_at(x, i * beat, 0.10 * tick)
        note_at(x, i * beat + beat * 0.66, 0.05 * tick)
        # walking bass, quarter notes
        n = walk[i % len(walk)]
        note_at(x, i * beat, 0.5 * env_tone(hz(n), 0.45, 0.01, 0.35,
                                            partials=((1, 1), (2, 0.3))))
    for i, chord in enumerate([["G3", "Bb3", "D4", "F4"], ["G3", "Bb3", "C4", "E4"],
                               ["A3", "C4", "E4", "F4"], ["A3", "C4", "D4", "Gb4"]]):
        stab = sum(env_tone(hz(n), 1.2, 0.05, 0.5) for n in chord)
        note_at(x, i * 8.0 + 1.0, 0.09 * stab)
        note_at(x, i * 8.0 + 5.0, 0.07 * stab)
    return x


TRACKS = {
    "menu": menu,
    "space_federation": space_federation,
    "space_rebel": space_rebel,
    "space_freefrontier": space_freefrontier,
    "space_helios": space_helios,
    "space_bastion": space_bastion,
    "space_order": space_order,
    "space_pirate": space_pirate,
    "surface_garden": surface_garden,
    "surface_ice": surface_ice,
    "surface_rocky": surface_rocky,
    "surface_desert": surface_desert,
    "surface_interior": surface_interior,
    "bar": bar,
}


def write_ogg(name, x):
    x = norm(loopify(x))
    wav_path = os.path.join(OUT_DIR, f"{name}.wav")
    ogg_path = os.path.join(OUT_DIR, f"{name}.ogg")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", wav_path,
                    "-c:a", "libvorbis", "-q:a", "3", ogg_path], check=True)
    os.remove(wav_path)
    print(f"  {name}.ogg  ({os.path.getsize(ogg_path) // 1024} KiB)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Synthesizing {len(TRACKS)} placeholder loops → assets/music/")
    for name, gen in TRACKS.items():
        write_ogg(name, gen())
    print("Done.")
