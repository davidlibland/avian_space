#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "imageio-ffmpeg",
# ]
# ///
"""Synthesize the planet-ambience sound pack (assets/sounds/ambience/).

Two kinds of files, all license-free in-house synthesis:
  * beds — seamless ~12s loops the AmbienceDirector crossfades by context
    (biome wind, water/lava proximity, venue machinery, bar murmur)
  * one-shots — short positional critter calls and venue accents, played
    near their source with pitch jitter

Stylized FM/noise synthesis on purpose: toon fauna want toon calls.

Run:  uv run scripts/synth_ambience.py
"""
import numpy as np, wave, subprocess, os
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 44100
LOOP = 12.0
rng = np.random.default_rng(0xAB1E4CE)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "sounds", "ambience")


def t(d):
    return np.linspace(0, d, int(SR * d), endpoint=False)


def sine(f, d, ph=0.0):
    return np.sin(2 * np.pi * f * t(d) + ph)


def fm(f, d, mod_hz, mod_depth):
    """Sine with vibrato/FM: instantaneous freq f + depth*sin(mod)."""
    inst = f + mod_depth * np.sin(2 * np.pi * mod_hz * t(d))
    return np.sin(2 * np.pi * np.cumsum(inst) / SR)


def sweep(f0, f1, d, curve=1.0):
    f = f0 + (f1 - f0) * (t(d) / d) ** curve
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def noise(d):
    return rng.uniform(-1, 1, int(SR * d))


def fft_lowpass(x, cutoff):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    gain = 1.0 / (1.0 + (freqs / max(cutoff, 1.0)) ** 2)
    return np.fft.irfft(spec * gain, n=len(x))


def fft_bandpass(x, center, width):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    gain = np.exp(-((freqs - center) ** 2) / (2 * width**2))
    return np.fft.irfft(spec * gain, n=len(x))


def env(x, attack, tau):
    tt = np.arange(len(x)) / SR
    return x * np.minimum(tt / max(attack, 1e-3), 1.0) * np.exp(-tt / tau)


def note_at(x, start, tone):
    i = int(start * SR) % len(x)
    end = i + len(tone)
    if end <= len(x):
        x[i:end] += tone
    else:
        k = len(x) - i
        x[i:] += tone[:k]
        x[: end - len(x)] += tone[k:]
    return x


def lfo(rate_hz, d, depth=1.0, base=1.0):
    return base + depth * 0.5 * (1 + np.sin(2 * np.pi * rate_hz * t(d) - np.pi / 2))


def loopify(x, xfade=1.5):
    n = int(SR * xfade)
    body = x[: int(SR * LOOP)].copy()
    tail = x[int(SR * LOOP) : int(SR * LOOP) + n]
    ramp = np.linspace(0, 1, n)
    body[:n] = body[:n] * ramp + tail[: len(ramp)] * (1 - ramp)
    return body


def fade_ends(x, ms=6):
    n = int(SR * ms / 1000)
    if len(x) > 2 * n:
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
    return x


def norm(x, peak=0.6):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


BD = LOOP + 2.0  # bed generation length (pre-loopify)

# ── Beds ──────────────────────────────────────────────────────────────────
def wind_garden():  # soft breeze + leaf rustle band
    base = fft_lowpass(noise(BD), 350) * lfo(1 / 6, BD, 0.6, 0.5)
    leaves = fft_bandpass(noise(BD), 2400, 900) * lfo(1 / 3.7, BD, 0.8, 0.25)
    return base + leaves * 0.5


def wind_ice():  # howling, resonant
    x = fft_lowpass(noise(BD), 900) * lfo(1 / 5, BD, 0.7, 0.4)
    howl = fft_bandpass(noise(BD), 620, 60) * lfo(1 / 7.3, BD, 0.9, 0.2)
    howl += fft_bandpass(noise(BD), 880, 50) * lfo(1 / 4.1, BD, 0.9, 0.15)
    return x * 0.6 + howl * 2.2


def wind_rocky():  # low desolate rumble-wind
    x = fft_lowpass(noise(BD), 220) * lfo(1 / 8, BD, 0.5, 0.6)
    grit = fft_bandpass(noise(BD), 1300, 500) * lfo(1 / 2.9, BD, 0.7, 0.12)
    return x + grit * 0.4


def wind_desert():  # dry hiss, sand on wind
    hiss = fft_bandpass(noise(BD), 3400, 1600) * lfo(1 / 5.3, BD, 0.7, 0.35)
    low = fft_lowpass(noise(BD), 300) * lfo(1 / 9, BD, 0.5, 0.4)
    return hiss * 0.7 + low * 0.7


def water_lap():  # gentle shoreline
    wash = fft_lowpass(noise(BD), 700) * lfo(1 / 4.7, BD, 0.8, 0.3)
    x = np.array(wash)
    lap = env(fft_lowpass(noise(1.0), 900), 0.15, 0.25)
    for s in np.cumsum(rng.uniform(1.2, 2.6, 8)):
        note_at(x, s, lap * rng.uniform(0.5, 1.0))
    return x


def lava_bubble():  # magma: brown floor + bubble pops + hiss
    floor = fft_lowpass(noise(BD), 130) * lfo(1 / 6.1, BD, 0.4, 0.8)
    x = np.array(floor)
    for s in np.cumsum(rng.uniform(0.4, 1.4, 14)):
        f0 = rng.uniform(110, 220)
        pop = env(sweep(f0, f0 * 0.4, 0.12), 0.01, 0.05)
        note_at(x, s, pop * rng.uniform(0.4, 0.9))
    hiss = fft_bandpass(noise(BD), 2600, 1400) * lfo(1 / 3.3, BD, 0.9, 0.06)
    return x + hiss


def mine_rumble():  # deep earth
    x = fft_lowpass(noise(BD), 75) * lfo(1 / 7, BD, 0.4, 0.9)
    x += 0.25 * sine(52, BD) * lfo(1 / 5.2, BD, 0.6, 0.5)
    airy = fft_bandpass(noise(BD), 500, 250) * 0.1
    return x + airy


def substation_hum():  # mains hum + electrical air
    x = 0.7 * sine(60, BD) + 0.4 * sine(120, BD) + 0.18 * sine(180, BD)
    x *= lfo(1 / 4.3, BD, 0.25, 0.8)
    crackle = fft_bandpass(noise(BD), 5200, 2200) * lfo(11, BD, 0.7, 0.10)
    return x + crackle * 0.5 + fft_lowpass(noise(BD), 200) * 0.25


def warehouse_fans():  # big slow fans + motor
    blades = fft_bandpass(noise(BD), 420, 260) * lfo(9.0, BD, 0.45, 0.7)
    motor = 0.5 * sine(90, BD) + 0.2 * sine(182, BD)
    air = fft_lowpass(noise(BD), 500) * 0.4
    return blades + motor * lfo(1 / 6, BD, 0.2, 0.85) + air


def interior_hum():  # quiet life-support
    hum = 0.5 * sine(110, BD) + 0.25 * sine(220, BD) + 0.12 * sine(165, BD)
    hum *= lfo(1 / 8, BD, 0.2, 0.85)
    vent = fft_lowpass(noise(BD), 400) * lfo(1 / 5.5, BD, 0.4, 0.4)
    return hum * 0.6 + vent * 0.6


def bar_murmur():  # crowd voices + clinks
    x = np.zeros(int(SR * BD))
    for _ in range(9):  # murmuring "voices": syllabic band-noise
        f = rng.uniform(160, 340)
        voice = fft_bandpass(noise(BD), f * 2.2, f * 0.9)
        syllables = np.clip(fft_lowpass(noise(BD), 4.5), 0, 1)
        x += voice * syllables * rng.uniform(0.5, 1.0)
    for s in np.cumsum(rng.uniform(1.5, 3.5, 5)):  # glass clinks
        f = rng.uniform(1800, 3200)
        clink = env(sine(f, 0.25) + 0.4 * sine(f * 2.51, 0.25), 0.002, 0.06)
        note_at(x, s, clink * rng.uniform(0.10, 0.2))
    return x


# ── One-shots ─────────────────────────────────────────────────────────────
def bird_song():  # 3-phrase warble
    x = np.zeros(int(SR * 1.2))
    for i, (f, dur) in enumerate([(3400, 0.14), (3900, 0.10), (3100, 0.18)]):
        chirp = env(fm(f, dur, 28, 320), 0.01, dur * 0.6)
        note_at(x, 0.1 + i * 0.28, chirp * 0.9)
    return fade_ends(x)


def deer_huff():  # breathy snort
    x = env(fft_lowpass(noise(0.3), 380), 0.02, 0.08)
    return fade_ends(x)


def rustle():  # undergrowth crinkle
    x = fft_bandpass(noise(0.35), 2100, 1200)
    x *= np.clip(fft_lowpass(noise(0.35), 22), 0, 1) * 2.0
    return fade_ends(env(x, 0.03, 0.15))


def fox_yip():  # two quick yips
    x = np.zeros(int(SR * 0.5))
    for i in range(2):
        yip = env(sweep(1250, 620, 0.12, 0.7), 0.008, 0.05)
        note_at(x, i * 0.22, yip)
    return fade_ends(x)


def stone_grumble():  # rock monster: low grind + cracks
    grind = env(fft_lowpass(noise(0.9), 140), 0.1, 0.35)
    x = np.array(grind)
    for s in [0.15, 0.4, 0.62]:
        crack = env(noise(0.03), 0.001, 0.01)
        note_at(x, s, crack * 0.5)
    x += 0.4 * env(fm(68, 0.9, 6, 8), 0.1, 0.3)
    return fade_ends(x)


def lava_hiss():  # salamander: steam vent
    x = fft_bandpass(noise(0.6), 3000, 1800) * np.exp(-t(0.6) / 0.25)
    x += 0.3 * env(sweep(200, 90, 0.6), 0.05, 0.2)
    return fade_ends(x)


def ice_groan():  # glacier creak, descending
    x = env(fm(190, 1.4, 4.5, 40) * 0.7 + fm(95, 1.4, 3.2, 18) * 0.5, 0.15, 0.5)
    x *= 1.0 + 0.5 * np.sin(2 * np.pi * 13 * t(1.4))  # stick-slip judder
    return fade_ends(x)


def gull_cry():  # petrel: harsh two-note cry
    a = env(fm(1350, 0.28, 9, 90), 0.02, 0.12)
    a = np.tanh(2.5 * a)
    b = env(fm(1080, 0.22, 9, 70), 0.02, 0.10)
    b = np.tanh(2.5 * b)
    x = np.zeros(int(SR * 0.65))
    note_at(x, 0.0, a)
    note_at(x, 0.32, b)
    return fade_ends(x)


def vulture_caw():  # ragged croak
    x = env(np.sign(sine(420, 0.3)) * 0.5 + fm(420, 0.3, 30, 60) * 0.5, 0.01, 0.12)
    x *= 1.0 + 0.6 * fft_lowpass(noise(0.3), 60)
    return fade_ends(np.tanh(2.0 * x))


def skitter():  # chitinous tick train
    x = np.zeros(int(SR * 0.45))
    s = 0.0
    while s < 0.38:
        tick = env(fft_bandpass(noise(0.02), 4200, 1800), 0.001, 0.006)
        note_at(x, s, tick * rng.uniform(0.5, 1.0))
        s += rng.uniform(0.03, 0.07)
    return fade_ends(x)


def rat_squeak():
    x = env(fm(3300, 0.16, 22, 500), 0.01, 0.06)
    return fade_ends(x)


def bat_flutter():  # rapid wing flaps
    flaps = fft_bandpass(noise(0.45), 900, 500)
    flaps *= np.clip(np.sin(2 * np.pi * 14 * t(0.45)), 0, 1)
    return fade_ends(env(flaps, 0.02, 0.2) * 1.6)


def crab_clack():  # two shell clicks
    x = np.zeros(int(SR * 0.35))
    for s in [0.0, 0.14]:
        click = env(sine(820, 0.05) + 0.5 * sine(1970, 0.05), 0.001, 0.012)
        note_at(x, s, click)
    return fade_ends(x)


def bot_beep():  # cheerful sweeper-bot double beep
    x = np.zeros(int(SR * 0.4))
    note_at(x, 0.0, env(sine(1180, 0.09), 0.004, 0.05))
    note_at(x, 0.14, env(sine(1570, 0.11), 0.004, 0.06))
    return fade_ends(x)


def drone_whir():  # rotor pass-by
    d = 0.8
    rotor = fft_bandpass(noise(d), 300, 160) * lfo(52, d, 0.5, 0.7)
    motor = fm(230, d, 2.0, 25) * 0.4
    swell = np.sin(np.pi * t(d) / d)  # rise and fade
    return fade_ends((rotor + motor) * swell)


def gecko_chirp():  # 3 descending blips
    x = np.zeros(int(SR * 0.5))
    for i, f in enumerate([2300, 2050, 1800]):
        note_at(x, i * 0.13, env(fm(f, 0.08, 40, 180), 0.005, 0.035))
    return fade_ends(x)


def drip():  # cave drip plink
    x = env(sweep(1350, 420, 0.16, 0.4), 0.002, 0.05)
    echo = np.zeros(int(SR * 0.6))
    note_at(echo, 0.0, x)
    note_at(echo, 0.22, x * 0.3)
    return fade_ends(echo)


def pipe_ping():  # metallic pipe knock, ringing
    partials = [(1.0, 1.0), (2.76, 0.55), (5.40, 0.3), (8.93, 0.15)]
    x = sum(a * env(sine(640 * r, 0.9), 0.002, 0.18 / (i + 1))
            for i, (r, a) in enumerate(partials))
    return fade_ends(x)


BEDS = {
    "wind_garden": wind_garden,
    "wind_ice": wind_ice,
    "wind_rocky": wind_rocky,
    "wind_desert": wind_desert,
    "water_lap": water_lap,
    "lava_bubble": lava_bubble,
    "mine_rumble": mine_rumble,
    "substation_hum": substation_hum,
    "warehouse_fans": warehouse_fans,
    "interior_hum": interior_hum,
    "bar_murmur": bar_murmur,
}

SHOTS = {
    "bird_song": bird_song,
    "deer_huff": deer_huff,
    "rustle": rustle,
    "fox_yip": fox_yip,
    "stone_grumble": stone_grumble,
    "lava_hiss": lava_hiss,
    "ice_groan": ice_groan,
    "gull_cry": gull_cry,
    "vulture_caw": vulture_caw,
    "skitter": skitter,
    "rat_squeak": rat_squeak,
    "bat_flutter": bat_flutter,
    "crab_clack": crab_clack,
    "bot_beep": bot_beep,
    "drone_whir": drone_whir,
    "gecko_chirp": gecko_chirp,
    "drip": drip,
    "pipe_ping": pipe_ping,
}


def write_ogg(name, x):
    wav_path = os.path.join(OUT_DIR, f"{name}.wav")
    ogg_path = os.path.join(OUT_DIR, f"{name}.ogg")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", wav_path,
                    "-c:a", "libvorbis", "-q:a", "2", ogg_path], check=True)
    os.remove(wav_path)
    print(f"  {name}.ogg  ({os.path.getsize(ogg_path) // 1024} KiB)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Beds ({len(BEDS)} seamless {LOOP:.0f}s loops):")
    for name, gen in BEDS.items():
        write_ogg(name, norm(loopify(gen()), 0.55))
    print(f"One-shots ({len(SHOTS)}):")
    for name, gen in SHOTS.items():
        write_ogg(name, norm(gen(), 0.7))
    print("Done.")
