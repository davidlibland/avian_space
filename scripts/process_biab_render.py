#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "imageio-ffmpeg",
# ]
# ///
"""Turn a Band-in-a-Box render into a seamless in-game music loop.

Expects a single 32-bar chorus (the brief-pack form) rendered with no
generated ending: the file is LOOP + reverb tail. The loop length is
128 beats at the render tempo; the tail is overlap-added onto the head
so the reverb continues across the seam. Output: stereo ogg at
assets/music/<stem>.ogg, peak-normalized.

Usage:  uv run scripts/process_biab_render.py <in.wav> <stem> <tempo_bpm>
   e.g. uv run scripts/process_biab_render.py "The Fleet_Render.wav" space_federation 84
"""
import os
import subprocess
import sys
import wave

import imageio_ffmpeg
import numpy as np

BEATS = 128  # 32 bars of 4/4 — the brief-pack form


def main(in_path, stem, tempo):
    # Decode via ffmpeg to float32 — BIAB exports 24-bit WAVs, which the
    # wave module would happily hand us as garbage if read as int16.
    with wave.open(in_path) as w:
        sr = w.getframerate()
        ch = w.getnchannels()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run(
        [ff, "-loglevel", "error", "-i", in_path, "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).astype(np.float64)

    loop_secs = BEATS * 60.0 / float(tempo)
    n = round(loop_secs * sr)
    if n > len(x):
        sys.exit(f"file shorter ({len(x)/sr:.1f}s) than the 32-bar loop ({loop_secs:.1f}s) "
                 f"— wrong tempo?")
    tail = x[n:]
    body = x[:n].copy()
    # Reverb tail continues over the head — that IS the seamless seam.
    k = min(len(tail), n)
    body[:k] += tail[:k]
    # Short safety crossfade in case the downbeat cut clicks.
    f = round(0.02 * sr)
    ramp = np.linspace(0.0, 1.0, f)[:, None]
    body[:f] *= ramp
    body[-f:] *= ramp[::-1]

    peak = np.max(np.abs(body))
    if peak > 0:
        body *= 0.90 / peak

    out_dir = os.path.join(os.path.dirname(__file__), "..", "assets", "music")
    tmp_wav = os.path.join(out_dir, f"{stem}_tmp.wav")
    out_ogg = os.path.join(out_dir, f"{stem}.ogg")
    with wave.open(tmp_wav, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((body * 32767).astype(np.int16).tobytes())
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp_wav,
                    "-c:a", "libvorbis", "-q:a", "5", out_ogg], check=True)
    os.remove(tmp_wav)
    rms = float(np.sqrt(np.mean(body**2)))
    crest = 0.90 / max(rms, 1e-9)
    print(f"{stem}.ogg  {n/sr:.1f}s loop  rms={rms:.3f} crest={crest:.1f}  "
          f"({os.path.getsize(out_ogg)//1024} KiB)")
    if crest < 2.0:
        print(f"  WARNING: crest factor {crest:.1f} looks like noise, not music")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
