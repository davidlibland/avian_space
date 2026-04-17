#!/usr/bin/env bash
# Convert all .wav files under assets/sounds/ to .ogg (Vorbis).
# Skips kenney_* library directories.
# Requires ffmpeg and oggenc (installs via Homebrew if missing).

set -euo pipefail

SOUNDS_DIR="$(cd "$(dirname "$0")/../assets/sounds" && pwd)"

for cmd in ffmpeg oggenc; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "$cmd not found — installing via Homebrew..."
        case "$cmd" in
            oggenc) brew install vorbis-tools ;;
            *)      brew install "$cmd" ;;
        esac
    fi
done

find "$SOUNDS_DIR" -name "*.wav" -not -path "*/kenney_*" | while read -r wav; do
    ogg="${wav%.wav}.ogg"
    echo "Converting: $(basename "$wav") -> $(basename "$ogg")"
    # Normalize to 16-bit 44.1kHz stereo WAV, then encode to Vorbis via oggenc.
    ffmpeg -y -i "$wav" -f wav -acodec pcm_s16le -ar 44100 -ac 2 /tmp/_wav2ogg_tmp.wav 2>/dev/null
    oggenc -q 5 -o "$ogg" /tmp/_wav2ogg_tmp.wav 2>/dev/null
    rm "$wav"
done

rm -f /tmp/_wav2ogg_tmp.wav
echo "Done. Remember to update any .wav references in code/yaml to .ogg."
