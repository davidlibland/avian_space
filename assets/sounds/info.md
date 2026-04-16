To find sounds use:
- (freesound.org)[https://freesound.org] — huge library, CC0 and CC-BY. Search "laser", "explosion", "thruster", "spaceship". CC-BY requires attribution.
- (OpenGameArt.org)[https://opengameart.org] — curated for games, many CC0 sci-fi SFX packs (e.g. "Space Shooter SFX", Kenney's packs).
- (Kenney.nl)[https://kenney.nl/assets?q=audio] — CC0, high-quality, well-organized packs ("Sci-Fi Sounds", "Impact Sounds", "UI Audio"). Best starting point — zero attribution hassle.
- (sonniss.com/gameaudiogdc)[https://sonniss.com/gameaudiogdc] — free GDC bundles, royalty-free, tens of GB of pro SFX.
- (pixabay.com/sound-effects)[https://pixabay.com/sound-effects/] — permissive license, no attribution.

To convert sounds: run `ffmpeg -i in.wav -c:a libvorbis -q:a 5 out.ogg`