"""
generate_3d_ships.py — render the prototype 3D ships to PNGs.

Produces, for each ship:
  * a game-sized sprite (matching ships.yaml radius * 2.2), nose pointing UP
  * a large preview (256px) so the aesthetics are easy to judge

Output goes to scripts/ship3d/out/.
"""

import os

from render3d import render
from ships3d import build_asteroid_miner, build_fighter, build_hauler

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

# fighter radius 12 -> 28, hauler radius 40 -> 88, asteroid_miner radius 20 -> 44.
# world_half is the object-space half-extent that fills the frame; our meshes
# span roughly y in [-1.1, 1.1] so 1.25 leaves a little margin.
SHIPS = [
    ("fighter",        build_fighter,        28,  1.25, "blue"),
    ("hauler",         build_hauler,         88,  1.45, "warm"),
    ("asteroid_miner", build_asteroid_miner, 44,  1.25, "warm"),
]


def main():
    for name, builder, game_size, wh, glow in SHIPS:
        mesh = builder()
        print(f"{name}: {len(mesh.tris)} tris")
        common = dict(world_half=wh, ss=4, cel_bands=0,
                      outline_strength=0.7, ambient=0.36, fill=0.20,
                      spec=0.55, shininess=32)
        # big preview
        render(mesh, 256, **common).save(os.path.join(OUT, f"{name}_preview.png"))
        # game-sized sprite (lighter outline at small size)
        render(mesh, game_size, **{**common, "outline_strength": 0.55}).save(
            os.path.join(OUT, f"{name}.png"))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
