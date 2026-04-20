"""
Tests for Blob-47 auto-tile bitmask logic.

Verifies that the Python implementation matches Boris The Brave's
canonical pick_tile dictionary and the shared YAML test cases.

Reference: https://www.boristhebrave.com/2013/07/14/tileset-roundup/
"""
import yaml
from pathlib import Path

TEST_CASES_PATH = Path(__file__).parent / "blob47_test_cases.yaml"

# ── Bit constants (Boris The Brave's standard) ───────────────────────────

TL, T, TR = 1, 2, 4
L, R = 8, 16
BL, B, BR = 32, 64, 128


def reduce_to_47(mask: int) -> int:
    if not (mask & L) or not (mask & T):
        mask &= ~TL
    if not (mask & R) or not (mask & T):
        mask &= ~TR
    if not (mask & L) or not (mask & B):
        mask &= ~BL
    if not (mask & R) or not (mask & B):
        mask &= ~BR
    return mask


# Build LUT: sorted unique reduced masks → column index
_BLOB47_SORTED = sorted({reduce_to_47(m) for m in range(256)})
BLOB47_LUT = [255] * 256
for _col, _mask in enumerate(_BLOB47_SORTED):
    BLOB47_LUT[_mask] = _col


# ── Boris The Brave's canonical pick_tile dict ───────────────────────────
# https://www.boristhebrave.com/2013/07/14/tileset-roundup/
PICK_TILE = {
    0: 0, 2: 1, 8: 2, 10: 3, 11: 4, 16: 5, 18: 6,
    22: 7, 24: 8, 26: 9, 27: 10, 30: 11, 31: 12, 64: 13,
    66: 14, 72: 15, 74: 16, 75: 17, 80: 18, 82: 19, 86: 20,
    88: 21, 90: 22, 91: 23, 94: 24, 95: 25, 104: 26, 106: 27,
    107: 28, 120: 29, 122: 30, 123: 31, 126: 32, 127: 33,
    208: 34, 210: 35, 214: 36, 216: 37, 218: 38, 219: 39,
    222: 40, 223: 41, 248: 42, 250: 43, 251: 44, 254: 45, 255: 46,
}


def test_lut_matches_pick_tile():
    """Every entry in Boris The Brave's pick_tile must match our LUT."""
    for mask, expected_col in PICK_TILE.items():
        actual = BLOB47_LUT[mask]
        assert actual == expected_col, (
            f"mask={mask}: LUT gives col {actual}, expected {expected_col}"
        )


def test_exactly_47_mapped():
    """Exactly 47 entries in the 256-element LUT should be mapped."""
    mapped = sum(1 for v in BLOB47_LUT if v != 255)
    assert mapped == 47, f"Expected 47 mapped entries, got {mapped}"


def test_reduce_to_47_idempotent():
    """reduce_to_47 applied twice gives the same result as once."""
    for m in range(256):
        once = reduce_to_47(m)
        twice = reduce_to_47(once)
        assert once == twice, f"mask={m}: reduce({m})={once}, reduce({once})={twice}"


def test_reduce_to_47_always_maps():
    """Every reduced mask should map to a valid column (never 255)."""
    for m in range(256):
        reduced = reduce_to_47(m)
        col = BLOB47_LUT[reduced]
        assert col != 255, f"mask={m} reduces to {reduced} which is unmapped"


def _parse_grid(grid_str: str) -> int:
    """Parse a 3x3 ASCII grid into a bitmask.

    Grid layout (rows top-to-bottom):
        TL T TR
        L  X  R
        BL B BR

    '#' = same terrain (bit set), '.' = different (bit clear).
    """
    rows = [line.strip() for line in grid_str.strip().splitlines()]
    assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
    assert all(len(r) == 3 for r in rows), f"Each row must be 3 chars: {rows}"

    bit_map = [
        (0, 0, TL), (0, 1, T), (0, 2, TR),
        (1, 0, L),             (1, 2, R),
        (2, 0, BL), (2, 1, B), (2, 2, BR),
    ]
    mask = 0
    for row, col, bit in bit_map:
        if rows[row][col] == '#':
            mask |= bit
    return mask


def test_yaml_cases():
    """Verify each YAML test case: grid → raw mask → reduce → LUT column."""
    cases = yaml.safe_load(TEST_CASES_PATH.read_text())
    for case in cases:
        name = case["name"]
        grid = case["grid"]
        expected_mask = case["mask"]
        expected_col = case["column"]

        raw_mask = _parse_grid(grid)
        reduced = reduce_to_47(raw_mask)
        assert reduced == expected_mask, (
            f"[{name}] raw mask from grid = {raw_mask}, "
            f"reduced = {reduced}, expected {expected_mask}"
        )
        col = BLOB47_LUT[reduced]
        assert col == expected_col, (
            f"[{name}] mask {reduced} → col {col}, expected {expected_col}"
        )


if __name__ == "__main__":
    test_lut_matches_pick_tile()
    print("PASS: LUT matches Boris The Brave's pick_tile")

    test_exactly_47_mapped()
    print("PASS: Exactly 47 mapped entries")

    test_reduce_to_47_idempotent()
    print("PASS: reduce_to_47 is idempotent")

    test_reduce_to_47_always_maps()
    print("PASS: Every reduced mask maps to a valid column")

    test_yaml_cases()
    print("PASS: All YAML test cases match")

    print("\nAll tests passed.")
