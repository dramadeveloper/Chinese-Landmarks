"""OPRL v5 shared, screen-aligned stone paving (40 model units per map tile)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PALETTE = np.array(json.loads((ROOT / 'ttd_dos_palette.json').read_text()), dtype=np.uint8)
TILE = 40
SCALE = .8
STONE, JOINT, CURB, SHADOW, SHADOW_JOINT = 11, 10, 12, 8, 7
SAFE = {0} | (set(range(1, 215)) - set(range(80, 88)) - set(range(198, 206)))


def paving(x, y, shadow=False, *, shadow_colors=(SHADOW, SHADOW_JOINT)):
    """20 x 10 staggered slabs, both axes repeat every 40 units.

    Coordinates are in the rotated world plane. Rotating scenery never rotates
    the shared paving or its seam phase. No per-tile frame or texture noise.
    """
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)
    seams = (y % 10 == 0) | ((x + ((y // 10) % 2) * 10) % 20 == 0)
    return np.where(shadow, np.where(seams, shadow_colors[1], shadow_colors[0]),
                    np.where(seams, JOINT, STONE)).astype(np.uint8)


def save_p(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(array.astype(np.uint8)).convert('P')
    im.putpalette(PALETTE.ravel().tolist())
    im.save(path, transparency=0, optimize=False, bits=8)
    return im


def validate_png(path):
    with Image.open(path) as im:
        assert im.mode == 'P' and im.info.get('transparency') == 0, path
        assert im.getpalette() == PALETTE.ravel().tolist(), path
        assert set(np.unique(np.array(im))) <= SAFE, path
        assert Path(path).read_bytes()[24:26] == bytes([8, 3]), path


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', size)
