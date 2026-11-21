"""Museum materials and planar-face correction; shared projection/lighting."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'chinese-landmarks'))
from render_core import (DEFAULT_SCENE, DEFAULT_SAMPLING, trace_model,
                         surface_shadow, shade, indexed, stone_ground, finish)
from export_plaza import RAMPS as PROP_RAMPS

RAMPS = {
    1: [6, 7, 8, 9, 10, 11, 12, 13],
    2: [178, 179, 180, 162, 163, 164, 165],
    3: [178, 179, 180, 181, 182, 183],
    4: [1, 3, 4, 5, 6, 7],
    5: [5, 7, 9, 10, 11, 12, 13],
    6: [5, 6, 7, 8, 9, 10, 11, 12],
    7: [2, 3, 4, 5, 6, 7, 8],
    8: [89, 90, 91, 92, 93],
    9: [154, 155, 156, 157, 158, 158, 159],
    10: [5, 6, 7, 7, 8, 9, 10, 11],
    **{key+20: ramp for key, ramp in PROP_RAMPS.items()},
}


def render(model, view, *, lighting=DEFAULT_SCENE, canvas=(352, 240), origin=(176, 56)):
    frame = trace_model(model, view, canvas, origin)
    normal = frame.normal.copy()
    # Pixel-centre ties can select an internal voxel side on a continuous plane.
    # Replace confirmed shared faces with the exposed top or adjoining riser.
    x, y, z = frame.xyz.transpose(2, 0, 1)
    padded = np.pad(frame.model, ((0, 1), (0, 1), (0, 1)))
    interior = padded[x+(frame.face == 0), y+(frame.face == 1), z] != 0
    top_open = padded[x, y, z+1] == 0
    shared = frame.hit & (frame.mat < 20) & (frame.face != 2) & interior
    normal[shared & top_open] = [0, 0, 1]
    normal[shared & ~top_open & (padded[x+1, y, z] == 0)] = [1, 0, 0]
    normal[shared & ~top_open & (padded[x, y+1, z] == 0)] = [0, 1, 0]
    shadow = surface_shadow(frame.model, frame.point, normal, frame.hit, DEFAULT_SAMPLING)
    pixels = indexed(frame.mat, shade(normal, shadow, lighting.standard), frame.hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), pixels)
