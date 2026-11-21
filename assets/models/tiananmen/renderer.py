"""Material treatment for the gate tower in the accepted Tiananmen camera."""
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import tiananmen_tower as tower
from plaza_material import SCALE, paving
from render_core import DEFAULT_SCENE, finish, indexed, shade, trace_model

RAMPS = {
    1: [4, 5, 6, 7, 8, 9, 10, 11],
    2: [178, 179, 180, 181, 162, 163, 164],
    3: [60, 61, 62, 63, 64, 65, 66, 67],
    4: [178, 179, 180, 181, 162, 163],
    5: [96, 97, 98, 99, 100, 101, 102],
    6: [1, 2, 3, 4],
    7: [154, 155, 156, 157, 158, 159, 160],
    8: [88, 89, 90, 91, 92, 93, 94],
    9: [88, 89, 90, 91, 92, 93],
    10: [32, 33, 34, 35, 36],
    11: [7, 8, 9, 10, 11, 12, 13, 14],
    12: [114, 115, 116, 117, 118, 120, 121],
    13: [16, 17, 18, 19, 20],
    tower.TILE: [61, 61, 62, 62, 63, 63, 64, 64, 64, 192, 192, 192, 193, 65, 65, 195, 195],
    tower.CAP: [61, 62, 63, 64, 65, 193, 195, 196],
    tower.FASCIA: [70, 71, 72, 73, 74, 75],
    tower.POST: [41, 42, 43, 162, 163, 163],
    tower.PANEL: [40, 41, 42, 43, 44, 162],
    tower.WINDOW: [70, 71, 72, 73, 74],
    tower.BEAM: [16, 17, 156, 98, 99, 100],
    tower.GILT: [55, 56, 64, 65, 195, 196],
    tower.GABLE: [34, 35, 36, 37, 38, 39],
}


def render(model, view, *, lighting=DEFAULT_SCENE):
    nx = model.shape[view % 2]
    frame = trace_model(model, view, (416, 288), (round(nx * SCALE) + 16, 48))
    cx, cy = tower.CENTRE
    width, depth = model.shape[:2]
    for _ in range(view):
        cx, cy = depth-cy, cx
        width, depth = depth, width
    px, py, pz = frame.point.transpose(2, 0, 1)
    x, y = px-cx, py-cy
    for _ in range(view):
        x, y = y, -x

    normal = frame.normal.copy()
    tiles = frame.mat == tower.TILE
    for surface, mask in ((tower.lower_roof, tiles & (pz < tower.UPPER_EAVE-2)),
                          (tower.upper_roof, tiles & (pz >= tower.UPPER_EAVE-2))):
        u, v = x[mask], y[mask]
        # Use the generating surface for the tile normals on both sides of each hip.
        dx = (surface(u+.05, v) - surface(u-.05, v)) / .1
        dy = (surface(u, v+.05) - surface(u, v-.05)) / .1
        for _ in range(view):
            dx, dy = -dy, dx
        n = np.column_stack((-dx, -dy, np.ones_like(dx)))
        normal[mask] = n / np.linalg.norm(n, axis=1, keepdims=True)
    light = shade(normal, frame.shadow, lighting.standard)
    # Tile runs follow the fall line of each face, perpendicular to its eave.
    end_face = np.where(pz < tower.UPPER_EAVE-2,
                        (tower.LOWER_HALF[0]-abs(x))/(tower.LOWER_HALF[0]-tower.UPPER_WALL[0])
                        < (tower.LOWER_HALF[1]-abs(y))/(tower.LOWER_HALF[1]-tower.UPPER_WALL[1]), abs(x) > tower.GABLE_X+1)
    across = np.where(end_face, y, x)
    light[tiles & (np.floor(across + .5).astype(int) % 4 == 0)] -= .12
    # Restrained ochre motifs on the shaded end gable.
    end_panel = frame.mat == tower.GABLE
    lattice = (np.floor(y + pz).astype(int) % 5 == 0) | (np.floor(y - pz).astype(int) % 5 == 0)
    light[end_panel & lattice] += .34
    # The accepted trees, river and screen-aligned paving use their original material rules.
    leaves = np.isin(frame.mat, (8, 9))
    i, j, k = frame.xyz[leaves].T
    light[leaves] += ((i*17+j*29+k*11) % 13-6)/75
    building = indexed(frame.mat, light, frame.hit, RAMPS)
    i, j = frame.gi, frame.gj
    water_light = .57 + (((i//2)*17+(j//2)*31) % 19-9)/500
    water_light += ((i+j*5) % 23 < 3) * .16
    water_light -= frame.gshadow * lighting.ground_darkening
    water = frame.gm & (frame.gmat == 7)
    ground = indexed(frame.gmat, water_light, water, RAMPS)
    paved = frame.gm & (frame.gmat == 1)
    ground[paved] = paving(frame.gx[paved], frame.gy[paved], frame.gshadow[paved],
                           shadow_colors=lighting.paving_shadow)
    return finish(frame, ground, building)
