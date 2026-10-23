"""Bell Tower material/roof adapter; camera, light, shadow and paving are shared."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_core import trace_model, shade, indexed, stone_ground, finish, DEFAULT_SCENE, Sampling
from export_plaza import RAMPS as PROP_RAMPS
from xian_bell_tower_model import (ROOFS, UNITS_PER_METRE, GROUND, BRICK, STONE, RED, DARK,
                   GREEN, RIDGE, GOLD, PAINT, STAIR_STONE, WOOD_TRIM, PROP_OFFSET, roof_height)

# Palette first: quiet masonry, red timber, green glazed tile, restrained gold.
RAMPS = {
    GROUND: [7, 8, 9, 10, 11, 12],
    BRICK: [4, 5, 6, 7, 8, 9, 10],
    STONE: [7, 8, 9, 10, 11, 12, 13],
    STAIR_STONE: [5, 6, 7, 8, 9, 10],
    WOOD_TRIM: [35, 36, 37, 38, 39],
    RED: [178, 179, 180, 181, 182, 183],
    DARK: [1, 2, 3, 4, 5],
    # Merge the steep roof's midtone band; retain the lit/shaded face contrast.
    GREEN: [96, 96, 97, 97, 98, 99, 99, 100],
    RIDGE: [96, 97, 98, 99, 100, 101],
    GOLD: [61, 62, 63, 64, 65, 66, 67],
    PAINT: [96, 96, 97, 98, 99],
    **{key+PROP_OFFSET: ramp for key, ramp in PROP_RAMPS.items()},
}
SAMPLING = Sampling(building_step=.125)


def render(model, view, *, lighting=DEFAULT_SCENE, canvas=(224, 176), origin=(112, 52)):
    frame = trace_model(model, view, canvas, origin, SAMPLING)
    size = model.shape[0]
    x, y, z = frame.point.transpose(2, 0, 1)
    # Roof/base textures are symmetric; no inverse rotation needed. World paving
    # remains untouched, and the normals below are already in the rotated frame.
    x, y, z = (x-size/2)/UNITS_PER_METRE, (y-size/2)/UNITS_PER_METRE, (z-1)/UNITS_PER_METRE
    normal = frame.normal.copy()
    roofmask = frame.hit & np.isin(frame.mat, (GREEN, RIDGE))
    for roof in ROOFS:
        surface = roof_height(x, y, roof)
        m = roofmask & (abs(z-surface) < 1.2) & (np.maximum(abs(x), abs(y)) <= roof[0]+.5)
        dx = (roof_height(x+.01, y, roof)-roof_height(x-.01, y, roof))/.02
        dy = (roof_height(x, y+.01, roof)-roof_height(x, y-.01, roof))/.02
        n = np.stack((-dx, -dy, np.ones_like(x)), axis=2)
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        normal[m] = n[m]
    light = shade(normal, frame.shadow, lighting.standard)
    # Low-contrast masonry courses. Roofs keep broad tone areas at native 1x.
    masonry = frame.hit & (frame.mat == BRICK) & (frame.face != 2)
    course = np.floor(z/1.2)
    across = np.where(frame.face == 0, y, x)
    joints = (np.mod(z, 1.2)<.12) | (np.mod(across+course*1.5, 3.0)<.10)
    light[masonry & joints] -= .065
    building = indexed(frame.mat, light, frame.hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), building)
