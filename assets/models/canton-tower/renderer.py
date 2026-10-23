"""Public projection, lighting, ground and palette; no private rendering pipeline."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_core import DEFAULT_SCENE, trace_model, shade, indexed, stone_ground, finish
from export_plaza import RAMPS as PROP_RAMPS
from canton_tower_model import (STONE, COLUMN, RING, BRACE, CORE, GLASS, DECK, RED,
                                PROP_OFFSET, section, UNITS_PER_METRE)

RAMPS = {
    STONE: [7, 8, 9, 10, 11, 12, 13],
    COLUMN: [7, 9, 11, 12, 13, 14, 15],
    RING: [4, 6, 7, 8, 9, 10, 11],
    BRACE: [4, 5, 6, 7, 8, 9, 10],
    CORE: [2, 3, 4, 5, 6],
    GLASS: [1, 2, 154, 155, 156, 157],
    DECK: [5, 6, 7, 8, 9, 10, 11],
    RED: [178, 179, 180, 181, 182],
    **{k+PROP_OFFSET: ramp for k, ramp in PROP_RAMPS.items()},
}


def render(model, view, *, lighting=DEFAULT_SCENE):
    frame = trace_model(model, view, canvas=(224, 448), origin=(112, 332))
    normal = frame.normal.copy()
    # Smooth the 24 cylindrical columns only; projection, ray visibility,
    # cast shadows and light response remain entirely in render_core.
    mask = frame.hit & (frame.mat == COLUMN) & (frame.point[:, :, 2] > 12) & (frame.point[:, :, 2] < 448*UNITS_PER_METRE)
    points = frame.point[mask].copy()
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if view == 1:
        points[:, 0], points[:, 1] = y, 120-x
    elif view == 2:
        points[:, 0], points[:, 1] = 120-x, 120-y
    elif view == 3:
        points[:, 0], points[:, 1] = 120-y, x
    vectors, distances = [], []
    for theta in np.arange(24)*2*np.pi/24:
        a, b = section(0, theta), section(1, theta)
        axis = b-a
        along = np.clip(((points-a)*axis).sum(axis=1)/(axis@axis), 0, 1)
        radial = points-a-along[:, None]*axis
        vectors.append(radial)
        distances.append((radial**2).sum(axis=1))
    nearest = np.argmin(distances, axis=0)
    smooth = np.array(vectors)[nearest, np.arange(len(points))]
    smooth /= np.linalg.norm(smooth, axis=1)[:, None]
    nx, ny = smooth[:, 0].copy(), smooth[:, 1].copy()
    if view == 1:
        smooth[:, 0], smooth[:, 1] = -ny, nx
    elif view == 2:
        smooth[:, 0], smooth[:, 1] = -nx, -ny
    elif view == 3:
        smooth[:, 0], smooth[:, 1] = ny, -nx
    normal[mask] = smooth
    building = indexed(frame.mat, shade(normal, frame.shadow, lighting.standard), frame.hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), building)
