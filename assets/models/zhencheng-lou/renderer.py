"""Zhencheng Lou surface normals/materials using the shared scene pipeline."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'chinese-landmarks'))
from render_core import (DEFAULT_SCENE, DEFAULT_SAMPLING, trace_model,
                         surface_shadow, shade, indexed, stone_ground, finish)
from plaza_material import SCALE
from export_plaza import RAMPS as PROP_RAMPS
# The package loads several model directories in one process; use this
# generator's own module name rather than another landmark's `model`.
spec = importlib.util.spec_from_file_location('zhencheng_lou_model', Path(__file__).with_name('model.py'))
geometry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = geometry
spec.loader.exec_module(geometry)
from zhencheng_lou_model import (DIMENSIONS, CENTER, EARTH, WOOD, DARK, OUTER_ROOF,
                   INNER_ROOF, PLASTER, BRICK, COURT, HALL_ROOF, EAR_ROOF, RIDGE,
                   ring_roof_height)

RAMPS = {
    1: [6, 7, 8, 9, 10, 11, 12],
    EARTH: [34, 35, 36, 37, 38, 38, 39],
    WOOD: [70, 104, 105, 106, 107, 108, 109],
    DARK: [1, 2, 2, 3, 4],
    OUTER_ROOF: [2, 3, 4, 5, 5, 6, 7],
    INNER_ROOF: [2, 3, 4, 5, 5, 6, 7],
    PLASTER: [6, 8, 10, 11, 12, 13, 14],
    BRICK: [4, 5, 6, 7, 8, 9, 10],
    COURT: [5, 6, 7, 8, 9, 10, 11],
    HALL_ROOF: [3, 4, 5, 6, 6, 7, 8],
    EAR_ROOF: [2, 3, 4, 5, 5, 6, 7],
    RIDGE: [3, 4, 5, 6, 7, 8, 9],
    13: [178, 179, 180, 181],
    **{key+30: ramp for key, ramp in PROP_RAMPS.items()},
}


def render(model, view, *, lighting=DEFAULT_SCENE, dimensions=DIMENSIONS,
           canvas=(288, 200), origin=(144, 48)):
    frame = trace_model(model, view, canvas, origin)
    normal = frame.normal.copy()
    shadow_point = frame.point.copy()
    # Centre coordinates rotate with the model, while the light remains fixed.
    cx, cy = CENTER
    for _ in range(view):
        cx, cy = 160-cy, cx
    dx, dy = frame.point[:, :, 0]-cx, frame.point[:, :, 1]-cy
    radius = np.hypot(dx, dy)
    radial = np.stack((dx/np.maximum(radius, .01), dy/np.maximum(radius, .01), np.zeros_like(dx)), axis=2)
    k, d = dimensions.art_scale/2, dimensions
    # Only the confirmed cylindrical earth envelope receives smooth normals.
    exterior = frame.hit & (frame.mat == EARTH) & (radius > 27*k)
    normal[exterior] = radial[exterior]
    for material, inner, ridge, outer, eave, top in (
        (OUTER_ROOF, d.outer_eave_inner_radius_m, d.outer_ridge_radius_m, d.outer_eave_outer_radius_m, d.outer_eave_height_m, d.outer_height_m),
        (INNER_ROOF, d.inner_eave_inner_radius_m, d.inner_ridge_radius_m, d.inner_eave_outer_radius_m, d.inner_eave_height_m, d.inner_ridge_height_m),
        (EAR_ROOF, 32, 36.5, 41, d.ear_eave_height_m, d.ear_ridge_height_m)):
        mask = frame.hit & (frame.mat == material)
        derivative = np.where(radius < ridge*k, (top-eave)/(ridge-inner), -(top-eave)/(outer-ridge))
        n = -radial*derivative[:, :, None]
        n[:, :, 2] = 1
        n /= np.linalg.norm(n, axis=2)[:, :, None]
        normal[mask] = n[mask]
        # Trace from the continuous roof envelope, above voxel stair treads.
        # This removes sampling self-shadow speckles while retaining shadow
        # cast by the tall outer ring onto the lower roofs and courtyard.
        top_z = 1+k*ring_roof_height(radius/k, inner, ridge, outer, eave, top)
        shadow_point[:, :, 2][mask] = np.maximum(frame.point[:, :, 2][mask], top_z[mask]+.6)
    hx, hy = CENTER[0], CENTER[1]-9.5*k
    for _ in range(view):
        hx, hy = 160-hy, hx
    rx, ry = frame.point[:, :, 0]-hx, frame.point[:, :, 1]-hy
    half = [(d.hall_width_m/2+.7)*k, (d.hall_depth_m/2+.7)*k]
    if view % 2:
        half.reverse()
    x_pitch = np.abs(rx)/half[0] > np.abs(ry)/half[1]
    hall_normal = np.zeros_like(normal)
    hall_normal[:, :, 0] = np.where(x_pitch, np.sign(rx)*(d.hall_roof_top_m-7.6)*k/half[0], 0)
    hall_normal[:, :, 1] = np.where(x_pitch, 0, np.sign(ry)*(d.hall_roof_top_m-7.6)*k/half[1])
    hall_normal[:, :, 2] = 1
    hall_normal /= np.linalg.norm(hall_normal, axis=2)[:, :, None]
    hall = frame.hit & (frame.mat == HALL_ROOF)
    normal[hall] = hall_normal[hall]
    hall_top = 1+k*(7.6+(d.hall_roof_top_m-7.6)*(1-np.maximum(np.abs(rx)/half[0], np.abs(ry)/half[1])))
    shadow_point[:, :, 2][hall] = np.maximum(frame.point[:, :, 2][hall], hall_top[hall]+.6)
    normal[frame.hit & (frame.mat == RIDGE)] = [0, 0, 1]
    shadow = surface_shadow(frame.model, shadow_point, normal, frame.hit, DEFAULT_SAMPLING)
    light = shade(normal, shadow, lighting.standard)
    # Broad, low-contrast tile runs, no random weathering or high-contrast noise.
    roofs = np.isin(frame.mat, [OUTER_ROOF, INNER_ROOF, EAR_ROOF])
    tile_run = np.cos(np.arctan2(dy, dx)*96)
    light[roofs] += .01*tile_run[roofs]
    pixels = indexed(frame.mat, light, frame.hit, RAMPS)
    ground = stone_ground(frame, lighting)
    court = frame.gm & (frame.gmat == COURT)
    ground[court] = np.where(frame.gshadow[court], 6, 9)
    return finish(frame, ground, pixels)
