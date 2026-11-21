"""Copper, brickwork and glazing for the selected 3×3 church."""
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import sophia_model as g
from export_plaza import RAMPS as PROP_RAMPS
from render_core import DEFAULT_SAMPLING, DEFAULT_SCENE, finish, indexed, shade, stone_ground, surface_shadow, trace_model

BRICK_RAMP = [71, 72, 123, 73, 74, 75, 76, 77]
RAMPS = {
    g.GRANITE: [3, 4, 5, 6, 7, 8, 9],
    g.BRICK: BRICK_RAMP, g.DRUM: BRICK_RAMP,
    g.MOULDING: [72, 73, 124, 74, 75, 76, 77, 78, 119],
    g.REVEAL: [70, 71, 72, 73, 74],
    g.SASH: [1, 2, 16, 17],
    g.GLASS: [16, 17, 18, 19, 20, 21, 22],
    g.DOOR: [70, 104, 105, 106, 107, 108],
    g.ROOF: [1, 96, 97, 98, 98, 99, 100],
    g.TENT_METAL: [1, 96, 97, 97, 98, 99, 100, 101],
    g.CUPOLA: [1, 96, 96, 97, 97, 98, 99, 99, 100, 101],
    g.RIDGE: [1, 96, 97, 97, 98, 99],
    g.GILT: [54, 55, 63, 64, 65, 66, 67, 68],
    g.BRONZE: [54, 55, 56, 57, 58, 59, 38],
    g.BELL_BRICK: [71, 72, 73, 74, 75, 76, 77],
    g.BELL_TRIM: [72, 73, 74, 75, 76, 77, 78],
    g.BELL_REVEAL: [1, 70, 71, 72, 73],
    g.BELL_WOOD: [70, 71, 104, 106, 107],
    g.BELL_EAVE: [96, 97, 97, 98, 99, 99, 100],
    g.BELL_TENT: [96, 97, 97, 98, 99, 99, 100],
    g.SELECTED_GOLD: [54, 55, 63, 65, 66, 67, 68],
    g.SANDSTONE: [114, 115, 110, 36, 37, 38],
    **{i+g.PEOPLE: ramp for i, ramp in PROP_RAMPS.items()},
}


def roof_normals(frame, mask):
    # Average the solid/air boundary over two units for the shallow compound
    # roofs. This keeps the angle of the roof instead of its stair-tread faces.
    field = (frame.model != 0).astype(np.float32)
    for axis in range(3):
        field = (6*field + 4*(np.roll(field, 1, axis)+np.roll(field, -1, axis))
                 + np.roll(field, 2, axis)+np.roll(field, -2, axis))/16
    ix, iy, iz = frame.xyz[mask].T
    normals = -np.stack([v[ix, iy, iz] for v in np.gradient(field)], axis=1)
    return normals/np.maximum(.001, np.linalg.norm(normals, axis=1, keepdims=True))


def render(model, view, *, lighting=DEFAULT_SCENE):
    frame = trace_model(model, view, (216, 204), (108, 100))
    cx, cy = g.CENTRE
    for _ in range(view):
        cx, cy = 120-cy, cx
    px, py, pz = frame.point.transpose(2, 0, 1)
    k = g.ART_SCALE/2
    dx, dy, h = (px-cx)/k, (py-cy)/k, (pz-1)/k
    u, v = dx, dy
    for _ in range(view):
        u, v = v, -u
    normal = frame.normal.copy()
    roof = frame.mat == g.ROOF
    normal[roof] = roof_normals(frame, roof)
    # Nearest tent axes in original plan coordinates. Each of the four roofs
    # is a metal-clad octagon, with its own pitch and a raised hip seam.
    west = frame.mat == g.BELL_TENT
    tent = (frame.mat == g.TENT_METAL) | west
    axes = ((21.45, 0, 1.75), (.1, 10.85, 2.15), (.1, -10.85, 2.15), (-13, 0, 1.95))
    nearest = np.argmin([(u-x)**2+(v-y)**2 for x, y, _ in axes], axis=0)
    seam = np.zeros_like(h)
    for i, (tx, ty, pitch) in enumerate(axes):
        m = tent & (nearest == i)
        sector = np.arctan2(v[m]-ty, u[m]-tx)/(np.pi/4)
        angle = (np.round(sector)+view*2)*np.pi/4
        normal[m] = np.stack([pitch*np.cos(angle), pitch*np.sin(angle), np.ones_like(angle)], axis=1)/np.hypot(1, pitch)
        off_axis = np.abs(sector-np.round(sector))
        seam[m] = .12*(off_axis > .435) if i == 0 else .075*(off_axis > .38)
    dome = frame.mat == g.CUPOLA
    theta = np.arctan2(dy, dx)
    slope = np.interp(h, g.BULB_Z, np.gradient(g.BULB_R, g.BULB_Z))
    normal[dome] = np.stack((np.cos(theta[dome]), np.sin(theta[dome]), -slope[dome]), axis=1)/np.hypot(1, slope[dome])[:,None]
    # The drum's glazing and brickwork sit on sixteen flat facets.
    drum = (np.hypot(dx, dy) < 7.62) & (h > 23.65) & (h < 35.9) & (frame.face != 2)
    angle = np.round(theta[drum]*8/np.pi)*np.pi/8
    normal[drum] = np.stack((np.cos(angle), np.sin(angle), np.zeros_like(angle)), axis=1)
    curved = dome | roof | tent
    shadow = frame.shadow.copy()
    start = frame.point.copy()
    start[curved] += 1.1*normal[curved]
    start[west] += .05*normal[west]
    shadow[curved] = surface_shadow(frame.model, start, normal, curved, DEFAULT_SAMPLING)[curved]
    light = shade(normal, shadow, lighting.standard)
    light[tent] += seam[tent]
    # Staggered diamond sheets. Decrease the seam contrast near the narrow tip.
    az = (theta-view*np.pi/2)*12/np.pi
    course = (h-36.3)/2.1
    joints = np.minimum(abs(az+course-np.rint(az+course)), abs(az-course-np.rint(az-course)))
    strength = np.clip(np.hypot(dx, dy)/7, .25, 1)
    light[dome] += np.where(joints[dome] < .085, -.075, .009)*strength[dome]
    masonry = np.isin(frame.mat, (g.BRICK, g.MOULDING, g.DRUM)) & (frame.face != 2)
    # Short joints break up broad walls, leaving openings and sills legible.
    along = np.where(frame.face == 0, py, px)
    courses = np.floor(pz/1.38)
    joints = ((along+2.1*(courses%2)) % 4.2 < .62) | (pz%1.38 < .18)
    light[masonry] += np.where(joints[masonry], -.055, .018)
    brick = (frame.mat == g.BELL_BRICK) & (frame.face != 2)
    joints = (np.floor(along/1.35)+2*np.floor(pz/1.6))%5 == 0
    light[brick] += np.where(joints[brick], -.04, .012)
    pixels = indexed(frame.mat, light, frame.hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), pixels)
