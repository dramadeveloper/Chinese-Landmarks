"""Material and surface normals for the National Stadium's folded envelope."""
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import stadium_model as g
from export_plaza import RAMPS as PROP_RAMPS
from plaza_material import SCALE
from render_core import DEFAULT_SCENE, finish, indexed, shade, stone_ground, trace_model

RAMPS = {
    g.PODIUM: [5, 6, 7, 8, 9, 10, 11],
    g.CONCRETE: [16, 17, 18, 7, 9, 10, 11],
    g.GALLERY: [1, 2, 16, 17, 18],
    g.BOWL_WALL: [70, 71, 72, 73, 124, 75, 76],
    g.CHERRY: [41, 42, 43, 162, 163, 165],
    g.CORAL: [42, 43, 162, 44, 45, 46, 47],
    g.PALE: [17, 18, 19, 20, 21, 22],
    g.AISLE: [17, 18, 19, 8, 9, 10],
    g.TRACK: [72, 73, 124, 75, 76, 77],
    g.TRACK_LINE: [73, 124, 75, 76, 77, 78],
    g.TURF: [89, 90, 91, 92, 93], g.MOWN: [88, 89, 90, 91, 92],
    g.CHALK: [7, 8, 9, 10, 11, 12],
    g.MEMBRANE: [17, 18, 19, 20, 20, 21],
    g.FASCIA: [18, 19, 20, 21, 22, 23, 13], g.SOFFIT: [16, 17, 18, 19, 20],
    g.WEB: [17, 18, 19, 20, 9, 10, 11],
    g.SECONDARY: [18, 19, 20, 9, 10, 11, 12],
    g.STEEL: [18, 19, 20, 10, 11, 12, 13, 14],
    **{k + g.PROP_OFFSET: ramp for k, ramp in PROP_RAMPS.items()},
}


def render(model, view, *, lighting=DEFAULT_SCENE):
    nx, ny = model.shape[:2] if view%2 == 0 else model.shape[1::-1]
    origin = (round(nx*SCALE)+10, 20)
    canvas = (round((nx+ny)*SCALE)+20, 210)
    frame = trace_model(model, view, canvas, origin)
    normal = frame.normal.copy()
    # Level grass, track and painted lines share one uninterrupted plane.
    # Seat treads and landing tops likewise keep a flat top wherever the
    # sampled side is an interior voxel face shared with the next cell.
    flat = np.isin(frame.mat, (g.TRACK, g.TRACK_LINE, g.TURF, g.MOWN, g.CHALK))
    terraces = np.isin(frame.mat, (g.PODIUM, g.CHERRY, g.CORAL, g.PALE, g.AISLE)) & (frame.face != 2)
    ix, iy, iz = frame.xyz[terraces].T
    f = frame.face[terraces]
    normal[flat] = (0, 0, 1)
    enclosed = frame.model[ix+(f==0), iy+(f==1), iz] != 0
    normal[terraces] = np.where(enclosed[:, None], (0, 0, 1), normal[terraces])
    # The cap of each box section lies tangent to the shell. Shade it on the
    # continuous surface, leaving the webs and concourse edges visibly faceted.
    shell = np.isin(frame.mat, (g.MEMBRANE, g.SECONDARY, g.STEEL))
    x, y, z = frame.point[shell].T
    x, y = x-nx/2, y-ny/2
    for _ in range(view):
        x, y = y, -x
    theta = np.arctan2(y/g.AXES[1], x/g.AXES[0])
    radius = np.hypot(x/g.AXES[0], y/g.AXES[1])
    roof = ((radius <= g.SHOULDER+.011) & (z > g.roof_height(x,y)-4.1))
    _, n = g.roof_surface(np.column_stack((x, y)))
    side = ~roof
    eave = g.roof_height(g.AXES[0]*g.SHOULDER*np.cos(theta[side]),
                         g.AXES[1]*g.SHOULDER*np.sin(theta[side]))
    fraction = np.clip((z[side]-2)/(eave-2), 0, 1)
    _, n[side] = g.wall_surface(theta[side], 1-(1-fraction)**(1/1.55))
    for _ in range(view):
        n[:, :2] = np.column_stack((-n[:, 1], n[:, 0]))
    normal[shell] = n
    inner = (frame.mat == g.FASCIA) | (frame.mat == g.SOFFIT)
    ix, iy = frame.point[inner, :2].T
    axes = g.OPENING if view%2 == 0 else g.OPENING[::-1]
    uv = np.column_stack((ix-nx/2, iy-ny/2))/axes
    radial = -np.sign(uv)*np.abs(uv)**(g.LIP_POWER-1)/axes
    normal[inner] = g.unit(np.column_stack((radial, np.zeros(len(radial)))))
    light = shade(normal, frame.shadow, lighting.standard)
    building = indexed(frame.mat, light, frame.hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), building)
