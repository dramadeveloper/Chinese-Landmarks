"""Daylight materials for the 3×3 tower, using the common projection and lighting."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pearl_model as tower
from export_plaza import RAMPS as PEOPLE_RAMPS
from render_core import DEFAULT_SCENE, finish, indexed, shade, stone_ground, trace_model

RAMPS = {
    tower.STONE: [5, 7, 9, 10, 11, 12], tower.STEP: [6, 8, 10, 11, 12, 13],
    tower.ROOF: [17, 19, 20, 21, 22, 23, 14], tower.WINDOW: [16, 154, 156, 158, 159],
    tower.FRAME: [5, 8, 10, 12, 14], tower.PLANT: [88, 89, 90, 91, 92, 93],
    tower.COLUMN: [5, 7, 9, 10, 12, 13, 14], tower.BRACE: [5, 7, 9, 10, 12, 13, 14],
    tower.PEARL: [40, 41, 42, 43, 44, 45, 46, 47],
    tower.SILVER: [16, 17, 18, 19, 20, 21, 22, 23],
    tower.GLASS_SHADE: [40, 41, 42, 43, 44, 45, 46],
    tower.LOBBY_ROOF: [17, 18, 19, 20, 21, 22, 23], tower.DOOR: [1, 16, 17, 154, 156],
    tower.HOTEL_GLASS: [1, 16, 17, 154, 156, 19], tower.HOTEL_BAND: [40, 42, 43, 44, 45],
    tower.GALLERY: [16, 18, 19, 20, 21, 22, 23],
    tower.OBSERVATION_RIM: [17, 18, 126, 127, 21, 22],
    tower.BELT: [16, 17, 18, 19, 20, 21], tower.HOTEL: [17, 19, 20, 21, 22, 23],
    tower.HUB: [16, 17, 18, 19, 20, 21, 22], tower.AERIAL: [5, 8, 10, 12, 14],
    tower.IDENT_RED: [41, 43, 44, 162, 165], tower.BEAM: [4, 6, 8, 10, 11, 12],
    tower.RIB: [17, 18, 19, 20, 21, 22], tower.CAP_GLASS: [16, 17, 18, 19, 20, 21],
    **{tower.PEOPLE+k: ramp for k, ramp in PEOPLE_RAMPS.items()},
}


def render(model, view, *, lighting=DEFAULT_SCENE):
    frame = trace_model(model, view, (206, 314), (103, 207))
    hit = frame.hit
    # Return surface samples to author coordinates so panel joints rotate with the tower.
    p = (frame.point[hit]-[tower.CENTER, tower.CENTER, 1])/tower.U
    quarter = np.array([[0, -1], [1, 0]])
    rotation = np.linalg.matrix_power(quarter, view)
    p[:, :2] = p[:, :2] @ rotation
    m = frame.mat[hit].copy()
    normal = frame.normal[hit].copy()
    normal[:, :2] = normal[:, :2] @ rotation
    detail = np.zeros(len(m))

    def curved(mask, radial):
        v = radial[mask]
        normal[mask] = v / np.linalg.norm(v, axis=1)[:, None]

    for cx, cy in tower.COLUMNS:
        radial = p - [cx, cy, 0]
        radial[:, 2] = 0
        curved((m == tower.COLUMN) & (np.linalg.norm(radial, axis=1) < 7.2), radial)
    for direction in tower.BRACE_DIRS:
        foot = np.r_[tower.COLUMN_SHIFT+tower.FOOT_R*direction, 0]
        axis = np.r_[(tower.JOIN_R-tower.FOOT_R)*direction, 93]
        rel = p-foot
        radial = rel - ((rel @ axis)/(axis @ axis))[:, None]*axis
        curved((m == tower.BRACE) & (np.linalg.norm(radial, axis=1) < 6.5), radial)

    radial = p.copy()
    radial[:, 2] = 0
    curved(m == tower.HUB, radial)
    # Smooth the sloping lobby roof and terraces where exposed voxel steps
    # would break the circular bands into alternating bright and dark pixels.
    ix, iy, iz = frame.xyz[hit].T
    exposed = frame.model[ix, iy, iz+1] == 0
    for material, slope in [(tower.LOBBY_ROOF, .5), (tower.PLANT, .18),
                            (tower.STEP, .30), (tower.ROOF, .08), (tower.STONE, .04)]:
        mask = (m == material) & exposed
        v = radial[mask].copy()
        v[:, 2] = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1)/slope
        normal[mask] = v/np.linalg.norm(v, axis=1)[:, None]

    globe = np.isin(m, (tower.PEARL, tower.GLASS_SHADE, tower.SILVER, tower.BELT, tower.OBSERVATION_RIM,
                        tower.HOTEL, tower.HOTEL_GLASS, tower.HOTEL_BAND, tower.CAP_GLASS))
    for cx, cy, cz, r in tower.MAIN_BALLS + tower.HOTELS + tower.BRACE_BALLS:
        radial = p-[cx, cy, cz]
        mask = globe & (np.linalg.norm(radial, axis=1) < r+2)
        if cz == 272.5:
            # Sample the circular gallery on the sphere itself. The voxel hit
            # still supplies its silhouette, occlusion, shadow and tile owner.
            ray = np.r_[np.array([1, 1]) @ rotation, 1]/np.sqrt(3)
            nearest = radial[mask] - (radial[mask] @ ray)[:, None]*ray
            depth = np.sqrt(np.maximum(0, r*r-np.sum(nearest**2, axis=1)))
            radial[mask] = nearest+depth[:, None]*ray
            m[mask] = tower.upper_glazing(radial[mask, 2]/r)
        curved(mask, radial)
        if r < 20:
            continue
        longitude = np.arctan2(radial[:, 1], radial[:, 0])/(2*np.pi)
        lat = np.arcsin(np.clip(radial[:, 2]/r, -1, 1))
        panel = ((m == tower.PEARL) | (m == tower.GLASS_SHADE)) & mask
        # Two mesh diagonals per pane, with a broad value change around the latitude.
        u, v = longitude*14, lat*1.85
        seam = np.minimum((u+v)%1, (u-v)%1) < .15
        detail[panel & seam] += .065
        detail[panel & ((np.floor(u+v)+np.floor(u-v)) % 2 == 0)] -= .04
        mullion = mask & (m == tower.BELT) & ((longitude*24)%1 < .21)
        m[mullion] = tower.FRAME
        if cz == 93:
            # Small portholes are punched around the silver cap above the rose glazing.
            ports = mask & (m == tower.SILVER) & (abs(radial[:, 2]/r-.55)<.066)
            m[ports & ((longitude*18)%1 < .34)] = tower.BELT

    normal[:, :2] = normal[:, :2] @ rotation.T
    frame.normal[hit] = normal
    paint = frame.mat.copy()
    paint[hit] = m
    light = shade(frame.normal, frame.shadow, lighting.standard)
    light[hit] += detail
    building = indexed(paint, light, hit, RAMPS)
    return finish(frame, stone_ground(frame, lighting), building)
