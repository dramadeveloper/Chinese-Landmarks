"""Shanghai Oriental Pearl. Architectural dimensions are in metres above the paving."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / 'assets/chinese-landmarks'))
from export_plaza import make_model
from plaza_material import TILE

WIDTH = 3 * TILE
ART_SCALE = 1.312
U = ART_SCALE / 2
TIP = 468
CENTER = WIDTH / 2
(STONE, ROOF, WINDOW, FRAME, PLANT, COLUMN, BRACE, PEARL, SILVER, BELT,
 HOTEL, HUB, AERIAL, IDENT_RED, STEP, BEAM, RIB, CAP_GLASS) = range(1, 19)
PEOPLE = 30
LOBBY_ROOF, GLASS_SHADE, DOOR = 19, 20, 21
HOTEL_GLASS, HOTEL_BAND, GALLERY = 22, 23, 24
OBSERVATION_RIM = 25

# (x, y, z, radius), metres. Five hotel pods sit inside the tripod of shafts.
MAIN_BALLS = [(0, 0, 93, 25), (0, 0, 272.5, 22.5), (0, 0, 342, 8)]
HOTELS = [(0, 0, z, 8.2) for z in (142, 165, 188, 211, 234)]
ANGLES = np.radians([90, 210, 330])


def radial_points(angles, radius):
    return np.column_stack((np.cos(angles), np.sin(angles))) * radius


# Centre the visible extent of the three shafts in both isometric axes.
# Translating the equilateral group keeps the three column spacings identical.
COLUMN_RADIUS = 11.5
COLUMN_SHIFT = np.array([0, 2.2])  # Raster-centred at native 1× in all four views.
COLUMNS = radial_points(ANGLES, COLUMN_RADIUS) + COLUMN_SHIFT
# Each diagonal rises to the space between two adjacent vertical shafts.
BRACE_DIRS = radial_points(ANGLES + np.pi/3, 1)
JOIN_R = COLUMN_RADIUS/2
FOOT_R = JOIN_R + 93 / np.tan(np.radians(60))
BRACE_BALLS = [(*(COLUMN_SHIFT+d*(FOOT_R + (JOIN_R-FOOT_R)*.37)), 93*.37, 7.3)
               for d in BRACE_DIRS]


def podium(layer, x, y, radius, angle, z):
    stairs = (abs(angle) < .42) | (abs(angle) > np.pi-.19)
    walks = abs(np.sin(3*(angle-.20))) < .20
    stair_top = np.clip(np.floor((78-radius)/3)*1.55, 0, 9.3)
    bank_top = np.clip((72-radius)*.37, 0, 6.3)
    layer[(radius < 73) & ~stairs & (z <= 3)] = STONE
    layer[(radius < 70) & ~stairs & (z <= bank_top)] = ROOF
    garden = (radius > 47) & (radius < 64) & ~walks & ~stairs
    layer[garden & (z <= bank_top+1.3)] = PLANT
    layer[stairs & (z <= stair_top)] = STEP
    if z <= 10.4:
        layer[radius < 46] = STONE if z < 8.5 else ROOF
    if 9.4 <= z < 19:
        lobby = radius < 32
        layer[lobby] = WINDOW
        layer[lobby & (np.cos(26*angle) > .84)] = FRAME
        vestibule = (x > 27) & (x < 37) & (abs(y) < 12)
        layer[vestibule] = WINDOW
        layer[vestibule & (abs(y) < 5.5)] = DOOR
        layer[vestibule & (abs(abs(y)-6.1) < .9)] = FRAME
    if 16.8 < z < 19:
        layer[(x > 26) & (x < 41) & (abs(y) < 14)] = ROOF
    if 19 <= z < 22:
        layer[radius < 33.5-(z-19)*2] = LOBBY_ROOF
    for azimuth in (-1.05, 1.13):
        along = x*np.cos(azimuth)+y*np.sin(azimuth)
        across = abs(y*np.cos(azimuth)-x*np.sin(azimuth))
        lane = (along > 35) & (along < 70) & (across < 6)
        cover = lane & (abs(z-(12-(along-35)*.17)) < 1.1)
        layer[cover] = WINDOW
        layer[cover & ((along % 8 < 1.3) | (across > 4.7))] = FRAME


def upper_glazing(t):
    """Height courses shared by the saved voxels and the curved glass surface."""
    z = 272.5+22.5*t
    return np.select([(z >= 265.5) & (z < 268.2), (z >= 258.5) & (z < 264),
                      (t >= .93) | (t <= -.73), t < .04],
                     [OBSERVATION_RIM, BELT, SILVER, GLASS_SHADE], default=PEARL)


def ball_material(level, t):
    if level == 272.5:
        return upper_glazing(t)
    if level == 93:
        if abs(t+.13) < .105:
            return BELT
        if abs(abs(t+.13)-.13) < .038:
            return SILVER
        if -.73 < t < .42:
            return GLASS_SHADE if t < .04 else PEARL
        return SILVER
    if level == 342:
        return CAP_GLASS if -.24 < t < .12 else SILVER
    if level < 93:
        return SILVER
    if -.28 < t < .30:
        return HOTEL_GLASS
    return HOTEL_BAND if .30 <= t < .56 else HOTEL


def generate_model():
    top = 1 + round(TIP*U)
    model = np.zeros((WIDTH, WIDTH, top+2), np.uint8)
    model[:, :, 0] = STONE
    ix, iy = np.ogrid[:WIDTH, :WIDTH]
    x, y = (ix+.5-CENTER)/U, (iy+.5-CENTER)/U
    radius, angle = np.hypot(x, y), np.arctan2(y, x)
    shafts = [np.hypot(x-cx, y-cy) <= 4.5 for cx, cy in COLUMNS]
    sx, sy = x-COLUMN_SHIFT[0], y-COLUMN_SHIFT[1]
    for iz in range(1, top):
        z = (iz-.5)/U
        layer = model[:, :, iz]
        if z < 22:
            podium(layer, x, y, radius, angle, z)
        if z < 286:
            for shaft in shafts:
                layer[shaft] = COLUMN
        if 112 < z < 255:
            # The slim service line visible between hotel levels in the original.
            layer[radius < 2.55] = HUB
        if z <= 93:
            brace_r = FOOT_R + (JOIN_R-FOOT_R)*z/93
            for dx, dy in BRACE_DIRS:
                along, across = sx*dx+sy*dy, sy*dx-sx*dy
                distance2 = ((along-brace_r)*np.sin(np.pi/3))**2 + across**2
                layer[distance2 <= 3.5**2] = BRACE
                if z <= 6:
                    layer[np.hypot(sx-FOOT_R*dx, sy-FOOT_R*dy) <= 6.8-.28*z] = STONE
        for _, _, level, _ in HOTELS:
            dz = z-level
            if -5 <= dz < .6:
                # Glazed passageways between each pair of shafts. Floor and
                # roof project beyond the glazing to read at native resolution.
                rim = dz < -3.2 or dz >= -1
                for dx, dy in BRACE_DIRS:
                    across, along = sx*dx+sy*dy, sy*dx-sx*dy
                    half_width = 2.7 if rim else 1.8
                    bridge = (abs(across-JOIN_R)<half_width) & (abs(along)<10.3)
                    layer[bridge] = GALLERY if rim else BEAM
        if 246 < z < 259:
            layer[radius < 10+(z-246)*.56] = HUB
        if 285 < z < 342:
            neck = np.interp(z, [285, 297, 309, 327, 342], [7, 4.1, 3.5, 2.9, 2.7])
            layer[radius < neck] = HUB
            if 301 < z < 325 and (z-301)%8 < 1.6:
                layer[radius < neck+1.1] = RIB
        for cx, cy, level, r in BRACE_BALLS + HOTELS + MAIN_BALLS:
            t = (z-level)/r
            if abs(t) < 1:
                disk = (x-cx)**2 + (y-cy)**2 <= r*r*(1-t*t)
                layer[disk] = ball_material(level, t)
        if z >= 349:
            # The terminal section is thickened enough to keep a continuous pixel.
            half = np.interp(z, [349, 366, 367, 402, 403, 434, 435, TIP],
                             [1.95, 1.95, 1.9, 1.8, 1.65, 1.5, 1.2, 1.2])
            mast = (abs(x) <= half) & (abs(y) <= half)
            layer[mast] = IDENT_RED if int((z-349)/12)%2 == 1 else AERIAL
            if abs(z-368) < 1.4 or abs(z-406) < 1.4:
                layer[radius < half+1.7] = RIB

    walkers = make_model('walkers')
    # Two visitors approach the fan stair, a third passes on the side.
    for sx, sy, px, py in [(10,12,111,61), (10,19,113,65), (26,26,83,106)]:
        person = walkers[sx:sx+3, sy:sy+3, 1:6]
        target = model[px:px+3, py:py+3, 1:6]
        target[person > 0] = person[person > 0] + PEOPLE
    return model


if __name__ == '__main__':
    np.savez_compressed(HERE/'model.npz', model=generate_model())
