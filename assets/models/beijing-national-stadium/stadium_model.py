"""Daylight National Stadium geometry. Run this file to save model.npz."""
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[2]
sys.path.insert(0, str(ROOT / 'assets/chinese-landmarks'))
from export_plaza import make_model

SHAPE = (240, 200, 56)
CENTRE = np.array(SHAPE[:2]) / 2
ART_SCALE = 1.25
UNITS_PER_METRE = ART_SCALE / 2
AXES = np.array([333., 296.]) * UNITS_PER_METRE / 2
OPENING = np.array([182.393, 124.103]) * UNITS_PER_METRE / 2
LIP_POWER = 2.45
SHOULDER = .955
ROOF_LIFT = 6  # Extra facade height in model units at native game scale.
PODIUM, CONCRETE, GALLERY, BOWL_WALL, CHERRY, CORAL, PALE, AISLE = range(1, 9)
TRACK, TRACK_LINE, TURF, MOWN, CHALK, MEMBRANE, WEB, SECONDARY, STEEL = range(9, 18)
FASCIA, SOFFIT = 18, 19
PROP_OFFSET = 40


def roof_height(x, y):
    # High eaves on the short axis, low eaves at the ends of the long axis.
    u, v = x / (AXES[0] * SHOULDER), y / (AXES[1] * SHOULDER)
    return 32.0 + ROOF_LIFT - 6.2*u*u + 11.1*v*v


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def roof_surface(xy):
    x, y = xy.T
    z = roof_height(x, y)
    nx = 12.4 * x / (AXES[0]*SHOULDER)**2
    ny = -22.2 * y / (AXES[1]*SHOULDER)**2
    return np.column_stack((x, y, z)), unit(np.column_stack((nx, ny, np.ones_like(z))))


def wall_radius(t):
    a = 1 - t
    return .854*a**3 + 3*.95*a*a*t + 3*1.075*a*t*t + SHOULDER*t**3


def wall_surface(theta, t):
    # Cubic section: narrow feet, flared belly, rounded shoulder, flat arrival
    # at the roof. Its derivatives give the outside faces of the steel boxes.
    a = 1 - t
    radius = wall_radius(t)
    dr = 3*(.096*a*a + .25*a*t + (SHOULDER-1.075)*t*t)
    co, si = np.cos(theta), np.sin(theta)
    eave = roof_height(AXES[0]*SHOULDER*co, AXES[1]*SHOULDER*si)
    fraction = 1 - a**1.55
    z = 2 + (eave-2)*fraction
    xyz = np.column_stack((AXES[0]*radius*co, AXES[1]*radius*si, z))
    along = np.column_stack((-AXES[0]*radius*si, AXES[1]*radius*co, 34.6*si*co*fraction))
    up = np.column_stack((AXES[0]*dr*co, AXES[1]*dr*si, (eave-2)*1.55*a**.55))
    return xyz, unit(np.cross(along, up))


def box_beam(model, xyz, normals, width, depth, face):
    """Sweep a rectangular steel section, retaining its face and recessed web."""
    start = xyz.copy()
    start[:, :2] += CENTRE
    for p, q, n in zip(start[:-1], start[1:], normals[:-1] + normals[1:]):
        tangent = unit(q - p)
        side = unit(np.cross(tangent, n))
        outer = np.cross(side, tangent)
        pad = (width + depth) / 2 + 1
        low = np.maximum(0, np.floor(np.minimum(p, q)-pad).astype(int))
        high = np.minimum(model.shape, np.ceil(np.maximum(p, q)+pad).astype(int))
        region = tuple(slice(l, h) for l, h in zip(low, high))
        sample = np.moveaxis(np.indices(high-low), 0, -1) + low + .5 - (p+q)/2
        height = sample @ outer
        inside = ((np.abs(sample @ tangent) <= np.linalg.norm(q-p)/2 + .17)
                  & (np.abs(sample @ side) <= width/2) & (np.abs(height) <= depth/2))
        block = model[region]
        block[inside & (block != STEEL) & (block != SECONDARY)] = WEB
        block[inside & (height > -depth*.13)] = face


def lip_xy(theta):
    cs = np.column_stack((np.cos(theta), np.sin(theta)))
    return OPENING * np.sign(cs) * np.abs(cs)**(2/LIP_POWER)


def traverse(model, angle, clearance, sweep, width, depth, face, inset):
    # A straight line grazes the oculus in plan. Both ends round the shoulder
    # and become inclined legs, continuously from one footing to the other.
    contact = lip_xy([angle])[0] * clearance
    gradient = np.sign(contact)*(np.abs(contact)/OPENING)**(LIP_POWER-1) / OPENING
    direction = unit(np.array([-gradient[1], gradient[0]]))
    c, d = contact / AXES, direction / AXES
    roots = np.sort(np.roots((d@d, 2*(c@d), c@c-SHOULDER**2)))
    along = np.linspace(*roots, int((roots[1]-roots[0])/1.4)+2)
    xy = contact + along[:, None]*direction
    roof, roof_n = roof_surface(xy)
    parts = []
    for end, sign, reach in ((0, -1, sweep[0]), (-1, 1, sweep[1])):
        ex, ey = xy[end]/AXES
        phi = np.arctan2(ey, ex)
        t = np.linspace(0, 1, 33) if end == 0 else np.linspace(1, 0, 33)
        bay = 2*np.pi/24
        foot = round((phi+sign*reach)/bay)*bay
        wall, wall_n = wall_surface(phi + (foot-phi)*(1-t), t)
        parts.append((wall, wall_n))
    points = np.vstack((parts[0][0][:-1], roof, parts[1][0][1:]))
    normals = np.vstack((parts[0][1][:-1], roof_n, parts[1][1][1:]))
    box_beam(model, points - inset*normals, normals, width, depth, face)


def steelwork(model):
    # Secondary wall braces cross several bays at varying heights.
    for i in range(25):
        s = np.linspace(0, 1, 44)
        phi = 2*np.pi*i/25 + (.50+.26*np.sin(2.3*i))*s
        h0 = .16 + .42*(.5+.5*np.sin(2.51*i))
        h1 = .35 + .47*(.5+.5*np.sin(1.81*i+.8))
        wall, normal = wall_surface(phi, h0*(1-s)+h1*s)
        box_beam(model, wall-1.3*normal, normal, 1.35, 1.55, SECONDARY)

    for i in range(22):
        angle = 2*np.pi*(i+.42+.14*np.sin(4.1*i))/22
        clearance = 1.13 + .50*(.5+.5*np.sin(2.24*i+.4))
        reach = (.10+.48*(.5+.5*np.sin(1.92*i)), .10+.43*(.5+.5*np.sin(3.21*i)))
        traverse(model, angle, clearance, reach, 1.55, 1.75, SECONDARY, .70)

    for i in range(24):
        angle = 2*np.pi*(i+.12)/24
        reach = (.11+.30*(.5+.5*np.sin(1.76*i)), .12+.24*(.5+.5*np.sin(2.13*i)))
        traverse(model, angle, 1.032, reach, 2.05, 2.50, STEEL, 0.)

    lip, normals = roof_surface(lip_xy(np.linspace(0, 2*np.pi, 241)))
    box_beam(model, lip-.55*normals, normals, 1.65, 2.7, STEEL)


def interiors(model, x, y, r, theta):
    # A track with two straights and two semicircles, inside the red bowl.
    track_dist = np.hypot(np.maximum(np.abs(x)-26.4, 0), y)
    level = model[:, :, 1]
    level[r < .893] = PODIUM
    level[track_dist <= 31.5] = TRACK
    for lane in (25.5, 27.8, 30.1):
        level[np.abs(track_dist-lane) < .32] = TRACK_LINE
    level[track_dist < 24.8] = TURF
    pitch = (np.abs(x) <= 32.8) & (np.abs(y) <= 21.2)
    level[pitch & (np.floor((x+32.8)/6.56).astype(int)%2 == 0)] = MOWN
    markings = ((np.abs(x) > 31.9) | (np.abs(y) > 20.4) | (np.abs(x)<.65)
                | (np.abs(np.hypot(x, y)-5.7)<.45)
                | ((np.abs(x)>22) & (np.abs(np.abs(y)-12.6)<.5))
                | ((np.abs(np.abs(x)-22.4)<.5) & (np.abs(y)<12.8)))
    level[pitch & markings] = CHALK

    # Seating ascends outwards in three decks and clears the outer concourse.
    dx, dy = AXES[0]*np.cos(theta), AXES[1]*np.sin(theta)
    inner = ((np.abs(dx)/59.8)**2.55 + (np.abs(dy)/33.2)**2.55)**(-1/2.55)
    outer = .84
    t = (r-inner)/(outer-inner)
    seats = (t >= 0) & (t <= 1)
    row = np.clip(np.floor(17*t), 0, 16).astype(int)
    # Lift the top tier by half the roof adjustment, tapering to zero at the pitch.
    high = 17 + ROOF_LIFT/2 + 12*np.sin(theta)**2
    terrace = 3 + np.rint(high*(row/16)**1.16) + (row>=7)*1.8 + (row>=12)*2.1
    sector = np.mod(theta*32/(2*np.pi) + .5, 1) - .5
    aisle = np.abs(sector) < .054
    # Recessed stairs connect the concourse levels behind the lattice. Each
    # pair rises toward a landing above an entrance bay.
    stair = 1 - np.abs(((theta/(2*np.pi)*12+.5) % 1)*2-1)
    stair_z = 4 + np.floor(11*stair)
    for z in range(2, int(terrace.max())+1):
        floor = model[:, :, z]
        top = seats & (z <= terrace) & (z > terrace-1)
        support = seats & (z <= terrace) & ((r < .786) | (z >= terrace-2.8))
        floor[support] = BOWL_WALL
        fascia = seats & ((row == 7) | (row == 12)) & (z < terrace) & (z >= terrace-2.9)
        floor[fascia] = GALLERY
        floor[top] = CHERRY
        floor[top & ((row%4 == 2) | ((row>=12) & (row%3 == 1)))] = CORAL
        floor[top & (row>=14)] = PALE
        floor[top & (aisle | (row == 7) | (row == 12))] = AISLE
        gate = (np.abs(sector) < .16) & ((row == 6) | (row == 11))
        floor[top & gate] = GALLERY

        # Slab noses and open door bays in the independent concourse wall.
        back = (r >= .772) & (r < .798) & seats & (z < terrace-2.8)
        floor[back] = BOWL_WALL
        if z in (7, 14, 21):
            floor[(r >= .755) & (r < .827) & (z < terrace-2.8)] = CONCRETE
        if z in (4, 5, 11, 12, 18, 19):
            floor[back & (np.abs(sector) < .38)] = GALLERY
        flight = (r>.807) & (r<.874) & (stair>.09)
        floor[flight & (z==stair_z)] = CONCRETE
        floor[flight & (z==stair_z-1)] = BOWL_WALL

    # Low entrance landings and flights on the four axes.
    for along, cross, radius in ((np.abs(x), y, AXES[0]), (np.abs(y), x, AXES[1])):
        gate = (along > radius*.755) & (np.abs(cross) < 9.5)
        for z, reach in ((1, .997), (2, .969), (3, .936)):
            model[:, :, z][gate & (along < radius*reach)] = PODIUM
        for z in range(4, 9):
            jambs = (np.abs(cross)>7.8) | (np.abs(cross)<.8)
            model[:, :, z][gate & (along < radius*.83)] = GALLERY
            model[:, :, z][gate & (along > radius*.82) & (along < radius*.85) & jambs] = CONCRETE
        model[:, :, 9][gate & (along < radius*.88)] = CONCRETE


def generate_model():
    model = np.zeros(SHAPE, dtype=np.uint8)
    model[:, :, 0] = PODIUM
    x, y = np.indices(SHAPE[:2], dtype=float) + .5 - CENTRE[:, None, None]
    r = np.hypot(x/AXES[0], y/AXES[1])
    theta = np.arctan2(y/AXES[1], x/AXES[0])
    interiors(model, x, y, r, theta)

    hole = (np.abs(x)/OPENING[0])**LIP_POWER + (np.abs(y)/OPENING[1])**LIP_POWER
    cover = (hole >= 1.) & (r < SHOULDER)
    top = roof_height(x, y) - 2.4
    eave = roof_height(AXES[0]*SHOULDER*np.cos(theta), AXES[1]*SHOULDER*np.sin(theta))
    for z in range(2, model.shape[2]):
        sheet = cover & (z+.5 <= top) & (z+.5 > top-1.45)
        # Fold the last roof panels down behind the eave members. The lower
        # facade stays open to the independent concourses and entrance stairs.
        fraction = (z+.5+2.4-2)/(eave-2)
        t = 1-(1-np.clip(fraction, 0, 1))**(1/1.55)
        edge = wall_radius(t)
        sheet |= (t>.79) & (fraction<=1) & (r<edge) & (r>edge-.017)
        model[:, :, z][sheet] = MEMBRANE
        # Visible inside face of the deep roof truss, with dark acoustic
        # openings between the inner fascia's uprights.
        rim = (hole >= 1.) & (hole < 1.13) & (z+.5 < top+2) & (z+.5 > top-3.5)
        model[:, :, z][rim] = FASCIA
        slot = (z+.5 < top+.1) & (z+.5 > top-1.6) & (np.cos(24*theta) > -.35)
        model[:, :, z][rim & slot] = SOFFIT
    steelwork(model)

    walkers = make_model('walkers')
    for ox, oy, sx, sy in ((226, 104, 10, 12), (230, 100, 10, 19),
                           (99, 189, 26, 26), (14, 81, 10, 19), (126, 5, 10, 12)):
        person = walkers[sx:sx+3, sy:sy+3, 1:6]
        block = model[ox:ox+3, oy:oy+3, 1:6]
        block[person>0] = person[person>0] + PROP_OFFSET
    return model


if __name__ == '__main__':
    np.savez_compressed(SOURCE / 'model.npz', model=generate_model())
