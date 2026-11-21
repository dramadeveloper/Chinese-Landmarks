"""Harbin Saint Sophia: photographic study with the selected west belfry and cross.

Coordinates in metres: +x is the west entrance, +y a transept. z=0 is
the upper surface of the public pavement. The people use native model units.
"""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'chinese-landmarks'))
from export_plaza import make_model

(PAVING, GRANITE, BRICK, MOULDING, REVEAL, SASH, GLASS, DOOR, ROOF,
 CUPOLA, RIDGE, GILT, BRONZE, DRUM, SANDSTONE) = range(1, 16)
PEOPLE = 30
TENT_METAL = 16
(BELL_BRICK, BELL_TRIM, BELL_REVEAL, BELL_WOOD, BELL_EAVE, BELL_TENT,
 SELECTED_GOLD) = range(17, 24)
CENTRE = (47, 60)
ART_SCALE = 4
# Outer silhouette from the side photograph. The broadest part is low down.
BULB_Z = [36.25, 37.1, 38, 39.1, 40.3, 41.4, 42.7, 44.1, 45.4, 46.65, 47.7, 48.55]
BULB_R = [7.18, 7.73, 8.03, 8.18, 8.10, 7.83, 7.21, 6.13, 4.76, 2.96, 1.56, .70]


def facet(x, y, sides):
    """Distance to nearest facet plane and transverse offset along its face."""
    sector = 2*np.pi/sides
    angle = sector*np.round(np.arctan2(y, x)/sector)
    return x*np.cos(angle)+y*np.sin(angle), y*np.cos(angle)-x*np.sin(angle)


def generate_model(art_scale=ART_SCALE):
    k = art_scale/2
    a = np.zeros((120, 120, int(np.ceil(53.35*k))+3), dtype=np.uint8)
    ix, iy, iz = np.ogrid[:120, :120, :a.shape[2]]
    x, y = (ix+.5-CENTRE[0])/k, (iy+.5-CENTRE[1])/k
    z = (iz-.5)/k
    a[:, :, 0] = PAVING

    def fill(plan, bottom, top, mat):
        a[plan & (z >= bottom) & (z < top)] = mat

    def arch(t, base, spring, width):
        return (z >= base) & (t*t + np.maximum(z-spring, 0)**2 < width*width)

    def opening(depth, t, face, sill, spring, width):
        # Relief around a recessed pane: moulded hood, dark jamb, glass and mullion.
        hood = (depth >= face-.6) & (depth <= face+.48)
        a[hood & arch(t, sill-.3, spring, width+.46)] = MOULDING
        hole = arch(t, sill, spring, width)
        a[hood & hole] = 0
        inset = (depth >= face-1.28) & (depth < face-.45) & hole
        a[inset] = SASH
        pane = inset & arch(t, sill+.32, spring, width-.22)
        a[pane] = GLASS
        a[pane & ((abs(t) < .18) | (abs(z-spring+.48) < .22))] = SASH
        fill((abs(t) < width+.42) & hood, sill-.48, sill-.12, SANDSTONE)

    def entry(depth, t, face, width, spring):
        # Four concentric archivolts step backwards to the paired oak doors.
        for inset, extend, mat in ((0, 1.65, MOULDING), (.58, 1.13, REVEAL),
                                   (1.04, .92, MOULDING), (1.61, .56, BRICK),
                                   (1.97, .25, REVEAL)):
            ring = arch(t, 1.45, spring, width-inset)
            a[(depth >= face-.75) & (depth < face+extend) & ring] = mat
            a[(depth >= face+extend) & (depth < face+1.7) & ring] = 0
        doors = arch(abs(t)-.91, 1.45, spring-.38, .88)
        a[(depth >= face-.45) & (depth <= face+.55) & doors] = DOOR
        for h in range(3):
            step = (depth > face) & (depth <= face+3.4-h*.66) & (abs(t) <= width+.4)
            fill(step, h*.48, (h+1)*.48, GRANITE)

    def gable(depth, t, face, spring, span):
        # Keel-shaped kokoshnik above a window bay, with a metal drip edge.
        v = abs(t)/span
        crown = spring+span*(np.sqrt(np.maximum(0, 1-v*v)) + .25*np.maximum(0, 1-3*v))
        plan = (abs(t) <= span) & (depth > face-.93) & (depth < face+.34)
        fill(plan, spring-.35, crown, MOULDING)
        fill(plan, crown-.51, crown+.13, RIDGE)

    def envelope(distance, radius, eave):
        for lo, hi, offset, mat in (
            (0, 1.6, .22, GRANITE), (1.6, eave, 0, BRICK),
            (2.05, 2.48, .20, MOULDING), (5.7, 6.21, .32, MOULDING),
            (6.21, 6.47, .36, RIDGE), (eave-.9, eave-.48, .14, MOULDING),
            (eave-.47, eave+.04, .34, MOULDING)):
            fill(distance <= radius+offset, lo, hi, mat)

    # Long west nave. The entrance projects beyond both side walls as a
    # three-part brick screen; its ridged portal is the widest doorway.
    nave = np.maximum(abs(y)-6.30, abs(x-13.15)-11.25)
    envelope(nave, 0, 17.5)
    roof = 20.2-.40*abs(y)
    fill((abs(y) < 6.75) & (x > .8) & (x < 24.55), 17.35, roof+.15, ROOF)
    fill((abs(y) < .34) & (x > 4) & (x < 24.55), 20, 20.55, RIDGE)
    for direction in (-1, 1):
        face = direction*y
        for pos in (9.0, 13.5, 18.0, 22.3):
            opening(face, x-pos, 6.30, 6.9, 12.95, 1.28)
            opening(face, x-pos, 6.30, 2.6, 4.25, .85)
            gable(face, x-pos, 6.4, 17.45, 1.65)
        for pos in (6.9, 11.25, 15.75, 20.25, 24.08):
            fill((abs(x-pos) < .43) & (face > 6.05) & (face <= 6.72), 6.5, 17.0, MOULDING)
    for side in (-1, 1):
        pillar = (abs(y-side*5.05) < 1.12) & (x >= 21.1) & (x < 24.8)
        fill(pillar, 1.5, 18.55, BRICK)
        for h in (8.5, 11, 13.5, 16):
            fill((abs(y-side*5.05) < .54) & (x > 24.55) & (x < 25), h, h+1, REVEAL)
        gable(x, y-side*5.1, 24.6, 18.45, 1.13)
    fill((abs(y) < 3.9) & (x >= 20) & (x < 24.9), 13.0, 22.20, BRICK)
    fill((abs(y) < 4.10) & (x >= 20) & (x < 25.10), 21.45, 22.20, MOULDING)
    gable(x, y, 24.9, 22.2, 1.68)
    for t in (-2.45, 2.45):
        opening(x, y-t, 24.9, 13.5, 16.1, .62)
    entry(x, y, 24.4, 5.03, 6.55)
    eye = y*y+(z-12.5)**2
    a[(x > 23.65) & (x <= 25.0) & (eye < 1.42**2)] = MOULDING
    a[(x > 23.65) & (x <= 25.0) & (eye < .84**2)] = SASH

    # Broad, chamfered transept ends carry lower tents. Each has its own
    # entrance and three prominent arcaded bays wrapping the polygonal face.
    for sign in (-1, 1):
        out = sign*y
        ax = abs(x)
        cut = np.maximum.reduce(np.broadcast_arrays(ax-7.15, out-13.80, 1.8-out,
                                                    (ax+out-18.5)/np.sqrt(2)))
        envelope(cut, 0, 15.85)
        fill(cut <= .4, 15.82, 19.45-.54*np.maximum(ax, abs(out-8.4)), ROOF)
        for side in (-1, 1):
            angled = (side*x+out)/np.sqrt(2)
            tangent = (side*x-out+8.4)/np.sqrt(2)
            opening(angled, tangent, 18.5/np.sqrt(2), 7.0, 12.75, 1.32)
            opening(out, x-side*3.05, 13.8, 7.15, 13.0, 1.0)
        entry(out, x, 13.8, 3.32, 4.0)

    # Eastern apse and lower axial chapel. Faceted brick bays below a half hip.
    radius, lateral = facet(x+6.9, y, 10)
    envelope(radius, 6.60, 14.9)
    for band in (-1, 1):
        angled = -(x+6.9)*np.cos(.628)+band*y*np.sin(.628)
        tangent = (x+6.9)*np.sin(.628)+band*y*np.cos(.628)
        opening(angled, tangent, 6.6, 6.75, 11.25, 1.36)
    fill(radius < 7, 14.8, 19.5-.60*radius, ROOF)
    chapel, _ = facet(x+12.85, y, 8)
    envelope(chapel, 3.20, 8.7)
    fill(chapel < 3.48, 8.65, 11.15-.6*chapel, ROOF)

    def finial(cx, cy, base, tip, bulb):
        xx, yy = x-cx, y-cy
        rad = np.hypot(xx, yy)
        f = (z-base)/(tip-base)
        r = np.interp(f, [0, .10, .21, .29, .36, .43, .52, .59],
                     [bulb*.85, bulb*.48, bulb*.43, bulb*.93, bulb, bulb*.77, bulb*.29, .26])
        fill(rad < r, base, base+.59*(tip-base), GILT)
        fill((abs(xx) < .38) & (abs(yy) < .50), base+.50*(tip-base), tip, GILT)
        cross_height = base+.80*(tip-base)
        fill((abs(xx) < .40) & (abs(yy) < max(1.25, bulb*1.45)),
             cross_height-.34, cross_height+.34, GILT)

    def tent(cx, cy, r, foot, shoulder, peak, tip):
        distance, t = facet(x-cx, y-cy, 8)
        envelope_plan = distance <= r
        fill(envelope_plan, foot, shoulder, BRICK)
        fill(distance <= r+.24, foot, foot+.62, MOULDING)
        sill, spring, w = foot+.5, shoulder-1, r*.31
        opening(distance, t, r, sill, spring, w)
        lip = np.sqrt(np.maximum(0, (w+.57)**2-t*t))
        hood = (distance >= r-.21) & (distance < r+.28) & (abs(t) <= w+.57)
        fill(hood, spring+lip-.54, spring+lip, MOULDING)
        radius = np.interp(z, [shoulder-.4, shoulder+.9, peak-.75, peak],
                           [r+.18, r-.22, .74, .6])
        fill(distance <= radius, shoulder-.4, peak, TENT_METAL)
        gable(distance, t, r, shoulder-.72, r*.39)
        fill(distance < .86, peak-.64, peak+.32, RIDGE)
        finial(cx, cy, peak+.24, tip, max(.80, r*.34))

    def west_belfry():
        # Original A geometry, from the octagonal base through the gilded finial.
        dx = x-21.45
        radius, floor, shoulder, peak, top = 4.35, 18.1, 27.5, 35.0, 41.2
        phase = (np.arctan2(y, dx)+np.pi/8) % (np.pi/4)-np.pi/8
        r = np.hypot(dx, y)*np.cos(phase)
        t = np.tan(phase)*radius
        spring = shoulder-1.1
        fill(r <= radius, floor, shoulder, BELL_BRICK)
        rim = (r <= radius+.35) & (t*t+np.maximum(z-spring, 0)**2 <= (radius*.39)**2)
        fill(rim, floor+.7, shoulder+1, BELL_TRIM)
        fill(r <= radius+.36, floor, floor+2.1, BELL_BRICK)
        fill(r <= radius+.51, floor+1.9, floor+2.65, BELL_TRIM)
        void = (r < radius-1) | (t*t+np.maximum(z-spring, 0)**2 <= (radius*.33)**2)
        fill((r <= radius+.36) & void, floor+2.65, shoulder, 0)
        fill((r >= radius-1) & (r < radius-.52) & ~void, floor+2.65, shoulder, BELL_REVEAL)
        mouth = shoulder-4.05
        bell = np.interp(z, [mouth, mouth+.58, mouth+2.25], [1.9, 1.5, .7])
        radial = np.hypot(dx, y)
        fill(radial < bell, mouth, mouth+2.25, BRONZE)
        fill(radial < .5, mouth+2.25, shoulder, BELL_WOOD)
        profile = np.interp(z, [shoulder-.5, shoulder+.55, shoulder+1.45, peak],
                            [radius+.45, radius+.35, radius*.84, .42])
        fill(r <= profile, shoulder-.5, peak, BELL_TENT)
        cap = np.sqrt(np.maximum(0, (radius*.36)**2-t*t))
        bay = (r >= radius-.2) & (r <= radius+.43) & (abs(t) <= radius*.36)
        fill(bay, shoulder-.55, shoulder+cap, BELL_TRIM)
        fill(bay, shoulder+cap-.35, shoulder+cap+.25, BELL_EAVE)
        bulb = radius*.23
        h = top-peak
        outline = np.interp((z-peak)/h, [0, .12, .21, .32, .44, .58],
                            np.array([.46, .5, 1, .92, .35, .25])*bulb)
        fill(radial <= outline, peak, peak+h*.58, SELECTED_GOLD)
        post = max(.50, .85/k)
        base = peak+h*.31
        fill((abs(dx) <= post) & (abs(y) <= post), base, top, SELECTED_GOLD)
        for frac, span in ((.59, 1), (.85, .55)):
            level = base+(top-base)*frac
            fill((abs(dx) <= post) & (abs(y) <= span*max(1.65, bulb*1.85)),
                 level-.29, level+.29, SELECTED_GOLD)

    west_belfry()
    for sign in (-1, 1):
        tent(.10, sign*10.85, 3.18, 17.6, 20.0, 26.05, 31.55)
    tent(-13.0, 0, 1.94, 10.2, 12.1, 15.0, 18.1)

    # The crossing has an octagonal shoulder, a narrower glazed sixteen-sided
    # drum and an overhanging corbelled cornice supporting the copper shell.
    lower, _ = facet(x, y, 8)
    fill(lower < 7.55, 0, 22.2, BRICK)
    fill(lower < 7.82, 20.7, 21.25, MOULDING)
    fill(lower < 7.93, 21.95, 24.45-.28*lower, ROOF)
    drum, t = facet(x, y, 16)
    fill(drum <= 6.94, 22.12, 36.70, DRUM)
    fill(drum <= 7.18, 22.30, 22.95, RIDGE)
    fill(drum <= 7.24, 23.02, 23.62, MOULDING)
    opening(drum, t, 6.94, 24.22, 30.4, .89)
    pilaster = (drum >= 6.72) & (drum <= 7.22) & (abs(t) > 1.00)
    fill(pilaster, 23.45, 32.25, MOULDING)
    for h, r, mat in ((32.40, 7.23, MOULDING), (33.05, 7.18, BRICK),
                      (35.1, 7.37, MOULDING), (35.70, 7.50, BRICK)):
        fill(drum <= r, h, h+.65, mat)
    bead = (drum >= 6.69) & (drum < 7.23) & arch(t, 33.4, 34.0, .62)
    a[bead] = REVEAL

    radial = np.hypot(x, y)
    fill(radial < np.interp(z, BULB_Z, BULB_R), BULB_Z[0], BULB_Z[-1], CUPOLA)
    fill(radial <= 1.23, 48.3, 48.95, SELECTED_GOLD)
    fill((abs(x) <= .5) & (abs(y) <= .5), 48.75, 53.35, SELECTED_GOLD)
    fill((abs(x+y) <= .63) & (abs(x-y) <= 3.0), 51.0, 51.9, SELECTED_GOLD)
    pair = make_model('walkers')[10:13, 12:21, 1:6]
    block = a[106:109, 65:74, 1:6]
    block[pair != 0] = pair[pair != 0]+PEOPLE
    return a


if __name__ == '__main__':
    np.savez_compressed(HERE/'model.npz', model=generate_model())
