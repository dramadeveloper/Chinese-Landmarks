"""Xi'an Bell Tower: measured base/height, photo-estimated timber and roofs."""
import numpy as np
from export_plaza import make_model

ART_SCALE = 4.0
UNITS_PER_METRE = ART_SCALE / 2
SIZE, HEIGHT = 120, 96
GROUND, BRICK, STONE, RED, DARK, GREEN, RIDGE, GOLD, PAINT = range(1, 10)
STAIR_STONE = 10
WOOD_TRIM = 11
PROP_OFFSET = 20
# outer half-span, inner half-span, eave height, rise, corner lift (metres).
# Three eaves, two timber storeys, one four-sided pointed roof.
ROOFS = ((13.4, 9.0, 15.8, 1.8, .65),
         (12.3, 9.0, 24.5, 1.5, .65),
         (11.5, .75, 27.9, 11.5, .7))


def roof_height(x, y, roof):
    outer, inner, base, rise, lift = roof
    radius = np.maximum(abs(x), abs(y))
    t = np.clip((outer-radius)/(outer-inner), 0, 1)
    corner = (np.minimum(abs(x), abs(y))/outer)**8
    return base + rise*t**1.65 + lift*corner


def make_building():
    a = np.zeros((SIZE, SIZE, HEIGHT), np.uint8)
    a[:, :, 0] = GROUND
    factor = UNITS_PER_METRE

    def box(x0, y0, z0, x1, y1, z1, material):
        low = np.rint([SIZE/2+x0*factor, SIZE/2+y0*factor, 1+z0*factor]).astype(int)
        high = np.rint([SIZE/2+x1*factor, SIZE/2+y1*factor, 1+z1*factor]).astype(int)
        a[low[0]:high[0], low[1]:high[1], low[2]:high[2]] = material

    def rail(half, z, material=STONE, north_gap=False, cap_material=STONE, posts=11):
        for axis in (0, 1):
            for sign in (-1, 1):
                for c in np.linspace(-half, half, posts):
                    if north_gap and axis == 1 and sign == 1 and 3.2 < abs(c) < 7.4:
                        continue
                    xy = (c, sign*half) if axis == 1 else (sign*half, c)
                    box(xy[0]-.3, xy[1]-.3, z, xy[0]+.4, xy[1]+.4, z+1.3, material)
                spans = ((-half, -7.4), (-3.2, 3.2), (7.4, half)) if north_gap and axis == 1 and sign == 1 else ((-half, half),)
                for lo, hi in spans:
                    if axis == 1:
                        box(lo, sign*half-.3, z+.65, hi, sign*half+.35, z+1.1, cap_material)
                    else:
                        box(sign*half-.3, lo, z+.65, sign*half+.35, hi, z+1.1, cap_material)

    # 35.5 m square masonry plinth, 8.6 m above the foundation datum.
    box(-17.75, -17.75, 0, 17.75, 17.75, 8.6, BRICK)
    box(-17.95, -17.95, 0, 17.95, 17.95, .45, STONE)
    box(-17.95, -17.95, 8.1, 17.95, 17.95, 8.6, STONE)
    xx, yy, zz = np.indices(a.shape, dtype=float)
    x, y, z = (xx+.5-SIZE/2)/factor, (yy+.5-SIZE/2)/factor, (zz+.5-1)/factor
    # Both crossing tunnels are real voids, including their semicircular vaults.
    for transverse, along in ((x, y), (y, x)):
        arch = (abs(transverse)<3) & (z>=0) & ((z<3) | ((z-3)**2+transverse**2<9))
        rim = ((abs(transverse)<3.65) & (z>=0) & ((z<3) | ((z-3)**2+transverse**2<3.65**2))
               & (abs(along)>17.0) & (abs(along)<18.0) & (a != 0))
        a[rim] = STONE
        a[arch & (abs(along)<18.1) & (zz>=1)] = 0
    rail(17.2, 8.6, north_gap=True)

    # North stair flights climb inward from the two corners to a shared landing.
    # Merge the documented 59 small steps into readable one-voxel risers.
    steps = round(8.6*factor)
    for side in (-1, 1):
        for i in range(steps):
            lo = 4.2 + (steps-1-i)*12.8/steps
            hi = 4.2 + (steps-i)*12.8/steps
            ztop = 8.6*(i+1)/steps
            xa, xb = sorted((side*lo, side*hi))
            # Grey masonry supports with stone tread caps, not a pale solid wedge.
            box(xa, 17.3, 0, xb, 20.4, ztop, BRICK)
            box(xa, 17.3, ztop-1/factor, xb, 20.4, ztop, STAIR_STONE)
            box(xa, 20.1, ztop, xb, 20.6, ztop+.85, STONE)
        box(side*4.7-.5, 20.0, 8.6, side*4.7+.5, 20.8, 10.0, STONE)

    # Three-bay hall (16.57 m), surrounding gallery (21.39 m).
    box(-10.9, -10.9, 8.6, 10.9, 10.9, 9.1, STONE)
    # Lift the upper gallery above the first roof; its old floor intersected it.
    # This local vertical change exposes both storeys in the fixed 2:1 camera.
    box(-10.5, -10.5, 17.5, 10.5, 10.5, 19.5, PAINT)
    for bottom, top, gallery in ((9.1, 15.8, True), (19.5, 24.5, True), (25.9, 28.0, False)):
        half = 8.285 if gallery else 9.0
        box(-half, -half, bottom, half, half, top, RED)
        # Dark inset lattice doors; red mullions kept broad enough for native 1x.
        for axis in (0, 1):
            for sign in (-1, 1):
                for c in (-5.5, 0, 5.5):
                    if axis == 0:
                        box(sign*half-.45, c-1.7, bottom+.6, sign*half+.45, c+1.7, top-.6, DARK)
                    else:
                        box(c-1.7, sign*half-.45, bottom+.6, c+1.7, sign*half+.45, top-.6, DARK)
        if gallery:
            for axis in (0, 1):
                for sign in (-1, 1):
                    for c in (-10.4, -6.2, 0, 6.2, 10.4):
                        px, py = (sign*10.4, c) if axis == 0 else (c, sign*10.4)
                        box(px-.45, py-.45, bottom, px+.45, py+.45, top, RED)
                        box(px-.8, py-.8, top-1.0, px+.8, py+.8, top-.2, PAINT)
            # A continuous muted green beam reads as painted brackets at 1x.
            box(-10.9, -10.9, top-.8, 10.9, 10.9, top, PAINT)
            if bottom > 10:
                box(-10.9, -10.9, bottom-.4, 10.9, 10.9, bottom, RED)
                rail(10.6, bottom, WOOD_TRIM, cap_material=WOOD_TRIM, posts=9)

    # The steep top roof is solid below its surface: a thin vertical shell leaves
    # diagonal voxel gaps at this pitch. Lower apron roofs retain their shells.
    for index, roof in enumerate(ROOFS):
        outer = roof[0]
        surface = roof_height(x, y, roof)
        bottom = roof[2]-.95 if index == 2 else surface-.95
        region = (np.maximum(abs(x), abs(y)) <= outer) & (z <= surface) & (z > bottom)
        a[region] = GREEN
        hips = region & (abs(abs(x)-abs(y)) < .4)
        a[hips] = RIDGE
        eave = region & (np.maximum(abs(x), abs(y)) > outer-.65) & (z > surface-.55)
        a[eave] = RIDGE
    # Small lotus support and gold finial: one unified accent, no random ornament.
    box(-1.0, -1.0, 39.1, 1.0, 1.0, 39.9, RIDGE)
    for z0, z1, half in ((39.9, 40.3, 1.0), (40.3, 41.8, .75), (41.8, 42.3, .4)):
        box(-half, -half, z0, half, half, z1, GOLD)

    # Two sparse visitors use the shared adult geometry, independent of art scale.
    person = make_model('walkers')[10:13, 12:14, 1:6]
    for px, py in ((91, 109), (105, 92)):
        target = a[px:px+3, py:py+2, 1:6]
        target[person != 0] = person[person != 0]+PROP_OFFSET
    return a
