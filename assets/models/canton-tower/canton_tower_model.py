"""Canton Tower: a ruled, eccentric elliptical lattice, in model units."""
import numpy as np

# Match the current Pearl's measured 307 u / 468 m total-height scale.
UNITS_PER_METRE = 307 / 468
ART_SCALE = 2 * UNITS_PER_METRE
SIZE = (120, 120, 400)
STONE, COLUMN, RING, BRACE, CORE, GLASS, DECK, RED = range(1, 9)
PROP_OFFSET = 20
BASE_AXES = (80, 72)  # Base minor axis enlarged by 20% for the model silhouette.
TOP_AXES = (50, 49)   # Top minor axis enlarged by 8.9% for the model silhouette.
TWIST = np.deg2rad(135)
ROTATION = np.array([[np.cos(TWIST), -np.sin(TWIST)],
                     [np.sin(TWIST), np.cos(TWIST)]])


def section(t, theta):
    """24 straight generators between rotated ellipses; no symmetric hourglass."""
    p = np.array([BASE_AXES[0]/2 * np.cos(theta), BASE_AXES[1]/2 * np.sin(theta)])
    q = ROTATION @ np.array([TOP_AXES[0]/2 * np.cos(theta), TOP_AXES[1]/2 * np.sin(theta)])
    # The offset is a documented visual estimate, not a surveyed dimension.
    xy = (1-t) * p + t * (q + np.array([6, -3]))
    return np.array([60 + xy[0]*UNITS_PER_METRE,
                     60 + xy[1]*UNITS_PER_METRE,
                     (3 + t*447)*UNITS_PER_METRE])


def make_model():
    model = np.zeros(SIZE, np.uint8)
    model[:, :, 0] = STONE

    def tube(a, b, radius, material):
        a, b = np.array(a), np.array(b)
        lo = np.maximum(np.floor(np.minimum(a, b)-radius).astype(int), 0)
        hi = np.minimum(np.ceil(np.maximum(a, b)+radius).astype(int), SIZE)
        grid = np.moveaxis(np.indices(hi-lo, dtype=float), 0, -1) + lo + .5
        direction = b-a
        t = np.clip(((grid-a)*direction).sum(axis=-1)/(direction@direction), 0, 1)
        distance = ((grid-a-t[..., None]*direction)**2).sum(axis=-1)
        target = model[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        target[distance <= radius*radius] = material

    def ellipse(cx, cy, rx, ry, z0, z1, material, angle=0):
        x, y = np.indices(SIZE[:2], dtype=float) + .5
        dx, dy = x-cx, y-cy
        u, v = dx*np.cos(angle)+dy*np.sin(angle), -dx*np.sin(angle)+dy*np.cos(angle)
        mask = (u/rx)**2 + (v/ry)**2 <= 1
        model[:, :, z0:z1][mask] = material

    def loop(points, radius, material):
        for a, b in zip(points, np.roll(points, -1, axis=0)):
            tube(a, b, radius, material)

    # Low entrance terrace, entirely inside the 3x3 open public-paving boundary.
    ellipse(60, 60, 34, 27, 1, 2, STONE)
    ellipse(60, 60, 31, 24, 2, 3, DECK)
    model[49:71, 28:36, 1:2] = STONE
    model[50:70, 31:38, 2:3] = STONE
    model[51:69, 34:40, 3:4] = DECK
    ellipse(60, 60, 19, 14, 3, 10, GLASS)
    ellipse(60, 60, 20, 15, 10, 12, DECK)
    model[56:64, 45:48, 3:9] = CORE
    model[53:67, 43:49, 9:10] = COLUMN

    # Vertical, dark concrete/elevator core. Its axis differs from the base ellipse.
    cx, cy = 62.5, 58.8
    ellipse(cx, cy, 8.5*UNITS_PER_METRE, 7*UNITS_PER_METRE,
            3, round(448*UNITS_PER_METRE), CORE)
    # Glazed program clusters read as separate buildings inside the open cage.
    # Heights/plan setbacks are photo/section estimates, not floor-by-floor survey.
    for lower, upper, fraction in [(24, 36, .72), (104, 122, .69),
                                   (155, 173, .70), (335, 354, .75), (375, 443, .79)]:
        for z in range(round(lower*UNITS_PER_METRE), round(upper*UNITS_PER_METRE)):
            t = (z/UNITS_PER_METRE-3)/447
            matrix = (1-t)*np.diag(np.array(BASE_AXES)/2) + t*ROTATION@np.diag(np.array(TOP_AXES)/2)
            u, singular, _ = np.linalg.svd(matrix)
            angle = np.arctan2(u[1, 0], u[0, 0])
            material = DECK if (z-round(lower*UNITS_PER_METRE)) % 6 == 0 else GLASS
            ellipse(60+6*t*UNITS_PER_METRE, 60-3*t*UNITS_PER_METRE,
                    singular[0]*UNITS_PER_METRE*fraction,
                    singular[1]*UNITS_PER_METRE*fraction, z, z+1, material, angle)

    # A restrained zigzag denotes the 170-330 m open-air stair beside the core.
    for i, z in enumerate(np.arange(173, 330, 7)*UNITS_PER_METRE):
        x0, x1 = (cx-5, cx+5) if i % 2 == 0 else (cx+5, cx-5)
        tube([x0, cy-5.2, z], [x1, cy-5.2, z+7*UNITS_PER_METRE], .5, RING)

    # 46 rings: closer near the waist. Their stations are a visual approximation.
    h = np.linspace(0, 1, 2001)
    density = 1 + .55*np.exp(-((h-.62)/.21)**2)
    cdf = np.cumsum(density)
    stations = np.interp(np.linspace(cdf[0], cdf[-1], 46), cdf, h)
    angles = np.arange(24)*2*np.pi/24
    # Alternate 12 of the 24 secondary braces per tier at native 1x. All primary
    # columns/rings remain; this avoids a white, visually solid mesh at the waist.
    for row, (lower, upper) in enumerate(zip(stations[:-1], stations[1:])):
        for theta in angles[row % 2::2]:
            tube(section(lower, theta), section(upper, theta+2*np.pi/24), .52, BRACE)
    for t in stations:
        loop(np.array([section(t, a) for a in np.linspace(0, 2*np.pi, 100, endpoint=False)]),
             .57, RING)
    for theta in angles:
        # Two diameter zones are an intentional raster simplification of tapering tubes.
        tube(section(0, theta), section(.62, theta), .78, COLUMN)
        tube(section(.62, theta), section(1, theta), .65, COLUMN)

    # Original sloping roof rim and small static pods, on the widened ellipse.
    roof = np.array([section(1, a) for a in np.linspace(0, 2*np.pi, 120, endpoint=False)])
    roof[:, 2] += (roof[:, 0]-cx)*.18
    loop(roof, .7, COLUMN)
    for theta in np.arange(16)*2*np.pi/16:
        p = section(1, theta)
        p[2] += (p[0]-cx)*.18 + 1.2
        ellipse(p[0], p[1], 1.2, 1.2, round(p[2])-1, round(p[2])+1, GLASS)
        tube(p+[0, 0, .6], p+[0, 0, 1.3], .9, COLUMN)
    ellipse(cx, cy, 6.2, 5.5, round(450*UNITS_PER_METRE)-1,
            round(450*UNITS_PER_METRE)+1, DECK)

    # Octagonal lattice mast from roof to 520 m, then slim red/white antenna.
    mast_bottom, mast_top = 450*UNITS_PER_METRE, 520*UNITS_PER_METRE
    for theta in np.arange(8)*np.pi/4:
        a = [cx+4.6*np.cos(theta), cy+4.6*np.sin(theta), mast_bottom]
        b = [cx+2.0*np.cos(theta), cy+2.0*np.sin(theta), mast_top]
        tube(a, b, .62, COLUMN)
    for z in np.arange(mast_bottom, mast_top, 5):
        r = 4.6-2.6*(z-mast_bottom)/(mast_top-mast_bottom)
        loop(np.array([[cx+r*np.cos(a), cy+r*np.sin(a), z]
                       for a in np.arange(8)*np.pi/4]), .52, RING)
    ellipse(cx, cy, 6.8, 6, round(488*UNITS_PER_METRE)-1,
            round(488*UNITS_PER_METRE)+1, DECK)
    # Twin drop-tower tracks: static, only the characteristic vertical rails.
    for dx in (-2.8, 2.8):
        tube([cx+dx, cy-4, 455*UNITS_PER_METRE],
             [cx+dx, cy-4, 485*UNITS_PER_METRE], .6, COLUMN)
    for z in range(round(mast_top), 394):
        radius = 1.55 if z < 356 else (.95 if z < 378 else .65)
        material = RED if ((z-round(mast_top))//6) % 2 == 0 else COLUMN
        ellipse(cx, cy, radius, radius, z, z+1, material)
    # Explicit last pixel-column fixes the model top at 394 units.
    model[62, 58, 378:394] = COLUMN

    # Reuse the public 5-unit adults, unchanged in scale.
    from export_plaza import make_model as public_model
    people = public_model('walkers')
    target = model[73:113, 16:56, :24]
    occupied = people != 0
    target[occupied] = people[occupied] + PROP_OFFSET
    return model
