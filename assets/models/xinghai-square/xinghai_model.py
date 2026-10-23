"""Xinghai Square before 2016: central star plaza and the Nine-Dragon Huabiao."""
import numpy as np
from PIL import Image, ImageDraw
from export_plaza import make_model

SIZE, HEIGHT = 200, 88
GROUND, STONE, RELIEF, RED_PAVING, GOLD_PAVING, GRASS, BRONZE, GOLD, LAMP_GLASS = range(1, 10)
DARK_STONE, LAMP_FRAME, LAMP_STONE = 10, 11, 12
PROP_OFFSET = 32
CENTER = 100
OUTER_RADIUS, INNER_RADIUS, RED_RADIUS = 90, 75, 20
PLAZA_TOP = 4
HUABIAO_ENLARGEMENT = 1.75
HUABIAO_TOP = 74  # 70 units above the raised plaza, 56 vertical pixels.
LAMP_TOP = 38
STAR_ANGLE = -np.pi / 2
LAMP_CENTERS = [(round(CENTER + 61*np.cos(STAR_ANGLE + (k+.5)*2*np.pi/5)),
                 round(CENTER + 61*np.sin(STAR_ANGLE + (k+.5)*2*np.pi/5))) for k in range(5)]


def huabiao_point(x, y, z):
    """Enlarge the entire monument about its base, leaving the plaza fixed."""
    return (CENTER+(x-CENTER)*HUABIAO_ENLARGEMENT,
            CENTER+(y-CENTER)*HUABIAO_ENLARGEMENT,
            PLAZA_TOP+(z-PLAZA_TOP)*HUABIAO_ENLARGEMENT)


def huabiao_z(z):
    return PLAZA_TOP+(z-PLAZA_TOP)*HUABIAO_ENLARGEMENT


def star_outline(radius=INNER_RADIUS):
    """Five broad points joined by concave curves, as in the historical aerial."""
    points = []
    for k in range(5):
        a, b = STAR_ANGLE+k*2*np.pi/5, STAR_ANGLE+(k+1)*2*np.pi/5
        start = radius*np.array([np.cos(a), np.sin(a)])
        end = radius*np.array([np.cos(b), np.sin(b)])
        control = radius*.40*np.array([np.cos((a+b)/2), np.sin((a+b)/2)])
        for t in np.linspace(0, 1, 49, endpoint=False):
            p = (1-t)**2*start + 2*t*(1-t)*control + t*t*end + CENTER
            points.append(tuple(p))
    return points


def make_model_array():
    model = np.zeros((SIZE, SIZE, HEIGHT), np.uint8)
    model[:, :, 0] = GROUND
    xx, yy = np.indices((SIZE, SIZE), dtype=float)
    radius = np.hypot(xx+.5-CENTER, yy+.5-CENTER)

    # Three low steps, a stone promenade and the five grass pockets.
    for z, r in ((1, OUTER_RADIUS), (2, OUTER_RADIUS-2), (3, OUTER_RADIUS-4)):
        model[:, :, z][radius < r] = STONE
    model[:, :, 3][(radius > 78) & (radius < 84)] = DARK_STONE
    star_image = Image.new('1', (SIZE, SIZE))
    ImageDraw.Draw(star_image).polygon(star_outline(), fill=1)
    star = np.array(star_image).T
    pockets = (radius < INNER_RADIUS) & ~star
    model[:, :, 3][pockets] = GRASS
    model[:, :, 3][star] = GOLD_PAVING
    model[:, :, 3][radius < RED_RADIUS+1.4] = STONE
    model[:, :, 3][radius < RED_RADIUS] = RED_PAVING

    def box(x0, y0, z0, x1, y1, z1, material):
        low = np.floor([x0, y0, z0]).astype(int)
        high = np.ceil([x1, y1, z1]).astype(int)
        model[low[0]:high[0], low[1]:high[1], low[2]:high[2]] = material

    def cylinder(cx, cy, z0, z1, r0, material, r1=None, sides=0):
        r1 = r0 if r1 is None else r1
        bound = max(r0, r1)
        lo = np.floor([cx-bound, cy-bound, z0]).astype(int)
        hi = np.ceil([cx+bound, cy+bound, z1]).astype(int)
        x, y, z = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dx, dy = x+.5-cx, y+.5-cy
        rr = np.sqrt(dx*dx+dy*dy)
        if sides:
            # Circumradius polygon, vertices at the diagonal half-angles.
            theta = np.arctan2(dy, dx)
            rr = rr*np.cos((theta+np.pi/sides)%(2*np.pi/sides)-np.pi/sides)/np.cos(np.pi/sides)
        r = r0+(r1-r0)*(z+.5-z0)/(z1-z0)
        mask = (rr <= r) & (z+.5 >= z0) & (z+.5 < z1)
        target = model[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        target[mask] = material

    def ellipsoid(cx, cy, cz, rx, ry, rz, material):
        lo = np.floor([cx-rx, cy-ry, cz-rz]).astype(int)
        hi = np.ceil([cx+rx, cy+ry, cz+rz]).astype(int)
        x, y, z = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        mask = ((x+.5-cx)/rx)**2+((y+.5-cy)/ry)**2+((z+.5-cz)/rz)**2 <= 1
        target = model[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        target[mask] = material

    def tube(points, radius, material):
        for start, end in zip(points[:-1], points[1:]):
            start, end = np.array(start), np.array(end)
            for t in np.linspace(0, 1, int(np.linalg.norm(end-start)*3)+2):
                p = start*(1-t)+end*t
                ellipsoid(*p, radius, radius, radius, material)

    def huabiao_ellipsoid(cx, cy, cz, rx, ry, rz, material):
        ellipsoid(*huabiao_point(cx,cy,cz),rx*HUABIAO_ENLARGEMENT,
                  ry*HUABIAO_ENLARGEMENT,rz*HUABIAO_ENLARGEMENT,material)

    def huabiao_tube(points, radius, material):
        tube([huabiao_point(*point) for point in points],radius*HUABIAO_ENLARGEMENT,material)

    # Huabiao: photo-estimated stepped octagonal foot and circular carved drum.
    for z0, z1, r0, r1 in ((4, 5, 8.5, 8.5), (5, 6, 7.5, 7.5),
                           (6, 7, 6.4, 6.4), (7, 8, 5.0, 4.5),
                           (8, 11, 4.2, 4.2), (11, 12, 5.0, 5.0),
                           (12, 14, 2.6, 2.3)):
        cylinder(CENTER, CENTER, huabiao_z(z0), huabiao_z(z1),
                 r0*HUABIAO_ENLARGEMENT, STONE, r1*HUABIAO_ENLARGEMENT, sides=8 if z0<8 else 0)
    cylinder(CENTER, CENTER, huabiao_z(14), huabiao_z(38), 2*HUABIAO_ENLARGEMENT, STONE)

    # Eight simplified relief scrolls around the pedestal; one spiral dragon.
    for angle in np.arange(8)*np.pi/4:
        path = []
        for t in np.linspace(0, 2*np.pi, 20):
            theta = angle + .16*np.sin(t)
            path.append((100+4.25*np.cos(theta), 100+4.25*np.sin(theta), 9.4+.9*np.cos(t)))
        huabiao_tube(path, .65, RELIEF)
    path = []
    for z in np.linspace(14.2, 35, 140):
        angle = (z-14.2)*2*np.pi/8.5
        path.append((100+2*np.cos(angle), 100+2*np.sin(angle), z))
    huabiao_tube(path, .70, RELIEF)
    huabiao_ellipsoid(*path[-1], 1.15, .9, 1.0, RELIEF)

    # Scrolled cloud board, with distinct upturned ends and a narrow waist.
    for side in (-1, 1):
        path = [(100+side*d, 100, z) for d, z in ((1,34), (3.5,33.7), (6,34.0), (8,35.2), (8.5,36.7))]
        huabiao_tube(path, .9, STONE)
        for d, z, rx in ((3.1,35.1,1.5), (5.2,35.2,1.4), (7.3,36.4,1.3)):
            huabiao_ellipsoid(100+side*d, 100, z, rx, .9, 1.3, STONE)
    cylinder(100, 100, huabiao_z(37), huabiao_z(38), 3.2*HUABIAO_ENLARGEMENT, STONE)
    cylinder(100, 100, huabiao_z(38), huabiao_z(40), 2.6*HUABIAO_ENLARGEMENT, STONE, 3.0*HUABIAO_ENLARGEMENT)

    # Small crouching beast: haunches, lifted head, forelegs, muzzle and tail.
    huabiao_ellipsoid(100, 99.6, 40.9, 1.5, 1.8, 1.2, STONE)
    box(*huabiao_point(99,100.3,40),*huabiao_point(100,101.3,42),STONE)
    box(*huabiao_point(101,100.3,40),*huabiao_point(102,101.3,42),STONE)
    huabiao_ellipsoid(100, 101, 42.3, 1.15, 1.15, 1.6, STONE)
    huabiao_ellipsoid(100, 102, 42.4, .85, 1.1, .6, STONE)
    huabiao_tube([(100,98,40.8), (101.5,97.7,41.5), (101.5,98,42.0)], .6, STONE)

    # Five monumental palace lanterns. These are historical architectural props,
    # distinct from the ordinary 16-unit public lamp reference. Integer centres
    # give all five the same voxel geometry; pale frames surround tall dark panes.
    for cx, cy in LAMP_CENTERS:
        cylinder(cx, cy, 4, 5, 4.3, LAMP_STONE, sides=8)
        cylinder(cx, cy, 5, 6, 3.0, LAMP_STONE, sides=8)
        cylinder(cx, cy, 6, 8, 2.5, LAMP_STONE)
        cylinder(cx, cy, 8, 26, 1.8, LAMP_STONE)
        cylinder(cx, cy, 25, 26, 2.4, LAMP_STONE)
        cylinder(cx, cy, 26, 27, 4.2, LAMP_STONE, sides=8)
        box(cx-3,cy-3,27,cx+3,cy+3,35,LAMP_STONE)
        box(cx-1,cy-3,28,cx+1,cy-2,34,LAMP_FRAME)
        box(cx-1,cy+2,28,cx+1,cy+3,34,LAMP_FRAME)
        box(cx-3,cy-1,28,cx-2,cy+1,34,LAMP_FRAME)
        box(cx+2,cy-1,28,cx+3,cy+1,34,LAMP_FRAME)
        cylinder(cx,cy,35,36,4.2,LAMP_STONE,sides=8)
        for angle in np.arange(4)*np.pi/2:
            tube([(cx+3.2*np.cos(angle), cy+3.2*np.sin(angle), 35.5),
                  (cx+4.1*np.cos(angle), cy+4.1*np.sin(angle), 37.4)], .65, LAMP_STONE)
        cylinder(cx,cy,36,38,.9,LAMP_STONE)

    # Nine small bronze ding on the outer stone promenade, with open handles.
    for k in range(9):
        angle = STAR_ANGLE+(k+.5)*2*np.pi/9
        cx, cy = 100+81*np.cos(angle), 100+81*np.sin(angle)
        cylinder(cx, cy, 4, 5, 3.2, STONE, sides=4)
        for theta in np.arange(3)*2*np.pi/3:
            box(cx+1.6*np.cos(theta)-.5, cy+1.6*np.sin(theta)-.5, 5,
                cx+1.6*np.cos(theta)+.5, cy+1.6*np.sin(theta)+.5, 7, BRONZE)
        cylinder(cx, cy, 6, 9, 1.5, BRONZE, 2.4, sides=4 if k%3==0 else 0)
        cylinder(cx, cy, 8, 9, 1.15, 0)
        for sign in (-1,1):
            tube([(cx+sign*2,cy,8), (cx+sign*2.5,cy,10), (cx+sign*1,cy,10), (cx+sign*1,cy,9)], .6, BRONZE)

    # Reuse the public 5-unit adult geometry with different poses/orientations.
    walkers = make_model('walkers')
    people = (walkers[10:13,12:14,1:6], walkers[10:13,19:21,1:6], walkers[26:29,26:29,1:6])
    positions = ((94,113,0), (101,115,1), (45,94,2), (49,96,0), (65,119,1),
                 (121,72,2), (146,120,0), (72,150,1), (129,153,2), (34,101,0), (161,87,1))
    for k, (cx,cy,variant) in enumerate(positions):
        prop = np.rot90(people[variant], k%4, axes=(0,1))
        nx, ny, nz = prop.shape
        base = int(np.nonzero(model[cx,cy])[0].max())+1
        target = model[cx:cx+nx,cy:cy+ny,base:base+nz]
        occupied = prop != 0
        target[occupied] = prop[occupied]+PROP_OFFSET
    return model
