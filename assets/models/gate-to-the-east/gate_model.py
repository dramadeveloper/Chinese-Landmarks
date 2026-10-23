"""Built Gate to the East: photo-estimated envelope, documented metric anchors."""
import numpy as np
from export_plaza import make_model as make_prop

SIZE = (160, 120, 244)
BODY_ENLARGEMENT = 1.2
UNITS_PER_METRE = 307 / 468 * BODY_ENLARGEMENT
ART_SCALE = 2 * UNITS_PER_METRE
CENTER = (80., 60.)
YAW = np.deg2rad(-20)
HALF_WIDTH = 180 * UNITS_PER_METRE / 2
HALF_DEPTH = 45 * UNITS_PER_METRE / 2
ARCH_HEIGHT = 230 * UNITS_PER_METRE
ARCH_HALF_SPAN = 60 * UNITS_PER_METRE / 2
ROOF = 281 * UNITS_PER_METRE
TIP = 301.8 * UNITS_PER_METRE
PAVING, GLASS, SPANDREL, METAL, PODIUM, ENTRY = 1, 2, 3, 4, 5, 6
PROP_OFFSET = 20


def envelope(z):
    """Front/back glazing rolls into a narrow roof; not a constant-depth arch."""
    roll = 18 * BODY_ENLARGEMENT
    t = np.clip((z - (ROOF - roll)) / roll, 0, 1)
    depth = HALF_DEPTH * np.sqrt(1 - .86 * t*t)
    width = HALF_WIDTH - .9 * BODY_ENLARGEMENT * (z / ROOF)**3
    return width, depth


def opening(z):
    t = np.clip(z / ARCH_HEIGHT, 0, 1)
    return ARCH_HALF_SPAN * np.sqrt(1 - t*t)


def make_model():
    a = np.zeros(SIZE, np.uint8)
    a[:, :, 0] = PAVING
    x, y = np.indices(SIZE[:2], dtype=float) + .5
    dx, dy = abs(x-CENTER[0]), abs(y-CENTER[1])
    for z in range(1, int(np.ceil(ROOF))):
        width, depth = envelope(z+.5)
        # Rounded corners are small; long elevations retain their broad faces.
        radius = 3 * BODY_ENLARGEMENT
        corner = np.maximum(dx-(width-radius), 0)**2 + np.maximum(dy-(depth-radius), 0)**2
        body = (dx < width) & (dy < depth) & (corner <= radius**2)
        body &= (z+.5 >= ARCH_HEIGHT) | (dx >= opening(z+.5))
        a[:, :, z][body] = GLASS
        band = dx < BODY_ENLARGEMENT * (27 - 19 * (z+.5)/ROOF)
        a[:, :, z][body & band] = SPANDREL
    # Roof coping, two rooftop plant strips and the two actual needle accents.
    roof = int(round(ROOF))
    for lo,hi in ((23,70),(90,137)):
        lo,hi = [round(CENTER[0]+(p-CENTER[0])*BODY_ENLARGEMENT) for p in (lo,hi)]
        a[lo:hi,58:62,roof:roof+2] = METAL
    for position in (23,136):
        px = round(CENTER[0]+(position-CENTER[0])*BODY_ENLARGEMENT)
        a[px:px+1,59:60,roof+1:int(round(TIP))] = METAL
    # Close-fitting U podium, open on the lake side. No surrounding mall roof.
    podium = ((x>=8)&(x<152)&(y>=31)&(y<91) & ((dx>=26)|(y>=80)))
    for z in range(1, 20):
        a[:, :, z][podium & (a[:, :, z]==0)] = PODIUM
    a[:, :, 19][podium] = METAL
    # Dark glazed entry bays and low canopies at each side of the east opening.
    for lo, hi in ((27,43),(117,133)):
        a[lo:hi,31:33,2:8] = ENTRY
        a[lo-2:hi+2,26:34,8:9] = METAL
    # Public-size adults: unchanged five-unit geometry, sparse entrance activity.
    source = make_prop('walkers')
    people = (source[10:13,12:14,1:6],source[10:13,19:21,1:6],source[26:29,26:29,1:6])
    for i,(px,py) in enumerate(((41,14),(46,16),(115,15),(80,34),(83,37),(25,104))):
        p = people[i%3]
        target = a[px:px+p.shape[0],py:py+p.shape[1],1:6]
        target[p!=0] = p[p!=0]+PROP_OFFSET
    # Turn the whole assembly, preserving proportions. The regular game camera
    # now shows a frontal opening in v0/v2 and a credible narrow side in v1/v3.
    lx = (x-CENTER[0])*np.cos(YAW)+(y-CENTER[1])*np.sin(YAW)+CENTER[0]
    ly = -(x-CENTER[0])*np.sin(YAW)+(y-CENTER[1])*np.cos(YAW)+CENTER[1]
    ix,iy = np.floor(lx).astype(int),np.floor(ly).astype(int)
    valid = (ix>=0)&(ix<SIZE[0])&(iy>=0)&(iy<SIZE[1])
    turned = np.zeros_like(a)
    turned[valid] = a[ix[valid],iy[valid]]
    turned[:,:,0] = PAVING
    return turned
