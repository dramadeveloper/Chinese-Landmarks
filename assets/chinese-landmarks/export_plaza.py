"""Generate modular paving and sparse static life as genuine 1x indexed sprites.

Furniture and people use deterministic small voxel models. Light stays fixed
in screen space in all four views. Ground pattern does not rotate with props.
"""
import numpy as np
from render_core import trace_model, shade, indexed, stone_ground, PROP_SAMPLING, DEFAULT_SCENE

KINDS = ['plain', 'walkers', 'meeting', 'bench', 'tree']
OBJECT_IDS = {'plain': 2, 'walkers': 6, 'meeting': 7, 'bench': 8, 'tree': 9}
VIEWS = {kind: 1 if kind == 'plain' else 4 for kind in KINDS}
ORIGIN = (48, 24)
# Model IDs and deliberate DOS ramps. Clothing is subdued and non-company-colour.
RAMPS = {1: [8, 10, 11, 12], 2: [32, 33, 34, 35], 3: [1, 2, 3, 4],
         4: [16, 17, 18, 19], 5: [178, 179, 180, 181], 6: [116, 118, 120, 121],
         7: [89, 90, 91, 92], 8: [7, 8, 9, 10], 9: [12, 13, 14, 14]}


def make_model(kind):
    a = np.zeros((40, 40, 24), np.uint8)
    def box(x1, y1, z1, x2, y2, z2, material):
        a[x1:x2+1, y1:y2+1, z1:z2+1] = material
    def person(x, y, clothing=4, facing=1, seated=False, child=False):
        # Adults occupy 5 model units; children occupy 4.
        if seated:
            box(x, y+1, 2, x+1, y+3, 2, 3)
            box(x, y+3, 1, x+1, y+3, 2, 3)
            box(x, y, 3, x+1, y+1, 4, clothing)
            box(x, y, 5, x+1, y+1, 5, 6)
            box(x, y, 5, x+1, y, 5, 3)
        else:
            w = 0 if child else 1
            top = 4 if child else 5
            box(x, y, 1, x, y, 1, 3)
            box(x+1, y+facing, 1, x+1, y+facing, 1, 3)
            box(x, y, 2, x+w, y+1, top-2, clothing)
            box(x-1, y+1, 2, x-1, y+1, 2, 6)
            box(x, y, top-1, x+w, y+1, top-1, 6)
            box(x, y, top, x+w, y, top, 3)
    if kind == 'walkers':
        person(11, 12, 4)
        person(11, 19, 9)
        person(27, 27, 5, facing=-1)
    elif kind == 'meeting':
        person(12, 12, 5)
        person(12, 20, 4, facing=-1)
        person(22, 12, 9)
        person(22, 20, 9, child=True)
        person(18, 28, 5, child=True)
    elif kind == 'bench':
        for x in (12, 23):
            box(x, 18, 1, x, 20, 2, 3)
        box(11, 18, 3, 25, 21, 3, 2)
        box(11, 18, 4, 25, 18, 5, 2)
        person(14, 20, 5, seated=True)
        person(21, 20, 9, seated=True)
        person(28, 27, 4)
    elif kind == 'tree':
        box(14, 14, 1, 25, 25, 1, 1)
        box(16, 16, 1, 23, 23, 1, 2)
        box(19, 19, 2, 20, 20, 9, 2)
        x, y, z = np.indices(a.shape)
        leaves = ((x-19.5)/5)**2 + ((y-19.5)/5)**2 + ((z-11)/5)**2 <= 1
        a[leaves] = 7
        person(29, 29, 4)
    return a


def render(a, kind, view, *, lighting=DEFAULT_SCENE):
    frame = trace_model(a, view, (96, 80), ORIGIN, PROP_SAMPLING)
    mat, point = frame.mat, frame.point
    light = shade(frame.normal, frame.shadow, lighting.standard)
    foliage = mat == 7
    grain = ((np.floor(point[:,:,0])*17+np.floor(point[:,:,1])*29+np.floor(point[:,:,2])*11)%13-6)/75
    light[foliage] += grain[foliage]
    building = indexed(mat, light, frame.hit, RAMPS)
    return stone_ground(frame, lighting), building
