"""4x4 Water Cube model: a low ETFE envelope with shallow polygonal cushions."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'assets/chinese-landmarks'))
from export_plaza import make_model

SIZE = (160, 160, 40)
LOW, HIGH = 14, 146
BASE, TOP = 2, 34
MEMBRANE, ROOF, JOINT, STONE, GLASS, METAL, VENT, WATER, COPING = range(1, 10)
PROP_OFFSET = 32


def sites(width, height, spacing, seed):
    """A staggered, varied foam drawing, simplified for the native game view."""
    rng = np.random.default_rng(seed)
    points = []
    for row, v in enumerate(np.arange(-spacing, height + spacing * 2, spacing * .86)):
        for u in np.arange(-spacing, width + spacing * 2, spacing):
            points.append((u + (row % 2) * spacing / 2 + rng.uniform(-.30, .30) * spacing,
                           v + rng.uniform(-.28, .28) * spacing))
    return np.array(points)


PANELS = {
    'roof': sites(132, 132, 8.4, 482),
    'west': sites(132, 23, 6.8, 144) * [1, (TOP-BASE)/23],
    'east': sites(132, 23, 6.8, 216) * [1, (TOP-BASE)/23],
    'north': sites(132, 23, 6.8, 355) * [1, (TOP-BASE)/23],
    'south': sites(132, 23, 6.8, 679) * [1, (TOP-BASE)/23],
}


def cushions(u, v, points):
    """Closest cells and their separating line; all coordinates follow the model."""
    first = np.full(u.shape, np.inf)
    second = np.full(u.shape, np.inf)
    owner = np.zeros(u.shape, int)
    runner = np.zeros(u.shape, int)
    for index, (sx, sy) in enumerate(points):
        distance = (u - sx) ** 2 + (v - sy) ** 2
        nearer = distance < first
        next_nearer = ~nearer & (distance < second)
        second = np.where(nearer, first, np.where(next_nearer, distance, second))
        runner = np.where(nearer, owner, np.where(next_nearer, index, runner))
        first = np.where(nearer, distance, first)
        owner = np.where(nearer, index, owner)
    separation = np.linalg.norm(points[owner] - points[runner], axis=-1)
    edge = (second - first) / (2 * separation)
    dx, dy = u - points[owner, 0], v - points[owner, 1]
    dome = .85 * (1 - np.exp(-edge / 1.25))
    return edge, dx, dy, dome, owner


def make_building():
    model = np.zeros(SIZE, np.uint8)
    model[:, :, 0] = STONE

    # Open outer paving surrounds a narrow, attached water belt. Bridges connect
    # it to the raised foundation; no border is applied to the map-tile edges.
    model[9:151, 9:151, 0] = WATER
    model[12:148, 12:148, :2] = STONE
    model[13:147, 13:147, 1:2] = COPING
    model[LOW:HIGH, LOW:HIGH, BASE:TOP] = MEMBRANE

    uv = np.arange(LOW, HIGH) + .5 - LOW
    u, v = np.meshgrid(uv, uv, indexing='ij')
    edge, _, _, dome, _ = cushions(u, v, PANELS['roof'])
    model[LOW:HIGH, LOW:HIGH, TOP - 1] = np.where(edge < .27, JOINT, ROOF)
    top_layer = model[LOW:HIGH, LOW:HIGH, TOP]
    top_layer[dome > .59] = ROOF

    u, v = np.meshgrid(uv, np.arange(BASE, TOP) + .5 - BASE, indexing='ij')
    for side, axis, plane, bump in [('west', 0, LOW, LOW - 1),
                                    ('east', 0, HIGH - 1, HIGH),
                                    ('north', 1, LOW, LOW - 1),
                                    ('south', 1, HIGH - 1, HIGH)]:
        edge, _, _, dome, _ = cushions(u, v, PANELS[side])
        color = np.where(edge < .27, JOINT, MEMBRANE).astype(np.uint8)
        if axis == 0:
            model[plane, LOW:HIGH, BASE:TOP] = color
            model[bump, LOW:HIGH, BASE:TOP][dome > .63] = MEMBRANE
        else:
            model[LOW:HIGH, plane, BASE:TOP] = color
            model[LOW:HIGH, bump, BASE:TOP][dome > .63] = MEMBRANE

    # Entrance placement is a photo-based composition, not a surveyed door plan.
    # South: a long recessed glass lobby; east: a shorter offset portal.
    model[42:114, HIGH-3:HIGH+1, BASE:7] = 0
    model[42:114, HIGH-4:HIGH-3, BASE:7] = GLASS
    model[42:114, HIGH-4:HIGH, 7:8] = METAL
    for x in range(43, 114, 7):
        model[x:x+1, HIGH-4:HIGH-3, BASE:7] = METAL
    model[55:76, HIGH-4:HIGH-3, BASE:6] = GLASS
    model[54:77, HIGH-3:153, 7:8] = STONE
    model[54:56, 151:153, BASE:7] = METAL
    model[75:77, 151:153, BASE:7] = METAL
    model[52:79, 146:154, :2] = STONE
    model[50:81, 154:156, 0] = STONE

    model[HIGH-3:HIGH+1, 85:116, BASE:8] = 0
    model[HIGH-4:HIGH-3, 85:116, BASE:8] = GLASS
    for y in range(86, 116, 6):
        model[HIGH-4:HIGH-3, y:y+1, BASE:8] = METAL
    model[HIGH-3:152, 84:117, 8:9] = STONE
    model[150:152, 84:86, BASE:8] = METAL
    model[150:152, 115:117, BASE:8] = METAL
    model[146:154, 83:118, :2] = STONE
    model[154:156, 81:120, 0] = STONE

    # Two modest secondary door groups keep the four orientations distinct.
    model[LOW-1:LOW+3, 42:63, BASE:7] = 0
    model[LOW+3:LOW+4, 42:63, BASE:7] = GLASS
    for y in range(43, 63, 6):
        model[LOW+3:LOW+4, y:y+1, BASE:7] = METAL
    model[8:LOW+3, 41:64, 7:8] = STONE
    model[6:14, 40:65, :2] = STONE
    model[92:112, LOW-1:LOW+3, BASE:6] = 0
    model[92:112, LOW+3:LOW+4, BASE:6] = GLASS
    for x in (94, 100, 106):
        model[x:x+1, LOW+3:LOW+4, BASE:6] = METAL
    model[91:113, 8:LOW+3, 6:7] = STONE
    model[89:115, 6:14, :2] = STONE

    # Low roof fittings are kept sparse. Their exact type/count is unverified.
    for x, y, length in [(33, 34, 8), (69, 31, 6), (116, 35, 8),
                         (37, 78, 7), (119, 84, 7), (53, 119, 8), (98, 120, 6)]:
        model[x:x+length, y:y+2, TOP:TOP+2] = VENT
        model[x:x+length, y:y+2, TOP+1] = METAL

    # Exact copies of current public adults: five model units tall.
    public = make_model('walkers')
    people = [public[10:13, 12:14, 1:6], public[10:13, 19:21, 1:6],
              public[26:29, 26:29, 1:6]]
    for i, (x, y, turn) in enumerate([(42, 154, 0), (47, 155, 0),
                                     (86, 153, 1), (155, 100, 1), (155, 108, 2)]):
        person = np.rot90(people[i % 3], turn, axes=(0, 1))
        target = model[x:x+person.shape[0], y:y+person.shape[1], 1:6]
        mask = person != 0
        target[mask] = person[mask] + PROP_OFFSET
    return model
