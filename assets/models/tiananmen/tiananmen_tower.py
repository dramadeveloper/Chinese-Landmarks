"""Rebuild the hall above the accepted Tiananmen platform, in model units."""
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parent
CENTRE = (160.5, 50.5)
DECK = 32
LOWER_HALF = (64, 28)
UPPER_HALF = (58, 23)
UPPER_WALL = (54, 20)
GABLE_X = 49
LOWER_EAVE, LOWER_HEAD = 45, 48
UPPER_EAVE, RIDGE = 58, 71
TILE, CAP, FASCIA, POST, PANEL, WINDOW, BEAM, GILT, GABLE = range(14, 23)


def corner_lift(x, y, half_width, half_depth, lift):
    wing = np.clip((abs(x) - half_width + 15) / 15, 0, 1)
    tip = np.clip((abs(y) - half_depth + 9) / 9, 0, 1)
    return lift * wing**2 * tip**1.5


def lower_roof(x, y):
    hx, hy = LOWER_HALF
    wx, wy = UPPER_WALL
    across = (hx - abs(x)) / (hx - wx)
    along = (hy - abs(y)) / (hy - wy)
    run = np.clip(np.minimum(across, along), 0, 1)
    return LOWER_EAVE + (LOWER_HEAD - LOWER_EAVE) * run**1.45 + corner_lift(x, y, hx, hy, 2)


def upper_roof(x, y):
    hx, hy = UPPER_HALF
    main = UPPER_EAVE + (RIDGE - UPPER_EAVE) * np.clip(1 - abs(y) / hy, 0, 1)**1.55
    # The end skirt terminates beneath the vertical gable and its projecting bargeboard.
    hip = UPPER_EAVE + 2.8 * np.clip((hx - abs(x)) / (hx - GABLE_X), 0, 1)**1.4
    surface = np.where(abs(x) <= GABLE_X + 1, main, np.minimum(main, hip))
    return surface + corner_lift(x, y, hx, hy, 2.7)


def generate_model():
    with np.load(SOURCE / 'site.npz') as saved:
        model = saved['model']
    hx, hy = LOWER_HALF
    cx, cy = (int(v) for v in CENTRE)
    hall = model[cx-hx:cx+hx+1, cy-hy:cy+hy+1, DECK:]
    x, y, z = np.ogrid[-hx:hx+1, -hy:hy+1, DECK:model.shape[2]]
    ax, ay = abs(x), abs(y)

    def fill(plan, bottom, top, material):
        hall[plan & (z >= bottom) & (z < top)] = material

    # Nine bays with a wider middle door. The doors stand behind the free-standing posts.
    post_x = (-57, -45, -32, -20, -8, 8, 20, 32, 45, 57)
    post_y = (-23, -14, -5, 5, 14, 23)
    enclosure = (ax <= 55) & (ay <= 21)
    fill(enclosure, DECK, LOWER_EAVE, PANEL)
    for left, right in zip(post_x, post_x[1:]):
        mid = (left + right) / 2
        door = (abs(x-mid) <= (right-left)/2-2) & (ax <= 54) & (ay == 21)
        fill(door, DECK+1, LOWER_EAVE-3, WINDOW)
        fill(door & (abs(x-mid) < 1.1), DECK+1, LOWER_EAVE-3, POST)
    for px in post_x:
        fill((abs(x-px) <= 1) & (abs(ay-23) <= 1), DECK, LOWER_EAVE, POST)
    for py in post_y[1:-1]:
        fill((abs(ax-57) <= 1) & (abs(y-py) <= 1), DECK, LOWER_EAVE, POST)
    colonnade = (ax <= 60) & (ay <= 26)
    fill(colonnade, LOWER_EAVE-5, LOWER_EAVE, BEAM)
    for px in post_x:
        fill((abs(x-px) <= 2) & (abs(ay-26) <= 1), LOWER_EAVE-6, LOWER_EAVE-3, BEAM)

    # Shallow lower apron around the clerestory, with raised diagonal hip seams.
    shell = (ax <= hx) & (ay <= hy)
    lower = np.rint(lower_roof(x, y)).astype(int)
    fill(shell, lower-2, lower+1, TILE)
    fill(shell & ((ax == hx) | (ay == hy)), lower-1, lower, FASCIA)
    wx, wy = UPPER_WALL
    hip = abs((hx-ax) / (hx-wx) - (hy-ay) / (hy-wy)) < .075
    fill(shell & hip & (ax >= wx) & (ay >= wy), lower+1, lower+2, CAP)

    # Red infill, posts, and a painted beam between the two eaves.
    upper_body = (ax <= wx) & (ay <= wy)
    fill(upper_body, LOWER_HEAD, UPPER_EAVE, PANEL)
    for px in np.linspace(-50, 50, 10):
        fill((abs(x-px) <= 1.1) & (abs(ay-wy) <= 1), LOWER_HEAD, UPPER_EAVE-2, POST)
    for py in (-15, -5, 5, 15):
        fill((abs(ax-wx) <= 1) & (abs(y-py) <= 1), LOWER_HEAD, UPPER_EAVE-2, POST)
    fill((ax <= wx+1) & (ay <= wy+1), UPPER_EAVE-5, UPPER_EAVE, BEAM)
    # South-facing national emblem, simplified to a gold rim and a red centre.
    badge = (x/3.2)**2 + ((z-54)/3.0)**2 <= 1
    hall[badge & (y >= wy+1) & (y <= wy+3)] = GILT
    hall[(abs(x) <= 1) & (y == wy+3) & (abs(z-54) <= 1)] = POST

    # Xieshan roof: end skirts and a recessed triangular gable under each verge.
    ux, uy = UPPER_HALF
    top = np.rint(upper_roof(x, y)).astype(int)
    upper = (ax <= ux) & (ay <= uy)
    fill((ax <= GABLE_X+1) & (ay <= uy), UPPER_EAVE, top, GABLE)
    fill(upper, top-1, top+1, TILE)
    fill(upper & ((ax == ux) | (ay == uy)), top-1, top, FASCIA)
    fill(upper & (ax == GABLE_X+1) & (ay < 13), top-2, top, PANEL)
    verge = (ax == GABLE_X+1) & (ay <= 13)
    fill(verge, top, top+2, CAP)
    hip_line = abs(ay - (13 + (ax-GABLE_X-1) * (uy-13)/(ux-GABLE_X-1))) < .8
    fill(upper & (ax > GABLE_X+1) & hip_line, top+1, top+2, CAP)
    fill((ax <= GABLE_X+1) & (ay <= 1), RIDGE, RIDGE+1, CAP)
    # Paired chiwen sit on broad feet and curl towards the ridge.
    fill((ax >= GABLE_X-2) & (ax <= GABLE_X+1) & (ay <= 1), RIDGE+1, RIDGE+2, CAP)
    fill((ax >= GABLE_X-1) & (ax <= GABLE_X) & (ay <= 1), RIDGE+2, RIDGE+3, CAP)
    return model


if __name__ == '__main__':
    np.savez_compressed(SOURCE / 'model.npz', model=generate_model())
