"""Parametric China Art Museum model. Run this file to save model.npz."""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parents[2] / 'assets/chinese-landmarks'))
from export_plaza import make_model

STONE, RED, RED_DARK, GLASS, ROOF, CONCRETE, METAL, FOLIAGE = range(1, 9)
CENTER_ROOF, PODIUM_JOINT = 9, 10
PROP_OFFSET = 20


@dataclass(frozen=True)
class Dimensions:
    art_scale: float = 2.0
    roof_span_m: float = 138.0
    support_span_m: float = 70.2
    core_width_m: float = 18.6
    crown_bottom_m: float = 33.3
    terrace_m: float = 60.3
    total_height_m: float = 69.9
    crown_extra_height_units: float = 26.0
    entrance_platform_m: float = 9.0
    # The following dimensions are photo-based modelling choices, not surveys.
    upper_roof_m: float = 98.0
    podium_width_m: float = 164.0
    podium_depth_m: float = 148.0
    podium_side_height_m: float = 13.0
    podium_recess_depth_m: float = 4.0
    stair_steps: int = 3
    stair_run_m: float = 20.7


DIMENSIONS = Dimensions()


def generate_model(d=DIMENSIONS):
    """Return material IDs in a fixed 200 x 200 x 104 map container.

    Building coordinates use reference metres times art_scale/2; ground is z=0,
    and construction begins at z=1. Positive y is the main stair approach.
    """
    a = np.zeros((200, 200, 104), np.uint8)
    a[:, :, 0] = STONE
    k = d.art_scale / 2

    def box(x1, y1, z1, x2, y2, z2, material):
        x1, x2 = round(100+x1*k), round(100+x2*k)
        y1, y2 = round(92+y1*k), round(92+y2*k)
        z1, z2 = round(1+z1*k), round(1+z2*k)
        a[x1:x2, y1:y2, z1:z2] = material

    w, depth = d.podium_width_m/2, d.podium_depth_m/2
    platform = d.entrance_platform_m
    # A cropped, immediate podium: two low side wings, rear connector, central
    # nine-metre landing. The much larger provincial pavilion is not reproduced.
    box(-w, -depth, 0, w, depth, platform, CONCRETE)
    box(-w, -depth, platform, -41, depth, d.podium_side_height_m, CONCRETE)
    box(41, -depth, platform, w, depth, d.podium_side_height_m, CONCRETE)
    box(-41, -depth, platform, 41, -55, d.podium_side_height_m, CONCRETE)
    box(-w+2, -depth+2, 12, -43, depth-2, 13, ROOF)
    box(43, -depth+2, 12, w-2, depth-2, 13, ROOF)
    box(-41, -depth+2, 12, 41, -57, 13, ROOF)
    # The aerial shows rectangular recessed bays on the wing to the left of
    # the main approach. Keep two broad bays with one low divider each; their
    # layout is a photo-based abbreviation of the much larger regional hall.
    podium_top = d.podium_side_height_m
    recess_floor = podium_top-d.podium_recess_depth_m
    for y in (-56, -10):
        box(48, y, recess_floor, w-5, y+34, podium_top, 0)
        box(48, y, recess_floor-1, w-5, y+34, recess_floor, PODIUM_JOINT)
        box(48, y+16, recess_floor, w-5, y+18, recess_floor+1, CONCRETE)
    # The opposite wing has roof planting in the photo. Use two low planted
    # strips on the podium itself, keeping the public paving open.
    for y in (-48, 6):
        box(-w+4, y, podium_top, -w+12, y+32, podium_top+1, CONCRETE)
        box(-w+5, y+1, podium_top+1, -w+11, y+31, podium_top+2, FOLIAGE)
    box(-40, -54, 8, 40, depth, platform, STONE)
    # Three broad symbolic treads retain the entrance platform and total run.
    tread = d.stair_run_m / d.stair_steps
    rise = platform / d.stair_steps
    for step in range(d.stair_steps):
        box(-40, depth+step*tread, 0, 40, depth+(step+1)*tread,
            platform-step*rise, STONE)
    for x in (-42, 40):
        for step in range(d.stair_steps):
            box(x, depth+step*tread, 0, x+2, depth+(step+1)*tread,
                platform-step*rise+1, CONCRETE)
    # Broad quiet podium faces: three subtle joints per long side, one per wing.
    for y in (-36, 12, 60):
        box(-w, y, 1, -w+1, y+1, 12, PODIUM_JOINT)
        box(w-1, y, 1, w, y+1, 12, PODIUM_JOINT)
    for x in (-62, 62):
        box(x, depth-1, 1, x+1, depth, 12, PODIUM_JOINT)

    half, core = d.support_span_m/2, d.core_width_m
    for x in (-half, half-core):
        for y in (-half, half-core):
            box(x, y, platform, x+core, y+core, d.crown_bottom_m, RED)
            # Grey glazed strips break up each of the four giant red cores.
            box(x+core*.40, y-.1, platform+1, x+core*.60, y+1,
                d.crown_bottom_m-1, GLASS)
            box(x+core*.40, y+core-1, platform+1, x+core*.60, y+core+.1,
                d.crown_bottom_m-1, GLASS)
            box(x-.1, y+core*.40, platform+1, x+1, y+core*.60,
                d.crown_bottom_m-1, GLASS)
            box(x+core-1, y+core*.40, platform+1, x+core+.1, y+core*.60,
                d.crown_bottom_m-1, GLASS)
    # A low glass entrance; diagonals sit behind the open front portal.
    box(-15, -12, platform, 15, 10, platform+5, GLASS)
    box(-16, -13, platform+5, 16, 11, platform+6, ROOF)
    for t in range(25):
        z = platform+5+t*.72
        x = -22+t*1.6
        box(x, -5, z, x+4, 1, z+2, METAL)
        box(-x-4, 5, z, -x, 11, z+2, METAL)
    box(-half, -half, 29, half, -half+3, 33.3, RED_DARK)
    box(-half, half-3, 29, half, half, 33.3, RED_DARK)

    # Seven silhouette bands; each has recessed glazing and protruding beam
    # ends. Widths preserve the crown's 70-ish to 138-metre widening.
    terrace = d.terrace_m + d.crown_extra_height_units/k
    roof_top = d.total_height_m + d.crown_extra_height_units/k
    widths = np.linspace(76, d.roof_span_m, 7)
    levels = np.linspace(d.crown_bottom_m, terrace, 8)
    for index, (span, bottom, top) in enumerate(zip(widths, levels[:-1], levels[1:])):
        r = span/2
        box(-r+3, -r+3, bottom, r-3, r-3, top, GLASS)
        box(-r, -r, top-2.4, r, r, top, RED)
        # Spaced bracket ends carry the recognisable timber-like construction.
        for p in np.linspace(-r+6, r-6, 7):
            box(p-1.6, -r-1, top-3.8, p+1.6, r+1, top-.3, RED)
            box(-r-1, p-1.6, top-3.8, r+1, p+1.6, top-.3, RED)

    # Roof terrace, inner clerestory and nine-part roof grid visible from above.
    r, upper = d.roof_span_m/2, d.upper_roof_m/2
    box(-r, -r, terrace, r, r, terrace+1, RED)
    box(-r+3, -r+3, terrace, r-3, r-3, terrace+1, ROOF)
    box(-upper, -upper, terrace+1, upper, upper, roof_top-1, GLASS)
    for p in np.arange(-upper+3, upper, 7):
        box(p, -upper, terrace+1, p+1, -upper+1, roof_top-1, RED)
        box(p, upper-1, terrace+1, p+1, upper, roof_top-1, RED)
        box(-upper, p, terrace+1, -upper+1, p+1, roof_top-1, RED)
        box(upper-1, p, terrace+1, upper, p+1, roof_top-1, RED)
    box(-upper-1, -upper-1, roof_top-1, upper+1, upper+1, roof_top, RED)
    for ix in range(3):
        for iy in range(3):
            x, y = -upper+3+ix*(2*upper-4)/3, -upper+3+iy*(2*upper-4)/3
            material = CENTER_ROOF if ix == 1 and iy == 1 else ROOF
            box(x, y, roof_top-1, x+27, y+27, roof_top, material)
    # Sparse, low roof planting visible in the aerial photo, no peripheral park.
    for p in range(-56, 57, 16):
        for side in (-1, 1):
            box(p, side*59-1, terrace+1, p+9, side*59+1,
                terrace+2, FOLIAGE)
            box(side*59-1, p, terrace+1, side*59+1, p+9,
                terrace+2, FOLIAGE)

    # Three public adults, copied at the existing 5-unit height.
    person = make_model('walkers')[10:13, 12:14, 1:6]
    for x, y, z in [(93, 157, 10), (99, 162, 10), (122, 191, 1)]:
        target = a[x:x+3, y:y+2, z:z+5]
        target[person != 0] = person[person != 0] + PROP_OFFSET
    return a


def save_model():
    model = generate_model()
    np.savez_compressed(SOURCE / 'model.npz', model=model)
    with np.load(SOURCE / 'model.npz') as saved:
        assert saved.files == ['model']
        assert np.array_equal(model, saved['model'])
    print(f'Saved {SOURCE / "model.npz"}: {model.shape}', flush=True)


if __name__ == '__main__':
    save_model()
