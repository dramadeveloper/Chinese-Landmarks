"""Parametric Zhencheng Lou; run this file to rebuild model.npz."""
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE.parents[2] / 'assets/chinese-landmarks'))
from export_plaza import make_model

STONE, EARTH, WOOD, DARK, OUTER_ROOF, INNER_ROOF = range(1, 7)
PLASTER, BRICK, COURT, HALL_ROOF, EAR_ROOF, RIDGE, RED = range(7, 14)
PROP_OFFSET = 30
CENTER = (80, 76)


@dataclass(frozen=True)
class Dimensions:
    art_scale: float = 3.6
    outer_diameter_m: float = 57.2
    outer_height_m: float = 16.0
    # All dimensions below are photo estimates, not published surveys.
    outer_room_inner_radius_m: float = 22.0
    outer_gallery_radius_m: float = 19.7
    outer_eave_inner_radius_m: float = 18.7
    outer_eave_outer_radius_m: float = 30.0
    outer_ridge_radius_m: float = 26.0
    outer_eave_height_m: float = 13.2
    inner_wall_diameter_m: float = 26.4
    inner_eave_outer_radius_m: float = 14.4
    inner_eave_inner_radius_m: float = 6.1
    inner_ridge_radius_m: float = 11.0
    inner_eave_height_m: float = 6.6
    inner_ridge_height_m: float = 8.0
    hall_width_m: float = 12.0
    hall_depth_m: float = 9.0
    hall_roof_top_m: float = 9.4
    ear_inner_radius_m: float = 33.0
    ear_outer_radius_m: float = 40.0
    ear_half_angle_degrees: float = 23.0
    ear_eave_height_m: float = 6.2
    ear_ridge_height_m: float = 8.0


DIMENSIONS = Dimensions()


def ring_roof_height(r, inner, ridge, outer, eave, top):
    """Two continuous conical pitches, meeting at a circular ridge."""
    return np.where(r < ridge, eave+(top-eave)*(r-inner)/(ridge-inner),
                    eave+(top-eave)*(outer-r)/(outer-ridge))


def generate_model(d=DIMENSIONS):
    """Return a 160×160×40 material array. +y is the south/main entrance.

    Architectural metres are multiplied by art_scale/2. Voxel centres are
    sampled; z=0 is public ground and construction begins at z=1.
    """
    a = np.zeros((160, 160, 40), np.uint8)
    a[:, :, 0] = STONE
    k = d.art_scale/2
    ix, iy = np.indices(a.shape[:2], dtype=float)
    x, y = (ix+.5-CENTER[0])/k, (iy+.5-CENTER[1])/k
    r, angle = np.hypot(x, y), np.arctan2(y, x)
    z = (np.arange(a.shape[2])+.5-1)/k

    def fill(mask, bottom, top, material):
        volume = mask[:, :, None] & (z >= np.asarray(bottom)[..., None]) & (z < np.asarray(top)[..., None])
        a[volume] = material

    def box(x1, y1, bottom, x2, y2, top, material):
        fill((x >= x1) & (x < x2) & (y >= y1) & (y < y2), bottom, top, material)

    def ring(inner, outer, bottom, top, material, sector=True):
        fill((r >= inner) & (r < outer) & sector, bottom, top, material)

    def roof(inner, ridge, outer, eave, top, material, sector=True):
        h = ring_roof_height(r, inner, ridge, outer, eave, top)
        footprint = (r >= inner) & (r < outer) & sector
        # Trim supporting walls to the roof envelope before laying its shell.
        fill(footprint, h, 26, 0)
        # A solid roof wedge avoids pinholes between diagonal voxel steps;
        # only the exposed eave is 0.75 m (1-2 voxels) thick.
        fill(footprint, eave-.75, h, material)
        fill((r >= ridge-.22) & (r < ridge+.22) & sector,
             top-.45, top+.15, RIDGE)

    # Courtyard and narrow entry apron: local stone, level with public paving.
    local_ground = (r < 29.1) | ((np.abs(x) < 5) & (y > 25) & (y < 34))
    a[:, :, 0][local_ground] = COURT

    # Four-storey rammed-earth outer ring with an actual continuous roof.
    ring(d.outer_room_inner_radius_m, d.outer_diameter_m/2, 0, 13.8, EARTH)
    ring(21.6, 22.2, .4, 13.1, WOOD)
    ring(27.8, 28.6, 0, .9, BRICK)
    # Exterior windows exist on the two upper storeys, not all four.
    bay = np.abs((angle*44/(2*np.pi)+.5) % 1-.5)
    for floor in (7.8, 11.2):
        ring(27.7, 28.8, floor, floor+1.4, DARK, bay < .13)
    for level in (0, 3.3, 6.6, 9.9):
        ring(d.outer_gallery_radius_m, 22.3, level, level+.55, WOOD)
        ring(21.4, 22.4, level+.6, level+2.6, DARK, bay < .28)
        if level:
            ring(19.5, 20.0, level+.65, level+1.05, WOOD)
    # Columns use one-voxel minimum thickness; room count remains 44 bays.
    for t in np.linspace(0, 2*np.pi, 44, endpoint=False):
        px, py = 19.8*np.cos(t), 19.8*np.sin(t)
        fill((np.abs(x-px) < .36) & (np.abs(y-py) < .36), .5, 13.0, WOOD)
    # Eight radial fire partitions stop beneath the outer eave. Their low
    # courtyard sections have through openings, rather than solid spokes.
    for t in np.arange(8)*np.pi/4+np.pi/8:
        along, cross = x*np.cos(t)+y*np.sin(t), -x*np.sin(t)+y*np.cos(t)
        partition = (along > 14.0) & (along < 22.3) & (np.abs(cross) < .37)
        fill(partition, 0, 3.4, BRICK)
        fill(partition & (along > 16.3) & (along < 18.0), 0, 2.5, 0)
        fill(partition & (along > 19.7), 3.4, 13.1, BRICK)
    roof(d.outer_eave_inner_radius_m, d.outer_ridge_radius_m,
         d.outer_eave_outer_radius_m, d.outer_eave_height_m,
         d.outer_height_m, OUTER_ROOF)

    # Two-storey inner ring: brick room band, open inward-facing colonnade.
    ring(9.5, d.inner_wall_diameter_m/2, 0, 6.7, BRICK)
    inner_bay = np.abs((angle*20/(2*np.pi)+.5) % 1-.5)
    for level in (0, 3.5):
        ring(9.1, 9.8, level+.4, level+2.8, DARK, inner_bay < .29)
        ring(6.9, 10.0, level, level+.45, PLASTER)
    ring(6.8, 7.3, 3.8, 4.5, WOOD)
    ring(6.7, 7.4, 4.5, 4.8, PLASTER)
    for t in np.linspace(0, 2*np.pi, 20, endpoint=False):
        px, py = 7.1*np.cos(t), 7.1*np.sin(t)
        fill((np.abs(x-px) < .38) & (np.abs(y-py) < .38), 0, 6.6, PLASTER)
    roof(d.inner_eave_inner_radius_m, d.inner_ridge_radius_m,
         d.inner_eave_outer_radius_m, d.inner_eave_height_m,
         d.inner_ridge_height_m, INNER_ROOF)

    # North ancestral hall replaces a segment of the inner ring. Four granite
    # columns support a taller pavilion; no roof or outer wall is cut away.
    hw, hy = d.hall_width_m/2, -9.5
    box(-hw, -14, 0, hw, -5, 10, 0)
    box(-hw, -14, 0, hw, -4.8, .7, PLASTER)
    box(-hw, -14, .7, hw, -13.2, 7.8, PLASTER)
    for side in (-hw, hw-.7):
        box(side, -14, .7, side+.7, -5, 7.8, PLASTER)
    box(-4.5, -13.1, .7, 4.5, -12.9, 6.2, WOOD)
    for px in (-4.8, -1.6, 1.6, 4.8):
        column = (x-px)**2+(y+5.2)**2 < .48**2
        fill(column, .7, 7.5, PLASTER)
        fill((np.abs(x-px) < .65) & (np.abs(y+5.2) < .65), .7, 1.1, PLASTER)
    box(-hw, -5.6, 7.0, hw, -4.6, 7.8, WOOD)
    for left, right in ((-6, -1.8), (1.8, 6)):
        box(left, -5.0, 1.2, right, -4.5, 1.6, PLASTER)
    box(-1.8, -4.8, 0, 1.8, -3.8, .4, PLASTER)
    hall_mask = (np.abs(x) < hw+.7) & (np.abs(y-hy) < d.hall_depth_m/2+.7)
    hall_eave = 7.6
    hall_h = hall_eave+(d.hall_roof_top_m-7.6)*(1-np.maximum(np.abs(x)/(hw+.7), np.abs(y-hy)/(d.hall_depth_m/2+.7)))
    fill(hall_mask, hall_h, 26, 0)
    fill(hall_mask, hall_eave-.7, hall_h, HALL_ROOF)
    diagonal = np.abs(np.abs(x)/(hw+.7)-np.abs(y-hy)/(d.hall_depth_m/2+.7)) < .055
    fill(hall_mask & diagonal, hall_h-.1, hall_h+.35, RIDGE)

    # Main axis passes through both rings. Portal surround is wider and pale;
    # lettering is a quiet plaque, not invented unreadable high-contrast text.
    box(-3.8, 27.7, 0, 3.8, 29.0, 6.6, PLASTER)
    box(-2.3, 28.6, 4.5, 2.3, 29.1, 5.8, BRICK)
    box(-1.7, 28.8, 4.8, 1.7, 29.2, 5.5, PLASTER)
    for ri, ro in ((19.0, 29.2), (6.5, 14.5)):
        opening = (np.abs(x) < 1.6) & (y > ri) & (y < ro)
        fill(opening, 0, 2.3+np.sqrt(np.maximum(0, 1.6**2-x*x)), 0)
    # Side access to the two ear courtyards.
    for side in (-1, 1):
        box(side*24-6 if side < 0 else 24, -1.4, 0,
            -24 if side < 0 else 30, 1.4, 3.0, 0)

    # Each 'half-moon' ear is one curved, six-bay, two-storey wing. A narrow
    # court and two low end walls join it to the main ring. Detached village
    # houses in the aerial are deliberately outside this object's scope.
    ear_sector = np.abs(np.arctan2(y, np.abs(x))) < np.deg2rad(d.ear_half_angle_degrees)
    ring(29.0, 41.0, 0, .2, COURT, ear_sector)
    ring(d.ear_inner_radius_m, d.ear_outer_radius_m, 0, d.ear_eave_height_m, BRICK, ear_sector)
    ear_bay = np.abs((np.arctan2(y, np.abs(x))+np.deg2rad(23))*6/np.deg2rad(46) % 1-.5)
    for level in (.7, 3.7):
        ring(32.7, 33.7, level, level+1.7, DARK, ear_sector & (ear_bay < .22))
        ring(39.4, 40.3, level, level+1.6, DARK, ear_sector & (ear_bay < .14))
    roof(32.0, 36.5, 41.0, d.ear_eave_height_m, d.ear_ridge_height_m, EAR_ROOF, ear_sector)
    for side in (-1, 1):
        for sign in (-1, 1):
            t = sign*np.deg2rad(d.ear_half_angle_degrees)
            end = (np.abs(y*np.cos(t)-np.abs(x)*np.sin(t)) < .45) & (x*side > 0)
            ring(28.5, 40.2, 0, 2.5, BRICK, end)

    # Four public adults, unchanged 5-unit stature; no scaled-up people.
    person = make_model('walkers')[10:13, 12:14, 1:6]
    for px, py in ((-5, 33), (2, 35), (4, 1), (-16, 5)):
        xx, yy = round(CENTER[0]+px*k), round(CENTER[1]+py*k)
        target = a[xx:xx+3, yy:yy+2, 1:6]
        target[person != 0] = person[person != 0]+PROP_OFFSET
    return a


def save_model():
    model = generate_model()
    np.savez_compressed(SOURCE/'model.npz', model=model)
    with np.load(SOURCE/'model.npz') as saved:
        assert saved.files == ['model']
        assert np.array_equal(saved['model'], model)
    print(f'Saved {SOURCE / "model.npz"}: {model.shape}', flush=True)


if __name__ == '__main__':
    save_model()
