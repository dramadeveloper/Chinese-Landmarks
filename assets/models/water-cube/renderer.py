"""Render and inspect the Water Cube model through the shared sprite pipeline."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[2]
OUT = ROOT / 'build/current/water-cube'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'assets/chinese-landmarks'))
from water_cube_model import (make_building, cushions, PANELS, LOW, HIGH, BASE, TOP,
                   MEMBRANE, ROOF, JOINT, STONE, GLASS, METAL, VENT, WATER,
                   COPING, PROP_OFFSET)
from render_core import (trace_model, shade, indexed, stone_ground, finish,
                         surface_shadow, DEFAULT_SCENE, DEFAULT_SAMPLING)
from plaza_material import PALETTE, SCALE, save_p, validate_png, paving
from sprite_pipeline import save_render, export_tiles
from voxel_io import native_vox
from render_assets import CURRENT, adapter
from export_plaza import make_model, render as render_prop, RAMPS as PROP_RAMPS

# Materials are selected before geometry. These are all static, safe DOS colors.
ETFE = [147, 148, 149, 150, 151, 152, 153, 153]
ROOF_JOINT = 10
RAMPS = {
    MEMBRANE: ETFE, ROOF: [154, 155, 156, 157, 158, 214, 159, 213, 160, 212],
    ROOF_JOINT: [154, 155, 156, 157, 158, 214, 159, 213, 160],
    JOINT: [146, 147, 148, 149, 150, 151, 152, 153],
    STONE: [6, 7, 8, 9, 10, 11, 12, 13],
    GLASS: [1, 154, 155, 156, 157, 158, 159],
    METAL: [4, 5, 6, 7, 8, 9, 10, 11],
    VENT: [2, 3, 4, 5, 6, 7, 8],
    WATER: [154, 155, 156, 157, 158],
    COPING: [7, 8, 9, 10, 11, 12, 13, 14],
    **{key + PROP_OFFSET: ramp for key, ramp in PROP_RAMPS.items()},
}
BG, PANEL, INK, MUTED, ACCENT = '#1b2832', '#243640', '#eef4f4', '#a9c3cc', '#95d2e5'


def render(model, view, *, lighting=DEFAULT_SCENE, canvas=(288, 176), origin=(144, 32), offset=0):
    frame = trace_model(model, view, canvas, origin)
    x, y, z = frame.point.transpose(2, 0, 1)
    size = model.shape[0]
    if view == 1:
        x, y = y, size-x
    elif view == 2:
        x, y = size-x, size-y
    elif view == 3:
        x, y = size-y, x
    x, y = x-offset, y-offset
    normal = frame.normal.copy()
    mat = frame.mat.copy()
    variation = np.zeros(mat.shape)
    membrane = frame.hit & np.isin(mat, [MEMBRANE, ROOF, JOINT])
    roof = membrane & (z >= TOP-1.01) & (x > LOW) & (x < HIGH) & (y > LOW) & (y < HIGH)
    walls = membrane & ~roof
    # The one-unit relief is an integer approximation of sub-unit cushions.
    # Trace their shadows against the uninflated carrier, retaining real roof
    # fittings and porticos, so voxel stair corners do not stipple the membrane.
    carrier = model.copy()
    cx, cy, cz = np.indices(model.shape)
    relief = np.isin(carrier, [MEMBRANE, ROOF, JOINT]) & ((cz >= TOP) |
             (cx < LOW+offset) | (cx >= HIGH+offset) |
             (cy < LOW+offset) | (cy >= HIGH+offset))
    carrier[relief] = 0
    smooth_shadow = surface_shadow(np.rot90(carrier, view, axes=(0, 1)),
                                   frame.point, frame.normal, membrane, DEFAULT_SAMPLING)
    shadow = np.where(membrane, smooth_shadow, frame.shadow)
    distances = np.stack([abs(x-LOW), abs(x-HIGH), abs(y-LOW), abs(y-HIGH)])
    side = distances.argmin(axis=0)
    for name, mask, u, v, axis, sign in [
        ('roof', roof, x-LOW, y-LOW, 2, 1),
        ('west', walls & (side == 0), y-LOW, z-BASE, 0, -1),
        ('east', walls & (side == 1), y-LOW, z-BASE, 0, 1),
        ('north', walls & (side == 2), x-LOW, z-BASE, 1, -1),
        ('south', walls & (side == 3), x-LOW, z-BASE, 1, 1),
    ]:
        edge, dx, dy, dome, owner = cushions(u[mask], v[mask], PANELS[name])
        # Analytic cushion normals smooth the shallow voxel relief; lighting is
        # still computed by render_core in fixed screen coordinates.
        slope = .085 if axis == 2 else .12
        du = dx*slope
        dv = dy*slope if axis == 2 else dy*slope*23/(TOP-BASE)
        n = np.zeros((len(edge), 3))
        if axis == 2:
            n[:, 0], n[:, 1], n[:, 2] = du, dv, 1
        elif axis == 0:
            n[:, 0], n[:, 1], n[:, 2] = sign, du, dv
        else:
            n[:, 0], n[:, 1], n[:, 2] = du, sign, dv
        for _ in range(view):
            n[:, 0], n[:, 1] = -n[:, 1].copy(), n[:, 0].copy()
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        normal[mask] = n
        seam = edge < .40
        mat[mask] = np.where(seam, ROOF_JOINT if axis == 2 else JOINT,
                            ROOF if axis == 2 else MEMBRANE)
        variation[mask] = ((owner*37 % 11)-5)*.004 - np.where(seam, .035, 0)
    light = shade(normal, shadow, lighting.standard) + variation
    glass = frame.hit & (mat == GLASS)
    light[glass] += .05*np.sin(z[glass]*1.8)
    building = indexed(mat, light, frame.hit, RAMPS)
    ground = stone_ground(frame, lighting)
    water = frame.gm & (frame.gmat == WATER)
    # Static muted water: no palette cycling and no full-tile border.
    ground[water] = np.where(frame.gshadow[water], 155, 157)
    glints = water & (((frame.gx+2*frame.gy) % 17) == 0)
    ground[glints] = np.where(frame.gshadow[glints], 156, 158)
    return finish(frame, ground, building)


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', size)


def rgba(pixels):
    return Image.fromarray(np.dstack((PALETTE[pixels], (pixels != 0).astype(np.uint8)*255)))


def cropped(pixels):
    image = rgba(pixels)
    return image.crop(image.getbbox())


def previews(results):
    native = Image.new('RGB', (1120, 230), BG)
    draw = ImageDraw.Draw(native)
    draw.text((22, 14), '水立方  /  4×4 正式模型  /  原生 1×', font=font(20), fill=INK)
    for view, result in enumerate(results):
        pic = cropped(result[4])
        native.paste(pic, (20+view*277, 187-pic.height), pic)
        draw.text((22+view*277, 198), f'方向 {view}', font=font(14), fill=MUTED)
    native.save(OUT/'four-views-1x.png')

    board = Image.new('RGB', (1100, 780), BG)
    draw = ImageDraw.Draw(board)
    draw.text((24, 18), '水立方  /  4×4 日间模型', font=font(25), fill=INK)
    draw.text((25, 58), '四方向 · 2× 最近邻  /  气枕表皮、低位门厅与贴身水带', font=font(15), fill=MUTED)
    for view, result in enumerate(results):
        pic = cropped(result[4])
        pic = pic.resize((pic.width*2, pic.height*2), Image.Resampling.NEAREST)
        x, y = 22+(view % 2)*545, 110+(view//2)*320
        board.paste(pic, (x, y), pic)
        draw.text((x, y+296), f'方向 {view}', font=font(14), fill=ACCENT)
    draw.text((25, 752), '离线精灵预览；入口位置与水带宽度为照片估算。', font=font(14), fill=MUTED)
    board.save(OUT/'four-views-2x.png')

    pic = cropped(results[0][4])
    hero = Image.new('RGB', (pic.width*4+48, pic.height*4+116), BG)
    enlarged = pic.resize((pic.width*4, pic.height*4), Image.Resampling.NEAREST)
    hero.paste(enlarged, (24, 76), enlarged)
    draw = ImageDraw.Draw(hero)
    draw.text((24, 15), '水立方  /  4×4 正式模型', font=font(25), fill=INK)
    draw.text((25, 49), '4× 最近邻细节  ·  原生素材为 TTD DOS 8bpp', font=font(14), fill=MUTED)
    hero.save(OUT/'preview-4x.png')


def comparison(result):
    board = Image.new('RGB', (1220, 520), BG)
    draw = ImageDraw.Draw(board)
    draw.text((24, 15), '同倍率体量比较  /  全部原生 1×', font=font(23), fill=INK)
    draw.text((25, 54), '水立方为 4×4 模型；立面单独加高约 39%，人物使用公共尺寸。', font=font(15), fill=MUTED)
    evidence = []
    # Align map-ground centres instead of independently fitting the objects.
    for asset, screen_x, label in zip(CURRENT, [100, 385, 650], ['东方明珠', '天安门', '天坛']):
        path = ROOT/'assets'/asset.source/asset.model
        with np.load(path) as data:
            inputs = {key: data[key] for key in asset.model_keys}
        arrays = adapter(asset).render(view=0, **inputs)
        picture = rgba(arrays[4])
        bounds = picture.getbbox()
        picture = picture.crop(bounds)
        ox, oy = arrays[7]
        nx, ny = arrays[8]
        anchor = (ox+16*(ny-nx)-bounds[0], oy+8*(nx+ny)-bounds[1])
        board.paste(picture, (round(screen_x-anchor[0]), round(340-anchor[1])), picture)
        draw.text((screen_x-38, 462), label, font=font(15), fill=MUTED)
        evidence.append(dict(name=asset.name, model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'Comparison: {asset.name}', flush=True)
    pic = rgba(result[4])
    bounds = pic.getbbox()
    pic = pic.crop(bounds)
    board.paste(pic, (int(878-(result[7][0]-bounds[0])), int(340-(result[7][1]+64-bounds[1]))), pic)
    draw.text((821, 462), '水立方', font=font(15), fill=ACCENT)
    for index, kind in enumerate(['walkers', 'bench', 'tree']):
        ground, building = render_prop(make_model(kind), kind, 0)
        pic = cropped(np.where(building != 0, building, ground))
        board.paste(pic, (1030+index*57, 358-pic.height), pic)
    draw.text((1030, 462), '公共人物与街具', font=font(14), fill=MUTED)
    draw.text((25, 494), '同一地块中心基线 · 原生精灵离线预览', font=font(13), fill=MUTED)
    board.save(OUT/'comparison-1x.png')
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = make_building()
    np.savez_compressed(SOURCE/'model.npz', model=model)
    xyz = np.argwhere(model != 0)
    voxels = np.column_stack((xyz, model[tuple(xyz.T)])).astype(np.uint8)
    palette = np.zeros((256, 4), np.uint8)
    palette[1:, 3] = 255
    for key, ramp in RAMPS.items():
        palette[key, :3] = PALETTE[ramp[round((len(ramp)-1)*.75)]]
    native_vox(OUT/'model.vox', model.shape, voxels, palette)
    results, measurements, records, graphics = [], [], [], []
    views = range(1) if args.preview_only else range(4)
    for view in views:
        arrays = render(model, view)
        save_render(arrays, OUT, view)
        results.append(arrays)
        bbox = rgba(arrays[4]).getbbox()
        assert 0 < bbox[0] < bbox[2] < 288 and 0 < bbox[1] < bbox[3] < 176
        if not args.preview_only:
            lines, tiles = export_tiles(OUT, 'water-cube', arrays, view, 'wc_', 'wc_tiles')
            records.extend(tiles)
            graphics.extend(lines)
        measurements.append(dict(view=view, tiles=list(arrays[8]),
                                 full_sprite_bbox_px=[bbox[2]-bbox[0], bbox[3]-bbox[1]],
                                 recomposition_exact=not args.preview_only))
        print(f'View {view}: {measurements[-1]}', flush=True)
    if args.preview_only:
        image = cropped(results[0][4])
        image.resize((image.width*4, image.height*4), Image.Resampling.NEAREST).save(OUT/'preview-4x.png')
        return
    previews(results)
    evidence = comparison(results[0])
    (OUT/'graphics.nml').write_text('\n'.join(graphics)+'\n', encoding='utf-8')
    (OUT/'sprite_offsets.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')

    connected = np.zeros((240, 240, model.shape[2]), np.uint8)
    connected[:, :, 0] = STONE
    connected[40:200, 40:200] = model
    context = render(connected, 0, canvas=(416, 256), origin=(208, 32), offset=40)
    save_p(context[4], OUT/'plaza-connection-1x-8bpp.png')
    # Compare the entire outer ground ring with the public paving at its world phase.
    yy, xx = np.indices(context[0].shape, dtype=float)
    u, v = (xx+.5-208)/SCALE, (yy+.5-32)/SCALE
    gx, gy = np.floor(v-u*.5).astype(int), np.floor(v+u*.5).astype(int)
    border = (gx >= 0) & (gy >= 0) & (gx < 240) & (gy < 240) & ((gx < 8) | (gy < 8) | (gx >= 232) | (gy >= 232))
    assert np.array_equal(context[4][border], paving(gx, gy)[border])
    pic = cropped(context[4])
    canvas = Image.new('RGB', (pic.width*2, pic.height*2), BG)
    enlarged = pic.resize(canvas.size, Image.Resampling.NEAREST)
    canvas.paste(enlarged, (0, 0), enlarged)
    canvas.save(OUT/'plaza-connection-2x.png')

    paths = [OUT/f'view{view}-1x-8bpp.png' for view in range(4)]
    paths += [OUT/'plaza-connection-1x-8bpp.png']
    paths += [OUT/record['file'] for record in records]
    for path in paths:
        validate_png(path)
    entity = np.argwhere(model[:, :, 1:] != 0) + [0, 0, 1]
    low, high = entity.min(axis=0), entity.max(axis=0)
    envelope = np.argwhere(np.isin(model, [MEMBRANE, ROOF, JOINT]))
    envelope_low, envelope_high = envelope.min(axis=0), envelope.max(axis=0)
    report = dict(footprint_tiles=[4, 4], container_units=list(model.shape),
                  reference_metres=[177, 177, 31], base_art_scale=1.5,
                  body_units=[HIGH-LOW, HIGH-LOW, TOP-BASE], body_base_z=BASE,
                  body_height_factor=(TOP-BASE)/23,
                  local_adjustments=['Horizontal body remains 132x132; height increased from 23 to 32 units for game-view readability (not uniform scaling).',
                                     'Two-unit raised foundation; one-unit cushion relief; low roof fittings.',
                                     'Facade cell centres stretched vertically with the taller envelope; doors, platforms and people keep their original dimensions.',
                                     'Roof retains its muted blue-grey ramp; facade uses a brighter sky-blue ramp with blue seams and broad cushion shading; common lighting unchanged.',
                                     'Merged small foam cells and approximately one-pixel seams.',
                                     'Photo-estimated openings, entrances, water belt and roof fittings.'],
                  above_ground_scene_bounds_inclusive=[low.tolist(), high.tolist()],
                  above_ground_scene_bbox_units=(high-low+1).tolist(),
                  membrane_bounds_inclusive=[envelope_low.tolist(), envelope_high.tolist()],
                  membrane_bbox_units=(envelope_high-envelope_low+1).tolist(),
                  highest_voxel_z=int(high[2]), top_from_z0_units=int(high[2])+1,
                  top_from_z0_projection_px=round((int(high[2])+1)*SCALE, 1),
                  voxel_count_including_ground=len(voxels), views=measurements,
                  indexed_pngs_checked=len(paths), outer_paving_phase_matches=True,
                  current_models_used_for_comparison=evidence,
                  model_sha256=hashlib.sha256((OUT/'model.vox').read_bytes()).hexdigest(),
                  package_integrated=True, object_id=1105,
                  package_build_report='../validation.json')
    (OUT/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['top_from_z0_units', 'indexed_pngs_checked', 'voxel_count_including_ground']}), flush=True)


if __name__ == '__main__':
    main()
