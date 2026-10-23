"""Build current model adapters through the shared renderer and sprite pipeline."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plaza_material import save_p, validate_png
from sprite_pipeline import (RESULT_KEYS, crop, save_render, write_sprite, export_tiles,
                             purchase_pixels, compile_grf)
import export_plaza
from render_core import SUN_VECTOR, DEFAULT_SCENE

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
PACKAGE_SOURCE = ASSETS / 'package'
REFERENCE = ASSETS / 'reference'
DEFAULT_OUTPUT = ASSETS.parent / 'build/current'
FRAGMENTS = ('grf.nml', 'plaza.nml', 'oriental-pearl-object.nml',
             'tiananmen.nml', 'temple-of-heaven.nml', 'water-cube.nml', 'xian-bell-tower.nml', 'canton-tower.nml',
             'xinghai-square.nml', 'yellow-crane-tower.nml', 'gate-to-the-east.nml')


@dataclass(frozen=True)
class Menu:
    scale: float | None = None
    max_size: tuple = (128, 88)
    allow_upscale: bool = False
    baseline: int | None = None
    filename: str = 'purchase{view}.png'


@dataclass(frozen=True)
class Asset:
    name: str
    source: str
    model: str
    folder: str
    prefix: str
    tile_switch: str
    fragment: str
    footer: str
    bands: tuple = ()
    model_keys: tuple = ('model',)
    menu: Menu = Menu()


CURRENT = (
    Asset('pearl', 'models/oriental-pearl', 'model.npz',
          'oriental-pearl', '', 'tile_view', 'oriental-pearl-object.nml',
          'world_graphics', (('low', 3, 160, 0), ('high', 160, 312, 127)), ('model', 'core'),
          Menu(scale=.5, baseline=32, filename='purchase-view{view}.png')),
    Asset('tiananmen', 'models/tiananmen', 'model.npz',
          'tiananmen', 'tam_', 'tam_tiles', 'tiananmen.nml', 'tam_world',
          menu=Menu(allow_upscale=True)),
    Asset('temple', 'models/temple-of-heaven', 'model.npz',
          'temple-of-heaven', 'toh_', 'toh_tiles', 'temple-of-heaven.nml', 'toh_world'),
    Asset('water-cube', 'models/water-cube', 'model.npz',
          'water-cube', 'wc_', 'wc_tiles', 'water-cube.nml', 'wc_world'),
    Asset('xian-bell-tower', 'models/xian-bell-tower', 'model.npz',
          'xian-bell-tower', 'xbt_', 'xbt_tiles', 'xian-bell-tower.nml', 'xbt_world'),
    Asset('canton-tower', 'models/canton-tower', 'model.npz',
          'canton-tower', 'ct_', 'ct_tiles', 'canton-tower.nml', 'ct_world',
          (('low', 1, 160, 0), ('high', 160, 400, 127)),
          menu=Menu(scale=.38, baseline=32)),
    Asset('xinghai-square', 'models/xinghai-square', 'model.npz',
          'xinghai-square', 'xh_', 'xh_tiles', 'xinghai-square.nml', 'xh_world'),
    Asset('yellow-crane-tower', 'models/yellow-crane-tower', 'model.npz',
          'yellow-crane-tower', 'yct_', 'yct_tiles', 'yellow-crane-tower.nml', 'yct_world'),
    Asset('gate-to-the-east', 'models/gate-to-the-east', 'model.npz',
          'gate-to-the-east', 'gte_', 'gte_tiles', 'gate-to-the-east.nml', 'gte_world'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter(asset):
    spec = importlib.util.spec_from_file_location('current_' + asset.name, ASSETS / asset.source / 'renderer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_package(output):
    package = output / 'grf'
    package.mkdir(parents=True, exist_ok=True)
    for name in FRAGMENTS:
        shutil.copy2(PACKAGE_SOURCE / name, package / name)
    shutil.copytree(PACKAGE_SOURCE / 'lang', package / 'lang', dirs_exist_ok=True)
    shutil.copy2(PACKAGE_SOURCE / 'LICENSE', package / 'LICENSE')
    combined = '\n\n'.join((package / name).read_text(encoding='utf-8-sig') for name in FRAGMENTS)
    (package / 'oriental-pearl.nml').write_text(combined, encoding='utf-8')
    return package


def menu_graphics(asset, composed, view, package):
    im = Image.fromarray(composed)
    im = im.crop(im.getbbox())
    menu = asset.menu
    scale = menu.scale
    if scale is None:
        scale = min(menu.max_size[0] / im.width, menu.max_size[1] / im.height)
        if not menu.allow_upscale:
            scale = min(scale, 1)
    pixels = purchase_pixels(composed, scale)
    height, width = pixels.shape
    yoffset = -height // 2 if menu.baseline is None else -height + menu.baseline
    filename = menu.filename.format(view=view)
    name = f'{asset.prefix}menu_{view}'
    path = f'sprites/{asset.folder}/{filename}'
    lines = write_sprite(package, path, name, pixels, (-width // 2, yoffset))
    lines += [f'spritelayout {asset.prefix}menu_layout_{view} {{', '    ground { sprite: 0; }',
              f'    building {{ sprite: {name}; zextent: {height}; }}', '}']
    return lines


def check_records(asset, records):
    old = json.loads((REFERENCE / asset.name / 'sprite_offsets.json').read_text())
    expected = {r['file']: r for r in old}
    assert len(records) == len(expected)
    for rec in records:
        before = expected[rec['file']]
        for key in ('view', 'tile', 'kind', 'width', 'height', 'xoffset', 'yoffset'):
            assert rec[key] == before[key], (asset.name, rec['file'], key)
        if asset.bands:
            for key in ('zoffset', 'zextent', 'source_rectangle'):
                assert rec[key] == before[key], (asset.name, rec['file'], key)


def render_asset(asset, output, package, verify, lighting=DEFAULT_SCENE):
    module = adapter(asset)
    source = ASSETS / asset.source
    model_path = source / asset.model
    with np.load(model_path) as loaded:
        inputs = {key: loaded[key] for key in asset.model_keys}
    lines, records, views = [], [], []
    for view in range(4):
        arrays = module.render(view=view, lighting=lighting, **inputs)
        ground, go, building, bo, composed, xyz, hit, origin, tiles = arrays
        save_render(arrays, output / asset.name, view)
        save_p(composed, package / f'sprites/{asset.folder}/view{view}-1x-8bpp.png')
        lines += menu_graphics(asset, composed, view, package)
        tile_lines, tile_records = export_tiles(
            package, asset.folder, arrays, view, asset.prefix, asset.tile_switch, asset.bands)
        lines += tile_lines
        records += tile_records
        if verify:
            with np.load(REFERENCE / asset.name / f'render-view{view}.npz') as before:
                for key, value in zip(RESULT_KEYS, arrays):
                    if key in before:
                        assert np.array_equal(value, before[key]), (asset.name, view, key)
        assert np.count_nonzero(ground) == tiles[0] * tiles[1] * 1024
        l, t, r, b = Image.fromarray(composed).getbbox()
        assert 0 < l < r < composed.shape[1] and 0 < t < b < composed.shape[0]
        views.append(dict(view=view, tiles=list(tiles), recomposition_exact=True,
                          existing_arrays_equal=True if verify else None))
        print(f'{asset.name} view {view}: rendered, exported and reassembled', flush=True)
    if verify:
        check_records(asset, records)
    fragment_path = package / asset.fragment
    original = fragment_path.read_text(encoding='utf-8-sig')
    marker = f'switch (FEAT_OBJECTS, SELF, {asset.footer}, view)'
    updated = '\n'.join(lines) + '\n' + original[original.index(marker):]
    combined_path = package / 'oriental-pearl.nml'
    combined = combined_path.read_text(encoding='utf-8-sig')
    assert combined.count(original) == 1, asset.fragment
    combined_path.write_text(combined.replace(original, updated), encoding='utf-8')
    fragment_path.write_text(updated, encoding='utf-8')
    (output / asset.name / 'sprite_offsets.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return dict(name=asset.name, model=str(model_path), model_sha256=sha(model_path), views=views,
                slice_offsets_equal=True if verify else None, slices=len(records))


def render_plaza(output, package, verify, lighting=DEFAULT_SCENE):
    records, graphics, checked = [], [], 0
    destination = output / 'plaza'
    destination.mkdir(parents=True, exist_ok=True)
    for kind in export_plaza.KINDS:
        model = export_plaza.make_model(kind)
        for view in range(export_plaza.VIEWS[kind]):
            ground, building = export_plaza.render(model, kind, view, lighting=lighting)
            composed = np.where(building != 0, building, ground)
            if verify:
                with np.load(REFERENCE / f'plaza/{kind}-{view}.npz') as before:
                    assert np.array_equal(ground, before['ground']), (kind, view, 'ground')
                    assert np.array_equal(building, before['building']), (kind, view, 'building')
                checked += 1
            save_p(composed, package / f'sprites/plaza/{kind}-v{view}.png')
            save_p(composed, destination / f'{kind}-v{view}.png')
            for layer, array in [('g', ground), ('b', building)]:
                pixels, offset, rectangle = crop(array, array != 0, export_plaza.ORIGIN)
                path = f'sprites/plaza/{kind}-{layer}-v{view}.png'
                name = f'plaza_{kind}_{layer}_{view}'
                graphics += write_sprite(package, path, name, pixels, offset)
                records.append(dict(kind=kind, view=view, layer=layer, file=path,
                                    width=pixels.shape[1], height=pixels.shape[0],
                                    xoffset=offset[0], yoffset=offset[1]))
            # Current 1x1 prop NML layout and IDs are preserved in the package.
    if verify:
        expected = json.loads((REFERENCE / 'plaza/sprite_offsets.json').read_text())
        for after, before in zip(records, expected, strict=True):
            for key, value in after.items():
                assert value == before[key], (key, after['file'])
    # Replace each declaration; this also supports changed sprite bounds after lighting work.
    import re
    fragment = package / 'plaza.nml'
    original = fragment.read_text(encoding='utf-8-sig')
    updated = original
    for start in range(0, len(graphics), 3):
        declaration = '\n'.join(graphics[start:start + 3])
        name = re.search(r'spriteset\((\w+),', declaration).group(1)
        updated, count = re.subn(r'spriteset\(' + name + r',.*?\n\}', lambda _: declaration, updated, count=1, flags=re.S)
        assert count == 1, name
    combined_path = package / 'oriental-pearl.nml'
    combined = combined_path.read_text(encoding='utf-8-sig')
    assert combined.count(original) == 1
    combined_path.write_text(combined.replace(original, updated), encoding='utf-8')
    fragment.write_text(updated, encoding='utf-8')
    (destination / 'sprite_offsets.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('plaza: 17 views rendered through the shared core', flush=True)
    return dict(name='plaza', views=17, existing_arrays_checked=checked)


def comparison_board(output):
    board = Image.new('RGB', (2480, 490), '#202a33')
    draw = ImageDraw.Draw(board)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
    draw.text((18, 15), '公共渲染管线 · 原生 1×', font=font, fill='#eff1ed')
    for name, x, title in [('pearl', 20, '东方明珠'), ('tiananmen', 190, '天安门'), ('temple', 620, '天坛'), ('water-cube', 775, '水立方'), ('xian-bell-tower', 1055, '西安钟楼'), ('canton-tower', 1270, '广州塔'), ('xinghai-square', 1495, '星海广场'), ('yellow-crane-tower', 1835, '黄鹤楼'), ('gate-to-the-east', 2050, '东方之门')]:
        im = Image.open(output / name / 'view0-1x-8bpp.png').convert('RGBA')
        im = im.crop(im.getbbox())
        board.paste(im, (x, 410 - im.height), im)
        draw.text((x, 429), title, font=font, fill='#b5c7cd')
    for i, kind in enumerate(('walkers', 'meeting', 'bench', 'tree')):
        im = Image.open(output / 'plaza' / f'{kind}-v0.png').convert('RGBA')
        im = im.crop(im.getbbox())
        board.paste(im, (2300 + (i % 2) * 80, 312 + (i // 2) * 66), im)
    draw.text((18, 463), '中国地标 · 九座地标与五种广场 · 原生精灵离线预览', font=font, fill='#b5c7cd')
    board.save(output / 'comparison-1x.png')
    board.resize((board.width * 2, board.height * 2), Image.Resampling.NEAREST).save(output / 'comparison-2x-nearest.png')


def build(output, verify=False, compile_package=False, *, lighting=DEFAULT_SCENE, comparison=True):
    output = Path(output).resolve()
    # CLI output must stay separate from source assets, drafts and release records.
    for protected in (ASSETS, ASSETS.parent / 'drafts', ASSETS.parent / 'releases', ASSETS.parent / 'archive'):
        assert not output.is_relative_to(protected) and not protected.is_relative_to(output), output
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p: sha(p) for folder in (ROOT, PACKAGE_SOURCE, REFERENCE, *(ASSETS / a.source for a in CURRENT))
                     for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    package = copy_package(output)
    reports = [render_asset(asset, output, package, verify, lighting) for asset in CURRENT]
    reports.append(render_plaza(output, package, verify, lighting))
    if comparison:
        comparison_board(output)
    pngs = list((package / 'sprites').rglob('*.png'))
    if verify:
        expected = json.loads((REFERENCE / 'package-pixels.json').read_text())
        assert {p.relative_to(package).as_posix() for p in pngs} == expected.keys()
    for path in pngs:
        validate_png(path)
        if verify:
            pixels = hashlib.sha256(np.array(Image.open(path)).tobytes()).hexdigest()
            assert pixels == expected[path.relative_to(package).as_posix()], path
    compilation = None
    if compile_package:
        run = compile_grf(package)
        compilation = dict(exit_code=run.returncode, sha256=sha(package / 'oriental-pearl.grf'))
        print(run.stdout + run.stderr, flush=True)
    for path, digest in source_hashes.items():
        assert sha(path) == digest, path
    report = dict(renderer='render_core.py',
                  baseline_acceptance=json.loads((REFERENCE / 'acceptance.json').read_text(encoding='utf-8')) if verify else None,
                  lighting_settings=dict(sun=SUN_VECTOR.tolist(), **asdict(lighting)),
                  assets=reports,
                  palette_pngs_checked=len(pngs), all_package_pngs_equal=True if verify else None,
                  source_assets_unchanged=True, compilation=compilation, in_game_tested=False)
    (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--verify-existing', action='store_true', help='Require active pixels, render arrays and offsets to match the fixed reference')
    parser.add_argument('--compile', action='store_true', dest='compile_package')
    parser.add_argument('--no-comparison', action='store_false', dest='comparison',
                        help='Skip the comparison board and its Windows font dependency')
    args = parser.parse_args()
    build(args.output, args.verify_existing, args.compile_package, comparison=args.comparison)

