"""Build current model adapters through the shared renderer and sprite pipeline."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plaza_material import PALETTE, save_p, validate_png
from sprite_pipeline import (RESULT_KEYS, crop, save_render, write_sprite, export_tiles,
                             purchase_pixels, compile_grf)
import export_plaza
from render_core import SUN_VECTOR, DEFAULT_SCENE

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
PACKAGE_SOURCE = ASSETS / 'package'
REFERENCE = ASSETS / 'reference'
DEFAULT_OUTPUT = ASSETS.parent / 'build/current'


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
          'world_graphics', (('low', 1, 160, 0), ('high', 160, 310, 127)),
          menu=Menu(scale=.5, baseline=32, filename='purchase-view{view}.png')),
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
    Asset('china-art-museum', 'models/china-art-museum', 'model.npz',
          'china-art-museum', 'cam_', 'cam_tiles', 'china-art-museum.nml', 'cam_world'),
    Asset('zhencheng-lou', 'models/zhencheng-lou', 'model.npz',
          'zhencheng-lou', 'zcl_', 'zcl_tiles', 'zhencheng-lou.nml', 'zcl_world'),
    Asset('harbin-saint-sophia', 'models/harbin-saint-sophia', 'model.npz',
          'harbin-saint-sophia', 'sophia_', 'sophia_tiles', 'harbin-saint-sophia.nml', 'sophia_world'),
    Asset('beijing-national-stadium', 'models/beijing-national-stadium', 'model.npz',
          'beijing-national-stadium', 'bns_', 'bns_tiles', 'beijing-national-stadium.nml', 'bns_world'),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_metadata():
    """Read the project's version expression from its canonical NML header."""
    header = (PACKAGE_SOURCE / 'grf.nml').read_text(encoding='utf-8-sig')
    major, minor, patch = map(int, re.search(
        r'^\s*version: \((\d+) << 16\) \| \((\d+) << 8\) \| (\d+);', header, re.M).groups())
    return dict(project_version=f'{major}.{minor}.{patch}',
                grfid=re.search(r'^\s*grfid: "([^"]+)";', header, re.M).group(1),
                grf_version=(major << 16) | (minor << 8) | patch,
                min_compatible_version=int(re.search(
                    r'^\s*min_compatible_version: (\d+);', header, re.M).group(1)))


def adapter(asset):
    return load_renderer(ASSETS / asset.source)


def load_renderer(source):
    spec = importlib.util.spec_from_file_location('current_' + source.name, source / 'renderer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_package(output):
    package = output / 'grf'
    package.mkdir(parents=True, exist_ok=True)
    fragments = ['grf.nml', 'plaza.nml', *(asset.fragment for asset in CURRENT)]
    for name in fragments:
        shutil.copy2(PACKAGE_SOURCE / name, package / name)
    shutil.copytree(PACKAGE_SOURCE / 'lang', package / 'lang', dirs_exist_ok=True)
    shutil.copy2(PACKAGE_SOURCE / 'LICENSE', package / 'LICENSE')
    combined = '\n\n'.join((package / name).read_text(encoding='utf-8-sig') for name in fragments)
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


def landmark_titles():
    language = (PACKAGE_SOURCE / 'lang/simplified_chinese.lng').read_text(encoding='utf-8-sig')
    labels = dict(re.findall(r'^(STR_\w+)\s*:(.*)$', language, re.M))
    titles = {}
    for asset in CURRENT:
        definition = (PACKAGE_SOURCE / asset.fragment).read_text(encoding='utf-8-sig')
        key = re.search(r'\bname:\s*string\((\w+)\)', definition).group(1)
        titles[asset.folder] = labels[key].strip()
    return titles


def native_picture(pixels, origin, tiles):
    image = Image.fromarray(np.dstack((PALETTE[pixels], (pixels != 0).astype(np.uint8) * 255)))
    box = image.getbbox()
    baseline = int(origin[1] + 8 * sum(tiles)) - box[1]
    return image.crop(box), baseline


def native_board(items):
    """Keep one sprite pixel per screen pixel and align the projected tile centres."""
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    widths = [max(im.width, round(font.getlength(title))) + 20 for im, _, title in items]
    baseline = 12 + max(anchor for _, anchor, _ in items)
    bottom = baseline + max(im.height - anchor for im, anchor, _ in items)
    board = Image.new('RGB', (sum(widths), bottom + 38), '#22313b')
    draw = ImageDraw.Draw(board)
    draw.line((6, baseline, board.width - 6, baseline), fill='#40525c')
    left = 0
    for (im, anchor, title), width in zip(items, widths):
        board.paste(im, (left + (width - im.width) // 2, baseline - anchor), im)
        draw.text((left + width // 2, bottom + 8), title, font=font, fill='#edf1ed', anchor='mt')
        left += width
    return board


def comparison_board(output):
    items = []
    titles = landmark_titles()
    for asset in CURRENT:
        with np.load(output / asset.name / 'render-view0.npz') as view:
            picture = native_picture(view['composed'], view['origin'], view['tiles'])
        items.append((*picture, titles[asset.folder]))
    for kind, title in [('walkers', '散步人群'), ('meeting', '结伴人群'), ('bench', '长椅'), ('tree', '树池')]:
        with Image.open(output / 'plaza' / f'{kind}-v0.png') as prop:
            picture = native_picture(np.asarray(prop), export_plaza.ORIGIN, (1, 1))
        items.append((*picture, title))
    board = native_board(items)
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
    report = dict(**package_metadata(), renderer='render_core.py',
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

