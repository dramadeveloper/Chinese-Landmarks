"""Render a saved landmark in isolation, including drafts without a package ID."""
import argparse

import numpy as np

import export_plaza
from render_assets import (ASSETS, CURRENT, DEFAULT_OUTPUT, landmark_titles, load_renderer,
                           native_board, native_picture, sha)


PACKAGE_MODELS = {asset.folder: asset for asset in CURRENT}


def pictures(folder, directions):
    source = ASSETS / 'models' / folder
    render = load_renderer(source).render
    asset = PACKAGE_MODELS.get(folder)
    keys = asset.model_keys if asset else ('model',)
    with np.load(source / 'model.npz') as saved:
        inputs = {key: saved[key] for key in keys}
    items = []
    for direction in directions:
        result = render(view=direction, **inputs)
        tiles = result[8]
        items.append((*native_picture(result[4], result[7], tiles), f'{tiles[0]}×{tiles[1]}'))
    return items


def compare_landmarks(folder, first_view, references, path):
    titles = landmark_titles()
    items = [(*first_view, titles.get(folder, folder))]
    evidence = []
    for name in dict.fromkeys(references):
        if name == folder:
            continue
        image, anchor, _ = pictures(name, (0,))[0]
        items.append((image, anchor, titles[name]))
        asset = PACKAGE_MODELS[name]
        model = ASSETS / asset.source / asset.model
        evidence.append(dict(name=asset.name, source=model.relative_to(ASSETS.parent).as_posix(),
                             sha256=sha(model)))
    for kind, title in [('walkers', '公共人物'), ('bench', '长椅'), ('tree', '小树')]:
        ground, building = export_plaza.render(export_plaza.make_model(kind), kind, 0)
        pixels = np.where(building != 0, building, ground)
        items.append((*native_picture(pixels, export_plaza.ORIGIN, (1, 1)), title))
    native_board(items).save(path)
    return evidence


def preview(folder, references):
    asset = PACKAGE_MODELS.get(folder)
    out = DEFAULT_OUTPUT / (asset.name if asset else folder)
    out.mkdir(parents=True, exist_ok=True)
    views = pictures(folder, range(4))
    items = [(im, anchor, f'方向 {i} · {size}') for i, (im, anchor, size) in enumerate(views)]
    native_board(items).save(out / 'four-views-1x.png')
    compare_landmarks(folder, views[0][:2], references, out / 'comparison-1x.png')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(p.parent.name for p in (ASSETS / 'models').glob('*/model.npz')))
    parser.add_argument('--compare', nargs='+', choices=[asset.folder for asset in CURRENT],
                        default=['oriental-pearl', 'tiananmen'], metavar='LANDMARK')
    args = parser.parse_args()
    preview(args.model, args.compare)
