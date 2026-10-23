"""Shared render cache, tiled sprites, NML graphics and compilation."""
from pathlib import Path
import math
import subprocess
import numpy as np
from PIL import Image
from plaza_material import SCALE, TILE, save_p, validate_png

RESULT_KEYS = ('ground', 'go', 'building', 'bo', 'composed', 'xyz', 'hit', 'origin', 'tiles')


def crop(array, mask, anchor):
    yy, xx = np.nonzero(mask)
    # Empty tiles are real NML sprites, represented by one transparent pixel.
    if not len(xx):
        return np.zeros((1, 1), np.uint8), (0, 0), (0, 0, 1, 1)
    l, t, r, b = map(int, (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1))
    return np.where(mask, array, 0)[t:b, l:r], (l - anchor[0], t - anchor[1]), (l, t, r, b)


def tile_anchor(origin, tx, ty):
    return (origin[0] + round(TILE * SCALE) * (ty - tx),
            origin[1] + round(TILE * SCALE / 2) * (tx + ty))


def save_render(arrays, directory, view):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f'render-view{view}.npz', **dict(zip(RESULT_KEYS, arrays)))
    path = directory / f'view{view}-1x-8bpp.png'
    save_p(arrays[4], path)
    validate_png(path)


def write_sprite(output, path, name, pixels, offset):
    save_p(pixels, Path(output) / path)
    validate_png(Path(output) / path)
    height, width = pixels.shape
    return [f'spriteset({name}, "{path}") {{',
            f'    [0, 0, {width}, {height}, {offset[0]}, {offset[1]}]', '}']


def export_tiles(output, folder, arrays, view, prefix, tile_switch, bands=()):
    """Export ground/building slices and one relative-position switch.

    bands contains (label, model_z_min, model_z_max, NML_zoffset), only for
    objects exceeding a single NML height volume (the current tower).
    Object IDs, introduction dates and package versions remain package data.
    """
    ground, go, building, bo, composed, xyz, hit, origin, tiles = arrays
    nx, ny = tiles
    lines, records = [], []
    for ty in range(ny):
        for tx in range(nx):
            tile = tx + nx * ty
            anchor = tile_anchor(origin, tx, ty)
            suffix = f'v{view}_{tx}_{ty}'
            layers = [('g', ground, (ground != 0) & (go == tile), 0, 0)]
            if bands:
                for label, zmin, zmax, zoffset in bands:
                    mask = hit & (bo == tile) & (xyz[:, :, 2] >= zmin) & (xyz[:, :, 2] < zmax)
                    if mask.any():
                        top = math.ceil((int(xyz[:, :, 2][mask].max()) + 1) * SCALE)
                        layers.append((label, building, mask, zoffset, top - zoffset))
            else:
                mask = hit & (bo == tile)
                extent = math.ceil((int(xyz[:, :, 2][mask].max()) + 1) * SCALE) if mask.any() else 1
                layers.append(('b', building, (building != 0) & (bo == tile), 0, extent))
            blocks = []
            for kind, array, mask, zoffset, extent in layers:
                filename = f'g_{suffix}' if kind == 'g' else f'b_{suffix}' + (f'_{kind}' if bands else '')
                pixels, offset, rectangle = crop(array, mask, anchor)
                xoff, yoff = offset
                if mask.any():
                    yoff += zoffset
                path = f'sprites/{folder}/{filename}.png'
                symbol = prefix + filename
                lines += write_sprite(output, path, symbol, pixels, (xoff, yoff))
                height, width = pixels.shape
                records.append(dict(view=view, tile=[tx, ty], kind=kind, file=path,
                                    width=width, height=height, xoffset=xoff, yoffset=yoff,
                                    zoffset=zoffset, zextent=extent, source_rectangle=list(rectangle)))
                if kind == 'g':
                    blocks.append(f'    ground {{ sprite: {symbol}; }}')
                else:
                    assert -128 <= zoffset <= 127 and 1 <= extent <= 255
                    if mask.any():
                        assert -32 <= xoff and xoff + width <= 32 and yoff >= -200
                    elevation = f' zoffset: {zoffset};' if bands else ''
                    blocks.append(f'    building {{ sprite: {symbol};{elevation} zextent: {extent}; }}')
            lines += [f'spritelayout {prefix}layout_{suffix} {{', *blocks, '}']
    # Check exported PNG pixels and offsets, including the tower's raised anchor.
    assert np.array_equal(reassemble(output, records, origin, composed.shape), composed)
    lines.append(f'switch (FEAT_OBJECTS, SELF, {tile_switch}_{view}, relative_pos) {{')
    for ty in range(ny):
        for tx in range(nx):
            lines.append(f'    relative_coord({tx},{ty}): {prefix}layout_v{view}_{tx}_{ty};')
    lines += [f'    {prefix}layout_v{view}_0_0;', '}']
    return lines, records


def reassemble(output, records, origin, shape):
    assembled = np.zeros(shape, np.uint8)
    for ground in (True, False):
        for rec in records:
            if (rec['kind'] == 'g') != ground:
                continue
            pixels = np.array(Image.open(Path(output) / rec['file']))
            if not pixels.any():
                continue
            tx, ty = rec['tile']
            ax, ay = tile_anchor(origin, tx, ty)
            x, y = ax + rec['xoffset'], ay - rec['zoffset'] + rec['yoffset']
            target = assembled[y:y + rec['height'], x:x + rec['width']]
            target[pixels != 0] = pixels[pixels != 0]
    return assembled


def purchase_pixels(composed, scale):
    im = Image.fromarray(composed)
    im = im.crop(im.getbbox())
    return np.array(im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.NEAREST))


def compile_grf(directory):
    run = subprocess.run(
        ['nmlc', '--no-cache', '--palette=DEFAULT', '--grf=oriental-pearl.grf',
         '--nfo=oriental-pearl.nfo', '--md5=oriental-pearl.md5', 'oriental-pearl.nml'],
        cwd=directory, text=True, capture_output=True)
    (Path(directory) / 'compile.log').write_text(run.stdout + run.stderr, encoding='utf-8')
    run.check_returncode()
    return run

