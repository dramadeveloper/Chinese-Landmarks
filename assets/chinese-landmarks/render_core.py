"""Shared native-1x voxel renderer. Adapters supply geometry/material detail."""
from dataclasses import dataclass
import numpy as np
from plaza_material import SCALE, TILE, SHADOW, SHADOW_JOINT, paving


# One screen-space light for diffuse shading AND cast-shadow rays.
SUN_VECTOR = np.array([.5, 1.5, 2.5])
SUN = SUN_VECTOR / np.linalg.norm(SUN_VECTOR)
SHADOW_RAY = SUN_VECTOR / SUN_VECTOR[2]


@dataclass(frozen=True)
class Lighting:
    ambient: float
    direct: float
    shadow_retention: float


# Model-specific lighting preserves the existing pixels. New models use STANDARD.
STANDARD = Lighting(.33, .67, .25)
PEARL_EXISTING = Lighting(.28, .72, .22)
PALE_STONE_EXISTING = Lighting(.50, .50, .70)


@dataclass(frozen=True)
class SceneLighting:
    """One explicit lighting choice shared by all landmarks, props and paving."""
    standard: Lighting
    pearl: Lighting
    pale_stone: Lighting
    ground_darkening: float
    paving_shadow: tuple


# Shared lighting configuration for the current models.
DEFAULT_SCENE = SceneLighting(
    STANDARD, PEARL_EXISTING, PALE_STONE_EXISTING, .30, (SHADOW, SHADOW_JOINT))


@dataclass(frozen=True)
class Sampling:
    building_z: int = 1
    surface_bias: float = .06
    surface_shadow_step: float = 1.0
    ground_shadow_step: float = .5
    self_shadow: bool = True
    building_step: float = .25


DEFAULT_SAMPLING = Sampling()
PEARL_SAMPLING = Sampling(3, .05, 1.25, .75)
PROP_SAMPLING = Sampling(self_shadow=False)


@dataclass
class Frame:
    model: np.ndarray
    origin: tuple
    tiles: tuple
    gx: np.ndarray
    gy: np.ndarray
    gm: np.ndarray
    gi: np.ndarray
    gj: np.ndarray
    gmat: np.ndarray
    go: np.ndarray
    hit: np.ndarray
    mat: np.ndarray
    xyz: np.ndarray
    point: np.ndarray
    face: np.ndarray
    normal: np.ndarray
    bo: np.ndarray
    shadow: np.ndarray
    gshadow: np.ndarray


def surface_shadow(a, point, cube_normal, hit, sampling):
    nx, ny, nz = a.shape
    shadow = np.zeros(hit.shape, bool)
    if not sampling.self_shadow:
        return shadow
    start = point + cube_normal * sampling.surface_bias
    for t in np.arange(1.25, nz, sampling.surface_shadow_step):
        pos = np.floor(start + SHADOW_RAY * t).astype(int)
        valid = (hit & ~shadow & (pos[:, :, 0] >= 0) & (pos[:, :, 0] < nx)
                 & (pos[:, :, 1] >= 0) & (pos[:, :, 1] < ny) & (pos[:, :, 2] < nz))
        yy, xx = np.nonzero(valid)
        shadow[yy, xx] |= a[pos[yy, xx, 0], pos[yy, xx, 1], pos[yy, xx, 2]] > 0
    return shadow


def ground_shadow(a, gx, gy, gm, sampling):
    nx, ny, nz = a.shape
    shadow = np.zeros(gm.shape, bool)
    for z in np.arange(sampling.building_z + .25, nz, sampling.ground_shadow_step):
        sx = np.floor(gx + .5 + z * SHADOW_RAY[0]).astype(int)
        sy = np.floor(gy + .5 + z * SHADOW_RAY[1]).astype(int)
        valid = gm & ~shadow & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
        yy, xx = np.nonzero(valid)
        shadow[yy, xx] |= a[sx[yy, xx], sy[yy, xx], int(z)] > 0
    return shadow


def trace_model(model, view, canvas, origin, sampling=DEFAULT_SAMPLING):
    """Rotate geometry, sample pixel centres, retain positions and tile ownership.

    canvas is (width, height); origin is the projected model (0,0,0).
    Ground coordinates stay in the world plane when the model rotates.
    """
    a = np.rot90(model, view, axes=(0, 1)).copy()
    nx, ny, nz = a.shape
    ntx, nty = nx // TILE, ny // TILE
    width, height = canvas
    py, px = np.indices((height, width), dtype=float)
    u, v = (px + .5 - origin[0]) / SCALE, (py + .5 - origin[1]) / SCALE
    gx = np.floor(v - u * .5).astype(int)
    gy = np.floor(v + u * .5).astype(int)
    gm = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gi, gj = np.clip(gx, 0, nx - 1), np.clip(gy, 0, ny - 1)
    gmat = a[gi, gj, 0].copy()
    for z in range(1, sampling.building_z):
        gmat = np.where(a[gi, gj, z] > 0, a[gi, gj, z], gmat)
    go = np.where(gm, gi // TILE + ntx * (gj // TILE), -1)
    hit = np.zeros(gm.shape, bool)
    mat = np.zeros(gm.shape, np.uint8)
    xyz = np.zeros((*gm.shape, 3), np.int16)
    point = np.zeros((*gm.shape, 3), float)
    face = np.zeros(gm.shape, np.uint8)
    for z in np.arange(nz - .25, sampling.building_z - .01, -sampling.building_step):
        ix = np.floor(z + v - u * .5).astype(int)
        iy = np.floor(z + v + u * .5).astype(int)
        yy, xx = np.nonzero(~hit & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny))
        c = a[ix[yy, xx], iy[yy, xx], int(z)]
        take = c > 0
        yy, xx, c = yy[take], xx[take], c[take]
        hit[yy, xx] = True
        mat[yy, xx] = c
        xyz[yy, xx] = np.stack([ix[yy, xx], iy[yy, xx], np.full(len(xx), int(z))], axis=1)
        times = np.stack([ix[yy, xx] + 1 - v[yy, xx] + u[yy, xx] * .5,
                          iy[yy, xx] + 1 - v[yy, xx] - u[yy, xx] * .5,
                          np.full(len(xx), int(z) + 1)], axis=1)
        face[yy, xx] = times.argmin(axis=1)
        t = times.min(axis=1)
        point[yy, xx] = np.stack([t + v[yy, xx] - u[yy, xx] * .5,
                                  t + v[yy, xx] + u[yy, xx] * .5, t], axis=1)
    normal = np.eye(3)[face]
    bo = xyz[:, :, 0] // TILE + ntx * (xyz[:, :, 1] // TILE)
    shadow = surface_shadow(a, point, normal, hit, sampling)
    gshadow = ground_shadow(a, gx, gy, gm, sampling)
    return Frame(a, tuple(origin), (ntx, nty), gx, gy, gm, gi, gj, gmat, go,
                 hit, mat, xyz, point, face, normal, bo, shadow, gshadow)


def shade(normal, shadow, lighting=STANDARD):
    diffuse = np.clip((normal * SUN).sum(axis=2), 0, 1)
    return lighting.ambient + lighting.direct * diffuse * np.where(shadow, lighting.shadow_retention, 1)


def indexed(material, light, mask, ramps):
    out = np.zeros(mask.shape, np.uint8)
    for material_id, ramp in ramps.items():
        m = mask & (material == material_id)
        index = np.rint(np.clip(light[m], 0, 1) * (len(ramp) - 1)).astype(int)
        out[m] = np.array(ramp, np.uint8)[index]
    return out


def stone_ground(frame, lighting=DEFAULT_SCENE):
    pixels = paving(frame.gx, frame.gy, frame.gshadow, shadow_colors=lighting.paving_shadow)
    return np.where(frame.gm, pixels, 0).astype(np.uint8)


def finish(frame, ground, building):
    """Canonical result used by the cache, previews and sprite exporter."""
    composed = np.where(frame.hit, building, ground).astype(np.uint8)
    return (ground, frame.go, building, frame.bo, composed, frame.xyz,
            frame.hit, frame.origin, frame.tiles)


def render_voxels(model, view, canvas, origin, ramps, lighting=DEFAULT_SCENE):
    """Complete pipeline for voxel-face normals and shared stone ground."""
    frame = trace_model(model, view, canvas, origin)
    building = indexed(frame.mat, shade(frame.normal, frame.shadow, lighting.standard), frame.hit, ramps)
    return finish(frame, stone_ground(frame, lighting), building)

