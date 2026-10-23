"""Shared projection/light with cylindrical stone and calm inlaid paving."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_core import trace_model, shade, indexed, stone_ground, finish, DEFAULT_SCENE, Sampling
from export_plaza import RAMPS as PROP_RAMPS
from xinghai_model import (STONE, RELIEF, RED_PAVING, GOLD_PAVING, GRASS, BRONZE,
                          GOLD, LAMP_GLASS, DARK_STONE, LAMP_FRAME, LAMP_STONE,
                          PROP_OFFSET, LAMP_CENTERS, HUABIAO_ENLARGEMENT, huabiao_z)

RAMPS = {
    1: [7,8,9,10,11,12],
    STONE: [7,9,11,12,13,14,15],
    RELIEF: [7,9,10,11,12,13,14],
    RED_PAVING: [71,72,73,74,75,76,77],
    GOLD_PAVING: [54,55,56,57,58,59,38,39],
    GRASS: [88,89,90,91,92],
    BRONZE: [1,2,3,4,5,6],
    GOLD: [55,56,57,58,59,38,39],
    LAMP_GLASS: [57,58,59,38,39],
    DARK_STONE: [5,6,7,8,9,10],
    LAMP_FRAME: [1,96,96,97,97,98,98],
    LAMP_STONE: [4,5,7,9,11,14,15],
    **{key+PROP_OFFSET:ramp for key,ramp in PROP_RAMPS.items()},
}
SAMPLING = Sampling(building_step=.125)


def render(model, view, *, lighting=DEFAULT_SCENE, canvas=(352,232), origin=(176,48)):
    frame = trace_model(model, view, canvas, origin, SAMPLING)
    x,y,z = frame.point.transpose(2,0,1)
    normal = frame.normal.copy()
    # A sampled ray can first land just past the edge of a coplanar voxel.
    # Its interior side is not a physical face of the continuous plaza slab.
    # Correct only these confirmed shared faces, keeping the real stair risers.
    ix,iy,iz = frame.xyz.transpose(2,0,1)
    plane = frame.hit & (iz<=3) & (frame.face!=2)
    py,px = np.nonzero(plane)
    adjacent = frame.model[ix[py,px]+(frame.face[py,px]==0),
                           iy[py,px]+(frame.face[py,px]==1),iz[py,px]]
    normal[py[adjacent!=0],px[adjacent!=0]] = [0,0,1]
    centers = [(100,100,huabiao_z(7),huabiao_z(40),6*HUABIAO_ENLARGEMENT),
               *((cx,cy,8,26,3) for cx,cy in LAMP_CENTERS)]
    # Rotate column centres with the model; sun stays in screen coordinates.
    for cx,cy,z0,z1,radius in centers:
        for _ in range(view):
            cx,cy = 200-cy,cx
        dx,dy = x-cx,y-cy
        distance = np.hypot(dx,dy)
        curved = (frame.hit & np.isin(frame.mat,(STONE,RELIEF,LAMP_STONE)) & (frame.face!=2)
                  & (distance<radius) & (z>=z0) & (z<z1))
        normal[:,:,0][curved] = dx[curved]/distance[curved]
        normal[:,:,1][curved] = dy[curved]/distance[curved]
        normal[:,:,2][curved] = 0
    light = shade(normal, frame.shadow, lighting.standard)
    # World-aligned slab joints; spacing stays in public visual units.
    top = frame.hit & (normal[:,:,2]==1)
    inlay = top & np.isin(frame.mat,(RED_PAVING,GOLD_PAVING,DARK_STONE))
    row = np.floor(y/10)
    joints = (np.mod(y,10)<.8) | (np.mod(x+(row%2)*10,20)<.8)
    light[inlay & joints] -= .18
    # Concentric bronze inlays suggest the zodiac/solar-term register at 1x.
    r = np.hypot(x-100,y-100)
    rings = top & (frame.mat==RED_PAVING) & ((abs(r-11.5)<.45)|(abs(r-17.3)<.45))
    light[rings] -= .15
    sectors = top & (frame.mat==RED_PAVING) & (r>11.5) & (r<17.3)
    angle = np.arctan2(y-100,x-100)
    ticks = abs(np.sin(angle*12)) < .07
    light[sectors & ticks] -= .15
    building = indexed(frame.mat, light, frame.hit, RAMPS)
    return finish(frame, stone_ground(frame,lighting), building)
