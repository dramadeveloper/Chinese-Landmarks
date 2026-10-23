"""Gate-specific surface normals and glazing; all tracing/lighting is shared."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_core import DEFAULT_SCENE, trace_model, shade, indexed, stone_ground, finish
from export_plaza import RAMPS as PROP_RAMPS
from gate_model import (CENTER, YAW, ROOF, ARCH_HEIGHT, HALF_DEPTH, BODY_ENLARGEMENT, GLASS, SPANDREL, METAL,
                        PODIUM, ENTRY, PROP_OFFSET, envelope, opening)

RAMPS = {
    1: [8,10,11,12],
    GLASS: [16,17,156,157,158,159,160,23,13],
    SPANDREL: [5,6,7,8,9,10,11,12],
    METAL: [6,7,8,9,10,11,12,13],
    PODIUM: [16,155,156,157,158,159],
    ENTRY: [1,2,3,4,5],
    7: [5,6,7,8,9,10,11,12,13],  # Aluminum ledges.
    8: [16,16,17,155,156,157,158,159,21],  # Side/inner glazing.
    **{key+PROP_OFFSET:ramp for key,ramp in PROP_RAMPS.items()},
}


def original_xy(x,y,view):
    if view == 0:
        return x,y
    if view == 1:
        return y,120-x
    if view == 2:
        return 160-x,120-y
    return 160-y,x


def rotate_normals(n,view):
    nx,ny = n[:,0].copy(),n[:,1].copy()
    n[:,0] = nx*np.cos(YAW)-ny*np.sin(YAW)
    n[:,1] = nx*np.sin(YAW)+ny*np.cos(YAW)
    for _ in range(view):
        n[:,0],n[:,1] = -n[:,1].copy(),n[:,0].copy()
    return n


def render(model,view,*,lighting=DEFAULT_SCENE):
    nx = model.shape[view%2]
    origin = (round(nx*.8)+16,180)
    frame = trace_model(model,view,(256,308),origin)
    x,y = original_xy(frame.point[:,:,0],frame.point[:,:,1],view)
    wx,wy = x-CENTER[0],y-CENTER[1]
    x = wx*np.cos(YAW)+wy*np.sin(YAW)+CENTER[0]
    y = -wx*np.sin(YAW)+wy*np.cos(YAW)+CENTER[1]
    z = frame.point[:,:,2]
    dx,dy = x-CENTER[0],y-CENTER[1]
    width,depth = envelope(z)
    shell = frame.hit & np.isin(frame.mat,[GLASS,SPANDREL])
    normals = frame.normal.copy()
    # Select the nearest analytic boundary of the rounded tower envelope/void.
    distances = np.stack((abs(abs(dx)-width),abs(abs(dy)-depth),
                          np.where(z<ARCH_HEIGHT,abs(abs(dx)-opening(z)),1000)),axis=2)
    boundary = np.argmin(distances,axis=2)
    n = np.zeros((*z.shape,3),float)
    n[:,:,0] = np.where(boundary==0,np.sign(dx),0)
    n[:,:,1] = np.where(boundary==1,np.sign(dy),0)
    roll = 18 * BODY_ENLARGEMENT
    roof_t = np.clip((z-(ROOF-roll))/roll,0,.999)
    slope = HALF_DEPTH*.86*roof_t/(roll*np.sqrt(1-.86*roof_t**2))
    n[:,:,2] = np.where(boundary==1,slope,0)
    inner = boundary==2
    n[:,:,0][inner] = -np.sign(dx[inner])
    dz = .25
    n[:,:,2][inner] = (opening(z[inner]+dz)-opening(z[inner]-dz))/(2*dz)
    n /= np.linalg.norm(n,axis=2,keepdims=True)
    normals[shell] = rotate_normals(n[shell],view)
    shadows = frame.shadow.copy()
    # Numerical staircase self-hits on outward convex faces are not real shade.
    shadows[shell & (boundary!=2)] = False
    light = shade(normals,shadows,lighting.standard)
    # Blue-cyan glazing strengthens the day-photo sky reflection, while central
    # arch panels and the aluminum-clad horizontal ledges retain neutral grey.
    central = shell & (frame.mat==SPANDREL) & (boundary==1)
    glass = shell & ~central
    side = glass & (boundary!=1)
    # Project texture coordinates back to the continuous facade plane. Rotated
    # voxel stairs otherwise turn horizontal ledges into disconnected speckles.
    vx,vy = ((1,1),(1,-1),(-1,-1),(-1,1))[view]
    vx,vy = vx*np.cos(YAW)+vy*np.sin(YAW),-vx*np.sin(YAW)+vy*np.cos(YAW)
    correction = np.zeros_like(z)
    correction[boundary==0] = ((np.sign(dx)*width-dx)/vx)[boundary==0]
    correction[boundary==1] = ((np.sign(dy)*depth-dy)/vy)[boundary==1]
    texture_z = z+correction
    facade_u = np.where(side,y+vy*correction,x+vx*correction)
    vertical = np.mod(facade_u,6.6)<.8
    floors = np.mod(texture_z-20,3.5)<1.1
    belt = glass & ((abs(z-75.6)<1)|(abs(z-145.2)<1)|(abs(z-208.8)<1))
    podium = frame.hit & (frame.mat==PODIUM)
    material = frame.mat.copy()
    material[glass] = GLASS
    material[side] = 8
    material[side & floors] = 7
    material[belt & side] = SPANDREL
    material[podium & (np.mod(z,4.7)<.8)] = 7
    building = indexed(material,light,frame.hit,RAMPS)
    # Encode window joints as one actual palette step. A fractional light change
    # can round to the same index and erase an entire facade's texture.
    for mat in (GLASS,SPANDREL,8):
        mask = shell & (material==mat)
        joints = (vertical | floors | belt) if mat!=SPANDREL else floors
        ramp = np.array(RAMPS[mat],np.uint8)
        level = np.rint(np.clip(light[mask],0,1)*(len(ramp)-1)).astype(int)
        level = np.maximum(level-joints[mask],0)
        building[mask] = ramp[level]
    return finish(frame,stone_ground(frame,lighting),building)
