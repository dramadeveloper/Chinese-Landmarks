"""Oriental Pearl surface adapter; projection and lighting live in render_core."""
from pathlib import Path
import json, math, sys
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'chinese-landmarks'))
from plaza_material import paving
from render_core import trace_model, shade, indexed, finish, DEFAULT_SCENE, PEARL_SAMPLING
W, H = 192, 336
ORIGIN = (96, 240)

def render(model, view, *, core, factor=1.3, lighting=DEFAULT_SCENE):
    """Pixel-centre orthographic export, material ramps and direct-light shadows."""
    frame = trace_model(model, view, (W, H), ORIGIN, PEARL_SAMPLING)
    acore = np.rot90(core, view, axes=(0, 1))
    point, xyz, hit, mat = frame.point, frame.xyz, frame.hit, frame.mat
    normal = frame.normal
    gx, gy, gm = frame.gx, frame.gy, frame.gm
    gi, gj, gmat = frame.gi, frame.gj, frame.gmat
    geometry=json.loads((ROOT/'geometry.json').read_text())
    # Inverse geometry transform for attached normals/panels only.
    # Map the translated 80x80 cropped model back to the 160x160 source
    # in each rotated frame. This preserves all existing analytic normals.
    crop_offsets = [(40,40,0)] * 4
    shape_point = (point + crop_offsets[view] - [80,80,0]) / factor + [80,80,0]
    core_hit = hit & (acore[xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]] != 0)
    def rotate(x,y):
        for _ in range(view): x,y=159-y,x
        return x,y
    # Cylindrical tubes, including continuous inclined struts.
    for j,(cx0,cy0) in enumerate(geometry['columns']):
        cx,cy=rotate(cx0,cy0)
        d=shape_point-[cx+.5,cy+.5,0]
        length=np.linalg.norm(d[:,:,:2],axis=2)
        m=core_hit&(mat==1)&(length<3.3)&(shape_point[:,:,2]<143)
        radial=d.copy(); radial[:,:,2]=0
        radial/=np.maximum(length[:,:,None],.01)
        normal[m]=radial[m]
        deg=[90,210,330][j]*math.pi/180
        fx0,fy0=80+round(25*math.cos(deg)),80+round(25*math.sin(deg))
        fx,fy=rotate(fx0,fy0)
        start=np.array([fx+.5,fy+.5,3.5])
        axis=np.array([cx-fx,cy-fy,34.])
        t=np.clip(((shape_point-start)*axis).sum(axis=2)/(axis@axis),0,1)
        radial=shape_point-(start+t[:,:,None]*axis)
        length=np.linalg.norm(radial,axis=2)
        m=core_hit&(mat==1)&(length<3.3)&(shape_point[:,:,2]<39)
        normal[m]=(radial/np.maximum(length[:,:,None],.01))[m]

    pearl_regions=[]
    for pearl in geometry['pearls']:
        cx,cy,cz=pearl['center']; cx,cy=rotate(cx,cy)
        r,rz=pearl['r'],pearl['rz']
        d=shape_point-[cx+.5,cy+.5,cz+.5]
        distance=(d[:,:,0]**2+d[:,:,1]**2)/(r+.9)**2+d[:,:,2]**2/(rz+.9)**2
        m=core_hit&(distance<=1.15)
        radial=d/np.array([r*r,r*r,rz*rz])
        radial/=np.maximum(np.linalg.norm(radial,axis=2)[:,:,None],.01)
        normal[m]=radial[m]
        phi=np.arctan2(d[:,:,1],d[:,:,0])-view*math.pi/2
        lat=np.arcsin(np.clip(d[:,:,2]/(rz+.5),-1,1))
        pearl_regions.append((m,phi,lat))
    light = shade(normal, frame.shadow, lighting.pearl)
    paint=mat.copy()
    for region,phi,lat in pearl_regions:
        # Surface-attached diagonal panel joints and narrow window mullions.
        phase1=(phi*8/math.pi+lat*7/math.pi)%1
        phase2=(phi*8/math.pi-lat*7/math.pi)%1
        seam=region&(mat==2)&((phase1<.10)|(phase2<.10))
        light[seam]+= .12
        mullion=region&(mat==3)&((phi*14/math.pi)%1<.14)
        paint[mullion]=6; light[mullion]=.70
    # Small pixel clusters on foliage; no generic image-wide noise.
    grain=((xyz[:,:,0]*17+xyz[:,:,1]*29+xyz[:,:,2]*11)%13-6)/65
    leaf=(mat==4)|(mat==7)
    light[leaf]+=grain[leaf]
    lamp=mat==10
    light[lamp]=1
    # Deliberate material-specific TTD ramps.
    ramps={1:[3,5,7,9,11,13,14],2:[40,42,43,44,45,46,48],
           3:[1,16,17,18,19,20],4:[88,89,90,91,92,93],
           5:[4,5,6,7,8,9,10],6:[16,17,18,19,20,21,22,23],
           7:[88,89,90,91,92,93,94],8:[32,33,34,35,36,37],
           9:[40,41,42,43,44],10:[13],11:[32,33,34,35]}
    building = indexed(paint, light, hit, ramps)
    gshadow = frame.gshadow
    pattern=(((gi//2)*17+(gj//2)*31)%19-9)/240
    glight=np.full((H,W),.72)+pattern
    glight[(gmat==7)|(gmat==4)]=.58+pattern[(gmat==7)|(gmat==4)]*2
    glight[gshadow]-=lighting.ground_darkening
    ground=indexed(gmat,glight,gm,ramps)
    paved = gm & ((gmat==5)|(gmat==8))
    ground[paved] = paving(gx,gy,gshadow,shadow_colors=lighting.paving_shadow)[paved]
    return finish(frame, ground, building)
