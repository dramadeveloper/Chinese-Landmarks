"""Temple roof/terrace adapter; shared projection, lighting and shadow pipeline."""
from pathlib import Path
import json
import numpy as np
from render_core import trace_model, shade, finish, DEFAULT_SCENE
from plaza_material import SCALE, paving
from render_core import indexed as index_material
RAMPS={1:[6,7,8,9,10,11,12],2:[178,179,180,181,162,163],
       3:[61,62,63,64,65,66,67],4:[178,179,180,181,162],
       5:[96,97,98,99,100,101],6:[1,2,3,4],7:[10,11,12,13,14,15],
       8:[144,145,146,147],9:[114,115,116,117,118,120,121],10:[16,17,18,19,20],11:[6,7,8,9,10,11,12]}

def indexed(material, light, mask):
 return index_material(material, light, mask, RAMPS)
ROOT=Path(__file__).resolve().parent
W,H=160,192

def render(model,view, *, lighting=DEFAULT_SCENE):
 nx = model.shape[view % 2]
 origin = (round(nx * SCALE) + 16, 64)
 frame = trace_model(model, view, (W, H), origin)
 point, xyz, hit, mat = frame.point, frame.xyz, frame.hit, frame.mat
 face, normal = frame.face, frame.normal
 gx, gy, gm = frame.gx, frame.gy, frame.gm
 gi, gj, gmat = frame.gi, frame.gj, frame.gmat
 # Surface patterns remain attached to the model when rotated.
 x,y,z=point[:,:,0].copy(),point[:,:,1].copy(),point[:,:,2]
 if view==1: x,y=y,80-x
 elif view==2: x,y=80-x,80-y
 elif view==3: x,y=80-y,x
 # Undo the uniform scale after undoing rotation, for original surface normals.
 x,y,z=80.5+(x-40)/1.3,75.5+(y-40)/1.3,z/1.3
 roofmask=np.zeros((H,W),bool)
 dx=x-80.5; dy=y-75.5; radius=np.hypot(dx,dy)
 for roof in json.loads((ROOT/'geometry.json').read_text())['roofs']:
  outer,inner,base,rise=(roof[k] for k in ['outer','inner','base','rise'])
  m=hit&(mat==8)&(z>=base)&(z<=base+rise+1)
  roofmask|=m
  slope=1.45*rise/(outer-inner)*np.clip((outer-radius)/(outer-inner),0,1)**.45
  n=np.stack([slope*dx/np.maximum(radius,1),slope*dy/np.maximum(radius,1),np.ones((H,W))],axis=2)
  for _ in range(view): n=np.stack([-n[:,:,1],n[:,:,0],n[:,:,2]],axis=2)
  n/=np.linalg.norm(n,axis=2)[:,:,None]; normal[m]=n[m]
 # Smooth the circular terrace risers, leaving posts and stairs crisp.
 for terrace in json.loads((ROOT/'geometry.json').read_text())['terraces']:
  rr,base,top=(terrace[k] for k in ['radius','base','riser_top'])
  m=hit&(mat==11)&(face!=2)&(z>=base)&(z<top+1)&(abs(radius-rr)<1.5)
  n=np.stack([dx/np.maximum(radius,1),dy/np.maximum(radius,1),np.zeros((H,W))],axis=2)
  for _ in range(view): n=np.stack([-n[:,:,1],n[:,:,0],n[:,:,2]],axis=2)
  normal[m]=n[m]
 light = shade(normal, frame.shadow, lighting.standard)
 # Pale stone receives reflected light; keep thin rail shadows one shade apart.
 marble=mat==7
 light[marble]=shade(normal,frame.shadow,lighting.pale_stone)[marble]
 # Radial tile ribs follow the architecture through all four rotations.
 phase=np.arctan2(dy,dx)*36/(2*np.pi)
 ribs=(phase-np.floor(phase))<.18
 light[roofmask&ribs]-=.14
 light[roofmask&(np.floor(radius).astype(int)%4==0)]-=.025
 building=indexed(mat,light,hit)
 gshadow = frame.gshadow
 ground=np.where(gm,paving(gi,gj,gshadow,shadow_colors=lighting.paving_shadow),0).astype(np.uint8)
 return finish(frame, ground, building)

