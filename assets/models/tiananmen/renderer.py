"""Tiananmen roof/river adapter; shared projection, lighting and shadow pipeline."""
from pathlib import Path
import json
import numpy as np
from render_core import trace_model, shade, finish, DEFAULT_SCENE
from plaza_material import SCALE, paving
from render_core import indexed as index_material
RAMPS={1:[4,5,6,7,8,9,10,11],2:[178,179,180,181,162,163,164],
 3:[60,61,62,63,64,65,66,67],4:[178,179,180,181,162,163],
 5:[96,97,98,99,100,101,102],6:[1,2,3,4],7:[154,155,156,157,158,159,160],
 8:[88,89,90,91,92,93,94],9:[88,89,90,91,92,93],10:[32,33,34,35,36],
 11:[7,8,9,10,11,12,13,14],12:[114,115,116,117,118,120,121],13:[16,17,18,19,20]}

def indexed(material, light, mask):
 return index_material(material, light, mask, RAMPS)
ROOT=Path(__file__).resolve().parent
W,H=416,288

def render(model,view, *, lighting=DEFAULT_SCENE):
 nx = model.shape[view % 2]
 origin = (round(nx * SCALE) + 16, 48)
 frame = trace_model(model, view, (W, H), origin)
 point, xyz, hit, mat = frame.point, frame.xyz, frame.hit, frame.mat
 face, normal = frame.face, frame.normal
 gx, gy, gm = frame.gx, frame.gy, frame.gm
 gi, gj, gmat = frame.gi, frame.gj, frame.gmat
 # Surface patterns remain attached to the model when rotated.
 x,y,z=point[:,:,0].copy(),point[:,:,1].copy(),point[:,:,2]
 if view==1: x,y=y,model.shape[1]-x
 elif view==2: x,y=model.shape[0]-x,model.shape[1]-y
 elif view==3: x,y=model.shape[0]-y,x
 # Map surfaces back through the uniform model transform for roof normals.
 x,y,z=(x-4)/1.3,(y-2)/1.3,z/1.3
 roofmask=np.zeros((H,W),bool)
 for roof in json.loads((ROOT/'geometry.json').read_text())['roofs']:
  cx,cy,hx,hy,base,rise,ridge=(roof[k] for k in ['cx','cy','hx','hy','base','rise','ridge'])
  dx=x-cx-.5; dy=y-cy-.5
  m=hit&(mat==3)&(abs(dx)<=hx+1)&(abs(dy)<=hy+1)&(z>=base)&(z<=base+rise+1)
  roofmask|=m
  slopey=1.45*rise/hy*np.maximum(0,1-abs(dy)/hy)**.45*np.sign(dy)
  slopex=rise/max(1,hx-ridge)*np.sign(dx)
  side=(abs(dx)>ridge+2)&((abs(dx)-ridge)/(hx-ridge)>abs(dy)/hy)
  n=np.stack([np.where(side,slopex,0),np.where(side,0,slopey),np.ones((H,W))],axis=2)
  for _ in range(view): n=np.stack([-n[:,:,1],n[:,:,0],n[:,:,2]],axis=2)
  n/=np.linalg.norm(n,axis=2)[:,:,None]; normal[m]=n[m]
 light = shade(normal, frame.shadow, lighting.standard)
 # Narrow tile ribs; variation is organized by surface, never image-wide noise.
 ribs=(np.floor(x).astype(int)%3==0)
 light[roofmask&ribs]-=.10
 leaf=(mat==8)|(mat==9)
 grain=((xyz[:,:,0]*17+xyz[:,:,1]*29+xyz[:,:,2]*11)%13-6)/75
 light[leaf]+=grain[leaf]
 building=indexed(mat,light,hit)
 gshadow = frame.gshadow
 pattern=(((gi//2)*17+(gj//2)*31)%19-9)/1000
 glight=np.full((H,W),.73)+pattern
 joints=(gj%6==0)|((gi+(gj//6%2)*4)%9==0)
 glight[joints&(gmat==1)]-=.08
 water=gmat==7
 glight[water]=.57+pattern[water]*2
 ripple=water&((gi+gj*5)%23<3); glight[ripple]+=.16
 glight[gshadow]-=lighting.ground_darkening
 ground=indexed(gmat,glight,gm)
 paved=gm&(gmat==1)
 ground[paved]=paving(gx,gy,gshadow,shadow_colors=lighting.paving_shadow)[paved]
 return finish(frame, ground, building)
