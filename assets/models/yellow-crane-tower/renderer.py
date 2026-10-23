"""Yellow Crane Tower roof normals; all lighting/projection/shadows are shared."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_core import trace_model, shade, indexed, stone_ground, finish, DEFAULT_SCENE, Sampling
from export_plaza import RAMPS as PROP_RAMPS
from tower_model import (ART_SCALE, PLINTH, ROOFS, GROUND, STONE, RED, DARK,
                         TILE, RIDGE, PAINT, GOLD, WOOD, PLAQUE, LANTERN, PLAQUE_TEXT,
                         PROP_OFFSET, roof_height, outline)

RAMPS = {GROUND:[7,8,9,10,11,12], STONE:[7,8,9,10,11,12,13],
         RED:[40,41,42,43,44], DARK:[1,2,3,4],
         TILE:[60,61,62,63,64,65,66], RIDGE:[61,62,63,64,65,66,67],
         PAINT:[96,97,98,99,100], GOLD:[62,63,64,65,66,67],
         WOOD:[35,36,37,38,39],
         PLAQUE:[54,55,56,57,58], PLAQUE_TEXT:[55,56,57,58,59],
         LANTERN:[179,180,181,182,183],
         **{key+PROP_OFFSET:ramp for key,ramp in PROP_RAMPS.items()}}


def render(model,view,*,lighting=DEFAULT_SCENE,art_scale=ART_SCALE,
           canvas=(224,208),origin=(112,80)):
    frame = trace_model(model,view,canvas,origin,Sampling(building_step=.125))
    # Roof forms are fourfold symmetric; derivatives are in rotated world space.
    px,py,pz = frame.point.transpose(2,0,1)
    f = art_scale/2
    x,y,z = (px-60)/f,(py-60)/f,(pz-1)/f-PLINTH
    normal = frame.normal.copy()
    mask = frame.hit&np.isin(frame.mat,(TILE,RIDGE))
    for index,roof in enumerate(ROOFS):
        h = roof_height(x,y,roof,index==4)
        m = mask&outline(x,y,*roof[:3])&(abs(z-h)<1.0)
        dx = (roof_height(x+.02,y,roof,index==4)-roof_height(x-.02,y,roof,index==4))/.04
        dy = (roof_height(x,y+.02,roof,index==4)-roof_height(x,y-.02,roof,index==4))/.04
        n = np.stack((-dx,-dy,np.ones_like(x)),axis=2)
        n /= np.linalg.norm(n,axis=2,keepdims=True)
        normal[m] = n[m]
    light = shade(normal,frame.shadow,lighting.standard)
    building = indexed(frame.mat,light,frame.hit,RAMPS)
    return finish(frame,stone_ground(frame,lighting),building)
