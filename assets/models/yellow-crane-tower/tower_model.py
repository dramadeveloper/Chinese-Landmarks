"""Present-day Wuhan Yellow Crane Tower; reference metres plus local gallery lift.

Storey widths/heights follow Huanghelou Cangsang chapter 9. Roof projections,
terrace and opening layout are photo estimates, not survey dimensions.
"""
import numpy as np
from export_plaza import make_model as make_prop

SIZE, HEIGHT = 120, 128
ART_SCALE = 3.5
# Add eight model units to each middle gallery. Footprint and overall height
# stay fixed while roof curvature and architectural details are refined.
GALLERY_EXTRA_UNITS = 8
GALLERY_EXTRA = GALLERY_EXTRA_UNITS/(ART_SCALE/2)
TOP_LIFT = 3*GALLERY_EXTRA
GROUND, STONE, RED, DARK, TILE, RIDGE, PAINT, GOLD, WOOD = range(1, 10)
PLAQUE, LANTERN, PLAQUE_TEXT = 10, 11, 12
PROP_OFFSET = 20
PLINTH = 1.8
# Square core half-span, projecting bay half-width, outer half-span,
# eave elevation above hall floor, roof rise, inner half-span, corner lift.
REFERENCE_ROOFS = ((15.2, 7.2, 18.2, 9.7, 2.7, 10.6, 2.0),
         (12.7, 6.0, 15.4, 17.2, 2.2, 9.0, 1.8),
         (12.7, 6.0, 15.4, 23.8, 2.2, 9.0, 1.8),
         (12.7, 6.0, 15.4, 30.4, 2.2, 8.8, 1.8),
         (11.7, 5.1, 14.2, 38.0, 8.4, 0.0, 2.0))
ROOFS = tuple((*roof[:3],roof[3]+lift*GALLERY_EXTRA,*roof[4:])
              for roof,lift in zip(REFERENCE_ROOFS,(0,1,2,3,3)))


def outline(x, y, core, bay, outer):
    return ((abs(x) <= core) & (abs(y) <= core) |
            (abs(x) <= bay) & (abs(y) <= outer) |
            (abs(y) <= bay) & (abs(x) <= outer))


def roof_height(x, y, roof, top=False):
    core, bay, outer, base, rise, inner, lift = roof
    q, s = np.maximum(abs(x), abs(y)), np.minimum(abs(x), abs(y))
    t = np.clip((core-q)/(core-inner), 0, 1)
    h = base + rise*t**(1.45 if top else 1.8) + lift*(np.clip(s/core, 0, 1)**8)
    # Four projecting bays make twelve outer tips instead of an octagonal pagoda.
    along = np.clip((outer-q)/(outer-(core-3)), 0, 1)
    if top:
        # Distinct front gables with a short ridge. The back half slopes toward
        # the central crown rather than spreading into one broad raised cap.
        ridge_run = np.clip((q-(core-5))/3.2,0,1)
        gable = base + 5.0*(1-np.clip(s/bay,0,1))**.85
        gable += lift*np.clip(s/bay,0,1)**8
        h = np.where(s<=bay,np.maximum(h,base+(gable-base)*ridge_run),h)
    else:
        porch = base + .45 + (rise-.45)*along**1.8
        porch += lift*np.clip(s/bay,0,1)**6*(1-along)**2
        h = np.where((s<=bay)&(q>=core-3), porch, h)
    return h


def make_model(art_scale=ART_SCALE):
    a = np.zeros((SIZE,SIZE,HEIGHT),np.uint8)
    a[:,:,0] = GROUND
    factor = art_scale/2
    xx,yy,zz = np.indices(a.shape,dtype=np.float32)
    x,y,z = (xx+.5-60)/factor,(yy+.5-60)/factor,(zz+.5-1)/factor
    floor_z = z-PLINTH

    def box(x0,y0,z0,x1,y1,z1,material):
        lo = np.rint([60+x0*factor,60+y0*factor,1+z0*factor]).astype(int)
        hi = np.rint([60+x1*factor,60+y1*factor,1+z1*factor]).astype(int)
        a[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]] = material

    def perimeter(core,bay,outer,bottom,top,material,posts=False,entry=False):
        mask = outline(x,y,core,bay,outer)&~outline(x,y,core-.45,bay-.45,outer-.45)
        if entry:
            mask &= ~((abs(y)<6.0)&(abs(x)>core-.6))
        if posts:
            mask &= ((np.mod(x+50,3.5)<.65)|(np.mod(y+50,3.5)<.65))
        a[mask&(z>=bottom)&(z<top)] = material

    # Necessary low terrace and broad west/east stairs; no scenic hill or park.
    mask = outline(x,y,17,8.3,19.8)&(z>=0)&(z<PLINTH)
    a[mask] = STONE
    perimeter(16.8,8.0,19.5,PLINTH,PLINTH+.7,STONE,entry=True)
    perimeter(16.8,8.0,19.5,PLINTH,PLINTH+1.1,STONE,posts=True,entry=True)
    for side in (-1,1):
        for step in range(5):
            lo,hi = sorted((side*(19.8+step*.8),side*(20.6+step*.8)))
            box(lo,-5.8,0,hi,5.8,PLINTH*(5-step)/5,STONE)

    for level,(bottom,top,half) in enumerate(((0,9.7,15),(12.3,17.2,11),
            (19.4,23.8,11),(26,30.4,11),(32.6,38.0,9))):
        bottom += PLINTH+max(level-1,0)*GALLERY_EXTRA
        top += PLINTH+min(level,3)*GALLERY_EXTRA
        inner = half-2.6
        box(-inner,-inner,bottom,inner,inner,top,RED)
        # Dark recessed openings; sparse frames remain legible at native 1x.
        for axis in (0,1):
            for sign in (-1,1):
                for c in (-inner*.67,0,inner*.67):
                    rect = [sign*inner-.35,c-1.45,bottom+.4,sign*inner+.35,c+1.45,top-.55]
                    if axis:
                        rect = [rect[1],rect[0],rect[2],rect[4],rect[3],rect[5]]
                    box(*rect,DARK)
                for c in (-half,-half*.5,0,half*.5,half):
                    if level==0 and c==0:
                        continue
                    px,py = (sign*half,c) if axis==0 else (c,sign*half)
                    box(px-.38,py-.38,bottom,px+.38,py+.38,top,RED)
                    box(px-.65,py-.65,top-.85,px+.65,py+.65,top-.35,PAINT)
        if level==0:
            # Front/back doors and a shallow gallery at the documented 6 m datum.
            for side in (-1,1):
                box(side*inner-.5,-3.7,bottom,side*inner+.5,3.7,bottom+5,DARK)
            perimeter(15,6.6,16.3,PLINTH+6,PLINTH+6.35,WOOD)
            # Framed west/east entry, forward porch posts and painted lintel.
            # Door glazing and sparse mullions are readable at native 1x.
            for side in (-1,1):
                door_x = side*12.6
                box(door_x-.45,-4.7,PLINTH+.35,door_x+.45,4.7,PLINTH+5.8,PAINT)
                for c in (-4.7,-1.6,1.6,4.7):
                    box(door_x-.55,c-.22,PLINTH+.2,door_x+.55,c+.22,PLINTH+6.1,RED)
                box(door_x-.55,-4.9,PLINTH+3.5,door_x+.55,4.9,PLINTH+3.9,RED)
                for c in (-5.6,5.6):
                    box(side*16.3-.45,c-.4,PLINTH,side*16.3+.45,c+.4,PLINTH+9.3,RED)
                box(side*16.3-.5,-5.9,PLINTH+7.1,side*16.3+.5,5.9,PLINTH+7.7,RED)
                box(side*16.3-.6,-5.5,PLINTH+7.75,side*16.3+.6,5.5,PLINTH+8.35,PAINT)
                for c in (-4.2,4.2):
                    lantern = ((x-side*16.6)/.65)**2+((y-c)/.85)**2+((z-PLINTH-6.25)/.85)**2<=1
                    a[lantern] = LANTERN
                    box(side*16.6-.25,c-.25,PLINTH+5.0,side*16.6+.25,c+.25,PLINTH+5.5,GOLD)
        else:
            a[outline(x,y,half,half*.55,half+1)&(z>=bottom-.5)&(z<bottom)] = RED
            perimeter(half,half*.55,half+1,bottom+.65,bottom+1.05,WOOD)
            perimeter(half,half*.55,half+1,bottom,bottom+1.3,WOOD,posts=True)
        perimeter(half,half*.55,half+1,top-.5,top,PAINT)

    # Five curved eaves. The crown shell is thicker to avoid voxel gaps while
    # keeping the four gable plaques visible beneath their overhangs.
    for index,roof in enumerate(ROOFS):
        h = roof_height(x,y,roof,index==4)
        footprint = outline(x,y,*roof[:3])
        bottom = h-(1.2 if index==4 else .7)
        q,s = np.maximum(abs(x),abs(y)),np.minimum(abs(x),abs(y))
        if index==4:
            face = (q>roof[2]-1.35)&(q<roof[2]-.7)&(s<roof[1]-.4)
            pediment = face&(floor_z>roof[3]+.6)&(floor_z<h-.65)
            a[pediment] = RED
            a[pediment&(floor_z>h-1.3)] = WOOD
        region = footprint&(floor_z<=h)&(floor_z>bottom)
        a[region] = TILE
        hips = (abs(abs(x)-abs(y))<.34)&(q<=roof[0])
        eave = ~outline(x,y,roof[0]-.42,roof[1]-.42,roof[2]-.42)
        a[region&(hips|eave)&(floor_z>h-.42)] = RIDGE
        if index==4:
            # A short lower eave separates the plaque face from the gallery.
            lip = roof[3]+.25+1.1*np.clip(s/roof[1],0,1)**6
            apron = (s<=roof[1])&(q>roof[2]-1.8)&(q<=roof[2])
            a[apron&(floor_z<=lip)&(floor_z>lip-.55)] = TILE

    # Four top gable plaques and first-floor entrance plaque, opaque inset boards.
    for axis in (0,1):
        for side in (-1,1):
            for zz0,zz1,extent,width in ((39.0,41.7,14.0,2.75),(7.65,9.25,17.0,3.2)):
                if zz0>30:
                    zz0 += TOP_LIFT
                    zz1 += TOP_LIFT
                rect = [side*extent-.35,-width,PLINTH+zz0,side*extent+.35,width,PLINTH+zz1]
                if axis:
                    rect = [rect[1],rect[0],rect[2],rect[4],rect[3],rect[5]]
                box(*rect,WOOD)
                inset = rect.copy()
                inset[2] += .2
                inset[5] -= .2
                inset[1-axis] += .2
                inset[4-axis] -= .2
                box(*inset,PLAQUE)
                # Muted gold marks suggest calligraphy without a strong focal point.
                for c in (-1.7,0,1.7):
                    glyph = [side*extent-.35,c-.18,PLINTH+zz0+.55,
                             side*extent+.35,c+.18,PLINTH+zz1-.45]
                    if axis:
                        glyph = [glyph[1],glyph[0],glyph[2],glyph[4],glyph[3],glyph[5]]
                    box(*glyph,PLAQUE_TEXT)

    # 5 m finial with 4.4 m lotus base, 2 m belly and small red top globe.
    radial = np.hypot(x,y)
    heights = [46.25,46.8,47.2,47.8,48.5,49.0,49.7,50.4,50.8,51.4]
    radii = [2.2,2.0,.7,1.0,.85,.4,.72,.45,.35,0]
    crown_z = floor_z-TOP_LIFT
    radius = np.interp(crown_z,heights,radii)
    a[(crown_z>=46.25)&(crown_z<51.4)&(radial<=radius)] = GOLD
    a[(crown_z>50.7)&(crown_z<=51.4)&(radial<.51)] = RED
    # Four public-size adults; no rescaling with the building.
    person = make_prop('walkers')[10:13,12:14,1:6]
    for px,py in ((16,48),(18,54),(96,84),(91,91)):
        target = a[px:px+3,py:py+2,1:6]
        target[person!=0] = person[person!=0]+PROP_OFFSET
    return a
