import sys, os, numpy as np
from PIL import Image, ImageDraw, ImageFilter
import terrain_gen as G
TILE=G.TILE; W3="../../assets/sprites/worlds"
def crop_c(im):
    a=np.asarray(im); ys,xs=np.where(a[...,3]>8); return im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
def shadow(spr):
    """Return sprite with a soft contact shadow ellipse at its base."""
    w,h=spr.size; pad=4; cv=Image.new("RGBA",(w+pad*2,h+pad),(0,0,0,0))
    sh=Image.new("RGBA",cv.size,(0,0,0,0)); d=ImageDraw.Draw(sh)
    ew=int(w*0.7); eh=max(3,int(w*0.22)); cx=cv.width//2; ey=h-2
    d.ellipse([cx-ew//2,ey-eh//2,cx+ew//2,ey+eh//2],fill=(15,30,15,110))
    sh=sh.filter(ImageFilter.GaussianBlur(1.5)); cv.alpha_composite(sh)
    cv.alpha_composite(spr,(pad,0)); return cv
atlas=Image.open(f"{W3}/garden_atlas.png").convert("RGB"); ftile=atlas.crop((46*TILE,3*TILE,47*TILE,4*TILE)).resize((TILE,TILE),Image.NEAREST)
fns=["oak","conifer","birch","willow","dead_tree","bush","fern","mushroom","rock"]
fns=[f for f in fns if os.path.exists(f"out/_o_{f}_34.png")]
sprs={f:shadow(crop_c(Image.open(f"out/_o_{f}_34.png").convert("RGBA"))) for f in fns}
def on_tile(spr,gscale=26):
    bg=ftile.resize((40,40),Image.NEAREST).convert("RGBA")
    sc=min(gscale/spr.width,gscale/spr.height); s2=spr.resize((max(1,int(spr.width*sc)),max(1,int(spr.height*sc))),Image.LANCZOS)
    bg.alpha_composite(s2,((40-s2.width)//2,40-s2.height-1)); return bg
cell=96; cv=Image.new("RGB",(len(fns)*(cell+6)+6, cell+40+50),(28,30,38)); d=ImageDraw.Draw(cv)
d.text((6,4),sys.argv[1] if len(sys.argv)>1 else "objects",fill=(245,245,255))
for i,f in enumerate(fns):
    x=6+i*(cell+6); spr=sprs[f]
    z=spr.resize((min(cell,int(spr.width*cell/max(spr.width,spr.height))),min(cell,int(spr.height*cell/max(spr.width,spr.height)))),Image.LANCZOS)
    cv.paste(Image.new("RGB",(cell,cell),(40,44,52)),(x,22)); cv.paste(z.convert("RGB"),(x+(cell-z.width)//2,22+(cell-z.height)//2),z)
    cv.paste(on_tile(spr).resize((cell,cell),Image.NEAREST).convert("RGB"),(x,22+cell+4))
    d.text((x+2,22+2*cell+6),f,fill=(220,225,235))
import random; random.seed(3); W,H=10,8; sc=TILE
scene=Image.new("RGBA",(W*sc,H*sc))
for yy in range(H):
    for xx in range(W): scene.paste(ftile,(xx*sc,yy*sc))
trees=["oak","conifer","birch","willow","dead_tree"]; under=["bush","fern","mushroom","rock"]
pool=[f for f in (trees+trees+under) if f in fns]; placed=[]
for _ in range(46):
    f=random.choice(pool); spr=sprs[f]; sc2=min(30/spr.width,30/spr.height)
    s2=spr.resize((max(1,int(spr.width*sc2)),max(1,int(spr.height*sc2))),Image.LANCZOS)
    placed.append((random.randint(8,H*sc-4),s2,random.randint(8,W*sc-8)))
for wy,s2,wx in sorted(placed,key=lambda p:p[0]): scene.alpha_composite(s2,(int(wx-s2.width/2),int(wy-s2.height)))
scene=scene.convert("RGB").resize((W*sc*2,H*sc*2),Image.NEAREST)
out=Image.new("RGB",(max(cv.width,scene.width),cv.height+scene.height+10),(28,30,38)); out.paste(cv,(0,0)); out.paste(scene,(0,cv.height+8))
out.save("out/_obj_iter.png"); print("saved")
