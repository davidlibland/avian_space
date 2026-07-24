import os, random, sys, numpy as np
from PIL import Image, ImageDraw, ImageFilter
import terrain_gen as G
TILE=G.TILE; BLOB47=G.BLOB47; r47=G.reduce_to_47
TL,T,TR,L,R_,BL,B,BR=G.TL,G.T,G.TR,G.L,G.R_,G.BL,G.B,G.BR
NB=[(TL,-1,-1),(T,0,-1),(TR,1,-1),(L,-1,0),(R_,1,0),(BL,-1,1),(B,0,1),(BR,1,1)]
W3="../../assets/sprites/worlds"; random.seed(9); np.random.seed(9)
def crop_c(im):
    a=np.asarray(im); ys,xs=np.where(a[...,3]>8); return im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1))
def shadow(spr,strength=115):
    w,h=spr.size; pad=6; cv=Image.new("RGBA",(w+pad*2,h+pad),(0,0,0,0)); sh=Image.new("RGBA",cv.size,(0,0,0,0)); d=ImageDraw.Draw(sh)
    ew=int(w*0.66); eh=max(3,int(w*0.2)); cx=cv.width//2; ey=h-2
    d.ellipse([cx-ew//2,ey-eh//2,cx+ew//2,ey+eh//2],fill=(14,28,14,strength)); sh=sh.filter(ImageFilter.GaussianBlur(2)); cv.alpha_composite(sh); cv.alpha_composite(spr,(pad,0)); return cv
S={}
for f in os.listdir("out"):
    if f.startswith("_o_") and f.endswith("_34.png"):
        n=f[3:-7]; S[n]=shadow(crop_c(Image.open(f"out/{f}").convert("RGBA")))
def terr_img(tidx):
    MW,MH=18,12; terr=np.full((MH,MW),tidx,int)
    atlas=Image.open(f"{W3}/garden_atlas.png").convert("RGB"); out=Image.new("RGB",(MW*TILE,MH*TILE))
    for y in range(MH):
        for x in range(MW):
            mask=255  # uniform terrain -> interior tile
            c=BLOB47.index(r47(mask)); c=46+((x*7+y*13)%G.N_VARIANTS) if c==G.INTERIOR_COL else c
            out.paste(atlas.crop((c*TILE,tidx*TILE,c*TILE+TILE,tidx*TILE+TILE)),(x*TILE,y*TILE))
    return out,MW,MH
def scene(tidx, spec, name):
    tim,MW,MH=terr_img(tidx); cv=tim.convert("RGBA")
    df=G.n01(G.pnoise(2.2,7,64)); df=np.asarray(Image.fromarray((df*255).astype('uint8')).resize((MW,MH),Image.BICUBIC),float)/255
    placed=[]
    for nm,dens,mx,ph in spec:
        if nm not in S: continue
        for y in range(MH):
            for x in range(MW):
                sp=dens*df[y,x]
                for _ in range(mx):
                    if random.random()<sp:
                        spr=S[nm]; sc=ph/spr.height; s2=spr.resize((max(1,int(spr.width*sc)),max(1,int(spr.height*sc))),Image.LANCZOS)
                        wx=x*TILE+TILE/2+random.uniform(-.4,.4)*TILE; wy=y*TILE+TILE/2+random.uniform(-.4,.4)*TILE
                        placed.append((wy,s2,wx))
    for wy,s2,wx in sorted(placed,key=lambda p:p[0]): cv.alpha_composite(s2,(int(wx-s2.width/2),int(wy-s2.height)))
    return cv.convert("RGB")
scenes={
 "water":(0,[("seaweed",0.18,2,16),("lilypad",0.12,1,12),("reed",0.14,1,30),("fish",0.05,1,16),("frog",0.04,1,12)]),
 "sand":(1,[("shell",0.1,1,10),("driftwood",0.06,1,12),("rock",0.1,2,14),("bush",0.05,1,16)]),
 "grass":(2,[("wildflower",0.5,2,14),("bush",0.25,1,18),("rock",0.08,1,14),("oak",0.06,1,30),("hole_creature",0.03,1,12),("alien_peek",0.02,1,14)]),
 "mountain":(4,[("rock",0.25,2,16),("boulder",0.06,1,26),("conifer",0.18,1,30),("alpine_scrub",0.15,2,14)]),
}
out=[]
for nm,(ti,sp) in scenes.items():
    s=scene(ti,sp,nm); out.append((nm,s))
W=max(s.width for _,s in out); H=sum(s.height for _,s in out)+len(out)*18+8
cv=Image.new("RGB",(W,H),(28,30,38)); d=ImageDraw.Draw(cv); y=4
for nm,s in out:
    d.text((6,y),nm,fill=(245,245,255)); y+=16; cv.paste(s,(0,y)); y+=s.height+2
cv.save("out/_terrscenes.png")
# peeker zoom
peek=["squirrel","bird_nest","frog","alien_peek","hole_creature"]
cell=120; pz=Image.new("RGB",(len(peek)*(cell+6)+6,cell+24),(46,50,58)); dd=ImageDraw.Draw(pz)
for i,f in enumerate(peek):
    s=S[f]; sc=min((cell-8)/s.width,(cell-8)/s.height); z=s.resize((int(s.width*sc),int(s.height*sc)),Image.LANCZOS)
    x=6+i*(cell+6); pz.paste(z.convert("RGB"),(x+(cell-z.width)//2,4+(cell-z.height)//2),z); dd.text((x+4,cell+6),f,fill=(235,235,245))
pz.save("out/_peekers.png")
print("saved _terrscenes.png + _peekers.png")
