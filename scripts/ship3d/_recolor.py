from PIL import Image, ImageDraw
OUT="out"
fed=['fed_patrol','fed_destroyer','fed_missile_cruiser','fed_carrier']
reb=['rebel_fighter','rebel_gunboat','rebel_frigate','rebel_carrier']
cell=200; pad=10
W=cell*4+pad*5; H=cell*2+pad*3+24
cv=Image.new("RGBA",(W,H),(0,0,0,255))  # BLACK space background
d=ImageDraw.Draw(cv)
d.text((pad,2),"FEDERATION (red & charcoal)",fill=(230,230,235,255))
def cellimg(n):
    im=Image.open(f"{OUT}/fleet_{n}.png").convert("RGBA"); im.thumbnail((cell,cell),Image.LANCZOS)
    c=Image.new("RGBA",(cell,cell),(0,0,0,0)); c.paste(im,((cell-im.width)//2,(cell-im.height)//2),im); return c
for i,n in enumerate(fed):
    cv.paste(cellimg(n),(pad+i*(cell+pad),20),cellimg(n))
d.text((pad,20+cell+pad),"REBELS (vivid blue & green)",fill=(230,230,235,255))
for i,n in enumerate(reb):
    cv.paste(cellimg(n),(pad+i*(cell+pad),20+cell+pad+14),cellimg(n))
cv.save(f"{OUT}/_recolor.png"); print("saved _recolor.png")
