from PIL import Image
OUT="out"; names=['fed_destroyer','fed_patrol','rebel_fighter','shuttle','pirate_missile_boat','asteroid_miner']
cell=200;pad=10;W=cell*3+pad*4;H=cell*2+pad*3
cv=Image.new("RGBA",(W,H),(0,0,0,255))
for i,n in enumerate(names):
    r,c=divmod(i,3)
    im=Image.open(f"{OUT}/fleet_{n}.png").convert("RGBA");im.thumbnail((cell,cell),Image.LANCZOS)
    cc=Image.new("RGBA",(cell,cell),(0,0,0,0));cc.paste(im,((cell-im.width)//2,(cell-im.height)//2),im)
    cv.paste(cc,(pad+c*(cell+pad),pad+r*(cell+pad)),cc)
cv.save(f"{OUT}/_vt.png");print("saved")
