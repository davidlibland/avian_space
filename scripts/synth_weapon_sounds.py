#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "imageio-ffmpeg",
# ]
# ///
"""Synthesize the distinct per-weapon-family SFX in assets/sounds/weapons/.

Clean, license-free sci-fi weapon sounds (mono 44.1kHz ogg/Vorbis via the ffmpeg
bundled by imageio-ffmpeg). Tweak a synth function and re-run to retune a weapon.
The 5 originally-recorded sounds (laser, proton_beam, ir_missile, javelin,
space_mine) are NOT regenerated here — only the synthesized de-duplication set.

Run:  uv run scripts/synth_weapon_sounds.py
"""
import numpy as np, wave, subprocess, os, struct
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe(); SR=44100
def t(d): return np.linspace(0,d,int(SR*d),endpoint=False)
def sine(f,d,ph=0):
    if np.isscalar(f): return np.sin(2*np.pi*f*t(d)+ph)
    tt=t(d); return np.sin(2*np.pi*np.cumsum(f)/SR+ph)
def sweep(f0,f1,d,curve=1.0):
    tt=t(d); f=f0+(f1-f0)*(tt/d)**curve; return np.sin(2*np.pi*np.cumsum(f)/SR)
def saw(f,d):
    tt=t(d); return 2*((f*tt)%1.0)-1
def noise(d): return np.random.uniform(-1,1,int(SR*d))
def edecay(x,tau): return x*np.exp(-t(len(x)/SR)/tau)
def fade(x,ms=4):
    n=int(SR*ms/1000); 
    if len(x)>2*n: x[:n]*=np.linspace(0,1,n); x[-n:]*=np.linspace(1,0,n)
    return x
def norm(x,peak=0.72):
    m=np.max(np.abs(x)); return x*(peak/m) if m>0 else x
def softclip(x,k=2.0): return np.tanh(k*x)/np.tanh(k)
def pad(x,d): 
    out=np.zeros(int(SR*d)); out[:len(x)]=x[:len(out)]; return out

def mass_driver():   # EM railgun: rising whine + sharp crack + thump
    whine=sweep(220,1600,0.09,1.6)*np.linspace(0.2,1,int(SR*0.09))
    crack=norm(edecay(noise(0.03),0.008),1.0)
    thump=edecay(sine(70,0.18),0.05)*0.6
    x=pad(whine,0.22)+pad(crack,0.22)*0.9+pad(thump,0.22)
    return fade(norm(x))
def helios_lance():  # bright clean energy lance, descending
    a=sweep(2100,820,0.30,0.7); a+=0.4*sweep(4200,1640,0.30,0.7)  # octave
    a*=np.exp(-t(0.30)/0.16); a+=0.15*noise(0.30)*np.exp(-t(0.30)/0.05)
    return fade(norm(a))
def siege_cannon():  # heavy boom
    boom=edecay(sine(55,0.6),0.18)+0.6*edecay(sine(82,0.6),0.12)
    blast=norm(edecay(noise(0.6),0.07),0.8)
    crack=pad(norm(edecay(noise(0.02),0.006),0.8),0.6)
    x=boom+blast*0.7+crack
    return fade(norm(softclip(x,1.4)))
def relic_lance():   # resonant inharmonic bell-beam (Order)
    car=500; bell=sum(np.sin(2*np.pi*car*r*t(0.5))*np.exp(-t(0.5)/(0.3/i)) for i,r in enumerate([1,1.41,2.07,2.78],1))
    drone=0.3*sine(250,0.5)*np.exp(-t(0.5)/0.25)
    return fade(norm(bell*0.25+drone))
def exotic_beam():   # alien warble (Precursor): ring-mod + pitch wobble
    base=600+80*np.sin(2*np.pi*7*t(0.4))
    car=sine(base,0.4); ringmod=np.sin(2*np.pi*34*t(0.4))
    x=car*(0.6+0.4*ringmod); x+=0.3*sweep(1200,300,0.4)
    x*=np.exp(-t(0.4)/0.22)
    return fade(norm(x))
def pirate_laser():  # grungy descending laser
    a=saw(1,0.18)  # placeholder
    a=softclip(sweep(950,260,0.18,1.3),3.0)+0.25*noise(0.18)
    a*=np.exp(-t(0.18)/0.07)
    return fade(norm(a))
def flak():          # airburst pop
    pop=norm(edecay(noise(0.18),0.045),0.9)
    boom=edecay(sine(120,0.18),0.05)*0.6
    return fade(norm(pop*0.8+boom))
def chaingun():      # 2 quick mechanical pulses
    def pulse():
        click=norm(edecay(noise(0.025),0.006),0.9); thud=edecay(saw(150,0.025),0.012)*0.5
        return click*0.8+thud
    x=np.zeros(int(SR*0.16)); 
    for off in (0.0,0.075):
        p=pulse(); i=int(SR*off); x[i:i+len(p)]+=p[:len(x)-i]
    return fade(norm(x))
def goose():         # scrappy rocket whoosh
    wh=noise(0.4)*np.linspace(0.3,1,int(SR*0.4)); 
    # crude lowpass via cumulative smoothing
    wh=np.convolve(wh,np.ones(40)/40,mode='same')
    rumble=edecay(sine(90,0.4),0.2)*0.5
    return fade(norm(wh*0.7+rumble))
def censer_charge(): # Order lob: soft fwoomp + chime
    fwoomp=sweep(120,300,0.18,0.6)*np.exp(-t(0.18)/0.1)
    chime=pad(sum(np.sin(2*np.pi*880*r*t(0.3))*np.exp(-t(0.3)/0.18) for r in [1,2.4])*0.15,0.35)
    return fade(norm(pad(fwoomp,0.35)*0.8+chime))
def seeker():        # precursor high-tech ascending zip
    zip_=sweep(700,2600,0.22,1.4)*np.exp(-t(0.22)/0.12)
    shimmer=0.2*np.sin(2*np.pi*60*t(0.22))*sine(2200,0.22)
    return fade(norm(zip_+shimmer))
def prox_mine():     # deploy tk + 2 arming beeps
    x=np.zeros(int(SR*0.34))
    tk=norm(edecay(noise(0.02),0.005),0.8); x[:len(tk)]+=tk
    for off,f in ((0.12,1400),(0.24,1700)):
        b=fade(sine(f,0.05))*0.6; i=int(SR*off); x[i:i+len(b)]+=b
    return fade(norm(x))

SOUNDS={"mass_driver":mass_driver,"helios_lance":helios_lance,"siege_cannon":siege_cannon,
        "relic_lance":relic_lance,"exotic_beam":exotic_beam,"pirate_laser":pirate_laser,
        "flak":flak,"chaingun":chaingun,"goose":goose,"censer_charge":censer_charge,
        "seeker":seeker,"prox_mine":prox_mine}
for name,fn in SOUNDS.items():
    x=fn().astype(np.float32); x=np.clip(x,-1,1)
    pcm=(x*32767).astype('<i2')
    wavp=f"/tmp/wpnsnd/{name}.wav"
    with wave.open(wavp,'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    ogg=f"assets/sounds/weapons/{name}.ogg"
    subprocess.run([FF,"-y","-loglevel","error","-i",wavp,"-c:a","libvorbis","-q:a","5",ogg],check=True)
    print(f"  {name}: {len(x)/SR:.2f}s -> {ogg} ({os.path.getsize(ogg)} bytes)")
print("done — 12 sounds")
