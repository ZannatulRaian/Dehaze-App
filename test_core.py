# Quick checks for dehaze_core.py; usage: python test_core.py
import cv2
import numpy as np
import dehaze_core as C
from evaluate import load_pairs,load_urban
rng=np.random.RandomState(0)
img=rng.randint(0,256,(97,131,3),dtype=np.uint8) # odd sizes on purpose
OFF=dict(C.P_BALANCED,use_dcp=False,hist=0,freq=0,smooth=0,sharpen=0,clahe=0,balance=0,restore=0,gamma=1.0,sat=1.0,far=0,deblur=0.0)
# 1. every stage leaves the image untouched when switched off
for name in C.STAGE_ORDER:
    assert not C.stage_active(name,OFF),name
    assert np.array_equal(C.apply_stage(name,img,OFF),img),f"{name} changed the image while off"
# 2. the full pipeline keeps shape and dtype
out=C.dehaze_pipeline(img,C.P_BALANCED)
assert out.shape==img.shape and out.dtype==np.uint8
# 3. simulated fog flattens edges (lower mean gradient); Wiener deblur undoes simulated blur
pairs=load_pairs(480)
clear=next(iter(pairs.values()))[1]
foggy=C.simulate_fog(clear,None,1.6)
assert C.quality_metrics(foggy)["gradient"]<C.quality_metrics(clear)["gradient"]
for kind in ("Gaussian","Motion"):
    blurred=C.simulate_blur(clear,2.0,kind)
    restored=C.wiener_deblur(blurred,2.0,kind,0.003)
    assert C.psnr(restored,clear)>C.psnr(blurred,clear)+1.0,kind
# 4. on the real foggy test photos the default settings beat doing nothing (PSNR and SSIM)
none_s=[C.ssim(h,c) for h,c in pairs.values()]
none_p=[C.psnr(h,c) for h,c in pairs.values()]
full_s=[C.ssim(C.dehaze_pipeline(h,C.P_BALANCED),c) for h,c in pairs.values()]
full_p=[C.psnr(C.dehaze_pipeline(h,C.P_BALANCED),c) for h,c in pairs.values()]
assert np.mean(full_s)>np.mean(none_s) and np.mean(full_p)>np.mean(none_p)
# 5. bright neutral regions receive a conservative DCP safeguard; output stays valid
white_patch=np.full((40,40,3),230,np.uint8)
protected=C.recover_scene(white_patch,np.full((40,40),0.25,np.float32),np.array([0.90,0.90,0.90],np.float32))
assert protected.mean()>=white_patch.mean()-1.0
assert protected.dtype==np.uint8 and protected.shape==white_patch.shape
# 6. the metrics themselves
assert abs(C.ssim(img,img)-1.0)<1e-6 and C.psnr(img,img)==99.0
assert cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).shape==img.shape[:2]
# 6. blur estimation: sharp photos read as sharp (also in fog); known blur is found, with its type, within 30 %
assert C.estimate_blur(clear)["kind"]=="none" and C.estimate_blur(foggy)["kind"]=="none"
for kind,amount in (("Gaussian",3.0),("Motion",3.0)):
    est=C.estimate_blur(C.simulate_blur(foggy,amount,kind))
    assert est["kind"]==kind and abs(est["amount"]-amount)<0.3*amount,(kind,est)
for ang in (0,45,90,135): # motion at any angle: type, direction and size
    est=C.estimate_blur(C.simulate_blur(foggy,3.0,"Motion",ang))
    assert est["kind"]=="Motion" and abs((est["angle"]-ang+90)%180-90)<12 and abs(est["amount"]-3.0)<1.2,(ang,est)
assert np.array_equal(C.motion_kernel(9,0)[4],np.full(9,1/9,np.float32)) # angle 0 = the plain 9-tap box
k45=C.motion_kernel(9,45)
assert abs(k45.sum()-1)<1e-5 and k45[4,4]>0 and k45[1,7]>0 and k45[7,1]>0 and k45[0,0]==k45[8,8]==0 # 45 deg runs bottom-left to top-right
b45=C.simulate_blur(clear,2.0,"Motion",45) # undoing a diagonal blur needs the right angle
assert C.psnr(C.wiener_deblur(b45,2.0,"Motion",0.003,45),clear)>C.psnr(C.wiener_deblur(b45,2.0,"Motion",0.003,0),clear)+3
assert C.estimate_blur(np.full((120,160,3),128,np.uint8))["kind"]=="none" # no edges at all
assert C.estimate_blur(img[:20,:20])["kind"]=="none" # too small to measure
P_auto,est=C.auto_deblur_params(C.simulate_blur(clear,3.0,"Motion",90),C.P_BALANCED)
assert P_auto["deblur"]>0 and P_auto["deblur_kind"]=="Motion" and abs(P_auto["deblur_angle"]-90)<12
assert C.auto_deblur_params(clear,C.P_BALANCED)[0]["deblur"]==0.0
# 7. urban test set: matches the generator on disk, depth is a 0..1 map, and the dark channel's transmission follows the true one
urban=load_urban()
assert len(urban)>=6
c0,d0=next(iter(urban.values()))
assert c0.shape[:2]==d0.shape and 0.0<=d0.min() and d0.max()<=1.0
assert np.array_equal(c0,C.make_urban_set(1)[0][0])
t_est,A=C.estimate_transmission(C.simulate_fog(c0,d0,1.6),0.87,0.2,0)
assert np.corrcoef(t_est.ravel(),np.clip(C.true_transmission(d0,1.6),0.2,1).ravel())[0,1]>0.8
assert np.array_equal(C.dark_channel_prior(foggy,0.87,0.2),C.dark_channel_prior(foggy,0.87,0.2))
# 8. downstream test: fog hides edges and lines, dehazing brings them back; a map matches itself perfectly
fog_c=C.simulate_fog(c0,d0,3.0)
dn_fog,dn_clear=C.downstream_metrics(fog_c,c0),C.downstream_metrics(c0,c0)
dn_out=C.downstream_metrics(C.dehaze_pipeline(fog_c,C.P_BALANCED),c0)
assert dn_out["edge_f1"]>dn_fog["edge_f1"] and dn_out["line_recall"]>dn_fog["line_recall"]
assert dn_clear["lines_vertical"]>0 and dn_clear["lines_oblique"]>0 # building and lane lines exist
assert abs(dn_clear["edge_f1"]-1.0)<1e-9 and dn_clear["line_recall"]==1.0
assert C.edge_f1(np.zeros((9,9),np.uint8),np.zeros((9,9),np.uint8))[2]==0.0 # empty maps do not crash
assert len(C.hough_lines(np.zeros((50,50),np.uint8)))==0
# 9. haze level and automatic strength: the level rises with fog, light haze gets less than the full default
levels=[C.haze_level(C.simulate_fog(c0,d0,b)) for b in (0.4,1.2,2.4)]
assert levels[0]<levels[1]<levels[2]
assert C.strength_scale(levels[0])<C.strength_scale(levels[2])<=1.0 and C.strength_scale(0.0)==C.STRENGTH_MIN
P_light,h,g=C.auto_strength_params(C.simulate_fog(c0,d0,0.4))
assert g<0.8 and P_light["sharpen"]<C.P_BALANCED["sharpen"] and P_light["omega"]<C.P_BALANCED["omega"]
assert C.auto_strength_params(fog_c,haze=0.9)[2]==1.0 and C.auto_strength_params(fog_c,C.P_MODULES,haze=0.9)[0]["use_dcp"] is False
assert C.P_TRAFFIC["freq"]==C.P_TRAFFIC["sharpen"]==C.P_TRAFFIC["balance"]==0 and C.P_TRAFFIC["use_dcp"]
# 10. visibility metrics: identical images give e = 0, r = 1; dehazing a foggy photo gains visible edges
vm=C.visibility_metrics(fog_c,fog_c)
assert vm["e"]==0 and abs(vm["r"]-1)<1e-6 and vm["sigma"]==0
vm=C.visibility_metrics(fog_c,C.dehaze_pipeline(fog_c,C.P_BALANCED))
assert vm["e"]>0.3 and vm["r"]>1.5
# 11. detector helpers (no model needed): matching by class and overlap
import detect
ref=[(10,10,50,50,.9,"car"),(100,100,140,140,.8,"person")]
assert detect.match([(12,11,52,49,.7,"car")],ref)[:2]==(0.5,1.0)
assert detect.match([(12,11,52,49,.7,"person")],ref)[2]==0 # right place, wrong class
r0,p0,n0=detect.match([],ref)
assert r0==0.0 and n0==0 and p0!=p0 # nothing found: recall 0, precision undefined
if detect.available():
    street=C.simulate_fog(C.make_sample_scene()[0],None,0.0)
    assert isinstance(detect.detect(street),list)
# 12. video: temporal processing flickers less than frame-by-frame, with similar quality; mp4 round trip
import tempfile
from pathlib import Path
from dehaze_video import TemporalDehazer,flicker,process_video
clip,dep=C.make_traffic_clip(10,320,200)
rng=np.random.default_rng(0)
foggy_clip=[np.clip(C.simulate_fog(f,dep,1.6).astype(np.float32)+rng.normal(0,2,f.shape),0,255).astype(np.uint8) for f in clip]
solo=[TemporalDehazer(smoothing=0.0).process(f) for f in foggy_clip]
d=TemporalDehazer()
temporal=[d.process(f) for f in foggy_clip]
assert flicker(foggy_clip,temporal)<0.6*flicker(foggy_clip,solo)
assert np.mean([C.psnr(o,c) for o,c in zip(temporal,clip)])>np.mean([C.psnr(o,c) for o,c in zip(solo,clip)])-0.5
assert np.array_equal(TemporalDehazer(smoothing=0.0).process(foggy_clip[0]),C.dehaze_pipeline(foggy_clip[0],C.P_BALANCED)) # no smoothing = still-image pipeline
with tempfile.TemporaryDirectory() as td:
    src,dst=Path(td)/"in.mp4",Path(td)/"out.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10,(320,200))
    for f in foggy_clip:
        w.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
    w.release()
    n_fr,_=process_video(src,dst,max_frames=6,auto=True,auto_blur=True)
    cap=cv2.VideoCapture(str(dst))
    try:
        assert n_fr==6 and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==6 and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==320
    finally:
        cap.release()  # Windows locks the MP4 until the capture handle is explicitly closed
print("all checks passed")
