# Video dehazing: the still-image stages with estimates smoothed over time (exponential moving average) to stop flicker
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import dehaze_core as C
# Stateful frame-by-frame dehazer: EMA-smoothed airlight, transmission, histogram range and colour gains, with a motion mask
class TemporalDehazer:
    # Stores the settings and clears the temporal state
    def __init__(self,P=None,auto=False,smoothing=0.9,motion_thr=10.0,denoise=0.6):
        self.base=dict(C.P_BALANCED if P is None else P)
        self.auto,self.smoothing,self.motion_thr=auto,float(smoothing),float(motion_thr)
        self.denoise=float(denoise) if smoothing>0 else 0.0
        self.reset()
    # Clears all temporal state
    def reset(self):
        self.prev_gray=self.t=self.A=self.hist_range=self.gains=self.haze=self.prev_in=None
        self.P=None
    # Exponential moving average: w * old + (1 - w) * new
    def _ema(self,old,new,w):
        return new if old is None else w*old+(1.0-w)*new
    # Processes one frame: motion mask, temporal denoise, then every stage with smoothed estimates
    def process(self,frame):
        P=dict(self.base)
        w=self.smoothing
        if self.auto: # strength follows the (smoothed) haze level
            self.haze=self._ema(self.haze,C.haze_level(frame),w)
            g=C.strength_scale(self.haze)
            for k in ("hist","freq","smooth","sharpen"):
                P[k]=P[k]*g
            P["far"]=P.get("far",0)*g
            if P["use_dcp"]:
                P["omega"]=min(0.95,P["omega"]*(0.35+0.65*g))
        # Blurred gray frame; pixels that changed little since the previous frame count as static
        gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY).astype(np.float32),(0,0),1.5)
        still=(np.abs(gray-self.prev_gray)<self.motion_thr) if self.prev_gray is not None else None
        self.prev_gray=gray
        img=frame
        if self.denoise>0 and self.prev_in is not None and self.prev_in.shape==frame.shape:
            # Temporal denoise: blend static pixels with the previous filtered frame, leave moving pixels untouched
            a=np.where(still,self.denoise,0.0).astype(np.float32)[...,None]
            img=np.clip(a*self.prev_in+(1.0-a)*frame.astype(np.float32)+0.5,0,255).astype(np.uint8)
        self.prev_in=img
        for name in C.STAGE_ORDER:
            if name=="dcp":
                if P["use_dcp"]:
                    t,A=C.estimate_transmission(img,P["omega"],P["t0"],P["patch"],P.get("adaptive",False))
                    if self.t is not None and self.t.shape==t.shape:
                        wt=np.where(still,w,0.0).astype(np.float32) # smooth only where nothing moved
                        t=wt*self.t+(1.0-wt)*t
                    # Smooth the airlight over time with an EMA
                    self.t,self.A=t,self._ema(self.A,A,w)
                    img=C.far_detail(C.recover_scene(img,self.t,self.A),self.t,P.get("far",0))
            elif name=="hist":
                if P["hist"]>0:
                    _,rng=C.histogram_stretch(img,P["hist"],return_range=True)
                    # Smooth the histogram-stretch range over time
                    self.hist_range=tuple(self._ema(None if self.hist_range is None else np.array(self.hist_range),
                            np.array(rng),w))
                    img=C.histogram_stretch(img,P["hist"],fixed_range=self.hist_range)
            elif name=="tone":
                if C.stage_active("tone",P):
                    if P.get("balance",0):
                        # Smooth the gray-world colour gains over time
                        self.gains=self._ema(self.gains,C.colour_balance_gains(C.colour_restore(img,P.get("restore",0)) if P.get("restore",0) else img,P["balance"]),w) # gains measured after restoration, as in the still pipeline
                    img=C.tone_adjust(img,P["gamma"],P["sat"],P.get("balance",0),gains=self.gains,
                        restore=P.get("restore",0))
            elif C.stage_active(name,P):
                img=C.apply_stage(name,img,P)
        self.P=P
        return img
# Flicker measure: mean absolute change between consecutive output frames on static (unchanged-input) pixels
def flicker(frames_in,frames_out,motion_thr=10.0):
    vals=[]
    for i in range(1,len(frames_out)):
        gi0,gi1=(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32) for f in (frames_in[i-1],frames_in[i]))
        still=cv2.GaussianBlur(np.abs(gi1-gi0),(0,0),1.5)<motion_thr
        go0,go1=(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY).astype(np.float32) for f in (frames_out[i-1],frames_out[i]))
        vals.append(float(np.abs(go1-go0)[still].mean()))
    return float(np.mean(vals))
# Re-encodes an mp4 as H.264 (libx264 via ffmpeg) so browsers can play it
def to_h264(path):
    try:
        import subprocess
        import imageio_ffmpeg
        tmp=str(path)+".h264.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-i",str(path),"-vcodec","libx264",
                "-pix_fmt","yuv420p","-preset","veryfast",tmp],check=True,timeout=600)
        Path(tmp).replace(path)
        return True
    except Exception:
        return False
# Dehazes a video file frame by frame with TemporalDehazer and writes an mp4; orig_dst and side_dst also write the scaled original and an original | dehazed side-by-side clip
def process_video(src,dst,P=None,auto=False,temporal=True,max_dim=0,max_frames=0,progress=None,auto_blur=False,orig_dst=None,side_dst=None):
    # Read frames with OpenCV, downscale with area interpolation, dehaze, write mp4v, then convert to H.264
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"cannot open {src}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dehazer=TemporalDehazer(P,auto=auto,smoothing=0.9 if temporal else 0.0)
    writer,n,t0=None,0,time.perf_counter()
    extra={} # optional second and third writers: key -> (writer, path)
    try: # the finally below frees every file even if the app stops the run, which Windows needs to delete the temp folder
        while True:
            ok,bgr=cap.read()
            if not ok or (max_frames and n>=max_frames):
                break
            s=min(1.0,max_dim/max(bgr.shape[:2])) if max_dim else 1.0
            if s<1.0:
                bgr=cv2.resize(bgr,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            if auto_blur and n==0:
                dehazer.base=C.auto_deblur_params(rgb,dehazer.base)[0]
            out=dehazer.process(rgb)
            if writer is None:
                fourcc=cv2.VideoWriter_fourcc(*"mp4v")
                writer=cv2.VideoWriter(str(dst),fourcc,fps,(out.shape[1],out.shape[0]))
                if orig_dst:
                    extra["orig"]=(cv2.VideoWriter(str(orig_dst),fourcc,fps,(out.shape[1],out.shape[0])),orig_dst)
                if side_dst:
                    extra["side"]=(cv2.VideoWriter(str(side_dst),fourcc,fps,(2*out.shape[1],out.shape[0])),side_dst)
            writer.write(cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
            if "orig" in extra:
                extra["orig"][0].write(bgr)
            if "side" in extra:
                extra["side"][0].write(np.hstack([bgr,cv2.cvtColor(out,cv2.COLOR_RGB2BGR)]))
            n+=1
            if progress:
                progress(n,min(n_total,max_frames) if max_frames and n_total else (max_frames or n_total))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for w,_ in extra.values():
            w.release()
    if writer is not None:
        to_h264(dst)
        for _,path in extra.values():
            to_h264(path)
    return n,(time.perf_counter()-t0)/max(n,1)
# Vehicle detection on every stride-th frame of the original and dehazed clips: boxes and counts drawn, written as an original | dehazed mp4; returns the (original, dehazed) count per frame
def detection_video(orig_path,dehazed_path,dst,conf=0.2,tiled=True,stride=1,progress=None,preview=None):
    import detect as D
    run=D.detect_tiled if tiled else D.detect
    cap_o,cap_d=cv2.VideoCapture(str(orig_path)),cv2.VideoCapture(str(dehazed_path))
    fps=cap_o.get(cv2.CAP_PROP_FPS) or 25.0
    writer,counts,boxes,i=None,[],([],[]),0
    try:
        while True:
            ok1,o=cap_o.read()
            ok2,d=cap_d.read()
            if not (ok1 and ok2):
                break
            o,d=cv2.cvtColor(o,cv2.COLOR_BGR2RGB),cv2.cvtColor(d,cv2.COLOR_BGR2RGB)
            if i%max(1,stride)==0: # between detections the last boxes are reused
                boxes=(run(o,conf),run(d,conf))
            frames=[C.draw_boxes(im,b) for im,b in zip((o,d),boxes)]
            for im,label,b in zip(frames,("Original","Dehazed"),boxes):
                # Count banner in the top-left corner
                cv2.rectangle(im,(0,0),(200,28),(0,0,0),-1)
                cv2.putText(im,f"{label}: {len(b)} found",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
            both=np.hstack(frames)
            if writer is None:
                writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*"mp4v"),fps,(both.shape[1],both.shape[0]))
            writer.write(cv2.cvtColor(both,cv2.COLOR_RGB2BGR))
            counts.append((len(boxes[0]),len(boxes[1])))
            i+=1
            if preview:
                preview(both,i)
            if progress:
                progress(i)
    finally:
        cap_o.release()
        cap_d.release()
        if writer is not None:
            writer.release()
    if writer is not None:
        to_h264(dst)
    return counts
# Command-line entry point
def main():
    ap=argparse.ArgumentParser(description="Dehazing for video (traffic cameras, construction-site cameras).")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--auto",action="store_true",help="scale the strength to the haze level")
    ap.add_argument("--auto-blur",action="store_true",help="undo the blur measured on the first frame")
    ap.add_argument("--no-temporal",action="store_true",help="process every frame independently (flickers)")
    ap.add_argument("--max-dim",type=int,default=0,help="downscale frames to this long edge")
    ap.add_argument("--max-frames",type=int,default=0)
    a=ap.parse_args()
    n,spf=process_video(a.src,a.dst,auto=a.auto,temporal=not a.no_temporal,max_dim=a.max_dim,
        max_frames=a.max_frames,auto_blur=a.auto_blur)
    print(f"{n} frames, {spf:.2f} s/frame -> {a.dst}")
if __name__=="__main__":
    main()
