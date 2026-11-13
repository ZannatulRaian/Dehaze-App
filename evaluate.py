# Evaluation of the Dehaze App against clear references (PSNR, SSIM, Canny, Hough, YOLOv5n); usage: python evaluate.py [--export-urban] [--fit-strength] [--charts]
import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np
import dehaze_core as C
import detect
from dehaze_video import TemporalDehazer,flicker
ROOT=Path(__file__).parent
TEST=ROOT/"assets"/"test images"
DATA=TEST/"outdoor_fog_with_clear_reference"
URBAN_REAL=TEST/"urban_with_clear_reference" # optional: your own street photos with clear references, see README
URBAN_REAL_FOGGY=TEST/"urban_real_foggy" # unpaired real fog/smog images; no clear references
URBAN_STREETS=TEST/"urban_real_clear" # clear real street photos, downloaded by get_street_photos.py
URBAN_SYN=TEST/"urban_synthetic"
OUT=ROOT/"results"
URBAN_BETA=1.6 # fog density of the hazy images stored in urban_synthetic
BETA=1.6 # fog density used for the simulated conditions
OFF=dict(C.P_BALANCED,use_dcp=False,hist=0,freq=0,smooth=0,sharpen=0,balance=0,clahe=0,restore=0,gamma=1.0,far=0)
auto_strength= lambda img:C.auto_strength_params(img)[0]
all_auto= lambda img:C.auto_deblur_params(img,C.auto_strength_params(img)[0])[0]
METHODS={ # name -> settings (strengths of the single modules are set to a medium 50), or a function image -> settings
    "Foggy input (no processing)":OFF,
    "Histogram stretching only":dict(OFF,hist=50),
    "Frequency-domain enhancement only":dict(OFF,freq=50,cutoff=60),
    "Adaptive smoothing + sharpening only":dict(OFF,smooth=50,sharpen=50),
    "The three course modules together":C.P_MODULES,
    "Dark channel prior only":dict(OFF,use_dcp=True,omega=C.P_BALANCED["omega"],t0=C.P_BALANCED["t0"]),
    "Full pipeline (Balanced default)":C.P_BALANCED,
    "Full pipeline, strength set from the haze level":auto_strength,
    "Traffic monitoring preset (fog removal + histogram stretching)":C.P_TRAFFIC,
    "Traffic monitoring preset, strength set from the haze level": lambda img:C.auto_strength_params(img,C.P_TRAFFIC)[0],
}
INPUT,FULL="Foggy input (no processing)","Full pipeline (Balanced default)"
AUTOSTR="Full pipeline, strength set from the haze level"
KNOWN,AUTO="Full pipeline + Wiener deblur (blur known)","Full pipeline + Wiener deblur (blur estimated)"
ALLAUTO="Full pipeline, haze level and blur both estimated"
ABLATION={
    "Full pipeline (Balanced default)":C.P_BALANCED,
    "without histogram stretching":dict(C.P_BALANCED,hist=0),
    "without frequency-domain enhancement":dict(C.P_BALANCED,freq=0),
    "without adaptive smoothing + sharpening":dict(C.P_BALANCED,smooth=0,sharpen=0),
    "without all three course modules":dict(C.P_BALANCED,hist=0,freq=0,smooth=0,sharpen=0),
    "without the dark channel prior":dict(C.P_BALANCED,use_dcp=False),
    "without colour balance":dict(C.P_BALANCED,balance=0),
    "without CLAHE":dict(C.P_BALANCED,clahe=0),
    "without colour restoration":dict(C.P_BALANCED,restore=0),
    "without the adaptive dark channel":dict(C.P_BALANCED,adaptive=False),
    "without the three course modules and CLAHE":dict(C.P_BALANCED,hist=0,freq=0,smooth=0,sharpen=0,clahe=0),
    "without CLAHE, colour restoration and the adaptive dark channel":dict(C.P_BALANCED,clahe=0,restore=0,adaptive=False),
    "without frequency, sharpening, smoothing and colour balance (Traffic preset)":C.P_TRAFFIC,
}
# The methods plus Wiener-deblur variants: blur known, blur estimated, and everything automatic
def blur_methods(kind,amount=2.0,angle=0.0):
    return dict(METHODS,**{KNOWN:dict(C.P_BALANCED,deblur=amount,deblur_kind=kind,deblur_angle=angle),
            AUTO: lambda img:C.auto_deblur_params(img,C.P_BALANCED)[0],
            ALLAUTO:all_auto})
# Downscales so the long edge is at most max_dim (area interpolation)
def _resize(img,max_dim):
    s=min(1.0,max_dim/max(img.shape[:2]))
    return cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA) if s<1 else img
# Loads {scene: (hazy, clear reference)} pairs and resizes both to max_dim
def load_pairs(max_dim,folder=DATA):
    pairs={}
    for hazy in sorted(p for p in folder.glob("*_hazy.*") if p.suffix.lower() in (".jpg",".jpeg",".png")):
        name=hazy.stem.replace("_hazy","")
        ref=next(iter(folder.glob(f"{name}_clear_reference.*")),None)
        if ref is None:
            continue
        h=cv2.cvtColor(cv2.imread(str(hazy)),cv2.COLOR_BGR2RGB)
        c=cv2.cvtColor(cv2.imread(str(ref)),cv2.COLOR_BGR2RGB)
        s=min(1.0,max_dim/max(h.shape[:2]))
        h=cv2.resize(h,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
        c=cv2.resize(c,(h.shape[1],h.shape[0]),interpolation=cv2.INTER_AREA)
        pairs[name]=(h,c)
    return pairs
# Loads the clear real street photos (empty if not downloaded)
def load_streets(max_dim):
    if not URBAN_STREETS.exists():
        return {}
    return {p.stem.split("_")[0]:_resize(cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB),max_dim)
        for p in sorted(URBAN_STREETS.glob("*.jpg"))}
# Writes the synthetic city test set: clear, fogged (true depth) and 16-bit depth PNGs
def export_urban(n=6):
    URBAN_SYN.mkdir(parents=True,exist_ok=True)
    for i,(img,depth) in enumerate(C.make_urban_set(n)):
        stem=URBAN_SYN/f"city{i+1:02d}"
        for suffix,im in (("_clear_reference",img),("_hazy",C.simulate_fog(img,depth,URBAN_BETA))):
            cv2.imwrite(f"{stem}{suffix}.png",cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{stem}_depth.png",np.round(depth*65535).astype(np.uint16))
# Reads {scene: (clear, depth)} of the synthetic set, writing it first if it is missing
def load_urban():
    if not list(URBAN_SYN.glob("city*_depth.png")):
        export_urban()
    out={}
    for d in sorted(URBAN_SYN.glob("city*_depth.png")):
        name=d.stem.replace("_depth","")
        clear=cv2.cvtColor(cv2.imread(str(URBAN_SYN/f"{name}_clear_reference.png")),cv2.COLOR_BGR2RGB)
        out[name]=(clear,cv2.imread(str(d),cv2.IMREAD_UNCHANGED).astype(np.float32)/65535.0)
    return out
# Runs every method on every scene and collects PSNR, SSIM, visibility, downstream and optional detection metrics
def run(methods,scenes,detection=False):
    rows,outputs,ref_boxes=[],{},{}
    for scene,(inp,clear) in scenes.items():
        base=C.quality_metrics(inp)
        if detection:
            ref_boxes[scene]=detect.detect(clear)
        for method,P in methods.items():
            t=time.perf_counter()
            out=C.dehaze_pipeline(inp,P(inp) if callable(P) else P)
            sec=time.perf_counter()-t
            q,vis=C.quality_metrics(out),C.visibility_metrics(inp,out)
            row=dict(scene=scene,method=method,psnr=C.psnr(out,clear),ssim=C.ssim(out,clear),
                contrast_x=q["contrast"]/base["contrast"],edges_pct=q["edges"],vis_e=vis["e"],vis_r=vis["r"],
                vis_clip=vis["sigma"],seconds=sec,**C.downstream_metrics(out,clear))
            if detection:
                boxes=detect.detect(out)
                rec,prec,_=detect.match(boxes,ref_boxes[scene])
                row.update(det_found=len(boxes),det_ref=len(ref_boxes[scene]),det_recall=rec,det_precision=prec)
            rows.append(row)
            outputs[(scene,method)]=out
    return rows,outputs
# Mean of one metric over all scenes for a method (NaN ignored)
def mean_of(rows,method,key):
    return float(np.nanmean([r[key] for r in rows if r["method"]==method]))
# Per-method means of the main metrics
def summarise(rows):
    methods=list(dict.fromkeys(r["method"] for r in rows))
    return [(m,*[mean_of(rows,m,k) for k in ("psnr","ssim","contrast_x","edges_pct","vis_e","vis_r","seconds")])
        for m in methods]
# Formats the summary as a markdown table
def md_table(summary):
    lines=["| Method | PSNR (dB) | SSIM | Contrast x | Edge pixels (%) | Visible-edge gain e | Gradient ratio r | Seconds/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m,p,s,c,e,ve,vr,t in summary:
        lines.append(f"| {m} | {p:.1f} | {s:.3f} | {c:.2f} | {e:.1f} | {ve:+.2f} | {vr:.2f} | {t:.2f} |")
    return "\n".join(lines)
# Markdown table of the Canny, Hough and detection results, and in how many scenes edge F1 beats the input
def downstream_table(rows):
    methods=list(dict.fromkeys(r["method"] for r in rows))
    scenes=list(dict.fromkeys(r["scene"] for r in rows))
    det="det_recall" in rows[0]
    base={r["scene"]:r["edge_f1"] for r in rows if r["method"]==INPUT}
    head=("| Method | Canny edges (% px) | Edge F1 vs clear | Edge F1, auto thresholds | Hough lines | Vertical (building) "
        "| Oblique (lane) | Line recall vs clear | F1 above input |"+(" Vehicles/people found | Detection recall vs clear | Detection precision |" if det else ""))
    lines=[head,"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"+(" ---: | ---: | ---: |" if det else "")]
    for m in methods:
        sel=[r for r in rows if r["method"]==m]
        mean= lambda k:float(np.nanmean([r[k] for r in sel]))
        better="-" if m==INPUT else f"{sum(r['edge_f1']>base[r['scene']]+1e-9 for r in sel)}/{len(scenes)}"
        line=(f"| {m} | {mean('canny_pct'):.1f} | {mean('edge_f1'):.3f} | {mean('edge_f1_auto'):.3f} | {mean('lines'):.0f} | "
            f"{mean('lines_vertical'):.0f} | {mean('lines_oblique'):.0f} | {mean('line_recall'):.3f} | {better} |")
        if det:
            line+=f" {mean('det_found'):.1f} (clear photo: {mean('det_ref'):.1f}) | {mean('det_recall'):.3f} | {mean('det_precision'):.3f} |"
        lines.append(line)
    return "\n".join(lines)
# Appends one section (summary table and downstream table) to the markdown report
def section(title,rows,md,level="##"):
    md+=[f"{level} {title}","",md_table(summarise(rows)),"","Downstream test (Canny + Hough"+
        (" + vehicle/pedestrian detection" if "det_recall" in rows[0] else "")+", means per image):","",
        downstream_table(rows),""]
# Writes result rows to a CSV file with floats rounded to 4 places
def write_csv(path,rows):
    names=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=names)
        w.writeheader()
        for r in rows:
            w.writerow({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
# Tiles images side by side with a text label bar and saves a PNG
def montage(images,labels,path,width=360):
    tiles=[]
    for img,lab in zip(images,labels):
        if img.ndim==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        t=cv2.resize(img,(width,int(width*img.shape[0]/img.shape[1])),interpolation=cv2.INTER_AREA)
        bar=np.full((28,width,3),255,np.uint8)
        cv2.putText(bar,lab,(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(30,30,30),1,cv2.LINE_AA)
        tiles.append(np.vstack([bar,t]))
    h=min(t.shape[0] for t in tiles)
    cv2.imwrite(str(path),cv2.cvtColor(np.hstack([t[:h] for t in tiles]),cv2.COLOR_RGB2BGR))
# Draws the Hough segments in red on the image
def lines_overlay(img):
    out=img.copy()
    for x1,y1,x2,y2 in C.hough_lines(C.canny_edges(img)):
        cv2.line(out,(int(x1),int(y1)),(int(x2),int(y2)),(255,40,40),1,cv2.LINE_AA)
    return out
# Draws the detector's boxes on the image (green vehicles, yellow people)
def boxes_overlay(img):
    out=img.copy()
    for x1,y1,x2,y2,sc,name in detect.detect(img):
        col=(255,220,0) if name=="person" else (30,220,30)
        cv2.rectangle(out,(int(x1),int(y1)),(int(x2),int(y2)),col,2)
    return out
# Compares the dark-channel transmission with the true exp(-beta * depth): MAE, correlation, airlight error
def transmission_rows(urban):
    rows=[]
    t0=C.P_BALANCED["t0"]
    for beta in (0.8,1.6,2.4,3.2):
        for name,(clear,depth) in urban.items():
            hazy=C.simulate_fog(clear,depth,beta)
            t_est,A=C.estimate_transmission(hazy,C.P_BALANCED["omega"],t0,C.P_BALANCED["patch"])
            t_true=np.clip(C.true_transmission(depth,beta),t0,1.0)
            rows.append(dict(scene=name,beta=beta,t_mae=float(np.abs(t_est-t_true).mean()),
                    t_corr=float(np.corrcoef(t_est.ravel(),t_true.ravel())[0,1]),
                    airlight_err=float(np.abs(A*255.0-C.FOG_AIRLIGHT).mean())))
    return rows
# Fog-density sweep: fixed Balanced default against the haze-adaptive strength
def sweep_table(datasets,betas=(0.4,0.8,1.6,2.4,3.2)):
    methods={INPUT:OFF,FULL:C.P_BALANCED,AUTOSTR:auto_strength}
    lines=["| Scenes | Fog beta | Haze level | Strength used | Input SSIM | Fixed SSIM | Adaptive SSIM | Fixed PSNR | Adaptive PSNR "
        "| Fixed edge F1 | Adaptive edge F1 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rows_all=[]
    for label,ds in datasets.items():
        for beta in betas:
            scenes={n:(C.simulate_fog(c,d,beta),c) for n,(c,d) in ds.items()}
            rows,_=run(methods,scenes)
            hz=[C.haze_level(inp) for inp,_ in scenes.values()]
            g= lambda m,k:mean_of(rows,m,k)
            lines.append(f"| {label} | {beta:g} | {np.mean(hz):.2f} | {np.mean([C.strength_scale(h) for h in hz]):.2f} | "
                f"{g(INPUT,'ssim'):.3f} | {g(FULL,'ssim'):.3f} | {g(AUTOSTR,'ssim'):.3f} | {g(FULL,'psnr'):.1f} | "
                f"{g(AUTOSTR,'psnr'):.1f} | {g(FULL,'edge_f1'):.3f} | {g(AUTOSTR,'edge_f1'):.3f} |")
            rows_all+=[dict(r,dataset=label,beta=beta) for r in rows]
    return rows_all,"\n".join(lines)
# Grid-search fit of strength = clip(a * haze - b) on two scene types, tested on the held-out third
def fit_strength_rule(datasets,betas=(0.4,0.8,1.2,1.6,2.4,3.2),gs=(0.0,0.25,0.5,0.75,1.0,1.25)):
    # Balanced settings with the strengths and omega scaled by g
    def scaled(g):
        P=dict(C.P_BALANCED)
        for k in ("hist","freq","smooth","sharpen"):
            P[k]=C.P_BALANCED[k]*g
        P["omega"],P["use_dcp"]=(min(0.95,C.P_BALANCED["omega"]*(0.35+0.65*g)),True) if g>0 else (0.0,False)
        return P
    recs=[]
    for label,ds in datasets.items():
        for beta in betas:
            for n,(c,d) in ds.items():
                f=C.simulate_fog(c,d,beta)
                sc=[(C.ssim(o,c)+C.psnr(o,c)/40.0) for o in (C.dehaze_pipeline(f,scaled(g)) if g>0 else f for g in gs)]
                recs.append((label,beta,C.haze_level(f),np.array(sc)))
    score= lambda r,g:float(np.interp(g,gs,r[3]))
    rule= lambda h,a,b:float(np.clip(a*h-b,C.STRENGTH_MIN,1.0))
    grid=[(a,b) for a in np.arange(1.0,3.01,0.1) for b in np.arange(0.0,1.01,0.05)]
    fit= lambda tr:max(grid,key=lambda p:np.mean([score(r,rule(r[2],*p)) for r in tr]))
    a,b=fit(recs)
    lines=[f"Fit on all scene types: strength = clip({a:.1f} x haze - {b:.2f}, {C.STRENGTH_MIN}, 1) "
        f"(in dehaze_core: {C.HAZE_SLOPE} and {C.HAZE_OFFSET}).","",
        "| Held-out scene type | Rule fit on the others | Score, fixed default | Score, adaptive | Score, per-image best | "
        "Light fog only (beta <= 0.8): fixed | adaptive |","| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for held in datasets:
        tr,te=[r for r in recs if r[0]!=held],[r for r in recs if r[0]==held]
        pa,pb=fit(tr)
        lt=[r for r in te if r[1]<=0.8]
        lines.append(f"| {held} | {pa:.1f} x haze - {pb:.2f} | {np.mean([score(r,1.0) for r in te]):.3f} | "
            f"{np.mean([score(r,rule(r[2],pa,pb)) for r in te]):.3f} | {np.mean([r[3].max() for r in te]):.3f} | "
            f"{np.mean([score(r,1.0) for r in lt]):.3f} | {np.mean([score(r,rule(r[2],pa,pb)) for r in lt]):.3f} |")
    return "\n".join(lines)
# Mean SSIM, edge F1 and visible-edge gain with each pipeline part removed in turn
def ablation_table(datasets,det_labels=()):
    res={label:run(ABLATION,scenes,detection=label in det_labels)[0] for label,scenes in datasets.items()}
    cols= lambda l:["SSIM","F1","e"]+(["detection recall"] if l in det_labels else [])
    lines=["| Pipeline | "+" | ".join(f"{l}: "+" | ".join(cols(l)) for l in res)+" |",
        "| --- | "+" | ".join(" | ".join("---:" for _ in cols(l)) for l in res)+" |"]
    for m in ABLATION:
        cells=[]
        for l,rows in res.items():
            cells+=[f"{mean_of(rows,m,'ssim'):.3f}",f"{mean_of(rows,m,'edge_f1'):.3f}",f"{mean_of(rows,m,'vis_e'):+.2f}"]
            if l in det_labels:
                cells.append(f"{mean_of(rows,m,'det_recall'):.3f}")
        lines.append(f"| {m} | "+" | ".join(cells)+" |")
    return "\n".join(lines)
# Smallest difference between two line angles (period 180 degrees)
def _angle_err(a,b):
    return abs((a-b+90.0)%180.0-90.0)
# Checks estimate_blur on photos blurred by known kinds, amounts and angles
def blur_estimation(real,urban):
    angles=(0,25,45,70,90,135,160)
    conds=[("none",0.0,b) for b in (0.0,1.6,3.0)]
    conds+=[(k,a,1.6) for k in ("Gaussian","Motion") for a in (1.0,2.0,3.0,4.0)]
    conds+=[("Gaussian",2.0,3.0),("Motion",2.0,3.0)]
    rows=[]
    for kind,amount,beta in conds:
        for ds_name,ds in (("real photos",real),("synthetic cities",urban)):
            for ang in (angles if kind=="Motion" else (0,)):
                for scene,clear in ds.items():
                    img=C.simulate_fog(clear,None,beta) if beta else clear
                    est=C.estimate_blur(C.simulate_blur(img,amount,kind,ang) if amount else img)
                    rows.append(dict(dataset=ds_name,true_kind=kind,true_amount=amount,true_angle=ang if kind=="Motion" else "",
                            fog_beta=beta,scene=scene,est_kind=est["kind"],est_amount=est["amount"],
                            est_angle=est["angle"] if est["kind"]=="Motion" else ""))
    lines=["| True blur | Fog beta | Real: estimate (px) | kind right | abs. error (px) | angle error (deg) "
        "| Synthetic: estimate (px) | kind right | abs. error (px) | angle error (deg) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind,amount,beta in conds:
        cells=[]
        for ds_name in ("real photos","synthetic cities"):
            sel=[r for r in rows if r["dataset"]==ds_name and r["true_kind"]==kind
                and r["true_amount"]==amount and r["fog_beta"]==beta]
            right=np.mean([r["est_kind"]==kind for r in sel])
            err=np.mean([abs(r["est_amount"]-amount) for r in sel])
            ang=[_angle_err(r["est_angle"],r["true_angle"]) for r in sel if kind=="Motion" and r["est_kind"]=="Motion"]
            cells+=[f"{np.mean([r['est_amount'] for r in sel]):.2f}",f"{right*100:.0f}%",f"{err:.2f}",
                f"{np.mean(ang):.1f}" if ang else "-"]
        label="none (sharp)" if kind=="none" else f"{kind} {amount:g} px"+(" (7 angles)" if kind=="Motion" else "")
        lines.append(f"| {label} | {beta:g} | "+" | ".join(cells)+" |")
    ang_lines=["| True motion angle | Real: mean angle error (deg) | Real: estimate at 3 px | Synthetic: mean angle error (deg) | Synthetic: estimate at 3 px |",
        "| ---: | ---: | ---: | ---: | ---: |"]
    for ang in angles:
        cells=[]
        for ds_name in ("real photos","synthetic cities"):
            sel=[r for r in rows if r["dataset"]==ds_name and r["true_kind"]=="Motion" and r["true_angle"]==ang
                and r["true_amount"]>=2.0 and r["fog_beta"]==1.6 and r["est_kind"]=="Motion"]
            s3=[r["est_amount"] for r in sel if r["true_amount"]==3.0]
            cells+=[f"{np.mean([_angle_err(r['est_angle'],ang) for r in sel]):.1f}",f"{np.mean(s3):.2f}"]
        ang_lines.append(f"| {ang} | "+" | ".join(cells)+" |")
    return rows,"\n".join(lines),"\n".join(ang_lines)
# Leave-one-scene-out tuning over random settings, against the shipped default and the per-scene best
def cross_validate(pairs,n=60,seed=0,max_dim=400):
    rng=np.random.default_rng(seed)
    space=dict(hist=(0,60),freq=(0,60),smooth=(0,50),sharpen=(0,80),omega=(0.6,0.95),balance=(0,100))
    # Random-search candidates drawn uniformly from the settings space
    cands=[dict(C.P_BALANCED,**{k:float(rng.uniform(*v)) for k,v in space.items()}) for _ in range(n)]
    small={k:(_resize(h,max_dim),_resize(c,max_dim)) for k,(h,c) in pairs.items()}
    sc= lambda P:np.array([[C.ssim(o,c),C.psnr(o,c)] for o,c in
            ((C.dehaze_pipeline(h,P),c) for h,c in small.values())])
    S=np.stack([sc(P) for P in cands]) # (candidate, scene, [ssim, psnr])
    D=sc(C.P_BALANCED)
    # Combined score = SSIM + PSNR / 40
    comb=S[...,0]+S[...,1]/40.0
    held=[]
    for i in range(len(small)):
        others=[j for j in range(len(small)) if j!=i]
        best=int(comb[:,others].mean(axis=1).argmax())
        held.append(S[best,i])
    held=np.array(held)
    oracle=S[comb.argmax(axis=0),np.arange(len(small))]
    return "\n".join([
            f"| Settings | SSIM | PSNR (dB) |","| --- | ---: | ---: |",
            f"| Shipped default (tuned on these photos) | {D[:,0].mean():.3f} | {D[:,1].mean():.2f} |",
            f"| Tuned on the other {len(small)-1} scenes, applied to the held-out one ({n} random candidates) | {held[:,0].mean():.3f} | {held[:,1].mean():.2f} |",
            f"| Best of the {n} candidates for each scene separately (upper bound) | {oracle[:,0].mean():.3f} | {oracle[:,1].mean():.2f} |",
            f"| Unprocessed | {np.mean([C.ssim(h,c) for h,c in small.values()]):.3f} | {np.mean([C.psnr(h,c) for h,c in small.values()]):.2f} |"])
# Compares independent-frame and temporal processing on a synthetic traffic clip (flicker, PSNR, SSIM)
def video_table(n=30,noise=2.0,seed=0):
    clear,depth=C.make_traffic_clip(n)
    rng=np.random.default_rng(seed)
    fog=[np.clip(C.simulate_fog(f,depth,BETA).astype(np.float32)+rng.normal(0,noise,f.shape),0,255).astype(np.uint8)
        for f in clear]
    lines=["| Processing | Flicker (grey levels) | PSNR (dB) | SSIM | Seconds/frame |","| --- | ---: | ---: | ---: | ---: |",
        f"| Foggy input | {flicker(fog,fog):.2f} | {np.mean([C.psnr(o,c) for o,c in zip(fog,clear)]):.1f} | "
        f"{np.mean([C.ssim(o,c) for o,c in zip(fog,clear)]):.3f} | - |"]
    outs={}
    for name,kw in (("Every frame on its own",dict(smoothing=0.0)),("Temporal (smoothed estimates + denoising)",dict()),
        ("Every frame on its own, strength from haze level",dict(smoothing=0.0,auto=True)),
        ("Temporal, strength from haze level",dict(auto=True)),
        ("Temporal, CLAHE off",dict(P=dict(C.P_BALANCED,clahe=0))),
        ("Temporal, CLAHE off, strength from haze level",dict(P=dict(C.P_BALANCED,clahe=0),auto=True))):
        d=TemporalDehazer(**kw)
        t=time.perf_counter()
        out=[d.process(f) for f in fog]
        sec=(time.perf_counter()-t)/n
        outs[name]=out
        lines.append(f"| {name} | {flicker(fog,out):.2f} | {np.mean([C.psnr(o,c) for o,c in zip(out,clear)]):.1f} | "
            f"{np.mean([C.ssim(o,c) for o,c in zip(out,clear)]):.3f} | {sec:.2f} |")
    k=n//2
    montage([fog[k],outs["Every frame on its own"][k],outs["Temporal (smoothed estimates + denoising)"][k],clear[k]],
        ["foggy frame","frame on its own","temporal","clear frame"],OUT/"figures"/"video_frame.png")
    return "\n".join(lines)
# Runs the nine experiments and writes the CSV files, figures and summary.md
# Escapes text for SVG
def _esc(t):
    return str(t).replace("&","&amp;").replace("<","&lt;")
SERIES_COLOURS=["#9a9a9a","#1e88e5","#d32f2f","#43a047","#f9a825"]
# Horizontal grouped bar chart as SVG text: one group per label, one bar per series (name, values), with the value printed on each bar
def bar_svg(title,labels,series,colours=SERIES_COLOURS,fmt="{:.3f}",vmin=0.0):
    lw,w,bh=300,820,15
    gh=len(series)*(bh+2)+12
    h=46+len(labels)*gh+22*len(series)
    vmax=max(max(v) for _,v in series)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/><text x="10" y="22" font-size="15" font-weight="bold" fill="#222">{_esc(title)}</text>']
    for j,(name,_) in enumerate(series): # legend
        out.append(f'<rect x="{10+j*190}" y="30" width="12" height="12" fill="{colours[j%len(colours)]}"/><text x="27" y="41" fill="#333" transform="translate({j*190},0)">{_esc(name)}</text>')
    for i,lab in enumerate(labels):
        y0=52+i*gh
        out.append(f'<text x="{lw-8}" y="{y0+gh/2:.0f}" text-anchor="end" fill="#333">{_esc(lab)}</text>')
        for j,(_,vals) in enumerate(series):
            v=vals[i]
            bw=max(1.0,(v-vmin)/max(vmax-vmin,1e-9)*(w-lw-80))
            y=y0+j*(bh+2)
            out.append(f'<rect x="{lw}" y="{y}" width="{bw:.1f}" height="{bh}" fill="{colours[j%len(colours)]}"/>'
                f'<text x="{lw+bw+5:.1f}" y="{y+bh-3}" fill="#333">{fmt.format(v)}</text>')
    return "".join(out)+"</svg>"
# Line chart as SVG text: series are (name, ys) over shared xs, with markers
def line_chart_svg(title,xs,series,xlabel,ylabel,colours=SERIES_COLOURS,fmt="{:.2f}"):
    w,h,l,b,t,r=640,340,60,50,56,20
    lo=min(min(v) for _,v in series)
    hi=max(max(v) for _,v in series)
    pad=(hi-lo)*0.08+1e-9
    lo,hi=lo-pad,hi+pad
    X= lambda x:l+(x-min(xs))/max(max(xs)-min(xs),1e-9)*(w-l-r)
    Y= lambda y:h-b-(y-lo)/(hi-lo)*(h-b-t)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/><text x="10" y="22" font-size="15" font-weight="bold" fill="#222">{_esc(title)}</text>']
    for k in range(5):
        v=lo+(hi-lo)*k/4
        out.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{w-r}" y2="{Y(v):.1f}" stroke="#e6e6e6"/><text x="{l-6}" y="{Y(v)+4:.1f}" text-anchor="end" fill="#666">{fmt.format(v)}</text>')
    for x in xs:
        out.append(f'<text x="{X(x):.1f}" y="{h-b+16}" text-anchor="middle" fill="#666">{x:g}</text>')
    out.append(f'<text x="{(l+w-r)/2:.0f}" y="{h-10}" text-anchor="middle" fill="#333">{_esc(xlabel)}</text>'
        f'<text x="14" y="{(t+h-b)/2:.0f}" fill="#333" transform="rotate(-90 14 {(t+h-b)/2:.0f})" text-anchor="middle">{_esc(ylabel)}</text>')
    for j,(name,ys) in enumerate(series):
        c=colours[j%len(colours)]
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="'+" ".join(f"{X(x):.1f},{Y(y):.1f}" for x,y in zip(xs,ys))+'"/>')
        out+= [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{c}"/>' for x,y in zip(xs,ys)]
        out.append(f'<rect x="{10+j*200}" y="30" width="12" height="12" fill="{c}"/><text x="{27+j*200}" y="41" fill="#333">{_esc(name)}</text>')
    return "".join(out)+"</svg>"
# Reads a results CSV back as dicts with numbers converted
def read_csv(path):
    rows=list(csv.DictReader(open(path,encoding="utf-8")))
    for r in rows:
        for k,v in r.items():
            try:
                r[k]=float(v)
            except (TypeError,ValueError):
                pass
    return rows
# Writes the SVG charts to results/figures: methods vs baselines, ablation, fog-density sweep (from strength_sweep.csv) and runtime vs image size
def make_charts(pairs):
    fig=OUT/"figures"
    fig.mkdir(parents=True,exist_ok=True)
    # Chart 1: simple one-technique baselines against our modules and the full pipeline, mean over the real fog scenes
    outs={}
    for n,(hazy,clear) in pairs.items():
        o={INPUT:hazy,"Histogram stretching only":C.dehaze_pipeline(hazy,METHODS["Histogram stretching only"]),
            "Three course modules together":C.dehaze_pipeline(hazy,C.P_MODULES),FULL:C.dehaze_pipeline(hazy,C.P_BALANCED)}
        o.update(C.baseline_methods(hazy))
        outs[n]=o
    names=list(next(iter(outs.values())))
    def mean_metric(name,fn):
        return float(np.nanmean([fn(outs[n][name],pairs[n][1]) for n in pairs]))
    psnr_v=[mean_metric(m,C.psnr) for m in names]
    ssim_v=[mean_metric(m,C.ssim) for m in names]
    f1_v=[mean_metric(m,lambda a,b:C.downstream_metrics(a,b)["edge_f1"]) for m in names]
    (fig/"chart_methods_ssim_f1.svg").write_text(bar_svg(f"Methods against simple baselines, mean over {len(pairs)} real fog scenes (higher is better)",
        names,[("SSIM",ssim_v),("Edge F1",f1_v)],[SERIES_COLOURS[1],SERIES_COLOURS[2]]),encoding="utf-8")
    (fig/"chart_methods_psnr.svg").write_text(bar_svg("PSNR against the clear photo (dB)",names,[("PSNR",psnr_v)],[SERIES_COLOURS[1]],fmt="{:.1f}"),encoding="utf-8")
    # Chart 2: ablation, what removing each part costs
    rows,_=run(ABLATION,pairs)
    labs=list(ABLATION)
    (fig/"chart_ablation.svg").write_text(bar_svg("Ablation on real fog: what each part adds",labs,
        [("SSIM",[mean_of(rows,m,"ssim") for m in labs]),("Edge F1",[mean_of(rows,m,"edge_f1") for m in labs])],
        [SERIES_COLOURS[1],SERIES_COLOURS[2]]),encoding="utf-8")
    # Chart 3: fog-density sweep from results/strength_sweep.csv (run evaluate.py once to create it)
    sweep=OUT/"strength_sweep.csv"
    if sweep.exists():
        rows=read_csv(sweep)
        for ds in dict.fromkeys(r["dataset"] for r in rows):
            sel=[r for r in rows if r["dataset"]==ds]
            betas=sorted({r["beta"] for r in sel})
            series=[(lab,[float(np.mean([r["ssim"] for r in sel if r["method"]==m and r["beta"]==b])) for b in betas])
                for lab,m in (("Foggy input",INPUT),("Fixed default",FULL),("Strength from haze level",AUTOSTR))]
            tag=ds.replace(" ","_")
            (fig/f"chart_sweep_{tag}.svg").write_text(line_chart_svg(f"SSIM against fog density: {ds}",betas,series,"fog density beta","SSIM vs clear photo",
                [SERIES_COLOURS[0],SERIES_COLOURS[1],SERIES_COLOURS[2]],"{:.2f}"),encoding="utf-8")
    # Chart 4: runtime against image size for the full and the Traffic preset
    sizes=(320,480,640,960,1280)
    times={"Full pipeline":[],"Traffic preset":[]}
    for w in sizes:
        img,_=C.make_sample_scene(w,round(w*0.625))
        for lab,P in (("Full pipeline",C.P_BALANCED),("Traffic preset",C.P_TRAFFIC)):
            best=1e9
            for _ in range(2):
                t=time.perf_counter()
                C.dehaze_pipeline(img,P)
                best=min(best,time.perf_counter()-t)
            times[lab].append(best)
    (fig/"chart_runtime.svg").write_text(line_chart_svg("Processing time against image width",sizes,list(times.items()),"image width (px)","seconds",
        [SERIES_COLOURS[2],SERIES_COLOURS[1]],"{:.2f}"),encoding="utf-8")
    print("charts written to",fig)

def load_unpaired_real_fog(max_dim):
    """Load real fog/smog photographs without assuming a clear reference exists."""
    supported={".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images={}
    if not URBAN_REAL_FOGGY.exists():
        return images
    for path in sorted(URBAN_REAL_FOGGY.iterdir()):
        if path.suffix.lower() not in supported:
            continue
        raw=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if raw is None:
            continue
        images[path.stem]=_resize(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),max_dim)
    return images

def colorfulness_metric(img):
    """Hasler-Susstrunk colourfulness score; descriptive, not a perceptual-quality score."""
    b,g,r=cv2.split(img.astype(np.float32))
    rg=np.abs(r-g)
    yb=np.abs(0.5*(r+g)-b)
    return float(np.sqrt(rg.std()**2+yb.std()**2)+0.3*np.sqrt(rg.mean()**2+yb.mean()**2))

def unpaired_metrics(inp,out):
    """Metrics available when a real foggy photo has no aligned clear reference."""
    q0,q1=C.quality_metrics(inp),C.quality_metrics(out)
    vis=C.visibility_metrics(inp,out)
    gray=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
    clipped=float(np.mean((gray<=1)|(gray>=254)))
    return {
        "visible_edge_gain_e":vis["e"], "gradient_ratio_r":vis["r"],
        "newly_clipped_share":vis["sigma"], "contrast_before":q0["contrast"],
        "contrast_after":q1["contrast"], "entropy_before":q0["entropy"],
        "entropy_after":q1["entropy"], "gradient_before":q0["gradient"],
        "gradient_after":q1["gradient"], "edge_pixels_before_pct":q0["edges"],
        "edge_pixels_after_pct":q1["edges"],
        "colorfulness_before":colorfulness_metric(inp),
        "colorfulness_after":colorfulness_metric(out),
        "output_clipped_pixel_pct":100.0*clipped,
    }

def contact_sheet(rows, path, tile_width=300):
    """Create a labelled multi-row contact sheet from (image, label) pairs."""
    tiles=[]
    for img,label in rows:
        h,w=img.shape[:2]
        scale=min(1.0,tile_width/w,190/h)
        resized=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
        canvas=np.full((220,tile_width,3),245,np.uint8)
        x=(tile_width-resized.shape[1])//2
        y=26+(190-resized.shape[0])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        cv2.putText(canvas,label[:42],(7,18),cv2.FONT_HERSHEY_SIMPLEX,0.43,(20,20,20),1,cv2.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    cols=3
    rows_n=(len(tiles)+cols-1)//cols
    sheet=np.full((rows_n*220,cols*tile_width,3),255,np.uint8)
    for i,tile in enumerate(tiles):
        y=(i//cols)*220; x=(i%cols)*tile_width
        sheet[y:y+220,x:x+tile_width]=tile
    cv2.imwrite(str(path),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))

def evaluate_unpaired_real_world(max_dim=800):
    """Evaluate real urban fog photos without inventing unavailable reference scores."""
    images=load_unpaired_real_fog(max_dim)
    if not images:
        print(f"No readable real-fog images found in {URBAN_REAL_FOGGY}")
        return
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"figures").mkdir(parents=True,exist_ok=True)
    methods={
        "Fog removal only (DCP)":dict(C.P_BALANCED,hist=0,freq=0,smooth=0,sharpen=0,clahe=0,
            balance=0,restore=0,gamma=1.0,sat=1.0,far=0,deblur=0.0),
        "Traffic monitoring":C.P_TRAFFIC,
        "Balanced (haze-adaptive)":lambda im:C.auto_strength_params(im,C.P_BALANCED)[0],
        "Course modules only":C.P_MODULES,
    }
    records=[]; sheets=[]; human=[]
    for name,inp in images.items():
        sheets.append((inp,f"{name} | original"))
        human.append({"image":name,"capture_date":"","capture_time":"","location_or_source":"",
            "weather_or_haze_description":"","lighting":"","scene_type":"",
            "visibility_score_1_5":"","artifact_score_1_5":"","color_naturalness_1_5":"","reviewer_notes":""})
        for method,settings in methods.items():
            start=time.perf_counter()
            P=settings(inp) if callable(settings) else settings
            out=C.dehaze_pipeline(inp,P)
            elapsed=time.perf_counter()-start
            metrics=unpaired_metrics(inp,out)
            records.append({"image":name,"method":method,"processing_seconds":elapsed,**metrics})
            if method=="Balanced (haze-adaptive)":
                sheets.append((out,f"{name} | balanced"))
    write_csv(OUT/"real_world_unpaired.csv",records)
    write_csv(OUT/"human_review_template.csv",human)
    contact_sheet(sheets,OUT/"figures"/"real_world_unpaired_contact_sheet.jpg",tile_width=300)
    report=["# Real-world urban fog validation (unpaired images)","",
        f"Images processed: **{len(images)}** from `{URBAN_REAL_FOGGY.relative_to(ROOT)}`. Maximum long edge: {max_dim}px.","",
        "These are real fog/smog photographs without aligned clear references. Therefore PSNR, SSIM, reference edge-F1, and true object-detection recall are deliberately not reported for this set.","",
        "## Mean no-reference indicators by method","",
        "| Method | Visible-edge gain e | Gradient ratio r | Newly clipped share | Contrast change | Entropy change | Processing (s/image) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        sel=[r for r in records if r["method"]==method]
        mean=lambda key:float(np.nanmean([r[key] for r in sel]))
        report.append(f"| {method} | {mean('visible_edge_gain_e'):+.3f} | {mean('gradient_ratio_r'):.3f} | {100*mean('newly_clipped_share'):.2f}% | {mean('contrast_after')-mean('contrast_before'):+.2f} | {mean('entropy_after')-mean('entropy_before'):+.3f} | {mean('processing_seconds'):.3f} |")
    report += ["", "## Interpretation and limitations", "",
        "- Visible-edge gain and gradient ratio measure contrast/edge changes, not recovery of the true scene. Sharpening and noise can increase them.",
        "- Entropy, contrast, and colourfulness are descriptive statistics; higher values are not automatically better.",
        "- Newly clipped share is a warning indicator for lost shadow/highlight detail.",
        "- The human-review template is intentionally blank. Complete it after independent visual review; do not treat generated scores as human ratings.",
        "- For stronger field validation, collect repeated daytime/nighttime samples across light, moderate, and dense fog; record capture conditions and source/licence; preserve original files; and obtain aligned clear references where feasible.",
        "- For traffic claims, annotate vehicles/pedestrians and calculate precision, recall, and mAP against those annotations. Detector agreement with another model is not ground truth.","",
        "## Output files", "",
        "- `real_world_unpaired.csv`: per-image, per-method metrics and runtime.",
        "- `human_review_template.csv`: blank reviewer metadata and scoring sheet.",
        "- `figures/real_world_unpaired_contact_sheet.jpg`: originals alongside haze-adaptive outputs.",
        ""]
    (OUT/"real_world_validation.md").write_text("\n".join(report),encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved real-world validation outputs to {OUT}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-dim",type=int,default=800,help="downscale test photos to this long edge")
    ap.add_argument("--export-urban",action="store_true",help="(re)write the synthetic city test set to assets/")
    ap.add_argument("--charts",action="store_true",help="only (re)write the SVG charts in results/figures")
    ap.add_argument("--fit-strength",action="store_true",help="re-run the fit of the haze-adaptive strength rule")
    ap.add_argument("--real-world-only",action="store_true",help="evaluate the bundled unpaired real urban fog photos and exit")
    args=ap.parse_args()
    (OUT/"figures").mkdir(parents=True,exist_ok=True)
    if args.real_world_only:
        evaluate_unpaired_real_world(args.max_dim)
        return
    if args.export_urban:
        export_urban()
    pairs=load_pairs(args.max_dim)
    if args.charts:
        make_charts(pairs)
        return
    urban=load_urban()
    streets=load_streets(args.max_dim)
    det_ok=bool(streets) and detect.available()
    md=["# Evaluation results","",
        f"{len(pairs)} real garden/park scenes with a clear reference, {len(streets)} real street photos fogged by simulation, "
        f"{len(urban)} synthetic streets with true depth; photos scaled to at most {args.max_dim}px on the long edge. "
        "PSNR and SSIM are measured against the clear photo; the contrast gain is relative to the foggy input and edge pixels "
        "are the share of pixels with a gradient above 12. *Visible-edge gain e* and *gradient ratio r* (Hautiere et al.) "
        "compare a result with its foggy input and need no clear photo: e is the relative gain in visible edges, r the mean "
        "gradient ratio on them (above 1 = contrast gained). Each value is the mean over the scenes. The downstream test runs "
        "a standard Canny edge detector (fixed thresholds 50/150, which is what makes fog costly) and a probabilistic Hough line "
        "detector on every result: *Edge F1* compares its edges with the clear photo's (2 px tolerance); *auto thresholds* repeats "
        "that with Canny thresholds set from each image's median brightness, which compensates for low contrast and so shows the "
        "gain that is not just a brighter or higher-contrast picture; *line recall* is the share of the clear photo's lines found "
        "again; *vertical* lines are within 15 degrees of vertical (building edges) and *oblique* lines are 20-70 degrees from "
        "horizontal (lane and roof lines). More edges or lines is not better by itself: noise and over-sharpening add false ones, "
        "and the Hough line count also depends on how segments merge, so compare with F1 and recall.",""]
    # Experiment 1: real fog photos against their clear references
    rows,outs=run(METHODS,pairs)
    write_csv(OUT/"real_fog.csv",rows)
    real_rows=rows
    section("1. Real fog (7 garden and park scenes)",rows,md)
    s="scene12" if "scene12" in pairs else next(iter(pairs))
    montage([pairs[s][0],outs[(s,"The three course modules together")],outs[(s,FULL)],pairs[s][1]],
        ["foggy photo","three modules only","full pipeline","clear reference"],OUT/"figures"/"real_fog_montage.png")
    s="scene14" if "scene14" in pairs else s # the most built-up of the real scenes
    montage([C.canny_edges(pairs[s][0]),C.canny_edges(outs[(s,FULL)]),C.canny_edges(pairs[s][1])],
        ["Canny: foggy","Canny: full pipeline","Canny: clear"],OUT/"figures"/"downstream_edges_real.png")
    montage([lines_overlay(pairs[s][0]),lines_overlay(outs[(s,FULL)]),lines_overlay(pairs[s][1])],
        ["Hough lines: foggy","Hough lines: full pipeline","Hough lines: clear"],OUT/"figures"/"downstream_lines_real.png")
    street_pairs=load_pairs(args.max_dim,URBAN_REAL) if URBAN_REAL.exists() else {}
    if street_pairs:
        rows,_=run(METHODS,street_pairs)
        write_csv(OUT/"street_pairs.csv",rows)
        section(f"1b. Real city street photos with real fog ({len(street_pairs)} pairs)",rows,md)
    # Experiment 2: real street photos with simulated fog, plus YOLOv5n vehicle and pedestrian detection
    if streets:
        md+=["## 2. Real city streets, simulated fog","",
            f"{len(streets)} clear photos of real streets (`python get_street_photos.py`; one is `assets/test images/2467248.jpg`), fogged "
            f"with the scattering model (beta {BETA}) and a rough depth guess (farther toward the top of the frame). "
            +("*Vehicles/people found* are the detections of a YOLOv5n detector (cars, buses, trucks, motorcycles, bicycles, "
                "pedestrians); recall and precision treat the detector's output on the clear photo as the reference (IoU 0.5). "
                if det_ok else "The vehicle detector (detect.py) could not be loaded, so that test was skipped. "),""]
        scenes={n:(C.simulate_fog(c,None,BETA),c) for n,c in streets.items()}
        rows,outs=run(METHODS,scenes,detection=det_ok)
        write_csv(OUT/"street_photos.csv",rows)
        section(f"2. Real streets, fog beta {BETA}",rows,md,"###")
        s=next(iter(scenes))
        montage([scenes[s][0],outs[(s,FULL)],scenes[s][1]],["foggy street","full pipeline","clear street"],
            OUT/"figures"/"street_montage.png",width=420)
        if det_ok:
            s=max(scenes,key=lambda n:sum(1 for b in detect.detect(streets[n]) if b[5] in detect.VEHICLES))
            montage([boxes_overlay(scenes[s][0]),boxes_overlay(outs[(s,FULL)]),boxes_overlay(scenes[s][1])],
                ["detections: foggy","detections: full pipeline","detections: clear"],OUT/"figures"/"detection_street.png",width=420)
    # Experiment 3: simulated fog and blur (Gaussian, horizontal and diagonal motion) on the clear garden photos
    conditions={
        "fog":(lambda c:C.simulate_fog(c,None,BETA),None,0.0),
        "fog + Gaussian blur (2 px)":(lambda c:C.simulate_blur(C.simulate_fog(c,None,BETA),2.0,"Gaussian"),"Gaussian",0.0),
        "fog + horizontal motion blur (2 px)":(lambda c:C.simulate_blur(C.simulate_fog(c,None,BETA),2.0,"Motion"),"Motion",0.0),
        "fog + diagonal motion blur (45 deg, 2 px)":(lambda c:C.simulate_blur(C.simulate_fog(c,None,BETA),2.0,"Motion",45.0),"Motion",45.0),
    }
    all_rows=[]
    for cond,(make,kind,ang) in conditions.items():
        scenes={n:(make(c),c) for n,(_,c) in pairs.items()}
        rows,outs=run(blur_methods(kind,2.0,ang) if kind else dict(METHODS),scenes)
        all_rows+=[dict(r,condition=cond) for r in rows]
        section(f"3. Simulated: {cond}",rows,md)
        if "diagonal" in cond:
            s=next(iter(scenes))
            montage([scenes[s][0],outs[(s,FULL)],outs[(s,AUTO)],scenes[s][1]],
                ["fog + diagonal motion blur","full pipeline","+ Wiener (blur estimated)","clear photo"],
                OUT/"figures"/"simulated_motion_blur_montage.png")
    write_csv(OUT/"simulated.csv",all_rows)
    # Experiment 4: synthetic cities with true depth, which also checks the transmission estimate directly
    md+=["## 4. Synthetic city streets with true depth","",
        f"{len(urban)} synthetic streets (assets/test images/urban_synthetic, 640x400, different building layouts, "
        "cars, lane lines), fogged with t = exp(-beta * depth) from their true depth map.",""]
    urban_rows,trans=[],transmission_rows(urban)
    for i,(cond,blur,ang) in enumerate((("fog",None,0.0),("fog + horizontal motion blur (2 px)","Motion",0.0),
            ("fog + diagonal motion blur (45 deg, 2 px)","Motion",45.0),
            ("fog + Gaussian blur (2 px)","Gaussian",0.0)),1):
        scenes={}
        for n,(c,d) in urban.items():
            f=C.simulate_fog(c,d,BETA)
            scenes[n]=(C.simulate_blur(f,2.0,blur,ang) if blur else f,c)
        rows,outs=run(blur_methods(blur,2.0,ang) if blur else dict(METHODS),scenes)
        urban_rows+=[dict(r,condition=cond) for r in rows]
        section(f"4.{i}. Synthetic streets: {cond} (beta {BETA})",rows,md,"###")
        if cond=="fog":
            s=next(iter(scenes))
            montage([scenes[s][0],outs[(s,"Dark channel prior only")],outs[(s,FULL)],scenes[s][1]],
                ["foggy city","dark channel only","full pipeline","clear city"],OUT/"figures"/"urban_montage.png")
            montage([lines_overlay(scenes[s][0]),lines_overlay(outs[(s,FULL)]),lines_overlay(scenes[s][1])],
                ["Hough lines: foggy","Hough lines: full pipeline","Hough lines: clear"],OUT/"figures"/"downstream_lines_urban.png")
    write_csv(OUT/"urban.csv",urban_rows)
    md+=["### Transmission estimate against the true depth","",
        "Mean over the synthetic set; the true transmission is exp(-beta * depth), clipped to the same minimum as the estimate.","",
        "| Beta | Transmission MAE | Correlation with true t | Airlight error (0-255) |","| ---: | ---: | ---: | ---: |"]
    for beta in (0.8,1.6,2.4,3.2):
        sel=[r for r in trans if r["beta"]==beta]
        md.append(f"| {beta:g} | {np.mean([r['t_mae'] for r in sel]):.3f} | {np.mean([r['t_corr'] for r in sel]):.3f} "
            f"| {np.mean([r['airlight_err'] for r in sel]):.1f} |")
    md.append("")
    write_csv(OUT/"transmission.csv",trans)
    # Experiment 5: fog-density sweep, fixed default against strength set from the haze level
    datasets={"7 garden photos":{n:(c,None) for n,(_,c) in pairs.items()},"synthetic streets":dict(urban)}
    if streets:
        datasets[f"{len(streets)} real streets"]={n:(c,None) for n,c in streets.items()}
    sweep_rows,sweep_md=sweep_table(datasets)
    write_csv(OUT/"strength_sweep.csv",sweep_rows)
    md+=["## 5. Fog density sweep: fixed default against strength set from the haze level","",
        "Fog is simulated at each density (garden photos and street photos with the rough depth guess, synthetic streets with their "
        "true depth). *Haze level* is `haze_level()` of the foggy image; *strength used* is the share of the default's strengths "
        "that `auto_strength_params` applies. The default's strengths are tuned for dense fog; on light haze they over-process.","",
        sweep_md,""]
    g= lambda m,k:mean_of(real_rows,m,k)
    md+=["On the 7 **real** fog photos (section 1; real fog was not used for the fit), fixed against adaptive default: "
        f"PSNR {g(FULL,'psnr'):.2f} / {g(AUTOSTR,'psnr'):.2f} dB, SSIM {g(FULL,'ssim'):.3f} / {g(AUTOSTR,'ssim'):.3f}, "
        f"edge F1 {g(FULL,'edge_f1'):.3f} / {g(AUTOSTR,'edge_f1'):.3f}.",""]
    if args.fit_strength:
        fit_md=fit_strength_rule(datasets)
        (OUT/"strength_fit.md").write_text(fit_md,encoding="utf-8")
    if (OUT/"strength_fit.md").exists():
        md+=["### Fit of the rule and held-out check","",
            "The rule was fit on simulated fog of two scene types and tested on the third (python evaluate.py --fit-strength). "
            "Score is SSIM + PSNR/40; per-image best is the best of six fixed scalings chosen with knowledge of the clear photo.","",
            (OUT/"strength_fit.md").read_text(encoding="utf-8"),""]
    # Experiment 6: ablation, what each module adds to the full pipeline
    abl_sets={"Real fog":{n:(h,c) for n,(h,c) in pairs.items()},
        "Synthetic streets":{n:(C.simulate_fog(c,d,BETA),c) for n,(c,d) in urban.items()}}
    if streets:
        abl_sets["Real streets"]={n:(C.simulate_fog(c,None,BETA),c) for n,c in streets.items()}
    md+=["## 6. What each module adds to the full pipeline","",
        "Each row removes one part of the Balanced pipeline; compare with the first row. SSIM and edge F1 are against the clear photo, "
        "e is the visible-edge gain over the foggy input; detection recall is the share of the clear street photos' vehicles and people "
        "that the YOLOv5n detector still finds.","",ablation_table(abl_sets,("Real streets",) if det_ok else ()),""]
    # Experiment 7: blur estimation and Wiener deblur with the estimated blur
    est_rows,est_md,ang_md=blur_estimation({n:c for n,(_,c) in pairs.items()},{n:c for n,(c,_) in urban.items()})
    write_csv(OUT/"blur_estimation.csv",est_rows)
    hazy_est=[C.estimate_blur(h) for h,_ in pairs.values()]
    md+=["## 7. Blur estimation","",
        "`estimate_blur` measures the width of the strongest edges along 8 directions and subtracts the width of a sharp photo. "
        "Each row blurs the clear photos by a known amount (fog added first, beta 1.6 unless the row says otherwise) and compares the "
        "estimate with it; motion blur rows pool 7 angles (0, 25, 45, 70, 90, 135, 160 degrees), length 3 x amount. The calibration "
        "factors were fit on the real photos, so the synthetic cities are an independent check.","",est_md,"",
        "Motion angle, by true angle (blur 2 to 4 px, fog beta 1.6):","",ang_md,"",
        f"On the {len(pairs)} real hazy photos (true blur unknown) the estimator found: "
        f"{', '.join(f'{k} {sum(e['kind']==k for e in hazy_est)}' for k in dict.fromkeys(e['kind'] for e in hazy_est))} "
        f"(largest estimate {max(e['amount'] for e in hazy_est):.2f} px).",""]
    # Experiment 8: leave-one-scene-out cross-validation of the settings
    md+=["## 8. Are the defaults overfit? Leave-one-scene-out tuning (real fog)","",
        "60 random settings (strengths, dark channel omega, colour balance) are scored on the 7 real fog scenes at 400 px. "
        "For each scene, the best setting on the other six is applied to it. The shipped default was tuned on a subset of these "
        "photos, so its row is optimistic; the street and synthetic results above are data it never saw.","",
        cross_validate(pairs),""]
    # Experiment 9: synthetic traffic video, per-frame against temporal processing
    md+=["## 9. Video: synthetic traffic clip","",
        "30 frames, 640x400, fixed camera, 3 cars driving across, fog (beta 1.6) from the true depth and sensor noise (sigma 2). "
        "*Flicker* is the mean frame-to-frame change of the output in pixels that did not change in the input (static background); "
        "the clip is processed by `dehaze_video.TemporalDehazer`.","",video_table(),""]
    (OUT/"summary.md").write_text("\n".join(md),encoding="utf-8")
    print("\n".join(md))
    make_charts(pairs)
if __name__=="__main__":
    main()
