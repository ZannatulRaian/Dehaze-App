# Dehaze App: Streamlit console around dehaze_core.py (upload a hazy photo, tune the filters, download the result); run: streamlit run dehaze_app.py
import base64
import glob
import io
import math
import random
import time
import tempfile
import zipfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image,ImageOps,UnidentifiedImageError
from dehaze_video import detection_video,process_video
from dehaze_core import ( # the image-processing code lives in dehaze_core.py
    P_BALANCED,P_MODULES,P_TRAFFIC,STAGE_ORDER,apply_stage,baseline_methods,draw_boxes,hist_clip_percent,histogram_stretch,edge_overlay,freq_response_curve,heat_image,radial_spectrum,sharpen_weight_map,smoothing_change_map,spectrum_image,auto_strength_params,dehaze_pipeline,
    downstream_metrics,edge_maps,estimate_blur as _estimate_blur,haze_level as _haze_level,strength_scale,make_sample_scene,psnr,quality_metrics,simulate_blur as _simulate_blur,
    simulate_fog as _simulate_fog,ssim,stage_active,transmission_map_image,visibility_metrics,
)
# Optional pillow-heif adds iPhone HEIC/HEIF support (pip install pillow-heif); the app works without it
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass
APP_DIR=Path(__file__).parent
# Streamlit-cached versions of the fog / blur simulations
simulate_fog=st.cache_data(show_spinner=False)(_simulate_fog)
simulate_blur=st.cache_data(show_spinner=False)(_simulate_blur)
estimate_blur=st.cache_data(show_spinner=False,max_entries=6)(_estimate_blur)
downstream=st.cache_data(show_spinner=False,max_entries=8)(downstream_metrics) # Canny + Hough counts
baselines=st.cache_data(show_spinner=False,max_entries=3)(baseline_methods) # one-technique baselines for the Compare view
haze_of=st.cache_data(show_spinner=False,max_entries=6)(_haze_level)
MAX_DIM=1200 # images larger than this (on the long edge) are downscaled
# Before/after slider image: original left of the divider, dehazed right, with a red divider line
def compare_composite(a,b,pos):
    h,w=a.shape[:2]
    x=int(round(w*pos/100))
    out=b.copy()
    out[:,:x]=a[:,:x]
    if 0<x<w:
        out[:,max(0,x-1):min(w,x+1)]=(211,47,47)
    return out
# Overlaid histogram line plot drawn as inline SVG polylines
def hist_svg(series,w=520,h=150,label="Histogram",marks=()):
    m=max(max(s[0].max() for s in series),1.0)
    # Optional vertical marks (bin, colour), e.g. the histogram-stretch clip points
    lines="".join(f'<line x1="{b*w/255:.1f}" y1="0" x2="{b*w/255:.1f}" y2="{h}" stroke="{c}" stroke-width="2" stroke-dasharray="4 3"/>' for b,c in marks)
    for hh,colour,dashed in series:
        # Polyline points: x over the 256 bins, y scaled to the tallest histogram
        pts=" ".join(f"{i*w/255:.1f},{h-4-(v/m)*(h-10):.1f}" for i,v in enumerate(hh))
        lines+=(f'<polyline fill="none" stroke="{colour}" stroke-width="2" '
            f'{"stroke-dasharray=\"6 4\" " if dashed else ""}points="{pts}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
        f'style="background:#fafafa;border:1px solid rgba(0,0,0,.08);border-radius:.35rem">{lines}</svg>')
# Gamma / tone-curve plot as inline SVG: input brightness on x, output brightness on y, one polyline per curve
def curve_svg(curves,w=520,h=240,label="Gamma curve"):
    p=26
    grid="".join(f'<line x1="{p+i*(w-p-6)/4:.1f}" y1="6" x2="{p+i*(w-p-6)/4:.1f}" y2="{h-p}" stroke="#e6e6e6"/>'
        f'<line x1="{p}" y1="{6+i*(h-p-6)/4:.1f}" x2="{w-6}" y2="{6+i*(h-p-6)/4:.1f}" stroke="#e6e6e6"/>' for i in range(5))
    lines=""
    for xs,ys,colour,dashed,width in curves:
        pts=" ".join(f"{p+x/255*(w-p-6):.1f},{h-p-y/255*(h-p-6):.1f}" for x,y in zip(xs,ys))
        lines+=(f'<polyline fill="none" stroke="{colour}" stroke-width="{width}" '
            f'{"stroke-dasharray=\"6 4\" " if dashed else ""}points="{pts}"/>')
    axes=(f'<text x="{p}" y="{h-8}" font-size="11" fill="#888">0</text><text x="{w-24}" y="{h-8}" font-size="11" fill="#888">255</text>'
        f'<text x="{w/2-30:.0f}" y="{h-8}" font-size="11" fill="#888">input brightness</text>'
        f'<text x="2" y="14" font-size="11" fill="#888">out</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
        f'style="background:#fafafa;border:1px solid rgba(0,0,0,.08);border-radius:.35rem">{grid}{lines}{axes}</svg>')
# Line chart as inline SVG on a 0..1 x axis: series are (xs, ys, colour, dashed); y is scaled to the data range
def line_svg(series,w=520,h=200,label="Chart",xlabel="",ylabel=""):
    p=30
    lo=min(float(np.min(s[1])) for s in series)
    hi=max(float(np.max(s[1])) for s in series)
    span=max(hi-lo,1e-6)
    grid="".join(f'<line x1="{p+i*(w-p-6)/4:.1f}" y1="6" x2="{p+i*(w-p-6)/4:.1f}" y2="{h-p}" stroke="#e6e6e6"/>'
        f'<line x1="{p}" y1="{6+i*(h-p-6)/4:.1f}" x2="{w-6}" y2="{6+i*(h-p-6)/4:.1f}" stroke="#e6e6e6"/>' for i in range(5))
    lines=""
    for xs,ys,colour,dashed in series:
        pts=" ".join(f"{p+x*(w-p-6):.1f},{h-p-(y-lo)/span*(h-p-6):.1f}" for x,y in zip(xs,ys))
        lines+=(f'<polyline fill="none" stroke="{colour}" stroke-width="2.5" '
            f'{"stroke-dasharray=\"6 4\" " if dashed else ""}points="{pts}"/>')
    ticks=(f'<text x="{p}" y="{h-8}" font-size="11" fill="#888">0</text><text x="{w-24}" y="{h-8}" font-size="11" fill="#888">1</text>'
        f'<text x="{w/2-40:.0f}" y="{h-8}" font-size="11" fill="#888">{xlabel}</text>'
        f'<text x="2" y="14" font-size="11" fill="#888">{hi:.1f}</text><text x="2" y="{h-p}" font-size="11" fill="#888">{lo:.1f}</text>'
        f'<text x="{p+4}" y="16" font-size="11" fill="#888">{ylabel}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
        f'style="background:#fafafa;border:1px solid rgba(0,0,0,.08);border-radius:.35rem">{grid}{lines}{ticks}</svg>')
# Clip points (low, high) the histogram stretch would use on an image, and the % clipped at each end; strength 0 shows the 20 default
def stretch_points(img,strength):
    strength=strength or 20
    return (*histogram_stretch(img,strength,return_range=True)[1],hist_clip_percent(strength))
# Effective gamma of an image: the g for which a mid-grey (0.5) maps to the image's median brightness, i.e. m = 0.5^(1/g), so g > 1 means a bright image
def effective_gamma(img):
    m=float(np.clip(np.median(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))/255.0,0.02,0.98))
    return float(np.clip(math.log(0.5)/math.log(m),0.2,6.0))
TEST_DIR=APP_DIR/"assets"/"test images"
# Finds the bundled test photos and pairs each hazy photo with its clear reference
@st.cache_data(show_spinner=False)
def list_test_images():
    found={}
    for f in sorted(glob.glob(str(TEST_DIR/"**"/"*.*"),recursive=True)):
        pth=Path(f)
        if (pth.suffix.lower() not in (".jpg",".jpeg",".png") or "clear_reference" in pth.name
            or "_depth" in pth.name or pth.parent.name=="urban_real_clear"):
            continue
        ref=pth.with_suffix("").with_name(pth.stem.replace("_hazy","_clear_reference")).with_suffix(pth.suffix)
        has_ref=ref!=pth and ref.exists() # a photo without "_hazy" in its name must not be its own reference
        label=pth.stem.replace("_hazy","").replace("_"," ").capitalize()
        found[label+(" (clear reference)" if has_ref else "")]=(pth,ref if has_ref else None)
    return found
# Lists the clear street photos on which fog is simulated
@st.cache_data(show_spinner=False)
def list_clear_photos():
    return {"Street photo "+Path(f).stem.split("_")[0][6:]+" (clear; fog simulated)":Path(f)
        for f in sorted(glob.glob(str(TEST_DIR/"urban_real_clear"/"*.jpg")))}
# Reads an image as EXIF-rotated RGB and resizes it
def _read_rgb(path,size=None):
    with Image.open(path) as im:
        arr=np.array(ImageOps.exif_transpose(im).convert("RGB"))
    if size is not None:
        return cv2.resize(arr,size,interpolation=cv2.INTER_AREA)
    return resize_max(arr)
# Converts core settings to the names of the app's widgets
def _ui(P):
    return dict(hist_strength=P["hist"],freq_strength=P["freq"],smooth_strength=P["smooth"],
        sharpen_strength=P["sharpen"],dcp_omega=P["omega"],dcp_t0=P["t0"],dcp_patch=P["patch"],
        freq_cutoff=P["cutoff"],tone_gamma=P["gamma"],tone_sat=P["sat"],tone_balance=P["balance"],
        clahe_strength=P.get("clahe",0),clahe_tiles=P.get("clahe_tiles",8),tone_restore=P.get("restore",0),
        dcp_adaptive=P.get("adaptive",False),dcp_far=P.get("far",0),
        deblur_amount=P["deblur"],deblur_kind=P["deblur_kind"],deblur_k=P["deblur_k"],deblur_auto=False,
        deblur_angle=0.0,strength_auto=False,use_dcp=P["use_dcp"],modules_only=False)
# Presets: Balanced comes from a parameter search (see evaluate.py), the others are hand-set around it
_BASE=_ui(P_BALANCED)
PRESETS={
    "Balanced (default)":dict(_BASE),
    "Light haze":{**_BASE,"hist_strength":15,"freq_strength":15,"smooth_strength":20,
        "sharpen_strength":25,"dcp_omega":0.80,"dcp_t0":0.25},
    "Dense fog":{**_BASE,"hist_strength":35,"freq_strength":30,"smooth_strength":30,
        "sharpen_strength":45,"dcp_omega":0.95,"dcp_t0":0.12},
    "Maximum detail":{**_BASE,"hist_strength":45,"freq_strength":40,"smooth_strength":35,
        "sharpen_strength":55,"dcp_omega":0.95,"dcp_t0":0.12},
    "Fog removal only":{**_BASE,"hist_strength":0,"freq_strength":0,"smooth_strength":0,"sharpen_strength":0,
        "clahe_strength":0,"tone_restore":0},
    # Traffic preset: fog removal plus histogram stretching, since the other stages lowered detection recall
    "Traffic monitoring":{**_ui(P_TRAFFIC),"strength_auto":True},
    "Course modules only":{**_ui(P_MODULES),"use_dcp":True,"modules_only":True},
}
# Runs the dehaze pipeline (cached) and returns the result and the seconds it took
@st.cache_data(show_spinner=False,max_entries=6)
def run_pipeline(img,P):
    t=time.perf_counter()
    out=dehaze_pipeline(img,P)
    return out,time.perf_counter()-t
# Three download buttons for an annotated pair: dehazed, original, and both side by side
def download_pair(a,b,tag,key):
    c1,c2,c3=st.columns(3)
    c1.download_button("Download dehazed",data=to_png_bytes(b),file_name=f"{tag}_dehazed.png",mime="image/png",key=f"dl_{tag}_b_{key}",width="stretch")
    c2.download_button("Download original",data=to_png_bytes(a),file_name=f"{tag}_original.png",mime="image/png",key=f"dl_{tag}_a_{key}",width="stretch")
    c3.download_button("Download side by side",data=to_png_bytes(np.hstack([a,b])),file_name=f"{tag}_side_by_side.png",mime="image/png",key=f"dl_{tag}_ab_{key}",width="stretch")
# Applications panel: traffic detection (YOLOv5n) and construction-inspection edge and line overlays, original against dehazed; returns the detector settings
def render_applications(a,b,key):
    t_traffic,t_build=st.tabs(["Traffic monitoring","Construction inspection"])
    with t_traffic:
        sens=st.slider("Detection confidence",0.10,0.60,0.20,0.05,key=f"det_conf_{key}",
            help="Lower finds more (and fainter) vehicles, but also more false ones")
        tiled=st.toggle("Search tiles for small and distant vehicles",value=True,key=f"det_tiled_{key}")
        try:
            d_in,d_out=detect_vehicles(a,sens,tiled),detect_vehicles(b,sens,tiled)
            box_a,box_b=draw_boxes(a,d_in),draw_boxes(b,d_out)
            col_b,col_a=st.columns(2)
            col_b.image(box_a,width="stretch",caption=f"Original: {len(d_in)} vehicles and people detected")
            col_a.image(box_b,width="stretch",caption=f"Dehazed: {len(d_out)} vehicles and people detected")
            download_pair(box_a,box_b,"detection",key)
        except Exception as exc:
            st.caption(f"Detection model not available here ({type(exc).__name__}); see detect.py to download yolov5n.onnx.")
    with t_build:
        e_in,e_out=downstream(a),downstream(b)
        ov_a,ov_b=edge_overlay(a),edge_overlay(b)
        col_b,col_a=st.columns(2)
        col_b.image(ov_a,width="stretch",
            caption=f"Original: edges {e_in['canny_pct']:.1f}% of pixels, {e_in['lines']} lines ({e_in['lines_vertical']} vertical)")
        col_a.image(ov_b,width="stretch",
            caption=f"Dehazed: edges {e_out['canny_pct']:.1f}% of pixels, {e_out['lines']} lines ({e_out['lines_vertical']} vertical)")
        download_pair(ov_a,ov_b,"edges_lines",key)
    return sens,tiled
# Graphs for an original and a dehazed image: brightness and colour histograms, gamma curves, FFT spectra and the mean spectrum by frequency
def render_graphs(a,b):
    m0,m1=quality_metrics(a),quality_metrics(b)
    c1,c2=st.columns(2)
    c1.markdown('<p class="ufc-cap">Brightness histogram: original (grey), dehazed (red)</p>'
        +hist_svg([(m0["hist"],"#9a9a9a",False),(m1["hist"],"#d32f2f",False)],label="Brightness histograms"),unsafe_allow_html=True)
    chan=["#e53935","#43a047","#1e88e5"]
    c2.markdown('<p class="ufc-cap">R, G, B histograms: original (dashed), dehazed (solid)</p>'
        +hist_svg([(m0["chans"][c],chan[c],True) for c in range(3)]+[(m1["chans"][c],chan[c],False) for c in range(3)],
            label="Colour channel histograms"),unsafe_allow_html=True)
    g0,g1=effective_gamma(a),effective_gamma(b)
    gx=np.arange(0,256,4.0)
    c1.markdown(f'<p class="ufc-cap">Gamma: normal 1.00 (dashed), original {g0:.2f} (grey), dehazed {g1:.2f} (red)</p>'
        +curve_svg([(gx,gx,"#555555",True,1.5),(gx,255*(gx/255)**(1/g0),"#9a9a9a",False,3),(gx,255*(gx/255)**(1/g1),"#d32f2f",False,3)],
            label="Gamma curves"),unsafe_allow_html=True)
    rx,r0=radial_spectrum(a)
    _,r1=radial_spectrum(b)
    c2.markdown('<p class="ufc-cap">Mean spectrum by frequency: original (grey), dehazed (red)</p>'
        +line_svg([(rx,r0,"#9a9a9a",False),(rx,r1,"#d32f2f",False)],label="Radial spectrum",xlabel="frequency (1 = Nyquist)",ylabel="log |F|"),
        unsafe_allow_html=True)
    s0,vmax=spectrum_image(a)
    s1,_=spectrum_image(b,vmax)
    c1.image(s0,caption="FFT spectrum of the original (log magnitude)",width="stretch")
    c2.image(s1,caption="FFT spectrum of the dehazed image (same scale)",width="stretch")
# Vertical bar chart as inline SVG, one bar per label with its value on top; a dashed line marks 1.0 when the values are ratios
def bars_svg(labels,values,colour="#d32f2f",fmt="{:.2f}",ref=None,w=900,h=170,label="Bars"):
    n=len(values)
    vmax=max(max(values),ref or 0.0,1e-6)*1.15
    bw=(w-20)/n
    out=""
    for i,(lab,v) in enumerate(zip(labels,values)):
        bh=v/vmax*(h-44)
        x=10+i*bw
        out+=(f'<rect x="{x+bw*0.15:.1f}" y="{h-24-bh:.1f}" width="{bw*0.7:.1f}" height="{bh:.1f}" fill="{colour}"/>'
            f'<text x="{x+bw/2:.1f}" y="{h-28-bh:.1f}" font-size="11" text-anchor="middle" fill="#333">{fmt.format(v)}</text>'
            f'<text x="{x+bw/2:.1f}" y="{h-8}" font-size="10" text-anchor="middle" fill="#888">{lab[:10]}</text>')
    if ref:
        out+=f'<line x1="10" y1="{h-24-ref/vmax*(h-44):.1f}" x2="{w-10}" y2="{h-24-ref/vmax*(h-44):.1f}" stroke="#555" stroke-dasharray="5 4"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
        f'style="background:#fafafa;border:1px solid rgba(0,0,0,.08);border-radius:.35rem">{out}</svg>')
# YOLOv5n detections (cached), optionally also on overlapping tiles for small objects
@st.cache_data(show_spinner="Detecting vehicles…",max_entries=64)
def detect_vehicles(img,conf,tiled):
    import detect as D
    return (D.detect_tiled if tiled else D.detect)(img,conf)
# Image after every stage, cumulatively, with whether each stage is active
@st.cache_data(show_spinner=False,max_entries=3)
def run_stages(img,P):
    steps,r=[("Original",img,True)],img
    for name in STAGE_ORDER:
        r=apply_stage(name,r,P)
        steps.append((name,r,stage_active(name,P)))
    return steps
# Area-interpolation downscale so the long edge is at most max_dim
def resize_max(img,max_dim=MAX_DIM):
    h,w=img.shape[:2]
    scale=max_dim/max(h,w)
    if scale<1:
        # max(1, ...) keeps a very thin image (e.g. 1 x 5000 px) from rounding to 0
        new_size=(max(1,round(w*scale)),max(1,round(h*scale)))
        img=cv2.resize(img,new_size,interpolation=cv2.INTER_AREA)
    return img
# Reads the original and dehazed clips side by side: per-frame metrics for the graphs and up to k evenly spaced frame pairs (as PNG bytes)
def extract_video_data(orig_path,dehazed_path,n,k=12):
    cap_o,cap_d=cv2.VideoCapture(str(orig_path)),cv2.VideoCapture(str(dehazed_path))
    keep=set(np.linspace(0,max(n-1,0),min(k,n)).astype(int).tolist())
    metrics,frames=[],[]
    try:
        for i in range(n):
            ok1,o=cap_o.read()
            ok2,d=cap_d.read()
            if not (ok1 and ok2):
                break
            o,d=cv2.cvtColor(o,cv2.COLOR_BGR2RGB),cv2.cvtColor(d,cv2.COLOR_BGR2RGB)
            q0,q1=quality_metrics(o),quality_metrics(d)
            metrics.append((q0["contrast"],q1["contrast"],q0["edges"],q1["edges"],float(o.mean()),float(d.mean())))
            if i in keep:
                frames.append((to_png_bytes(o),to_png_bytes(d)))
    finally:
        cap_o.release()
        cap_d.release()
    return np.array(metrics),frames
# Decodes PNG bytes back to an RGB array
def png_to_rgb(data):
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
# Encodes an RGB array as PNG bytes
def to_png_bytes(img):
    buf=io.BytesIO()
    Image.fromarray(img).save(buf,format="PNG")
    return buf.getvalue()
# Image formats Pillow can decode, minus EPS (reading one would call Ghostscript)
def _readable_formats():
    Image.init()
    return [name for name in Image.OPEN if name!="EPS"]
# Opens any Pillow-readable image as 8-bit RGB: EXIF rotation, 16-bit/float scaling, transparency on white, downscaling
def load_photo(file):
    img=Image.open(file,formats=_readable_formats())
    # Draft mode lets the JPEG decoder shrink huge camera photos while reading
    img.draft("RGB",(MAX_DIM,MAX_DIM))
    try:
        img=ImageOps.exif_transpose(img) # honour the camera's rotation flag
    except Exception:
        pass # broken EXIF block: show the picture as stored
    if img.mode in ("I;16","I;16L","I;16B","I;16N","I","F"):
        # 16/32-bit and float data: scale to 8 bits by hand, since Pillow's convert() would clip at 255
        data=np.asarray(img,dtype=np.float32)
        top=float(data.max()) if data.size else 0.0
        if img.mode=="F" and top<=1.0:
            gain=255.0 # float pixels in the 0..1 range
        elif top<=255.0:
            gain=1.0 # already 8-bit values in a bigger container
        else:
            # 16-bit files often hold 10/12/14-bit data: pick the full scale from the brightest value
            full_scale=next((2**bits-1 for bits in (10,12,14,16)
                    if top<=2**bits-1),top)
            gain=255.0/full_scale
        gray=np.clip(data*gain,0,255).astype(np.uint8)
        arr=np.stack([gray]*3,axis=-1)
    elif img.mode in ("RGBA","LA","PA") or "transparency" in img.info:
        # Flatten transparency onto white (the colour of fog) instead of black
        rgba=img.convert("RGBA")
        page=Image.new("RGBA",rgba.size,(255,255,255,255))
        arr=np.array(Image.alpha_composite(page,rgba).convert("RGB"))
    else:
        arr=np.array(img.convert("RGB"))
    return resize_max(arr)
_RAW_EXTENSIONS={".cr2",".cr3",".crw",".nef",".nrw",".arw",".srf",".sr2",".dng",
    ".raf",".orf",".rw2",".pef",".srw",".x3f",".3fr",".erf",".mrw",".raw"}
# Plain-language explanation for a file load_photo could not open
def unreadable_message(name,exc):
    Image.init() # fills Image.OPEN, which tells us if the optional codecs are installed
    ext=Path(name).suffix.lower()
    if isinstance(exc,Image.DecompressionBombError):
        return f"`{name}` is too large to open safely. Try a smaller copy of the photo."
    if ext in (".heic",".heif",".hif") and "HEIF" not in Image.OPEN:
        return (f"`{name}` is a HEIC/HEIF photo (the iPhone format), which needs one extra "
            "package: run `pip install pillow-heif`, restart the app and upload it again. "
            "Or save the photo as a JPG first.")
    if ext==".avif" and "AVIF" not in Image.OPEN:
        return (f"`{name}` is an AVIF image, which needs a newer Pillow: run "
            "`pip install --upgrade pillow` (or `pip install pillow-heif`), restart the app "
            "and upload it again. Or save the image as a JPG first.")
    if ext in _RAW_EXTENSIONS:
        return (f"`{name}` is a camera RAW file, which this app can't develop. "
            "Export the photo as a JPG or TIFF first and upload that.")
    if isinstance(exc,UnidentifiedImageError):
        return (f"`{name}` doesn't look like an image this app can open. "
            "JPG, PNG, WebP, GIF, BMP and TIFF files all work.")
    reason=" ".join(str(exc).split())[:120]
    return (f"Couldn't read `{name}`"+(f" ({reason})" if reason else "")+". The file may be "
        "damaged or incomplete — try saving it again as a JPG or PNG.")
# (label, Pillow format name) of the formats named in the uploader hint
_HINT_FORMATS=[("JPG","JPEG"),("PNG","PNG"),("WebP","WEBP"),("GIF","GIF"),
    ("BMP","BMP"),("TIFF","TIFF"),("AVIF","AVIF"),("HEIC","HEIF")]
# Text listing the image formats this install can open
def supported_formats_hint():
    Image.init()
    return ", ".join(label for label,key in _HINT_FORMATS if key in Image.OPEN)+" and more"
# Loads the car or cloud sprites as base64 PNG with their aspect ratios
@st.cache_resource(show_spinner=False)
def load_sprite_meta(kind):
    folder=APP_DIR/"assets"/kind
    files=sorted(glob.glob(str(folder/"*.png")))
    out=[]
    for f in files:
        with open(f,"rb") as fh:
            b64=base64.b64encode(fh.read()).decode("ascii")
        with Image.open(f) as im:
            w,h=im.size
        out.append((b64,w/h))
    return out
# Builds one randomised row of sprite <img> tags and returns its HTML and width
def _build_sequence(sprite_meta,count,height_range,gap_range,seed,top_range=None):
    rnd=random.Random(seed)
    n=len(sprite_meta)
    order=[rnd.randrange(n) for _ in range(count)]
    rnd.shuffle(order)
    parts=[]
    total_w=0.0
    for idx in order:
        b64,aspect=sprite_meta[idx]
        h=rnd.randint(*height_range)
        gap=rnd.randint(*gap_range)
        top=f"--t:{rnd.randint(*top_range)};" if top_range else ""
        # Sizes are authored in px of a 200px-tall strip; the iframe CSS scales them by --vu
        parts.append(
            f'<img class="sprite" style="--h:{h};--g:{gap};{top}" '
            f'src="data:image/png;base64,{b64}">'
        )
        total_w+=h*aspect+gap
    return "".join(parts),total_w
MIN_TRACK_WIDTH=8000 # authored px a lane must cover before it's safe on any realistic viewport
# One endlessly scrolling lane: a row repeated wider than the window and slid by one row-width in a CSS loop
def _lane_html(sprite_meta,count,height_range,gap_range,seed,duration,
    top_range=None,opacity=None):
    row_html,row_w=_build_sequence(sprite_meta,count,height_range,gap_range,seed,top_range)
    loops=max(3,math.ceil(MIN_TRACK_WIDTH/row_w)+1)
    style=f"animation-duration:{duration:.2f}s;--shift:{-100/loops:.4f}%;"
    if opacity is not None:
        style+=f"opacity:{opacity};"
    layer_class="cloud-layer" if top_range else "car-layer"
    return f'<div class="track {layer_class}" style="{style}">'+row_html*loops+"</div>"
# Embeds HTML with st.iframe (or components.html on older Streamlit)
def _embed_html(html,height):
    # st.iframe is the supported call; st.components.v1.html is deprecated and only a fallback
    if hasattr(st,"iframe"):
        st.iframe(html,height=height)
    else:
        st.components.v1.html(html,height=height,scrolling=False)
# n random lane speeds stratified across [lo, hi]
def _spread_speeds(seed,lo,hi,n):
    rnd=random.Random(seed)
    band=(hi-lo)/n
    return [rnd.uniform(lo+(i+0.15)*band,lo+(i+0.85)*band) for i in range(n)]
# Column splits shared by the title and the cards so their left edges line up
PAGE_SPLIT=[1,16,1] # [left margin, content, right margin]
CARD_SPLIT=[5,2] # [Preview card, Enhancement Pipeline card]
# Page title, placed in Streamlit's top bar
def render_title():
    _,title_col,_=st.columns(PAGE_SPLIT,gap="large")
    title_col.markdown('<div class="ufc-title">Dehaze App</div>',unsafe_allow_html=True)
# Animated strip of cars and clouds scrolling across the page
def render_header():
    cars=load_sprite_meta("cars")
    clouds=load_sprite_meta("clouds")
    if not cars or not clouds:
        # Without the assets folder the animated strip is skipped
        return
    # Two sparse car lanes with random gaps; speeds are stratified so the lanes differ
    car_speeds=_spread_speeds(seed=101,lo=7,hi=11,n=2)
    car_lane_specs=[(6,13),(5,31)] # (cars in lane, seed)
    car_layers_html="".join(
        _lane_html(cars,count=count,height_range=(44,48),gap_range=(160,270),
            seed=seed,duration=duration)
        for (count,seed),duration in zip(car_lane_specs,car_speeds)
    )
    # Two cloud layers (mid and far), emitted before the cars so the cars paint on top
    cloud_layers_html=(
        _lane_html(clouds,count=9,height_range=(51,75),gap_range=(31,71),
            seed=51,duration=61,top_range=(0,24),opacity=0.80)
        +_lane_html(clouds,count=9,height_range=(37,53),gap_range=(23,55),
            seed=83,duration=79,top_range=(20,46),opacity=0.62)
    )
    # Strip HTML and CSS: sprite sizes scale with --vu and every lane loops with a CSS keyframe animation
    html=f"""<div class="ufc-strip">{cloud_layers_html}{car_layers_html}</div><style>*{{box-sizing:border-box;}}
html,body{{margin:0;background:#f5f5f5;overflow:hidden;}}
:root{{--vu:0.5vh;}}
.ufc-strip{{position:relative;height:100vh;overflow:hidden;border-top:1px solid rgba(0,0,0,.07);}}
.track{{position:absolute;left:0;top:0;height:100%;display:flex;animation-name:scrollLTR;animation-timing-function:linear;animation-iteration-count:infinite;will-change:transform;}}
.cloud-layer{{align-items:flex-start;padding-top:0;}}
.car-layer{{align-items:flex-end;padding-bottom:calc(78 * var(--vu));}}
img.sprite{{image-rendering:pixelated;display:block;flex-shrink:0;height:calc(var(--h) * var(--vu));margin-right:calc(var(--g) * var(--vu));margin-top:calc(var(--t,0) * var(--vu));}}
@keyframes scrollLTR{{from{{transform:translateX(var(--shift));}}
to{{transform:translateX(0%);}}
}}
</style>"""
    _embed_html(html,height=200)
# Injects the page-wide CSS: root font scale, fonts, card layout, upload dropzone, preview images, pipeline card, sliders
def inject_css():
    st.markdown(
        """<style>@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');html{font-size:clamp(11px,min(0.968vw,1.84vh),48px)!important;--ufc-card-h:min(calc(100vh - 13.8rem),40rem);}
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]){font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stMain"]{overflow-x:hidden;}
[data-testid="stColumn"]{min-width:0;}
.st-key-upload_card{max-width:100%;}
.st-key-preview_images [data-testid="stImage"] img{max-width:100%;}
[data-testid="stMainBlockContainer"]{padding:0 2rem 1.2rem 2rem;max-width:112rem;margin-left:auto;margin-right:auto;}
.st-key-evaluation_shell{width:100%!important;max-width:100%!important;margin:0!important;box-sizing:border-box!important;}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]{gap:1.15rem!important;}
.st-key-batch_card,.st-key-video_card,.st-key-method_card,.st-key-evaluation_shell{width:100%!important;box-sizing:border-box!important;}
[data-testid="stHorizontalBlock"]{column-gap:1.15rem!important;}
[data-testid="stAppViewContainer"] .ufc-title,[data-testid="stAppViewContainer"] .pipeline-heading,[data-testid="stAppViewContainer"] .card-heading{font-family:'Big Shoulders Display',sans-serif;}
[data-testid="stAppViewContainer"] .stage-num,[data-testid="stAppViewContainer"] .stage-value{font-family:'IBM Plex Mono',monospace;}
.ufc-title{font-weight:800;font-size:2.3rem;line-height:1;text-transform:uppercase;letter-spacing:.03em;color:#111;margin:.725rem 0;}
[data-testid="stMarkdownContainer"]:has(> .ufc-title){position:relative;z-index:999991;}
[data-testid="stMarkdownContainer"]:has(> .ufc-title),.st-key-pipeline_card [data-testid="stMarkdownContainer"]{margin-bottom:0!important;}
div[data-testid="stElementContainer"]:has(iframe[data-testid="stIFrame"]){width:100vw!important;max-width:100vw!important;flex:0 0 auto!important;height:7.5rem!important;position:relative;left:50%;right:50%;margin-left:-50vw!important;margin-right:-50vw!important;margin-bottom:1.2rem;}
iframe[data-testid="stIFrame"]{height:7.5rem!important;display:block;}
.st-key-upload_card,.st-key-pipeline_card{padding:1.5rem 1.75rem;min-height:var(--ufc-card-h);position:relative;}
.st-key-upload_card button:not([data-testid="stBaseButton-elementToolbar"]),.st-key-pipeline_card button:not([data-testid="stBaseButton-elementToolbar"]){font-size:1.05rem;min-height:2.8rem;}
[data-testid="stLayoutWrapper"]:has(> .st-key-upload_card),[data-testid="stLayoutWrapper"]:has(> .st-key-pipeline_card){flex:1 1 auto;}
.st-key-upload_card > [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]){flex:1 1 auto;display:flex;flex-direction:column;}
.st-key-upload_card [data-testid="stFileUploader"]{flex:1 1 auto;display:flex;flex-direction:column;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"]{flex:1 1 auto;min-height:15rem;padding:1.5rem 1.75rem;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.85rem;border:2px dashed rgba(0,0,0,.18);border-radius:.9rem;background:#fafafa;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"]::before{content:"";order:1;width:3.6rem;height:3.6rem;opacity:.6;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15.5V4.5'/%3E%3Cpath d='M7.5 9 12 4.5 16.5 9'/%3E%3Cpath d='M4.5 14.5v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3'/%3E%3C/svg%3E") center / contain no-repeat;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"]::after{content:"Drop a foggy photo here";order:2;font-size:1.45rem;font-weight:600;line-height:1.2;color:#111;margin-bottom:.4rem;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"] > span{order:3;}
.st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"]{order:4;flex:0 0 auto;display:flex;flex-direction:column;align-items:center;}
.st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] span,.st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] small{font-size:.92rem;}
.st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"]::after{content:"__FORMAT_HINT__";font-size:.92rem;color:rgba(49,51,63,.6);margin-top:.15rem;}
.st-key-upload_card [data-testid="stAlert"] code{font-size:.92em;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]{background-color:#d32f2f;border-color:#d32f2f;color:#ffffff;padding:.7rem 2.1rem;font-size:1.1rem;border-radius:.4rem;}
.st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover{background-color:#b71c1c;border-color:#b71c1c;color:#ffffff;}
.st-key-upload_card{--pv-maxh:max(12rem,calc(var(--ufc-card-h) - 8.5rem));}
.st-key-preview_images{flex:0 1 auto!important;width:min(100%,calc(2 * var(--pv-maxh) * var(--pv-ar,1.25) + 1rem))!important;margin-left:auto;margin-right:auto;}
.st-key-preview_images [data-testid="stImage"] img{width:100%!important;height:auto;}
.st-key-preview_images [data-testid="stImageCaption"]{margin-top:.3rem;padding:0;}
.st-key-preview_images [data-testid="stImageCaption"] p{font-size:.9rem;line-height:1.3;}
[data-testid="stLayoutWrapper"]:has(> .st-key-preview_images){margin-top:auto;margin-bottom:auto;}
.st-key-upload_card > [data-testid="stElementContainer"]:has(> [data-testid="stSpinner"]){position:absolute;top:2.1rem;left:10.5rem;z-index:3;width:auto;padding:.3rem .9rem;background:#ffffff;border:1px solid rgba(0,0,0,.1);border-radius:999px;box-shadow:0 .1rem .5rem rgba(0,0,0,.12);}
.st-key-pipeline_card{display:flex;flex-direction:column;justify-content:space-between;gap:0;}
.st-key-stage_hist,.st-key-stage_freq,.st-key-stage_smooth,.st-key-stage_sharpen{gap:.15rem;margin-top:.9rem;}
.st-key-pipeline_actions{gap:.7rem;margin-top:1.2rem;}
.pipeline-heading{text-transform:uppercase;font-weight:800;font-size:2rem;line-height:1.1;letter-spacing:.02em;color:#111;border-bottom:1px solid rgba(0,0,0,.08);padding-bottom:.8rem;}
.card-heading{text-transform:uppercase;font-weight:800;font-size:2rem;line-height:1.1;letter-spacing:.02em;color:#111;}
.stage-row{display:flex;justify-content:space-between;align-items:baseline;line-height:1.3;}
.stage-num{color:#d32f2f;font-size:.85rem;margin-right:.55rem;}
.stage-label{font-weight:600;font-size:1.08rem;color:#111;}
.stage-value{color:#d32f2f;font-weight:600;font-size:1.08rem;}
[data-testid="stMarkdownContainer"] p.stage-readout{font-family:'IBM Plex Mono',monospace;font-size:.84rem;line-height:1.3;color:#777777;margin:0;}
.st-key-pipeline_card [data-testid="stSlider"] > div > div{padding:.75rem 0;}
.st-key-pipeline_card [data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]){width:1.1rem!important;height:1.1rem!important;}
[data-testid="stSliderTickBar"]{display:none;}
[data-testid="stSliderThumbValue"]{opacity:0;pointer-events:none;transition:opacity .12s ease;font-size:.88rem;font-weight:600;background:#ffffff;border:1px solid rgba(0,0,0,.12);border-radius:.35rem;padding:.05rem .4rem;box-shadow:0 .1rem .45rem rgba(0,0,0,.14);z-index:5;}
[data-testid="stSlider"]:hover [data-testid="stSliderThumbValue"],[data-testid="stSlider"]:active [data-testid="stSliderThumbValue"]{opacity:1;}
</style>""".replace("__FORMAT_HINT__",supported_formats_hint()),
        unsafe_allow_html=True,
    )
st.set_page_config(page_title="Dehaze App",page_icon="🌫️",layout="wide")
inject_css()
# CSS for the extra pipeline rows, the measurements table, the batch, video and method cards, and narrow windows
st.markdown(
    """<style>.st-key-stage_input,.st-key-stage_dcp,.st-key-stage_deblur,.st-key-stage_tone,.st-key-stage_clahe{gap:.15rem;margin-top:.9rem;}
.st-key-upload_card{--pv-maxh:max(12rem,calc(var(--ufc-card-h) - 12.5rem));}
.st-key-pipeline_card [data-testid="stSelectbox"] label p,.st-key-pipeline_card [data-testid="stToggle"] label p,.st-key-pipeline_card [data-testid="stCheckbox"] label p{font-size:.95rem;}
.st-key-upload_card [data-testid="stSegmentedControl"]{margin-bottom:.2rem;}
.st-key-method_card,.st-key-batch_card{padding:1.5rem 1.75rem;margin-top:1.2rem;}
table.ufc-m{width:100%;border-collapse:collapse;}
table.ufc-m th,table.ufc-m td{text-align:right;padding:.4rem .3rem;border-bottom:1px solid rgba(0,0,0,.08);}
table.ufc-m th{font-weight:600;font-size:.85rem;color:#777;}
table.ufc-m th:first-child,table.ufc-m td:first-child{text-align:left;}
[data-testid="stAppViewContainer"] table.ufc-m td:not(:first-child){font-family:'IBM Plex Mono',monospace;font-size:.9rem;}
table.ufc-m td b{color:#d32f2f;font-weight:600;}
.ufc-cap{font-size:.85rem;color:#777;margin:0 0 .4rem;}
.ufc-eq{font-family:'IBM Plex Mono',monospace;font-size:1.05rem;}
@media (max-width:1000px){.pipeline-heading{font-size:1.45rem!important;word-break:normal!important;overflow-wrap:normal!important;hyphens:none!important;}
}
.st-key-method_card [data-testid="stMarkdownContainer"] p{margin:0 0 .8rem;line-height:1.65;text-align:justify;hyphens:auto;text-indent:1.6rem;}
.st-key-method_card [data-testid="stMarkdownContainer"] p.ufc-sub{text-indent:0;text-align:left;font-weight:600;font-size:1.05rem;margin:.2rem 0 .7rem;}
.st-key-method_card [data-testid="stMarkdownContainer"] p.ufc-eq{text-indent:0;text-align:left;margin-bottom:1rem;}
.st-key-method_card [data-testid="stMarkdownContainer"] ol{margin:0;padding-left:1.9rem;}
.st-key-method_card [data-testid="stMarkdownContainer"] ol li{margin:0 0 .7rem;padding-left:.5rem;line-height:1.65;text-align:justify;hyphens:auto;}
</style>""",
    unsafe_allow_html=True,
)
render_title()
render_header()
SAMPLE_LABEL,PLACEHOLDER="Sample scene (synthetic city)","Or try a sample…"
VIEWS=["Side by side","Slider","Original","Dehazed","Stages","Edges","Transmission","Modules","Compare","Applications"]
STAGE_LABELS=["01 Dark Channel","02 Deblur","05 Smoothing","06 Sharpening","03 Histogram","03b CLAHE","04 Frequency","07 Tone"] # run order
for _k,_v in {"source_key":None,"source_img":None,"source_depth":None,"reference":None,
    "pre_foggy":True,"fog_density":1.6,"upload_round":0,"upload_error":None,
    "use_dcp":True,"source_choice":PLACEHOLDER,"blur_amount":0.0,"blur_kind":"Gaussian",
    "preset_choice":"Balanced (default)","blur_angle":0.0,
    **PRESETS["Balanced (default)"]}.items():
    st.session_state.setdefault(_k,_v)
if st.session_state.pop("_reset_source_choice",False): # must happen before the selectbox is created
    st.session_state.source_choice=PLACEHOLDER
# Auto-Enhance button: Balanced preset with strength and blur set automatically
def _apply_auto_enhance():
    for k,v in PRESETS["Balanced (default)"].items():
        st.session_state[k]=v
    st.session_state.use_dcp=True
    st.session_state.strength_auto=True # Auto-Enhance = Balanced, strength from the haze level, blur measured
    st.session_state.deblur_auto=True
    st.session_state.preset_choice="Balanced (default)"
# Reset button: switches every filter off
def _apply_reset():
    st.session_state.hist_strength=0
    st.session_state.freq_strength=0
    st.session_state.smooth_strength=0
    st.session_state.sharpen_strength=0
    st.session_state.tone_gamma=1.0
    st.session_state.dcp_far=0
    st.session_state.tone_sat=1.0
    st.session_state.tone_balance=0
    st.session_state.tone_restore=0
    st.session_state.clahe_strength=0
    st.session_state.deblur_amount=0.0
    st.session_state.deblur_auto=False
    st.session_state.strength_auto=False
# Loads the chosen preset into the widgets
def _apply_preset():
    for k,v in PRESETS[st.session_state.preset_choice].items():
        st.session_state[k]=v
# Loads the sample scene, a clear street photo or a bundled test photo into the session
def _choose_source():
    choice=st.session_state.source_choice
    if choice==PLACEHOLDER:
        return
    if choice==SAMPLE_LABEL:
        img,depth=make_sample_scene()
        st.session_state.update(source_img=img,source_key="sample",source_depth=depth,
            reference=("sample",img),pre_foggy=False)
        return
    if choice in list_clear_photos(): # a clear street photo: fog is simulated on it, the photo is the reference
        img=_read_rgb(list_clear_photos()[choice])
        st.session_state.update(source_img=img,source_key=f"street:{choice}",source_depth=None,
            reference=(f"street:{choice}",img),pre_foggy=False)
        return
    path,ref=list_test_images()[choice]
    img,key=_read_rgb(path),f"test:{choice}"
    st.session_state.update(
        source_img=img,source_key=key,source_depth=None,pre_foggy=True,
        reference=(key,_read_rgb(ref,(img.shape[1],img.shape[0]))) if ref else None)
# Pipeline settings from the sliders (modules-only mode keeps just the three course modules)
def current_params():
    ss=st.session_state
    only=ss.modules_only
    return dict(hist=ss.hist_strength,freq=ss.freq_strength,smooth=ss.smooth_strength,
        sharpen=ss.sharpen_strength,omega=ss.dcp_omega,t0=ss.dcp_t0,patch=ss.dcp_patch,
        use_dcp=bool(ss.use_dcp and not only),deblur=0.0 if only else ss.deblur_amount,
        deblur_kind=ss.deblur_kind,deblur_k=ss.deblur_k,deblur_angle=ss.deblur_angle,
        gamma=1.0 if only else ss.tone_gamma,sat=1.0 if only else ss.tone_sat,
        balance=0 if only else ss.tone_balance,cutoff=ss.freq_cutoff,
        clahe=0 if only else ss.clahe_strength,clahe_tiles=ss.clahe_tiles,
        restore=0 if only else ss.tone_restore,adaptive=bool(ss.dcp_adaptive and not only),
        far=0 if only else ss.dcp_far)
# Applies the automatic options: blur from estimate_blur and strengths scaled to the haze level
def effective_params(img,P):
    ss,info=st.session_state,{}
    if ss.deblur_auto and not ss.modules_only:
        # Blind blur estimate (edge-spread method) sets the Wiener deblur
        est=estimate_blur(img)
        info["blur"]=est
        if est["kind"] in ("Gaussian","Motion"):
            P=dict(P,deblur=round(est["amount"]*2)/2,deblur_kind=est["kind"],deblur_angle=est["angle"]) # slider step 0.5
        else:
            P=dict(P,deblur=0.0)
    if ss.strength_auto:
        # Haze level from the dark channel scales the enhancement strengths
        h=haze_of(img)
        info["haze"]=(h,strength_scale(h))
        P=auto_strength_params(img,P,haze=h)[0]
    return P,info
has_image=st.session_state.source_img is not None
pipe_info=None # filled in by the Preview card once a photo is loaded
hazy=result=None
auto_info={} # what the automatic options found for the current photo (see effective_params)
# Both cards sit in a narrower centred strip
_,center_col,_=st.columns(PAGE_SPLIT,gap="large")
left_col,right_col=center_col.columns(CARD_SPLIT,gap="medium")
# Left card: upload, preview views and measurements
with left_col:
    with st.container(border=True,key="upload_card"):
        if not has_image:
            st.selectbox("Or try a sample",[PLACEHOLDER,SAMPLE_LABEL,*list_clear_photos(),*list_test_images()],
                key="source_choice",on_change=_choose_source,label_visibility="collapsed")
            # No type filter, so any image is accepted; an unreadable file gets a message and a fresh uploader
            uploaded_file=st.file_uploader(
                "Drop a foggy photo here",
                label_visibility="collapsed",
                key=f"photo_uploader_{st.session_state.upload_round}",
            )
            if uploaded_file is not None:
                try:
                    photo=load_photo(uploaded_file)
                except Exception as exc:
                    st.session_state.upload_error=unreadable_message(uploaded_file.name,exc)
                    st.session_state.upload_round+=1
                else:
                    st.session_state.upload_error=None
                    st.session_state.source_img=photo
                    st.session_state.source_key=f"{uploaded_file.name}-{uploaded_file.size}"
                    st.session_state.source_depth=None
                    st.session_state.reference=None
                    st.session_state.pre_foggy=True
                    st.session_state._reset_source_choice=True
                st.rerun()
            if st.session_state.upload_error:
                st.error(st.session_state.upload_error)
        else:
            raw=st.session_state.source_img
            hazy=raw if st.session_state.pre_foggy else simulate_blur(
                simulate_fog(raw,st.session_state.source_depth,st.session_state.fog_density),
                st.session_state.blur_amount,st.session_state.blur_kind,st.session_state.blur_angle)
            P,auto_info=effective_params(hazy,current_params())
            src_h,src_w=hazy.shape[:2]
            view=st.session_state.get("view_mode") or "Side by side"
            # Only for a sliver of a picture (taller than 5:1): see the preview CSS.
            sliver_cap=(
                '.st-key-preview_images { width: min(100%, max('
                'calc(2 * var(--pv-maxh) * var(--pv-ar) + 1rem), 9rem)) !important; }'
                '.st-key-preview_images [data-testid="stImage"] img '
                '{ max-height: var(--pv-maxh); object-fit: contain; }'
                if src_w/src_h<0.2 else ""
            )
            single_cap=(
                '.st-key-preview_images { width: min(100%, calc(var(--pv-maxh) * var(--pv-ar, 1.25))) !important; }'
                if view not in ("Side by side","Stages","Edges","Transmission") else ""
            )
            top_l,top_r=st.columns([3,1])
            with top_l:
                st.markdown(
                    f'<style>.st-key-preview_images {{ --pv-ar: {src_w/src_h:.4f}; }}{sliver_cap}{single_cap}</style>'
                    '<style>.st-key-upload_card { min-height: 0 !important; }'
                    '[data-testid="stLayoutWrapper"]:has(> .st-key-upload_card) '
                    '{ flex: 0 0 auto !important; align-self: flex-start; width: 100%; position: sticky; top: 4.5rem; z-index: 1000050; }'
                    '[data-testid="stLayoutWrapper"]:has(> .st-key-preview_images) '
                    '{ margin-top: 0 !important; margin-bottom: 0 !important; }</style>'
                    '<div class="card-heading">Preview</div>',
                    unsafe_allow_html=True,
                )
            with top_r:
                if st.button("Change Image",width="stretch"):
                    st.session_state.source_img=None
                    st.session_state.source_key=None
                    st.session_state._reset_source_choice=True
                    st.rerun()
            st.segmented_control("View",VIEWS,default="Side by side",key="view_mode",
                label_visibility="collapsed")
            if view=="Slider":
                st.slider("Compare position",0,100,50,key="compare_pos",label_visibility="collapsed")
            elif view=="Stages":
                st.pills("Stage",STAGE_LABELS,default=STAGE_LABELS[0],key="stage_pick",
                    label_visibility="collapsed")
                st.segmented_control("Stage mode",["Cumulative","This stage only"],default="Cumulative",
                    key="stage_mode",label_visibility="collapsed")
                st.caption("Cumulative applies every stage up to the picked one, in run order, to the original. "
                    "This stage only applies just the picked stage to the original.")
            with st.spinner("Processing…"):
                result,proc_s=run_pipeline(hazy,P)
            # Only when fog (and blur) were simulated is there a clear reference: the photo itself.
            ref=raw if not st.session_state.pre_foggy else None
            if ref is None and st.session_state.reference is not None: # a bundled photo that comes with its clear version
                ref_key,ref_img=st.session_state.reference
                if ref_key==st.session_state.source_key and ref_img.shape==hazy.shape:
                    ref=ref_img
            ref_for_compare=ref
            with st.container(key="preview_images"):
                if view=="Side by side":
                    col_before,col_after=st.columns(2)
                    with col_before:
                        st.image(hazy,caption="Original",width="stretch")
                    with col_after:
                        st.image(result,caption="Dehazed",width="stretch")
                elif view=="Slider":
                    pos=st.session_state.get("compare_pos",50)
                    st.image(compare_composite(hazy,result,pos),
                        caption=f"Original (left) · Dehazed (right) · divider at {pos}%",width="stretch")
                elif view=="Stages":
                    pick=st.session_state.get("stage_pick") or STAGE_LABELS[0]
                    idx=STAGE_LABELS.index(pick)
                    name=STAGE_ORDER[idx]
                    if st.session_state.get("stage_mode")=="This stage only":
                        before,after,cap_b=hazy,apply_stage(name,hazy,P),"Original"
                    else:
                        # Cumulative: every stage up to and including the picked one, in run order, against the original
                        steps=run_stages(hazy,P)
                        before,after,cap_b=hazy,steps[idx+1][1],"Original"
                    col_b,col_a=st.columns(2)
                    with col_b:
                        st.image(before,caption=cap_b,width="stretch")
                    with col_a:
                        st.image(after,width="stretch",
                            caption=(f"Cumulative: stages up to {pick}" if st.session_state.get("stage_mode")!="This stage only"
                                else f"Only {pick}") if stage_active(name,P) else f"{pick} is off, so it adds nothing")
                elif view=="Edges":
                    # Edges view: Sobel gradient magnitude of both images on one scale
                    e0,e1=edge_maps(hazy,result)
                    col_b,col_a=st.columns(2)
                    with col_b:
                        st.image(e0,caption="Original edges",width="stretch")
                    with col_a:
                        st.image(e1,caption="Dehazed edges (same scale)",width="stretch")
                elif view=="Transmission":
                    col_b,col_a=st.columns(2)
                    with col_b:
                        st.image(hazy,caption="Original",width="stretch")
                    with col_a:
                        # Transmission view: the dark channel prior's estimated t as a colour map
                        st.image(transmission_map_image(hazy,P),width="stretch",
                            caption="Estimated transmission t (dark = thick haze, bright = clear)"
                            if P["use_dcp"] else "Dark channel is off, so this is the map it would estimate")
                elif view=="Modules":
                    # Course modules made visible: FFT spectrum, histogram clip points, adaptive smoothing and sharpening maps
                    steps=run_stages(hazy,P)
                    st.markdown("**Frequency-domain enhancement**")
                    s0,vmax=spectrum_image(hazy)
                    s1,_=spectrum_image(result,vmax)
                    col_b,col_a=st.columns(2)
                    col_b.image(s0,caption="Spectrum of the original (log magnitude, centre = low frequency)",width="stretch")
                    col_a.image(s1,caption="Spectrum of the dehazed image (same scale)",width="stretch")
                    rx,r0=radial_spectrum(hazy)
                    _,r1=radial_spectrum(result)
                    fx,fy=freq_response_curve(P["freq"],P["cutoff"],hazy.shape[:2])
                    col_b.markdown('<p class="ufc-cap">Mean spectrum by frequency: original (grey), dehazed (red)</p>'
                        +line_svg([(rx,r0,"#9a9a9a",False),(rx,r1,"#d32f2f",False)],label="Radial spectrum",xlabel="frequency (1 = Nyquist)",ylabel="log |F|"),unsafe_allow_html=True)
                    col_a.markdown(f'<p class="ufc-cap">Frequency filter gain (1 + (g - 1) &middot; Gaussian high-pass), freq strength {P["freq"]:.0f}, cutoff {P["cutoff"]}</p>'
                        +line_svg([(fx,fy,"#1e88e5",False),(fx,np.ones_like(fy),"#555555",True)],label="Frequency filter response",xlabel="frequency (1 = Nyquist)",ylabel="gain"),unsafe_allow_html=True)
                    st.markdown("**Histogram stretching**")
                    h_in,h_out=steps[STAGE_ORDER.index("hist")][1],steps[STAGE_ORDER.index("hist")+1][1]
                    lo,hi,pc=stretch_points(h_in,P["hist"])
                    y_in=quality_metrics(h_in)["hist"]
                    y_out=quality_metrics(h_out)["hist"]
                    col_b,col_a=st.columns(2)
                    col_b.markdown(f'<p class="ufc-cap">Before stretch, clip points {lo:.0f} and {hi:.0f} ({pc:.2f}% clipped each end)</p>'
                        +hist_svg([(y_in,"#9a9a9a",False)],marks=[(lo,"#1e88e5"),(hi,"#1e88e5")],label="Histogram before stretch"),unsafe_allow_html=True)
                    col_a.markdown('<p class="ufc-cap">After stretch: the range fills 0 to 255</p>'
                        +hist_svg([(y_out,"#d32f2f",False)],label="Histogram after stretch"),unsafe_allow_html=True)
                    st.markdown("**Adaptive smoothing and sharpening**")
                    sm_in,sh_in=steps[STAGE_ORDER.index("smooth")][1],steps[STAGE_ORDER.index("sharpen")][1]
                    sw=sharpen_weight_map(sh_in,P["sharpen"] or 50)
                    sc=smoothing_change_map(sm_in,P["smooth"] or 50)
                    col_b,col_a=st.columns(2)
                    col_b.image(heat_image(sc,max(float(np.percentile(sc,99.5)),1.0)),width="stretch",
                        caption="Smoothing: where the bilateral filter changes pixels (bright = flat noisy areas, dark = edges kept)")
                    col_a.image(heat_image(sw,max(float(sw.max()),1e-3)),width="stretch",
                        caption="Sharpening gain map (bright = real structure, dark = flat or noisy areas)")
                    if not (P["freq"]>0 and P["hist"]>0 and P["smooth"]>0 and P["sharpen"]>0):
                        st.caption("A module is set to 0 in the pipeline, so its panels show what it would do at the default strength or no change.")
                elif view=="Compare":
                    # Full pipeline against single-technique baselines
                    base=baselines(hazy)
                    cmp_items=[("Original",hazy)]+list(base.items())+[("Ours (full pipeline)",result)]
                    cols=st.columns(3)
                    for i,(nm,im) in enumerate(cmp_items):
                        q=quality_metrics(im)
                        cap=f"{nm} · contrast {q['contrast']:.1f} · edges {q['edges']:.1f}%"
                        if ref_for_compare is not None and ref_for_compare.shape==im.shape:
                            cap+=f" · PSNR {psnr(im,ref_for_compare):.1f} dB · SSIM {ssim(im,ref_for_compare):.3f}"
                        cols[i%3].image(im,caption=cap,width="stretch")
                elif view=="Applications":
                    render_applications(hazy,result,"prev")
                elif view=="Original":
                    st.image(hazy,caption="Original",width="stretch")
                else:
                    st.image(result,caption="Dehazed",width="stretch")
            # Measurements table and histogram comparison below the images
            st.markdown('<div class="card-heading" style="font-size:1.5rem;margin-top:1.4rem">Measurements</div>',
                unsafe_allow_html=True)
            m0,m1=quality_metrics(hazy),quality_metrics(result)
            # Canny edge and Hough line counts before and after
            d0,d1=downstream(hazy,ref),downstream(result,ref)
            vis=visibility_metrics(hazy,result)
            # One table row of original and dehazed values
            def _row(name,a,b,d):
                return f"<tr><td>{name}</td><td>{a:.{d}f}</td><td><b>{b:.{d}f}</b></td></tr>"
            # Measurement rows: contrast, entropy, gradient, Hautiere visibility metrics, Canny edges, Hough lines
            rows=(_row("Contrast (std. dev.)",m0["contrast"],m1["contrast"],1)
                +_row("Entropy (bits)",m0["entropy"],m1["entropy"],2)
                +_row("Mean gradient",m0["gradient"],m1["gradient"],2)
                +_row("Edge pixels (%)",m0["edges"],m1["edges"],1)
                +f'<tr><td>Visible-edge gain e</td><td>&ndash;</td><td><b>{vis["e"]:+.2f}</b></td></tr>'
                +f'<tr><td>Gradient ratio r̄</td><td>&ndash;</td><td><b>{vis["r"]:.2f}</b></td></tr>'
                +f'<tr><td>Newly clipped pixels (%)</td><td>&ndash;</td><td><b>{100*vis["sigma"]:.2f}</b></td></tr>'
                +_row("Canny edges (% px)",d0["canny_pct"],d1["canny_pct"],1)
                +_row("Lines found (Hough)",d0["lines"],d1["lines"],0)
                +_row("&nbsp;&nbsp;vertical (buildings)",d0["lines_vertical"],d1["lines_vertical"],0)
                +_row("&nbsp;&nbsp;oblique (lanes)",d0["lines_oblique"],d1["lines_oblique"],0))
            if ref is not None:
                # With a clear reference: PSNR, SSIM, edge F1 and line recall
                rows+=(_row("PSNR vs. clear (dB)",psnr(hazy,ref),psnr(result,ref),1)
                    +_row("SSIM vs. clear",ssim(hazy,ref),ssim(result,ref),3)
                    +_row("Edge F1 vs. clear",d0["edge_f1"],d1["edge_f1"],3)
                    +_row("Line recall vs. clear",d0["line_recall"],d1["line_recall"],3))
            rows+=f"<tr><td>Processing time (s)</td><td>&ndash;</td><td><b>{proc_s:.2f}</b></td></tr>"
            st.markdown(
                '<p class="ufc-cap">Structure visibility: edge pixels '
                f'{m0["edges"]:.1f}% &rarr; {m1["edges"]:.1f}% &middot; mean gradient &times;'
                f'{m1["gradient"]/max(m0["gradient"],1e-3):.1f} &middot; contrast &times;'
                f'{m1["contrast"]/max(m0["contrast"],1e-3):.1f}</p>',
                unsafe_allow_html=True)
            c_tab,c_h=st.columns([1,1.1],gap="medium")
            c_tab.markdown(f'<table class="ufc-m"><tr><th></th><th>Original</th><th>Dehazed</th></tr>{rows}</table>',
                unsafe_allow_html=True)
            c_h.markdown(
                '<p class="ufc-cap">Brightness histogram: original (grey), dehazed (red)</p>'
                +hist_svg([(m0["hist"],"#9a9a9a",False),(m1["hist"],"#d32f2f",False)],label="Brightness histograms"),
                unsafe_allow_html=True)
            chan=["#e53935","#43a047","#1e88e5"]
            c_h.markdown(
                '<p class="ufc-cap">R, G, B histograms: original (dashed), dehazed (solid)</p>'
                +hist_svg([(m0["chans"][c],chan[c],True) for c in range(3)]
                    +[(m1["chans"][c],chan[c],False) for c in range(3)],label="Colour channel histograms"),
                unsafe_allow_html=True)
            # Gamma curves y = x^(1/g) of the original and the dehazed image (g from the median brightness) against normal gamma 1.0
            g0,g1=effective_gamma(hazy),effective_gamma(result)
            gx=np.arange(0,256,4.0)
            c_h.markdown(
                f'<p class="ufc-cap">Gamma of the image: normal 1.00 (dashed), original {g0:.2f} (grey), '
                f'dehazed {g1:.2f} (red). Final Tone slider gamma: {float(P["gamma"]):.2f}</p>'
                +curve_svg([(gx,gx,"#555555",True,1.5),(gx,255*(gx/255)**(1/g0),"#9a9a9a",False,3),
                    (gx,255*(gx/255)**(1/g1),"#d32f2f",False,3)],label="Gamma curves, original and dehazed"),
                unsafe_allow_html=True)
            # Downloads: side-by-side PNG and the measurements as CSV
            csv_rows=[("metric","original","dehazed")]+[(k,f"{m0[k]:.4f}",f"{m1[k]:.4f}") for k in ("contrast","entropy","gradient","edges")]
            csv_rows+=[(k,f"{d0[k]:.4f}",f"{d1[k]:.4f}") for k in ("canny_pct","lines","lines_vertical","lines_oblique")]
            csv_rows+=[("visible_edge_gain_e","",f"{vis['e']:.4f}"),("gradient_ratio_r","",f"{vis['r']:.4f}"),("newly_clipped_pct","",f"{100*vis['sigma']:.4f}")]
            if ref is not None:
                csv_rows+=[("psnr_vs_clear_db",f"{psnr(hazy,ref):.4f}",f"{psnr(result,ref):.4f}"),("ssim_vs_clear",f"{ssim(hazy,ref):.4f}",f"{ssim(result,ref):.4f}")]
            dl1,dl2=st.columns(2)
            dl1.download_button("Download before/after PNG",data=to_png_bytes(np.hstack([hazy,result])),file_name="before_after.png",mime="image/png",width="stretch")
            dl2.download_button("Download measurements CSV",data="\n".join(",".join(r) for r in csv_rows),file_name="measurements.csv",mime="text/csv",width="stretch")
# Right card: enhancement pipeline controls
# HTML heading row of a pipeline stage
def _stage_row(num,label,value):
    return (
        f'<div class="stage-row"><span><span class="stage-num">{num}</span>'
        f'<span class="stage-label">{label}</span></span>'
        f'<span class="stage-value">{value}</span></div>'
    )
# HTML label and value row of a parameter
def _param_row(label,value):
    return (
        '<div class="stage-row" style="margin-top:.45rem"><span class="stage-label" '
        f'style="font-weight:500;font-size:.95rem">{label}</span>'
        f'<span class="stage-value" style="font-size:.95rem">{value}</span></div>'
    )
with right_col:
    with st.container(border=True,key="pipeline_card"):
        st.markdown('<div class="pipeline-heading">Enhancement Pipeline</div>',unsafe_allow_html=True)
        # Stage 00 input: photo as is, or fog (scattering model) and blur simulated on it
        with st.container(key="stage_input"):
            foggy=st.session_state.pre_foggy
            st.markdown(_stage_row("00","Input","photo as is" if foggy else f"β {st.session_state.fog_density:.2f}"),
                unsafe_allow_html=True)
            st.checkbox("Photo is already foggy (skip the fog simulation)",key="pre_foggy")
            st.markdown(_param_row("Simulated fog density",f"{st.session_state.fog_density:.2f}"),
                unsafe_allow_html=True)
            st.slider("Fog density",0.0,4.0,step=0.05,key="fog_density",
                disabled=st.session_state.pre_foggy,label_visibility="collapsed")
            blur_txt="off" if st.session_state.blur_amount==0 else f"{st.session_state.blur_amount:.1f}px"
            st.markdown(_param_row("Simulated blur",blur_txt),unsafe_allow_html=True)
            st.slider("Blur amount",0.0,10.0,step=0.5,key="blur_amount",
                disabled=st.session_state.pre_foggy,label_visibility="collapsed")
            st.selectbox("Blur type",["Gaussian","Motion"],key="blur_kind",disabled=st.session_state.pre_foggy)
            if st.session_state.blur_kind=="Motion":
                st.markdown(_param_row("Motion direction",f"{st.session_state.blur_angle:.0f}°"),unsafe_allow_html=True)
                st.slider("Motion direction",0.0,179.0,step=5.0,key="blur_angle",
                    disabled=st.session_state.pre_foggy,label_visibility="collapsed")
            st.toggle("Course modules only (skip dark channel, deblur and tone)",key="modules_only")
        # Stage 01: dark channel prior fog removal
        with st.container(key="stage_dcp"):
            st.markdown(_stage_row("01","Dark Channel Prior",
                    "On" if st.session_state.use_dcp and not st.session_state.modules_only else "Off"),
                unsafe_allow_html=True)
            st.toggle("Remove fog before the other stages",key="use_dcp")
            st.markdown(_param_row("Haze removed (ω)",f"{st.session_state.dcp_omega:.2f}"),unsafe_allow_html=True)
            st.slider("Haze removed",0.50,1.00,step=0.01,key="dcp_omega",label_visibility="collapsed")
            st.markdown(_param_row("Minimum transmission (t₀)",f"{st.session_state.dcp_t0:.2f}"),unsafe_allow_html=True)
            st.slider("Minimum transmission",0.05,0.50,step=0.01,key="dcp_t0",label_visibility="collapsed")
            patch=st.session_state.dcp_patch
            st.markdown(_param_row("Patch size","auto" if patch==0 else f"{patch}px"),unsafe_allow_html=True)
            st.slider("Patch size",0,61,step=1,key="dcp_patch",label_visibility="collapsed")
            st.toggle("Adaptive airlight and t₀ (measured from the photo's haze)",key="dcp_adaptive")
            # Distant detail: noise-gated local-contrast boost weighted by haze depth (1 - t)
            st.markdown(_param_row("Distant detail",f"{st.session_state.dcp_far}"),unsafe_allow_html=True)
            st.slider("Distant detail",0,100,key="dcp_far",label_visibility="collapsed")
        # Stage 02: Wiener deconvolution deblur
        with st.container(key="stage_deblur"):
            blur_est=auto_info.get("blur")
            auto_on=bool(st.session_state.deblur_auto and blur_est is not None)
            amt=st.session_state.deblur_amount
            if auto_on:
                amt=round(blur_est["amount"]*2)/2 if blur_est["kind"] in ("Gaussian","Motion") else 0.0
            st.markdown(_stage_row("02","Deblur (Wiener)","off" if amt==0 else f"{amt:.1f}px"),
                unsafe_allow_html=True)
            st.toggle("Detect the blur automatically",key="deblur_auto",disabled=st.session_state.modules_only)
            if auto_on:
                kind=blur_est["kind"]
                st.caption("Detected: the photo looks sharp, nothing to undo." if kind=="none" else
                    f"Detected: {kind.lower()} blur of about {blur_est['amount']:.1f} px"
                    +(f" at {blur_est['angle']:.0f}° (0° = horizontal)" if kind=="Motion" else "")+", undone below.")
            st.markdown(_param_row("Blur to undo","off" if amt==0 else f"{amt:.1f}px"),unsafe_allow_html=True)
            st.slider("Blur to undo",0.0,10.0,step=0.5,key="deblur_amount",label_visibility="collapsed",
                disabled=auto_on)
            st.selectbox("Assumed blur type",["Gaussian","Motion"],key="deblur_kind",disabled=auto_on)
            if st.session_state.deblur_kind=="Motion" or (auto_on and blur_est["kind"]=="Motion"):
                ang=blur_est["angle"] if auto_on and blur_est["kind"]=="Motion" else st.session_state.deblur_angle
                st.markdown(_param_row("Motion direction",f"{ang:.0f}°"),unsafe_allow_html=True)
                st.slider("Motion direction to undo",0.0,179.0,step=5.0,key="deblur_angle",
                    label_visibility="collapsed",disabled=auto_on)
            st.markdown(_param_row("Noise damping (K)",f"{st.session_state.deblur_k:.3f}"),unsafe_allow_html=True)
            st.slider("Noise damping",0.001,0.100,step=0.001,format="%.3f",key="deblur_k",
                label_visibility="collapsed")
        # Stage 03: histogram stretching
        with st.container(key="stage_hist"):
            st.markdown(_stage_row("03","Histogram Stretch",st.session_state.hist_strength),unsafe_allow_html=True)
            hist_strength=st.slider("Histogram strength",0,100,key="hist_strength",label_visibility="collapsed")
        # Stage 03b: CLAHE local contrast on the luminance
        with st.container(key="stage_clahe"):
            st.markdown(_stage_row("03b","CLAHE (local contrast)",st.session_state.clahe_strength),
                unsafe_allow_html=True)
            st.slider("CLAHE clip strength",0,100,key="clahe_strength",label_visibility="collapsed")
            st.markdown(_param_row("Tile grid",f"{st.session_state.clahe_tiles}×{st.session_state.clahe_tiles}"),
                unsafe_allow_html=True)
            st.slider("CLAHE tile grid",2,16,key="clahe_tiles",label_visibility="collapsed")
        # Stage 04: frequency-domain high-frequency emphasis (FFT)
        with st.container(key="stage_freq"):
            st.markdown(_stage_row("04","Frequency-Domain Filter",st.session_state.freq_strength),
                unsafe_allow_html=True)
            freq_strength=st.slider("Frequency strength",0,100,key="freq_strength",label_visibility="collapsed")
            st.markdown(_param_row("Cutoff radius",f"{st.session_state.freq_cutoff}px"),unsafe_allow_html=True)
            st.slider("Cutoff radius",5,80,key="freq_cutoff",label_visibility="collapsed")
        # Stage 05: adaptive smoothing (bilateral filter)
        with st.container(key="stage_smooth"):
            st.markdown(_stage_row("05","Adaptive Smoothing",st.session_state.smooth_strength),
                unsafe_allow_html=True)
            smooth_strength=st.slider("Smoothing strength",0,100,key="smooth_strength",label_visibility="collapsed")
        # Stage 06: adaptive sharpening (unsharp mask)
        with st.container(key="stage_sharpen"):
            st.markdown(_stage_row("06","Adaptive Sharpening",st.session_state.sharpen_strength),
                unsafe_allow_html=True)
            sharpen_strength=st.slider("Sharpening strength",0,100,key="sharpen_strength",
                label_visibility="collapsed")
        # Stage 07: final tone (colour restoration, gray-world balance, saturation, gamma)
        with st.container(key="stage_tone"):
            st.markdown(_stage_row("07","Final Tone",""),unsafe_allow_html=True)
            st.markdown(_param_row("Gamma",f"{st.session_state.tone_gamma:.2f}"),unsafe_allow_html=True)
            st.slider("Gamma",0.60,1.60,step=0.05,key="tone_gamma",label_visibility="collapsed")
            st.markdown(_param_row("Saturation",f"{st.session_state.tone_sat:.2f}"),unsafe_allow_html=True)
            st.slider("Saturation",0.50,2.00,step=0.05,key="tone_sat",label_visibility="collapsed")
            st.markdown(_param_row("Colour balance",f"{st.session_state.tone_balance}"),unsafe_allow_html=True)
            st.slider("Colour balance",0,100,key="tone_balance",label_visibility="collapsed")
            st.markdown(_param_row("Colour restoration",f"{st.session_state.tone_restore}"),unsafe_allow_html=True)
            st.slider("Colour restoration",0,100,key="tone_restore",label_visibility="collapsed")
        with st.container(key="pipeline_actions"):
            st.selectbox("Preset",list(PRESETS),key="preset_choice",on_change=_apply_preset)
            st.toggle("Set the strength from the haze level",key="strength_auto")
            if "haze" in auto_info:
                st.caption(f"Haze level {auto_info['haze'][0]:.2f}: the enhancement runs at "
                    f"{auto_info['haze'][1]*100:.0f}% of the sliders' strength.")
            b1,b2=st.columns(2)
            with b1:
                st.button("Auto-Enhance",width="stretch",on_click=_apply_auto_enhance)
            with b2:
                st.button("Reset",width="stretch",on_click=_apply_reset)
            if has_image and result is not None:
                st.download_button("Download Result",data=to_png_bytes(result),file_name="dehazed_result.png",
                    mime="image/png",type="primary",width="stretch")
            else:
                st.download_button("Download Result",data=b"",file_name="dehazed_result.png",
                    mime="image/png",type="primary",width="stretch",disabled=True)
# Below the cards: batch processing, video and method
_,lower_col,_=st.columns(PAGE_SPLIT,gap="large")
with lower_col:
    with st.container(border=True,key="batch_card"):
        st.markdown('<div class="card-heading">Batch Processing</div>',unsafe_allow_html=True)
        st.markdown('<p class="ufc-cap">Run several photos through the pipeline with the current settings, '
            'for example frames from a traffic camera or site-inspection photos. Every photo is '
            'treated as already foggy.</p>',unsafe_allow_html=True)
        def _clear_batch(): # a new uploader key drops every uploaded file at once
            st.session_state.batch_round=st.session_state.get("batch_round",0)+1
            st.session_state.batch_results=[]
        batch_files=st.file_uploader("Photos to process",accept_multiple_files=True,
            key=f"batch_up_{st.session_state.get('batch_round',0)}",
            label_visibility="collapsed")
        if batch_files or st.session_state.get("batch_results"):
            run_col,clear_col,_sp=st.columns([1,1,3])
            run=run_col.button("Process batch",type="primary",disabled=not batch_files)
            clear_col.button("Clear all",on_click=_clear_batch,help="Remove every uploaded photo and the results")
        else:
            run=False
        if batch_files and run:
            P_b,results,bar=current_params(),[],st.progress(0.0)
            for i,f in enumerate(batch_files):
                try:
                    img_b=load_photo(f)
                except Exception as exc:
                    results.append({"name":f.name,"error":unreadable_message(f.name,exc)})
                else:
                    t_b=time.perf_counter()
                    out_b=dehaze_pipeline(img_b,effective_params(img_b,P_b)[0])
                    sec=time.perf_counter()-t_b
                    q0,q1=quality_metrics(img_b),quality_metrics(out_b)
                    results.append({"name":f.name,"size":f"{img_b.shape[1]}x{img_b.shape[0]}","sec":sec,
                            "contrast":q1["contrast"]/max(q0["contrast"],1e-3),
                            "edges":q1["edges"]/max(q0["edges"],1e-3),"png":to_png_bytes(out_b),"orig":to_png_bytes(img_b)})
                bar.progress((i+1)/len(batch_files))
            bar.empty()
            st.session_state.batch_results=results
        done=[r for r in st.session_state.get("batch_results",[]) if "png" in r]
        for r in st.session_state.get("batch_results",[]):
            if "error" in r:
                st.error(r["error"])
        if done:
            st.segmented_control("Batch view",["Table","All images","Side by side","Slider","Applications","Graphs"],default="Table",
                key="batch_view",label_visibility="collapsed")
            bview=st.session_state.get("batch_view") or "Table"
            if bview=="Table":
                body="".join(f"<tr><td>{r['name']}</td><td>{r['size']}</td><td>{r['sec']:.2f}</td>"
                    f"<td>{r['contrast']:.2f}</td><td>{r['edges']:.2f}</td></tr>" for r in done)
                st.markdown('<table class="ufc-m"><tr><th>Photo</th><th>Size</th><th>Time (s)</th>'
                    f'<th>Contrast &times;</th><th>Edge pixels &times;</th></tr>{body}</table>',
                    unsafe_allow_html=True)
            elif bview=="All images":
                # Every photo in the batch: original on the left, dehazed on the right
                for r in done:
                    ca,cb=st.columns(2)
                    ca.image(r["orig"],caption=f"{r['name']} (original)",width="stretch")
                    cb.image(r["png"],caption=f"{r['name']} (dehazed)",width="stretch")
            else:
                # One photo at a time: pick it, then compare, run the application tests or look at its graphs
                pick=st.selectbox("Photo",[r["name"] for r in done],key="batch_pick")
                r=next(x for x in done if x["name"]==pick)
                if bview=="Slider":
                    pos=st.slider("Compare position",0,100,50,key="batch_pos",label_visibility="collapsed")
                    st.image(compare_composite(png_to_rgb(r["orig"]),png_to_rgb(r["png"]),pos),
                        caption=f"Original (left) · Dehazed (right) · divider at {pos}%",width="stretch")
                elif bview=="Applications":
                    sens,tiled=render_applications(png_to_rgb(r["orig"]),png_to_rgb(r["png"]),"batch")
                    if st.checkbox("Count vehicles and people in every photo",key="batch_count"):
                        try:
                            rows="".join(f"<tr><td>{x['name']}</td><td>{len(detect_vehicles(png_to_rgb(x['orig']),sens,tiled))}</td>"
                                f"<td><b>{len(detect_vehicles(png_to_rgb(x['png']),sens,tiled))}</b></td></tr>" for x in done)
                            st.markdown(f'<table class="ufc-m"><tr><th>Photo</th><th>Original</th><th>Dehazed</th></tr>{rows}</table>',unsafe_allow_html=True)
                        except Exception as exc:
                            st.caption(f"Detection model not available here ({type(exc).__name__}).")
                elif bview=="Graphs":
                    names=[x["name"] for x in done]
                    # Gain of contrast and of edge pixels for every photo in the batch (1.0 = unchanged)
                    st.markdown('<p class="ufc-cap">Contrast gain per photo (&times;, dashed line = 1)</p>'
                        +bars_svg(names,[x["contrast"] for x in done],ref=1.0,label="Contrast gain per photo"),unsafe_allow_html=True)
                    st.markdown('<p class="ufc-cap">Edge-pixel gain per photo (&times;, dashed line = 1)</p>'
                        +bars_svg(names,[x["edges"] for x in done],"#1e88e5",ref=1.0,label="Edge gain per photo"),unsafe_allow_html=True)
                    render_graphs(png_to_rgb(r["orig"]),png_to_rgb(r["png"]))
                else:
                    ca,cb=st.columns(2)
                    ca.image(r["orig"],caption="Original",width="stretch")
                    cb.image(r["png"],caption="Dehazed",width="stretch")
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
                for r in done:
                    zf.writestr(f"dehazed_{Path(r['name']).stem}.png",r["png"])
                zf.writestr("batch_results.csv","photo,size,seconds,contrast_x,edge_pixels_x\n"+"".join(
                        f"{r['name']},{r['size']},{r['sec']:.3f},{r['contrast']:.3f},{r['edges']:.3f}\n" for r in done))
            st.download_button("Download all as ZIP",data=buf.getvalue(),file_name="dehazed_batch.zip",
                mime="application/zip")
    with st.container(border=True,key="video_card"):
        st.markdown('<div class="card-heading">Video</div>',unsafe_allow_html=True)
        st.markdown('<p class="ufc-cap">Dehaze a short clip from a fixed camera with the current settings, including the '
            'automatic options. The fog, colour and contrast estimates are smoothed over time and static pixels are '
            'averaged, so the result does not flicker; moving vehicles are left alone. Frames are scaled to at most '
            '640 px.</p>',unsafe_allow_html=True)
        vid=st.file_uploader("Video",type=["mp4","mov","avi","mkv","webm"],key="video_up",label_visibility="collapsed")
        v1,v2=st.columns(2)
        v_frames=v1.slider("Frames to process",10,300,90,key="video_frames")
        v_smooth=v2.toggle("Smooth over time",value=True,key="video_temporal")
        if vid is not None and st.button("Process video",type="primary",key="video_go"):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td: # on Windows a still-open file must not stop the cleanup
                src,dst=Path(td)/("in"+Path(vid.name).suffix),Path(td)/"out.mp4"
                orig_dst,side_dst=Path(td)/"orig.mp4",Path(td)/"side.mp4"
                src.write_bytes(vid.getvalue())
                bar=st.progress(0.0)
                try:
                    # Temporal dehazing of the clip: smoothed estimates and denoising of static pixels
                    n_fr,spf=process_video(
                        src,dst,P=current_params(),auto=st.session_state.strength_auto,temporal=v_smooth,
                        max_dim=640,max_frames=v_frames,orig_dst=orig_dst,side_dst=side_dst,auto_blur=st.session_state.deblur_auto and not st.session_state.modules_only,
                        progress=lambda i,total:bar.progress(min(1.0,i/max(total,1))))
                    st.session_state.video_result=(dst.read_bytes(),n_fr,spf,orig_dst.read_bytes(),side_dst.read_bytes()) if n_fr else None
                    st.session_state.video_extra=extract_video_data(orig_dst,dst,n_fr) if n_fr else None
                    st.session_state.video_error=None if n_fr else "No frames could be read from that file."
                except Exception as exc:
                    st.session_state.video_result,st.session_state.video_error=None,f"Could not process that video: {exc}"
            bar.empty()
        if st.session_state.get("video_error"):
            st.error(st.session_state.video_error)
        if st.session_state.get("video_result"):
            vdata,n_fr,spf,odata,sdata=st.session_state.video_result
            st.segmented_control("Video view",["Side by side","Dehazed","Original","Applications","Live detection","Graphs"],default="Side by side",
                key="video_view",label_visibility="collapsed")
            vview=st.session_state.get("video_view") or "Side by side"
            vmetrics,vframes=st.session_state.get("video_extra") or (np.zeros((0,6)),[])
            if vview in ("Side by side","Dehazed","Original"):
                # One clip with the original on the left and the dehazed frames on the right, so both play in sync
                st.video({"Side by side":sdata,"Dehazed":vdata,"Original":odata}[vview])
                if vview=="Side by side":
                    st.caption("Original (left) · Dehazed (right)")
            elif vview=="Live detection":
                # Detection on every frame of the whole clip: live preview while it runs, then a playable video with boxes and counts
                st.caption("Runs the vehicle detector on every frame of the original and the dehazed clip and plays the result as a video, "
                    "original on the left and dehazed on the right.")
                d1,d2,d3=st.columns(3)
                v_conf=d1.slider("Detection confidence",0.10,0.60,0.20,0.05,key="vdet_conf")
                v_tiled=d2.toggle("Search tiles (small, distant)",value=True,key="vdet_tiled")
                v_stride=d3.selectbox("Detect every … frames",[1,2,3,5],key="vdet_stride",
                    help="Between detections the last boxes are reused: faster, but boxes lag fast traffic")
                if st.button("Run detection on the video",type="primary",key="vdet_go"):
                    live,bar=st.empty(),st.progress(0.0)
                    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                        po,pd_,pr=Path(td)/"o.mp4",Path(td)/"d.mp4",Path(td)/"det.mp4"
                        po.write_bytes(odata)
                        pd_.write_bytes(vdata)
                        try:
                            counts=detection_video(po,pd_,pr,v_conf,v_tiled,v_stride,
                                progress=lambda i:bar.progress(min(1.0,i/max(n_fr,1))),
                                preview=lambda im,i:live.image(im,caption=f"Frame {i} of {n_fr}",width="stretch"))
                            st.session_state.video_det=(pr.read_bytes(),counts) if counts else None
                            st.session_state.video_det_error=None
                        except Exception as exc:
                            st.session_state.video_det,st.session_state.video_det_error=None,f"Detection is not available here ({type(exc).__name__}: {exc})."
                    bar.empty()
                    live.empty()
                if st.session_state.get("video_det_error"):
                    st.error(st.session_state.video_det_error)
                if st.session_state.get("video_det"):
                    ddata,dcounts=st.session_state.video_det
                    st.video(ddata)
                    dc=np.array(dcounts,dtype=float)
                    st.caption(f"Vehicles and people per frame, on average: original {dc[:,0].mean():.1f}, dehazed {dc[:,1].mean():.1f}.")
                    xs=np.linspace(0,1,len(dc))
                    st.markdown('<p class="ufc-cap">Vehicles and people found per frame: original (grey), dehazed (red)</p>'
                        +line_svg([(xs,dc[:,0],"#9a9a9a",False),(xs,dc[:,1],"#d32f2f",False)],label="Detections per frame",xlabel="frame (first to last)",ylabel="count"),
                        unsafe_allow_html=True)
                    st.download_button("Download detection video",data=ddata,file_name="detection_video.mp4",mime="video/mp4",key="vdet_dl")
            elif not vframes:
                st.caption("No frames could be read back from the processed clip.")
            else:
                fi=st.select_slider("Frame",options=list(range(1,len(vframes)+1)),key="video_frame_pick",
                    help="Evenly spaced frames from the processed part of the clip") if len(vframes)>1 else 1
                fa,fb=png_to_rgb(vframes[fi-1][0]),png_to_rgb(vframes[fi-1][1])
                if vview=="Applications":
                    sens,tiled=render_applications(fa,fb,"video")
                    if st.checkbox("Count vehicles and people in every sample frame",key="video_count"):
                        try:
                            n0=[len(detect_vehicles(png_to_rgb(o),sens,tiled)) for o,_ in vframes]
                            n1=[len(detect_vehicles(png_to_rgb(d),sens,tiled)) for _,d in vframes]
                            xs=np.linspace(0,1,len(n0))
                            st.markdown('<p class="ufc-cap">Vehicles and people found per sample frame: original (grey), dehazed (red)</p>'
                                +line_svg([(xs,n0,"#9a9a9a",False),(xs,n1,"#d32f2f",False)],label="Detections per frame",xlabel="sample frame (first to last)",ylabel="count"),
                                unsafe_allow_html=True)
                        except Exception as exc:
                            st.caption(f"Detection model not available here ({type(exc).__name__}).")
                else:
                    # Metrics over the whole processed clip, then the histogram, gamma and spectrum graphs of the picked frame
                    xs=np.linspace(0,1,len(vmetrics))
                    g1,g2=st.columns(2)
                    g1.markdown('<p class="ufc-cap">Contrast (std. dev.) per frame: original (grey), dehazed (red)</p>'
                        +line_svg([(xs,vmetrics[:,0],"#9a9a9a",False),(xs,vmetrics[:,1],"#d32f2f",False)],label="Contrast per frame",xlabel="frame (first to last)",ylabel="contrast"),unsafe_allow_html=True)
                    g2.markdown('<p class="ufc-cap">Edge pixels (%) per frame: original (grey), dehazed (red)</p>'
                        +line_svg([(xs,vmetrics[:,2],"#9a9a9a",False),(xs,vmetrics[:,3],"#d32f2f",False)],label="Edge pixels per frame",xlabel="frame (first to last)",ylabel="edge %"),unsafe_allow_html=True)
                    g1.markdown('<p class="ufc-cap">Mean brightness per frame: original (grey), dehazed (red)</p>'
                        +line_svg([(xs,vmetrics[:,4],"#9a9a9a",False),(xs,vmetrics[:,5],"#d32f2f",False)],label="Brightness per frame",xlabel="frame (first to last)",ylabel="mean 0-255"),unsafe_allow_html=True)
                    render_graphs(fa,fb)
            st.caption(f"{n_fr} frames, {spf:.2f} s per frame.")
            st.download_button("Download video",data=vdata,file_name="dehazed_video.mp4",mime="video/mp4")
    with st.container(border=True,key="method_card"):
        st.markdown('<div class="card-heading">Method</div>',unsafe_allow_html=True)
        st.markdown('<p class="ufc-eq">I = J·t + A·(1 − t),&nbsp; t = e<sup>−βd</sup></p>',unsafe_allow_html=True)
        m_l,m_r=st.columns(2,gap="large")
        m_l.markdown('<p class="ufc-sub">Approach</p>',unsafe_allow_html=True)
        m_l.markdown(
            "Fog follows the atmospheric scattering model: *I* is the observed image, *J* the clear "
            "scene, *A* the airlight and *t* the transmission at depth *d* (Koschmieder, 1924; "
            "Narasimhan and Nayar, 2002). The dark channel prior estimates *A* from the haziest pixels "
            "and *t* from local colour minima, refines *t* with a guided filter, then solves for *J* "
            "(He et al.). Smoothing, sharpening, histogram stretch and a Gaussian high-frequency-emphasis "
            "filter in the Fourier domain then work on luminance only."
        )
        m_l.markdown(
            "Blur is undone with Wiener deconvolution in the Fourier domain, for a Gaussian or linear "
            "motion blur (at any angle), with a configurable size or automatic estimation from the width of the strongest "
            "edges along eight directions (*Detect the blur automatically*). *Course modules only* runs just the three project modules: "
            "frequency-domain enhancement, histogram stretching, and adaptive smoothing and sharpening. "
            "The Edges view shows the Sobel gradient of both images on one scale, highlighting how much "
            "more of the buildings and roads shows up."
        )
        m_l.markdown(
            "When fog and blur are simulated on a clear photo, that photo is the reference for PSNR and SSIM, "
            "and for how well a Canny edge detector and a Hough line detector find the same edges and lines as "
            "in the clear photo. Depth for simulated fog is assumed to "
            "grow toward the top of the frame. Contrast, entropy and gradient rise when structure becomes "
            "visible, but also when an image is over-sharpened, so judge them together with the picture. "
            "The default settings come from a parameter search on the bundled test scenes; evaluate.py "
            "reproduces the numbers."
        )
        m_r.markdown('<p class="ufc-sub">References</p>',unsafe_allow_html=True)
        m_r.markdown(
            "1. He, K., Sun, J. and Tang, X. (2011). Single image haze removal using dark channel prior. "
            "*IEEE TPAMI*, 33(12), 2341–2353.\n"
            "2. He, K., Sun, J. and Tang, X. (2013). Guided image filtering. *IEEE TPAMI*, 35(6), 1397–1409.\n"
            "3. Narasimhan, S. G. and Nayar, S. K. (2002). Vision and the atmosphere. *IJCV*, 48(3), 233–254.\n"
            "4. Koschmieder, H. (1924). Theorie der horizontalen Sichtweite. *Beitr. Phys. freien Atm.*, 12, 33–53.\n"
            "5. Gonzalez, R. C. and Woods, R. E. (2018). *Digital Image Processing*, 4th ed. Pearson.\n"
            "6. Tomasi, C. and Manduchi, R. (1998). Bilateral filtering for gray and color images. *Proc. ICCV*, 839–846.\n"
            "7. Wang, Z. et al. (2004). Image quality assessment: from error visibility to structural similarity. "
            "*IEEE TIP*, 13(4), 600–612."
        )


# -----------------------------------------------------------------------------
# WEB WORKSPACE: evaluation and diagnostics (no separate terminal commands)
# -----------------------------------------------------------------------------
_, evaluation_col, _ = st.columns(PAGE_SPLIT, gap="large")
with evaluation_col:
    with st.container(key="evaluation_shell"):
        with st.expander("Evaluation & Project Health · optional project tools", expanded=False):
            st.caption(
                "Use these tools for project-report evidence or installation checks. "
                "After the app is running, evaluation and checks happen here—no separate scripts or terminal commands are needed.")
            with st.expander("1. Real-world fog validation", expanded=False):
                st.caption(
                    "Uses the bundled real fog/smog set plus uploaded field photographs. "
                    "Because these images have no aligned clear references, the app reports "
                    "no-reference indicators and runtime—not PSNR/SSIM or ground-truth detection accuracy.")
                field_files = st.file_uploader(
                    "Upload real foggy urban photos (optional)",
                    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                    accept_multiple_files=True, key="web_validation_uploads")
                eval_dim = st.select_slider("Maximum image long edge", options=[480, 640, 800, 1000, 1200], value=800,
                                            key="web_eval_dim")
                run_eval = st.button("Run real-world evaluation", type="primary", key="web_run_eval")
                if run_eval:
                    try:
                        import csv
                        import evaluate as E
                        from dehaze_core import P_BALANCED as _PB, P_MODULES as _PM, P_TRAFFIC as _PT
                        with st.spinner("Evaluating real fog images with four processing methods…"):
                            images = E.load_unpaired_real_fog(int(eval_dim))
                            for uploaded in (field_files or []):
                                try:
                                    arr = load_photo(uploaded)
                                    images[f"Uploaded: {uploaded.name}"] = resize_max(arr, int(eval_dim))
                                except Exception as exc:
                                    st.warning(f"Could not read {uploaded.name}: {type(exc).__name__}: {exc}")
                            if not images:
                                st.error("No readable real-fog images were found. Add photos above or check the bundled test-image folder.")
                            else:
                                methods = {
                                    "Fog removal only (DCP)": dict(_PB, hist=0, freq=0, smooth=0, sharpen=0, clahe=0,
                                        balance=0, restore=0, gamma=1.0, sat=1.0, far=0, deblur=0.0),
                                    "Traffic monitoring": _PT,
                                    "Balanced (haze-adaptive)": lambda im: __import__('dehaze_core').auto_strength_params(im, _PB)[0],
                                    "Course modules only": _PM,
                                }
                                rows, sheet = [], []
                                for image_name, source in images.items():
                                    sheet.append((source, f"{image_name} | original"))
                                    for method_name, setting in methods.items():
                                        started = time.perf_counter()
                                        params = setting(source) if callable(setting) else setting
                                        restored = dehaze_pipeline(source, params)
                                        elapsed = time.perf_counter() - started
                                        m = E.unpaired_metrics(source, restored)
                                        rows.append({"image": image_name, "method": method_name,
                                                     "processing_seconds": round(elapsed, 5), **m})
                                        if method_name == "Balanced (haze-adaptive)":
                                            sheet.append((restored, f"{image_name} | balanced"))
                                out_dir = APP_DIR / "results"
                                fig_dir = out_dir / "figures"
                                fig_dir.mkdir(parents=True, exist_ok=True)
                                E.write_csv(out_dir / "web_real_world_unpaired.csv", rows)
                                E.contact_sheet(sheet, fig_dir / "web_real_world_contact_sheet.jpg", tile_width=300)
                                review_rows = [{"image": name, "capture_date": "", "capture_time": "",
                                    "location_or_source": "", "weather_or_haze_description": "", "lighting": "",
                                    "scene_type": "", "visibility_score_1_5": "", "artifact_score_1_5": "",
                                    "color_naturalness_1_5": "", "reviewer_notes": ""} for name in images]
                                E.write_csv(out_dir / "web_human_review_template.csv", review_rows)
                                st.session_state.web_eval_rows = rows
                                st.session_state.web_eval_images = len(images)
                                st.session_state.web_eval_csv = (out_dir / "web_real_world_unpaired.csv").read_bytes()
                                st.session_state.web_eval_review = (out_dir / "web_human_review_template.csv").read_bytes()
                                st.session_state.web_eval_sheet = (fig_dir / "web_real_world_contact_sheet.jpg").read_bytes()
                                st.success(f"Completed: {len(images)} images × {len(methods)} methods = {len(rows)} evaluations.")
                    except Exception as exc:
                        st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
                if st.session_state.get("web_eval_rows"):
                    import pandas as pd
                    rows = st.session_state.web_eval_rows
                    st.subheader("Per-image evaluation results")
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    df = pd.DataFrame(rows)
                    numeric = [c for c in df.columns if c not in ("image", "method")]
                    summary = df.groupby("method", as_index=False)[numeric].mean(numeric_only=True)
                    st.subheader("Average results by method")
                    st.dataframe(summary, width="stretch", hide_index=True)
                    c1, c2, c3 = st.columns(3)
                    c1.download_button("Download metrics CSV", st.session_state.web_eval_csv,
                                       "real_world_metrics.csv", "text/csv", key="web_dl_metrics")
                    c2.download_button("Download human-review sheet", st.session_state.web_eval_review,
                                       "human_review_template.csv", "text/csv", key="web_dl_review")
                    c3.download_button("Download comparison sheet", st.session_state.web_eval_sheet,
                                       "real_world_contact_sheet.jpg", "image/jpeg", key="web_dl_sheet")
                    st.info("Interpret edge gain, gradient ratio, contrast and entropy as visibility indicators—not proof that the original scene was perfectly reconstructed. Complete the human-review sheet with independent reviewers.")
            with st.expander("2. Run built-in system checks", expanded=False):
                st.caption("These checks run inside the Streamlit app. Each result includes the test performed and the observed output. No temporary video files or separate test commands are required.")
                if st.button("Run system checks", key="web_run_checks"):
                    checks = []
                    def _check(label, fn):
                        try:
                            detail = fn()
                            checks.append({"Check": label, "Result": "PASS",
                                           "Details": str(detail) if detail else "Test completed successfully; all assertions passed."})
                        except Exception as exc:
                            checks.append({"Check": label, "Result": "FAIL",
                                           "Details": f"{type(exc).__name__}: {exc}"})
                    rng = np.random.default_rng(7)
                    sample = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
                    def _imports():
                        import dehaze_core as core
                        return f"Imported dehaze_core successfully; {len(core.STAGE_ORDER)} processing stages are registered: {', '.join(core.STAGE_ORDER)}."
                    def _shape_dtype():
                        result = dehaze_pipeline(sample, P_BALANCED)
                        assert result.shape == sample.shape, f"Expected shape {sample.shape}, got {result.shape}."
                        assert result.dtype == np.uint8, f"Expected uint8 output, got {result.dtype}."
                        assert np.isfinite(result).all(), "Output contains NaN or infinite values."
                        return f"Input/output shape: {sample.shape}; output dtype: {result.dtype}; pixel range: {int(result.min())}–{int(result.max())}; no non-finite values."
                    def _disabled_stages():
                        params = dict(P_BALANCED, use_dcp=False, hist=0, freq=0, smooth=0, sharpen=0,
                                      clahe=0, balance=0, restore=0, gamma=1.0, sat=1.0, far=0, deblur=0.0)
                        changed = [stage for stage in STAGE_ORDER if not np.array_equal(apply_stage(stage, sample, params), sample)]
                        assert not changed, f"Disabled stages unexpectedly changed pixels: {changed}."
                        return f"All {len(STAGE_ORDER)} pipeline stages preserved every pixel when disabled; output is identical to input."
                    def _identity_metrics():
                        score, peak = ssim(sample, sample), psnr(sample, sample)
                        assert abs(score-1.0) <= 1e-6, f"Expected SSIM 1.0, got {score:.8f}."
                        assert peak == 99.0, f"Expected the app's identical-image PSNR sentinel 99.0, got {peak}."
                        return f"Identical-image reference test: SSIM = {score:.6f}; PSNR sentinel = {peak:.1f} dB."
                    def _bright_safeguard():
                        white = np.full((32, 32, 3), 230, np.uint8)
                        recovered = __import__('dehaze_core').recover_scene(white, np.full((32,32), .25, np.float32),
                            np.array([.9,.9,.9], np.float32))
                        before, after = float(white.mean()), float(recovered.mean())
                        assert after >= before-1, f"Bright-region mean fell from {before:.1f} to {after:.1f}."
                        return f"Bright-region safeguard preserved a test patch: mean intensity {before:.1f} → {after:.1f} (allowed drop: at most 1 level)."
                    def _dataset_readable():
                        count = len(__import__('evaluate').load_unpaired_real_fog(320))
                        assert count >= 15, f"Expected at least 15 bundled images; found {count}."
                        return f"Loaded {count} bundled real-fog/smog photographs at a 320 px maximum long edge."
                    _check("Core processing imports", _imports)
                    _check("Pipeline preserves image shape and dtype", _shape_dtype)
                    _check("Disabled stages preserve original pixels", _disabled_stages)
                    _check("Identity SSIM and PSNR", _identity_metrics)
                    _check("Bright-region DCP safeguard", _bright_safeguard)
                    _check("Bundled real-fog dataset readable", _dataset_readable)
                    st.session_state.web_check_rows = checks
                if st.session_state.get("web_check_rows"):
                    import pandas as pd
                    checks_df = pd.DataFrame(st.session_state.web_check_rows)
                    st.dataframe(checks_df, width="stretch", hide_index=True,
                                 column_config={"Details": st.column_config.TextColumn("What was tested / observed", width="large")})
                    passed = sum(x["Result"] == "PASS" for x in st.session_state.web_check_rows)
                    total = len(st.session_state.web_check_rows)
                    if passed == total:
                        st.success(f"All {total} system checks passed. Expand the Details column to see the evidence for each check.")
                    else:
                        st.warning(f"{passed}/{total} checks passed. The Details column explains each failure.")
            st.caption("Tip: Optional during normal image processing. Open this section for project-report evidence or installation checks.")

