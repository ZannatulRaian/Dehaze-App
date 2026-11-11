# Vehicle and pedestrian detection for the downstream test (evaluation only): YOLOv5n (COCO, ONNX) run with onnxruntime
import urllib.request
from pathlib import Path
import cv2
import numpy as np
MODEL_URL="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
MODEL_PATH=Path(__file__).parent/"models"/"yolov5n.onnx"
# COCO classes that matter for traffic monitoring: id -> name
CLASSES={0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck"}
VEHICLES={"bicycle","car","motorcycle","bus","truck"}
_session=None
# True if the model is present or can be downloaded
def available():
    try:
        _load()
        return True
    except Exception:
        return False
# Creates the ONNX session once, downloading yolov5n.onnx on first use
def _load():
    global _session
    if _session is None:
        import onnxruntime as ort
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        _session=ort.InferenceSession(str(MODEL_PATH),providers=["CPUExecutionProvider"])
    return _session
# YOLOv5n detection: letterbox to 640, forward pass, confidence filter, then non-maximum suppression
def detect(img,conf=0.25,iou=0.45,classes=None):
    sess=_load()
    h,w=img.shape[:2]
    s=640.0/max(h,w)
    nh,nw=round(h*s),round(w*s)
    # Letterbox: resize keeping the aspect ratio and pad to 640x640 with grey 114
    canvas=np.full((640,640,3),114,np.uint8)
    top,left=(640-nh)//2,(640-nw)//2
    canvas[top:top+nh,left:left+nw]=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
    x=canvas.astype(np.float32).transpose(2,0,1)[None]/255.0
    if "float16" in sess.get_inputs()[0].type: # the released yolov5n.onnx is half precision
        x=x.astype(np.float16)
    out=sess.run(None,{sess.get_inputs()[0].name:x})[0][0].astype(np.float32) # (25200, 85)
    keep=classes or CLASSES
    # Class score = objectness * class probability
    scores=out[:,4:5]*out[:,5:]
    cls=scores.argmax(1)
    best=scores[np.arange(len(out)),cls]
    sel=(best>=conf)&np.isin(cls,list(keep))
    if not sel.any():
        return []
    b,sc,cl=out[sel,:4],best[sel],cls[sel]
    # Centre/size boxes to corner boxes in original image coordinates
    xyxy=np.stack([(b[:,0]-b[:,2]/2-left)/s,(b[:,1]-b[:,3]/2-top)/s,
            (b[:,0]+b[:,2]/2-left)/s,(b[:,1]+b[:,3]/2-top)/s],1)
    wh=[[float(r[0]),float(r[1]),float(r[2]-r[0]),float(r[3]-r[1])] for r in xyxy]
    # Non-maximum suppression: drop boxes that overlap a better box by more than the IoU threshold
    idx=cv2.dnn.NMSBoxes(wh,sc.tolist(),conf,iou)
    return [(*np.clip(xyxy[i],0,[w,h,w,h]).tolist(),float(sc[i]),keep[int(cl[i])]) for i in np.array(idx).ravel()]
# Tiled detection: the full frame plus overlapping tiles of about 560 px, so small far-away vehicles and motorcycles are seen at a higher resolution; merged with NMS
def detect_tiled(img,conf=0.25,iou=0.45,classes=None,tile=560,overlap=0.3):
    h,w=img.shape[:2]
    nx,ny=-(-w//tile),-(-h//tile)
    found=detect(img,conf,iou,classes)
    if nx*ny>1:
        tw,th=min(w,int(w/nx*(1+overlap))),min(h,int(h/ny*(1+overlap)))
        for y0 in np.linspace(0,h-th,ny).astype(int):
            for x0 in np.linspace(0,w-tw,nx).astype(int):
                for b in detect(img[y0:y0+th,x0:x0+tw],conf,iou,classes):
                    # A box that ends on a tile edge inside the image is cut off, so skip it
                    if (b[0]<2 and x0>0) or (b[1]<2 and y0>0) or (b[2]>tw-2 and x0+tw<w) or (b[3]>th-2 and y0+th<h):
                        continue
                    found.append((b[0]+x0,b[1]+y0,b[2]+x0,b[3]+y0,b[4],b[5]))
    if len(found)<2:
        return found
    wh=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in found]
    # Non-maximum suppression over the union of all passes
    keep=np.array(cv2.dnn.NMSBoxes(wh,[b[4] for b in found],conf,iou)).ravel()
    return [found[i] for i in keep]
# Intersection over union of two boxes
def _iou(a,b):
    ix=max(0.0,min(a[2],b[2])-max(a[0],b[0]))
    iy=max(0.0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0
# Greedy matching of detections to reference boxes by class and IoU; returns recall, precision and matches
def match(boxes,ref_boxes,thr=0.5):
    used,hit=set(),0
    for r in ref_boxes:
        best,bi=0.0,None
        for i,b in enumerate(boxes):
            if i not in used and b[5]==r[5] and _iou(b,r)>max(best,thr-1e-9):
                best,bi=_iou(b,r),i
        if bi is not None:
            used.add(bi)
            hit+=1
    return (hit/len(ref_boxes) if ref_boxes else float("nan"),
        hit/len(boxes) if boxes else float("nan"),hit)
