# Image-processing core: dark channel prior, Wiener deblur, bilateral smoothing, unsharp masking, histogram stretch, CLAHE, FFT filter, tone
import hashlib
import math
import random
from collections import OrderedDict
import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2,fftshift,ifft2,ifftshift
FREQ_CUTOFF=30 # frequency-domain cutoff radius, in frequency-plane pixels
# Default settings: P_BALANCED from a parameter search on scenes with a clear reference, P_MODULES = the three course modules only
P_BALANCED=dict(hist=20,freq=20,smooth=23,sharpen=37,omega=0.87,t0=0.20,patch=0,use_dcp=True,
    deblur=0.0,deblur_kind="Gaussian",deblur_k=0.01,gamma=1.25,far=25,sat=1.0,balance=50,cutoff=60,
    adaptive=True,clahe=15,clahe_tiles=8,restore=30)
P_MODULES=dict(P_BALANCED,use_dcp=False,hist=20,freq=20,smooth=20,sharpen=80,balance=0,
    adaptive=False,clahe=0,restore=0,gamma=1.0,far=0)
# P_TRAFFIC: fog removal plus histogram stretching only, since the other stages lowered vehicle detection recall
P_TRAFFIC=dict(P_BALANCED,hist=20,freq=0,smooth=0,sharpen=0,balance=0,clahe=0,restore=0,gamma=1.0,far=0)
# Maps slider strength (0-100) to the % of pixels clipped at each end of the histogram
def hist_clip_percent(strength):
    return (strength/100.0)*2.0 # 0 .. 2 % clipped from each end
# Maps strength to the high-frequency gain of the frequency-domain filter
def freq_gain(strength):
    return 1.0+(strength/100.0)*1.5 # gain applied to fine detail
# Maps strength to the bilateral filter's colour sigma (how big a step still counts as an edge)
def smoothing_tolerance(strength):
    return 10.0+0.6*strength
# Maps strength to the unsharp-mask amount
def edge_gain(strength):
    return (strength/100.0)*2.0
# RGB to YCrCb conversion; returns the full image and its luminance Y ( brightness channel) scaled to 0..255
def _to_ycc(img):
    ycc=cv2.cvtColor(img.astype(np.float32)/255.0,cv2.COLOR_RGB2YCrCb)
    return ycc,ycc[:,:,0]*255.0
# Writes a new luminance back into YCrCb and converts to 8-bit RGB
def _from_ycc(ycc,y):
    out=ycc.copy()
    out[:,:,0]=np.clip(y,0.0,255.0)/255.0
    rgb=cv2.cvtColor(out,cv2.COLOR_YCrCb2RGB)
    return np.clip(rgb*255.0+0.5,0,255).astype(np.uint8)
# Histogram stretching: linear contrast stretch of the luminance between percentile-clipped levels
def histogram_stretch(img,strength=55,return_range=False,fixed_range=None):
    percent=hist_clip_percent(strength)
    ycc,y=_to_ycc(img)
    # Clip points: percentiles of the luminance (or a fixed range), mapped to black and white
    lo,hi=fixed_range if fixed_range is not None else np.percentile(y,[percent,100-percent])
    if strength<=0 or hi-lo<1.0:
        out=img
    else:
        # Linear stretch (y - lo) * 255 / (hi - lo)
        out=_from_ycc(ycc,(y-lo)*255.0/(hi-lo))
    return (out,(float(lo),float(hi))) if return_range else out
# Adaptive smoothing: edge-preserving bilateral filter that smooths flat areas and noise but keeps edges
def adaptive_smoothing(img,strength=50):
    if strength<=0:
        return img
    sigma_color=smoothing_tolerance(strength)
    sigma_space=3.0+0.05*strength
    # Bilateral filter (Tomasi and Manduchi): Gaussian weights in space and in colour difference
    return cv2.bilateralFilter(img,7,sigma_color,sigma_space)
# Immerkaer's fast noise estimate from a 3x3 Laplacian-difference mask
def noise_sigma(y):
    k=np.array([[1,-2,1],[-2,4,-2],[1,-2,1]],dtype=np.float32)
    h,w=y.shape
    # Convolve with the mask, then sigma = sqrt(pi/2) * sum|r| / (6 * (W-2) * (H-2))
    r=cv2.filter2D(y.astype(np.float32),-1,k)[1:-1,1:-1]
    return float(np.sqrt(math.pi/2.0)*np.abs(r).sum()/(6.0*max(1,(h-2)*(w-2))))
# Sharpening internals shared by the filter and its weight map: clipped unsharp detail and the activity-based weight
def _sharpen_parts(y):
    # Unsharp-mask detail = y - Gaussian blur (sigma 1.5), clipped to +-40 to avoid halos
    detail=np.clip(y-cv2.GaussianBlur(y,(0,0),1.5),-40.0,40.0)
    # Sobel gradient magnitude smoothed by a Gaussian = local activity map
    gx,gy=cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)
    activity=cv2.GaussianBlur(np.hypot(gx,gy),(0,0),2.0)
    sigma=noise_sigma(y)
    floor=float(np.clip(0.45-0.05*sigma,0.15,0.45)) # share of the detail kept in flat areas
    knee=6.0+4.0*sigma # gradient level that counts as real structure
    # Weight: small share of the detail in flat areas, rising to 1 on real structure
    return detail,floor+(1.0-floor)*np.clip(activity/knee,0.0,1.0)
# Adaptive sharpening: clipped unsharp mask on luminance, weighted by a Sobel activity map and the noise level
def adaptive_sharpening(img,strength=50):
    amount=edge_gain(strength)
    if amount<=0:
        return img
    ycc,y=_to_ycc(img)
    detail,weight=_sharpen_parts(y)
    # Sharpened luminance = y + amount * weight * detail
    return _from_ycc(ycc,y+amount*weight*detail)
# CLAHE (contrast-limited adaptive histogram equalisation) on the CIELAB L channel, blended with the input
def clahe_enhance(img,strength=50,tiles=8):
    if strength<=0:
        return img
    clip=1.0+3.0*strength/100.0
    tiles=int(np.clip(tiles,2,32))
    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
    # CLAHE on L: equalise per tile with a clip limit, interpolating between tiles
    eq=cv2.createCLAHE(clipLimit=clip,tileGridSize=(tiles,tiles)).apply(lab[:,:,0])
    # Blend the CLAHE result with the original L so the slider acts as an amount
    mix=min(1.0,strength/60.0)
    lab[:,:,0]=np.clip(lab[:,:,0].astype(np.float32)*(1-mix)+eq.astype(np.float32)*mix+0.5,0,255).astype(np.uint8)
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
# Frequency-domain enhancement: FFT with a Gaussian high-frequency-emphasis filter on mirror-padded luminance
def frequency_enhancement(img,strength=55,cutoff=FREQ_CUTOFF):
    gain=freq_gain(strength)
    if gain<=1.0:
        return img
    ycc,y=_to_ycc(img)
    h,w=y.shape
    pad=max(0,min(48,h-1,w-1))
    # Mirror padding keeps the FFT's wrap-around from leaking detail across borders
    yp=np.pad(y,pad,mode="reflect") if pad else y
    rows,cols=yp.shape
    # 2-D FFT with the zero frequency shifted to the centre
    spectrum=fftshift(fft2(yp))
    v=np.arange(rows)-rows//2
    u=np.arange(cols)-cols//2
    d2=v[:,None]**2+u[None,:]**2
    # Gaussian high-pass H = 1 - exp(-D^2 / (2 * cutoff^2)): smooth, so no ringing
    high_pass=1.0-np.exp(-d2/(2.0*cutoff**2))
    response=1.0+(gain-1.0)*high_pass # 1.0 at DC, `gain` at fine detail
    # Multiply the spectrum by the emphasis response and take the inverse FFT
    out=np.real(ifft2(ifftshift(spectrum*response)))
    out=out[pad:pad+h,pad:pad+w]
    return _from_ycc(ycc,out)
# Guided image filter (He et al.): edge-aware smoothing of src that follows the edges of guide, built from box filters
def _guided_filter(guide,src,radius,eps):
    k=(2*radius+1,2*radius+1)
    # Box-filter mean over the (2r+1) x (2r+1) window
    def mean(x):
        return cv2.boxFilter(x,-1,k,borderType=cv2.BORDER_REFLECT)
    mean_g,mean_s=mean(guide),mean(src)
    # Local linear model src ~ a * guide + b with a = cov / (var + eps), b = mean_s - a * mean_g
    cov=mean(guide*src)-mean_g*mean_s
    var=mean(guide*guide)-mean_g*mean_g
    a=cov/(var+eps)
    b=mean_s-a*mean_g
    # Average the coefficients over windows, then apply them to the guide
    return mean(a)*guide+mean(b)
# Dark channel prior: airlight A from the haziest pixels, transmission t from a min filter refined by a guided filter
def estimate_transmission(img,omega=0.95,t0=0.15,patch=0,adaptive=False):
    I=img.astype(np.float32)/255.0
    h,w=I.shape[:2]
    size=(max(3,int(patch))|1) if patch else (max(3,round(max(h,w)/80))|1) # odd window; auto ~15 px at 1200 px
    # Dark channel: minimum over colour channels, then a minimum filter over the patch
    dark=ndimage.minimum_filter(I.min(axis=2),size=size)
    # Airlight A: from the haziest 0.1% of pixels (brightest dark channel), one pixel or the mean of the brightest fifth if adaptive
    n=max(1,int(0.001*dark.size))
    idx=np.argpartition(dark.ravel(),-n)[-n:]
    candidates=I.reshape(-1,3)[idx]
    if adaptive:
        order=np.argsort(candidates.sum(axis=1))
        A=np.maximum(candidates[order[-max(1,len(order)//5):]].mean(axis=0),0.1)
    else:
        A=np.maximum(candidates[np.argmax(candidates.sum(axis=1))],0.1)
    # Transmission t = 1 - omega * min filter of (I / A), each colour channel divided by its own airlight value
    t=1.0-omega*ndimage.minimum_filter((I/A).min(axis=2),size=size)
    # Guided filter refinement so t follows the photo's real edges (removes blocks and halos of the raw min-filter map)
    gray=cv2.cvtColor(I,cv2.COLOR_RGB2GRAY)
    radius=max(1,min(size*3,(min(h,w)-1)//2))
    t=_guided_filter(gray,t.astype(np.float32),radius,1e-3)
    if adaptive:
        # Adaptive floor t0: lower in dense fog (fuller removal), higher in light haze (bright areas kept)
        haze=float(np.clip(1.0-t.mean(),0.0,1.0))
        t0=float(np.clip(t0*np.clip(1.5-1.6*haze,0.6,1.4),0.04,0.6))
    return np.clip(t,t0,1.0),A
# Single-image haze removal (He et al.): estimate A and t, then recover the scene J
def dark_channel_prior(img,omega=0.95,t0=0.15,patch=0,adaptive=False,far=0):
    t,A=estimate_transmission(img,omega,t0,patch,adaptive)
    return far_detail(recover_scene(img,t,A),t,far)
# Far-detail boost: noise-gated local-contrast (unsharp) gain on luminance weighted by (1 - t)^1.5, so distant hazy structure gains detail while flat sky and near areas stay clean
def far_detail(img,t,strength):
    if strength<=0:
        return img
    ycc,y=_to_ycc(img)
    # Lightly denoised luminance (Gaussian sigma 1.2), so the boost works on structure and not on pixel noise
    ys=cv2.GaussianBlur(y,(0,0),1.2)
    # Detail = denoised luminance minus a wide Gaussian blur (sigma ~ 1/120 of the long edge), clipped to +-30 to limit halos
    detail=np.clip(ys-cv2.GaussianBlur(y,(0,0),max(2.0,max(y.shape)/120.0)),-30.0,30.0)
    # Noise gate: Sobel activity of the denoised luminance against the Immerkaer noise level, ~0 on flat noisy sky and 1 on real edges
    gx,gy=cv2.Sobel(ys,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(ys,cv2.CV_32F,0,1,ksize=3)
    gate=np.clip(cv2.GaussianBlur(np.hypot(gx,gy),(0,0),2.0)/(6.0+4.0*noise_sigma(y)),0.0,1.0)
    # Depth weight: 0 where the air is clear (t=1), rising to 1 in dense haze (t<=0.2), times the noise gate
    w=np.clip((1.0-t)/0.8,0.0,1.0)**1.5*gate
    return _from_ycc(ycc,y+(strength/100.0)*2.0*w*detail)
# Inverts the scattering model: J = (I - A) / t + A. A soft safeguard
# blends back some input in bright, nearly-achromatic regions where DCP's
# dark-pixel assumption is unreliable (e.g. white facades and headlights).
def recover_scene(img,t,A,protect_bright=True):
    I=img.astype(np.float32)/255.0
    J=(I-A)/np.maximum(t[:,:,None],1e-3)+A
    if protect_bright:
        hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV).astype(np.float32)
        luminance=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        saturation=hsv[:,:,1]/255.0
        # Smooth masks avoid visible hard boundaries. This is a conservative
        # prior, not semantic white-building detection; bright haze may remain.
        bright=np.clip((luminance-0.72)/0.22,0.0,1.0)
        bright=bright*bright*(3.0-2.0*bright)  # smoothstep
        neutral=np.clip((0.24-saturation)/0.18,0.0,1.0)
        protection=(0.60*bright*neutral)[:,:,None]
        J=J*(1.0-protection)+I*protection
    return np.clip(J*255.0+0.5,0,255).astype(np.uint8)
_DCP_CACHE=OrderedDict()
# dark_channel_prior behind a small LRU cache, since it is the slow stage
def cached_dark_channel_prior(img,omega=0.95,t0=0.15,patch=0,adaptive=False,far=0):
    # Cache key: blake2b hash of the pixels plus the settings
    key=(hashlib.blake2b(np.ascontiguousarray(img).tobytes(),digest_size=16).hexdigest(),
        img.shape,round(float(omega),4),round(float(t0),4),int(patch),bool(adaptive),round(float(far),2))
    if key in _DCP_CACHE:
        _DCP_CACHE.move_to_end(key)
        return _DCP_CACHE[key]
    out=dark_channel_prior(img,omega,t0,patch,adaptive,far)
    _DCP_CACHE[key]=out
    # LRU eviction: drop the oldest entries beyond 4
    while len(_DCP_CACHE)>4:
        _DCP_CACHE.popitem(last=False)
    return out
# Gray-world colour balance: per-channel gains that pull the channel means together
def colour_balance_gains(img,balance):
    means=(img.astype(np.float32)/255.0).reshape(-1,3).mean(axis=0)+1e-6
    return (means.mean()/means)**(balance/100.0)
# Colour restoration: mid-tone gray-world white balance plus an adaptive saturation boost
def colour_restore(img,strength):
    if strength<=0:
        return img
    f=img.astype(np.float32)/255.0
    lum=f@np.array([0.299,0.587,0.114],dtype=np.float32)
    # Mid-tone mask: bright haze and deep shadows are left out of the gray-world average
    mask=(lum>0.12)&(lum<0.88)
    means=(f[mask].mean(axis=0) if mask.sum()>100 else f.reshape(-1,3).mean(axis=0))+1e-6
    gains=np.clip((means.mean()/means)**(strength/100.0),0.8,1.25)
    f=np.clip(f*gains,0.0,1.0)
    mx,mn=f.max(axis=2),f.min(axis=2)
    mean_s=float(((mx-mn)/np.maximum(mx,1e-3)).mean())
    # Adaptive saturation boost: larger the more washed-out (low mean saturation) the image is
    sat=1.0+(strength/100.0)*float(np.clip((0.35-mean_s)/0.35,0.0,1.0))*0.6
    lum=(f@np.array([0.299,0.587,0.114],dtype=np.float32))[...,None]
    f=np.clip(lum+(f-lum)*sat,0.0,1.0)
    return np.clip(f*255.0+0.5,0,255).astype(np.uint8)
# Final tone step: colour restoration, gray-world balance, saturation about luminance, and gamma correction
def tone_adjust(img,gamma=1.0,saturation=1.0,balance=0,gains=None,restore=0):
    if gamma==1.0 and saturation==1.0 and not balance and not restore:
        return img
    if restore:
        img=colour_restore(img,restore)
    f=img.astype(np.float32)/255.0
    if balance:
        if gains is None: # `gains` lets a video reuse colour-balance gains smoothed over time
            gains=colour_balance_gains(img,balance)
        f=np.clip(f*gains,0.0,1.0)
    if saturation!=1.0:
        lum=(f@np.array([0.299,0.587,0.114],dtype=np.float32))[...,None]
        # Saturation: scale the colour difference from luminance
        f=np.clip(lum+(f-lum)*saturation,0.0,1.0)
    if gamma!=1.0:
        # Gamma on luminance only: RGB is scaled by Y^(1/g)/Y, so hue and saturation stay natural while brightness changes
        lum=(f@np.array([0.299,0.587,0.114],dtype=np.float32))[...,None]+1e-4
        f=np.clip(f*(lum**(1.0/gamma)/lum),0.0,1.0)
    return np.clip(f*255.0+0.5,0,255).astype(np.uint8)
# Colour-maps (INFERNO) the estimated transmission t to show how the dark channel prior judges the haze
def transmission_map_image(img,P):
    t,_=estimate_transmission(img,P["omega"],P["t0"],P["patch"],P.get("adaptive",False))
    g=np.clip((t-t.min())/max(float(t.max()-t.min()),1e-3)*255.0+0.5,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
STAGE_ORDER=["dcp","deblur","smooth","sharpen","hist","clahe","freq","tone"] # the order they run
# Linear motion-blur kernel: normalised anti-aliased line of length n at the given angle
def motion_kernel(n,angle=0.0):
    n=max(3,int(n)|1)
    if angle%180.0==0.0:
        kern=np.zeros((n,n),np.float32)
        kern[n//2,:]=1.0/n
        return kern
    big,c=n*8,(n*8)//2
    dx,dy=math.cos(math.radians(angle))*(n*4-4),-math.sin(math.radians(angle))*(n*4-4)
    canvas=np.zeros((big,big),np.float32)
    # Anti-aliased line drawn 8x larger, then downsampled with area averaging
    cv2.line(canvas,(round(c-dx),round(c-dy)),(round(c+dx),round(c+dy)),1.0,8,cv2.LINE_AA)
    kern=cv2.resize(canvas,(n,n),interpolation=cv2.INTER_AREA)
    return kern/kern.sum()
# Wiener deconvolution of the luminance in the Fourier domain for Gaussian or linear motion blur (k damps noise)
def wiener_deblur(img,amount,kind="Gaussian",k=0.01,angle=0.0):
    if amount<=0:
        return img
    ycc=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
    y=ycc[...,0].astype(np.float32)/255.0
    pad=int(4*amount*(1.5 if kind=="Motion" and angle%180.0 else 1.0))+8
    # Mirror-pad to limit FFT border artefacts
    yp=cv2.copyMakeBorder(y,pad,pad,pad,pad,cv2.BORDER_REFLECT_101)
    fy=np.fft.fftfreq(yp.shape[0])[:,None]
    fx=np.fft.fftfreq(yp.shape[1])[None,:]
    if kind=="Motion" and angle%180.0!=0.0:
        # Any-angle motion blur: transfer function from the FFT of the line kernel
        kern=motion_kernel(int(round(3*amount)),angle)
        psf=np.zeros(yp.shape,np.float32)
        kh,kw=kern.shape
        psf[:kh,:kw]=kern
        H=np.real(np.fft.fft2(np.roll(psf,(-(kh//2),-(kw//2)),axis=(0,1)))) # centred kernel -> real response
    # Horizontal motion blur: frequency response of an n-tap box filter
    elif kind=="Motion":
        n=max(3,int(round(3*amount))|1)
        with np.errstate(divide="ignore",invalid="ignore"):
            h1=np.where(fx==0,1.0,np.sin(np.pi*n*fx)/(n*np.sin(np.pi*fx)))
        H=np.broadcast_to(h1,yp.shape)
    else:
        # Gaussian blur transfer function H = exp(-2 pi^2 sigma^2 (fx^2 + fy^2))
        H=np.exp(-2.0*np.pi**2*amount**2*(fx**2+fy**2))
    # Wiener filter X = Y * H / (H^2 + K), K = noise damping
    X=np.fft.fft2(yp)*H/(H*H+k)
    x=np.real(np.fft.ifft2(X))[pad:-pad,pad:-pad]
    ycc[...,0]=np.clip(x*255.0+0.5,0,255).astype(np.uint8)
    return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB)
HAZE_SLOPE,HAZE_OFFSET,STRENGTH_MIN=2.1,0.35,0.25 # strength = clip(2.1 * haze - 0.35, 0.25, 1); fit in evaluate.py
# Haze level 0..1 = 1 - mean dark-channel transmission
def haze_level(img):
    t,_=estimate_transmission(img,0.95,0.05,0)
    return float(1.0-t.mean())
# Linear rule mapping the haze level to the share of the base strength applied
def strength_scale(haze):
    return float(np.clip(HAZE_SLOPE*haze-HAZE_OFFSET,STRENGTH_MIN,1.0))
# Copy of the settings with the enhancement strengths and omega scaled to the measured haze level
def auto_strength_params(img,P=None,haze=None):
    P=dict(P_BALANCED if P is None else P)
    h=haze_level(img) if haze is None else haze # `haze`: a level already measured (the app caches it)
    g=strength_scale(h)
    for k in ("hist","freq","smooth","sharpen","clahe","far"):
        P[k]=P.get(k,0)*g
    if P["use_dcp"]:
        P["omega"]=min(0.95,P["omega"]*(0.35+0.65*g))
    return P,h,g
BLUR_DIRECTIONS=8 # profile directions measured, every 22.5 degrees
BLUR_BASELINE_AXIS,BLUR_BASELINE_DIAG=0.65,0.72 # edge spread (px, std) of a sharp photo along / between the image axes
BLUR_MIN=0.7 # excess spread (px) below which an image is called sharp
BLUR_ANISOTROPY=0.32 # 1 - mean/peak of the excess variance: above it the blur is directional (motion), below it round
GAUSS_SCALE=1.08 # Calibration factors fitted on the 7 real photos only: raw estimates read about 8% (Gaussian) and 10% (motion) low
MOTION_SCALE=1.10
# Blur measurement: median spread (std) of the gradient profile across the strongest edges along one axis
def _edge_spread(L,axis,top=0.08,reach=30):
    if axis==0:
        L=L.T
    h,w=L.shape
    if w<2*reach+3 or h<8:
        return None
    g=np.zeros_like(L)
    # Central-difference gradient along the axis
    g[:,1:-1]=(L[:,2:]-L[:,:-2])*0.5
    a=np.abs(g)
    # Edge centres = local gradient maxima along the axis
    peak=(a>=np.roll(a,1,1))&(a>np.roll(a,-1,1))&(a>0)
    peak[:,:reach+1]=peak[:,w-reach-1:]=False
    if peak.sum()<100:
        return None
    # Keep only the strongest edges (top 8%)
    ys,xs=np.nonzero(peak&(a>=np.quantile(a[peak],1.0-top)))
    sgn,pk=np.sign(g[ys,xs]),a[ys,xs]
    m0,m1,m2=pk.copy(),np.zeros(len(xs)),np.zeros(len(xs))
    for d in (1,-1): # walk out of the edge while the gradient keeps its sign
        alive=np.ones(len(xs),bool)
        for k in range(1,reach):
            v=g[ys,xs+d*k]*sgn
            alive&=v>0.05*pk
            wk=np.where(alive,v,0.0)
            m0+=wk
            m1+=wk*d*k
            m2+=wk*k*k
    # Gradient-weighted mean and variance of the edge profile; spread = sqrt(variance), median over edges
    c=m1/m0
    return float(np.median(np.sqrt(np.maximum(m2/m0-c*c,0.0))))
# Edge spread along any direction, by rotating the image so that direction is horizontal
def _spread_along(L,angle):
    if angle%180.0==0.0:
        return _edge_spread(L,1)
    if angle%180.0==90.0:
        return _edge_spread(L,0)
    h,w=L.shape
    M=cv2.getRotationMatrix2D((w/2.0,h/2.0),-angle,1.0)
    return _edge_spread(cv2.warpAffine(L,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101),1)
# Blind blur estimate: edge spread in 8 directions minus a sharp baseline gives Gaussian or motion type, size and angle
def estimate_blur(img):
    L=_lum(img)
    n=BLUR_DIRECTIONS
    angles=[k*180.0/n for k in range(n)]
    sp=[_spread_along(L,a) for a in angles]
    out={"kind":"none","amount":0.0,"angle":0.0,"spreads":sp}
    if any(x is None for x in sp):
        return out
    base=np.array([BLUR_BASELINE_AXIS if a%90.0==0.0 else BLUR_BASELINE_DIAG for a in angles])
    v=np.maximum(np.array(sp)**2-base**2,0.0) # excess variance per direction
    k=int(v.argmax())
    vm,v0,vp=v[(k-1)%n],v[k],v[(k+1)%n]
    den=vm-2.0*v0+vp
    off=float(np.clip(0.5*(vm-vp)/den,-0.5,0.5)) if den<0 else 0.0 # parabola through the peak
    peak=v0-0.25*(vm-vp)*off if den<0 else v0
    if math.sqrt(max(peak,0.0))<BLUR_MIN:
        return out
    if 1.0-v.mean()/peak>=BLUR_ANISOTROPY: # directional: linear motion blur
        out.update(kind="Motion",angle=round(((k+off)*180.0/n)%180.0,1),
            amount=MOTION_SCALE*math.sqrt(12.0*peak)/3.0) # box of length L: variance L^2/12, L = 3 * amount
    else:
        out.update(kind="Gaussian",amount=GAUSS_SCALE*math.sqrt(v.mean()))
    out["amount"]=round(out["amount"],2)
    return out
# Copy of the settings with the Wiener deblur set from estimate_blur (off if the photo is sharp)
def auto_deblur_params(img,P):
    est=estimate_blur(img)
    if est["kind"] in ("Gaussian","Motion"):
        return dict(P,deblur=est["amount"],deblur_kind=est["kind"],deblur_angle=est["angle"]),est
    return dict(P,deblur=0.0),est
# Runs one named pipeline stage on an image
def apply_stage(name,img,P):
    if name=="dcp":
        return cached_dark_channel_prior(img,P["omega"],P["t0"],P["patch"],P.get("adaptive",False),P.get("far",0)) if P["use_dcp"] else img
    if name=="deblur":
        return wiener_deblur(img,P["deblur"],P["deblur_kind"],P["deblur_k"],P.get("deblur_angle",0.0))
    if name=="smooth":
        return adaptive_smoothing(img,P["smooth"])
    if name=="sharpen":
        return adaptive_sharpening(img,P["sharpen"])
    if name=="hist":
        return histogram_stretch(img,P["hist"])
    if name=="clahe":
        return clahe_enhance(img,P.get("clahe",0),P.get("clahe_tiles",8))
    if name=="freq":
        return frequency_enhancement(img,P["freq"],P["cutoff"])
    return tone_adjust(img,P["gamma"],P["sat"],P.get("balance",0),restore=P.get("restore",0))
# True if the stage's settings would change the image
def stage_active(name,P):
    return {"dcp":P["use_dcp"],"deblur":P["deblur"]>0,"smooth":P["smooth"]>0,
        "sharpen":P["sharpen"]>0,"hist":P["hist"]>0,"clahe":P.get("clahe",0)>0,"freq":P["freq"]>0,
        "tone":P["gamma"]!=1.0 or P["sat"]!=1.0 or bool(P.get("balance",0)) or bool(P.get("restore",0))}[name]
# Full pipeline: runs every stage in STAGE_ORDER
def dehaze_pipeline(img,P):
    for name in STAGE_ORDER:
        img=apply_stage(name,img,P)
    return img
FOG_AIRLIGHT=np.array([216,223,229],dtype=np.float32)
# Synthetic city street with its depth map (0 near, 1 far), so fog can be simulated with a known clear scene
def make_sample_scene(w=640,h=400,seed=11,lane_lines=False):
    k=w/512.0
    rnd=random.Random(seed)
    hz=int(h*0.56)
    img=np.zeros((h,w,3),np.uint8)
    depth=np.ones((h,w),np.float32)
    # Sky as a vertical colour gradient
    t=np.linspace(0,1,hz)[:,None,None]
    img[:hz]=(np.array([58,108,176])*(1-t)+np.array([211,227,242])*t).astype(np.uint8)
    img[hz:]=(60,63,67)
    depth[hz:]=np.linspace(0.74,0.04,h-hz)[:,None]
    # Draws a row of random buildings with lit windows at one depth
    def buildings(dv,col,wmin,wr,hmin,hr,x0,x1):
        col=np.array(col,np.float32)
        win=tuple(int(c) for c in (col*0.5+np.array([255,235,170])*0.5))
        bx=x0
        while bx<x1:
            bw,bh=(wmin+rnd.random()*wr)*k,(hmin+rnd.random()*hr)*k
            xa,xb,ya,yb=int(bx),int(bx+bw),int(hz-bh),hz+int(8*k)
            cv2.rectangle(img,(xa,ya),(xb,yb),tuple(int(c) for c in col),-1)
            cv2.rectangle(depth,(xa,ya),(xb,yb),float(dv),-1)
            for wy in range(ya+int(8*k),hz-int(6*k),int(12*k)):
                for wx in range(xa+int(5*k),xb-int(6*k),int(9*k)):
                    if rnd.random()>0.45:
                        cv2.rectangle(img,(wx,wy),(wx+int(4*k),wy+int(6*k)),win,-1)
            bx+=bw+(2+rnd.random()*4)*k
    buildings(0.90,(109,129,150),26,50,60,110,int(-10*k),w)
    buildings(0.65,(78,95,114),34,50,50,150,int(-10*k),w)
    buildings(0.42,(53,66,79),40,40,40,170,int(-10*k),w)
    # Road as a polygon converging on the vanishing point
    cv2.fillPoly(img,[np.array([[w*.46,hz],[w*.54,hz],[w*1.1,h],[-w*.1,h]],np.int32)],(37,39,43))
    if lane_lines:
        for f in (0.2,0.8): # solid lane lines converging on the vanishing point
            cv2.line(img,(int(w*(0.46+0.08*f)),hz),(int(-0.1*w+1.2*w*f),h),(225,225,225),
                max(1,int(2*k)),cv2.LINE_AA)
    # Centre-line dashes that grow toward the camera (perspective)
    for i in range(1,9):
        tt=i/9
        py,sw=hz+(h-hz)*tt*tt,(1+tt*10)*k
        cv2.rectangle(img,(int(w/2-sw/2),int(py)),(int(w/2+sw/2),int(py+(2+tt*14)*k)),
            (232,211,74),-1)
    buildings(0.12,(29,36,44),70,60,170,90,int(-20*k),int(120*k))
    buildings(0.12,(29,36,44),70,60,170,90,int(w-130*k),int(w+20*k))
    # Cars as a body box, a cabin box and two wheels
    for cx,cy,sc,col in [(200,hz/k+40,.6,(179,38,30)),(300,hz/k+62,.9,(216,216,216)),
        (110,hz/k+104,1.3,(43,95,168)),(330,hz/k+120,1.5,(224,162,26))]:
        x0,y0,s_=int(cx*k),int(cy*k),sc*k
        cv2.rectangle(img,(x0,y0),(int(x0+60*s_),int(y0+16*s_)),col,-1)
        cv2.rectangle(img,(int(x0+12*s_),int(y0-10*s_)),(int(x0+48*s_),int(y0+s_)),col,-1)
        for wx in (6,44):
            cv2.rectangle(img,(int(x0+wx*s_),int(y0+13*s_)),(int(x0+(wx+10)*s_),int(y0+20*s_)),(17,17,17),-1)
        cv2.rectangle(depth,(x0,int(y0-10*s_)),(int(x0+60*s_),int(y0+20*s_)),0.157,-1)
    return img,depth
# n synthetic streets with different layouts, each as (clear image, true depth map)
def make_urban_set(n=6,w=640,h=400):
    return [make_sample_scene(w,h,seed=11+i,lane_lines=True) for i in range(n)]
# Synthetic fixed-camera clip: the city plus cars driving across the road at different speeds
def make_traffic_clip(n=30,w=640,h=400,seed=11,cars=3):
    base,depth=make_sample_scene(w,h,seed=seed,lane_lines=True)
    rnd=random.Random(seed+100)
    specs=[]
    for i in range(cars):
        y=int(h*(0.72+0.09*i)) # lower on screen = nearer = bigger
        cw=int(w*(0.07+0.04*i))
        specs.append(dict(y=y,w=cw,h=int(cw*0.5),x0=rnd.uniform(0,w),v=(3.0+2.0*i)*(1 if i%2==0 else -1),
                color=[(180,40,40),(40,70,160),(200,200,205)][i%3]))
    frames=[]
    for f in range(n):
        im=base.copy()
        for c in specs:
            x=int((c["x0"]+c["v"]*f)%(w+c["w"]))-c["w"]//2
            y0=c["y"]-c["h"]
            cv2.rectangle(im,(x,y0),(x+c["w"],c["y"]),c["color"],-1)
            cv2.rectangle(im,(x+c["w"]//6,y0+c["h"]//8),(x+c["w"]*5//6,y0+c["h"]*5//12),(30,35,45),-1)
            for wx in (x+c["w"]//5,x+c["w"]*4//5):
                cv2.circle(im,(wx,c["y"]),max(2,c["h"]//6),(15,15,15),-1)
        frames.append(im)
    return frames,depth
# Scattering-model transmission t = exp(-beta * depth)
def true_transmission(depth,density):
    return np.exp(-float(density)*depth).astype(np.float32)
# Fog simulation with the atmospheric scattering model I = J*t + A*(1 - t), t = exp(-beta*d)
def simulate_fog(img,depth,density):
    h,w=img.shape[:2]
    # Photos have no depth map, so assume depth grows toward the top of the frame
    if depth is None:
        depth=np.repeat(np.linspace(0.95,0.15,h,dtype=np.float32)[:,None],w,axis=1)
    t=np.exp(-float(density)*depth)[...,None]
    return np.clip(img.astype(np.float32)*t+FOG_AIRLIGHT*(1-t)+0.5,0,255).astype(np.uint8)
# Blur simulation: Gaussian filter or linear motion kernel
def simulate_blur(img,amount,kind,angle=0.0):
    if amount<=0:
        return img
    if kind=="Motion":
        # Motion blur: convolve with the line kernel
        return cv2.filter2D(img,-1,motion_kernel(int(round(3*amount)),angle),borderType=cv2.BORDER_REFLECT_101)
    # Gaussian blur with std = amount
    return cv2.GaussianBlur(img,(0,0),float(amount))
# Sobel gradient magnitude of both images on one shared scale
def edge_maps(a,b):
    # Sobel gradient magnitude from the x and y derivatives
    def mag(img):
        L=_lum(img)
        return np.hypot(cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3))
    ma,mb=mag(a),mag(b)
    # Shared scale: the 99.5th percentile of both maps maps to 255
    top=max(float(np.percentile(np.concatenate([ma.ravel(),mb.ravel()]),99.5)),1e-6)
    return [np.clip(m/top*255.0,0,255).astype(np.uint8) for m in (ma,mb)]
# Luminance (grayscale) as float32
def _lum(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)
# No-reference fog-removal metrics (Hautiere et al.): visible-edge gain e, gradient ratio r, newly clipped share sigma
def visibility_metrics(orig,restored,contrast=0.05):
    lo,hi=_lum(orig),_lum(restored)
    # Visible edges: pixels whose 3x3 max - min (dilate - erode) spans at least the contrast threshold
    def visible(L):
        k=np.ones((3,3),np.uint8)
        return (cv2.dilate(L,k)-cv2.erode(L,k))>=contrast*255.0
    vo,vr=visible(lo),visible(hi)
    n0,nr=int(vo.sum()),int(vr.sum())
    # Sobel gradient magnitude
    grad= lambda L:np.hypot(cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3),cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3))
    go,gr=grad(lo),grad(hi)
    # Gradient ratio restored / original on the restored image's visible edges
    ratio=(gr[vr]+1e-3)/(go[vr]+1e-3)
    # Saturated (black or white) pixels
    clip= lambda L:(L<=1)|(L>=254)
    return {"e":(nr-n0)/max(n0,1),"r":float(np.exp(np.log(ratio).mean())) if nr else float("nan"),
        "sigma":float((clip(hi)&~clip(lo)).mean())}
# No-reference image statistics: contrast, entropy, mean gradient, edge share and histograms
def quality_metrics(img):
    L=_lum(img)
    # 256-bin histogram and its entropy
    hist=np.bincount(L.astype(np.uint8).ravel(),minlength=256).astype(np.float64)
    p=hist/hist.sum()
    p=p[p>0]
    # Forward-difference gradient magnitude
    mag=np.hypot(L[:-1,1:]-L[:-1,:-1],L[1:,:-1]-L[:-1,:-1])
    chans=[np.bincount(img[...,c].ravel(),minlength=256).astype(np.float64) for c in range(3)]
    return {"contrast":float(L.std()),"entropy":float(-(p*np.log2(p)).sum()),
        "gradient":float(mag.mean()),"edges":float(100.0*(mag>12).mean()),
        "hist":hist,"chans":chans}
# Peak signal-to-noise ratio from the mean squared error
def psnr(a,b):
    mse=float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2))
    return 99.0 if mse<1e-6 else 10.0*math.log10(255.0**2/mse)
# Structural similarity (SSIM) on luminance with an 11x11 Gaussian window
def ssim(a,b):
    x,y=_lum(a),_lum(b)
    # Stabilising constants of the SSIM formula
    c1,c2=(0.01*255)**2,(0.03*255)**2
    # Gaussian local mean
    def g(z):
        return cv2.GaussianBlur(z,(11,11),1.5)
    mx,my=g(x),g(y)
    sxx,syy,sxy=g(x*x)-mx*mx,g(y*y)-my*my,g(x*y)-mx*my
    # SSIM map = luminance term * contrast-structure term, averaged over the image
    s=((2*mx*my+c1)*(2*sxy+c2))/((mx*mx+my*my+c1)*(sxx+syy+c2))
    return float(s.mean())
CANNY_LOW,CANNY_HIGH=50,150 # the textbook fixed thresholds, so fog's low contrast shows up
# Canny edge detector after a 5x5 Gaussian blur, with fixed or median-based thresholds
def canny_edges(img,auto=False):
    gray=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),(5,5),1.4)
    if auto:
        m=float(np.median(gray))
        return cv2.Canny(gray,max(5,0.66*m),max(15,1.33*m))
    return cv2.Canny(gray,CANNY_LOW,CANNY_HIGH)
# Probabilistic Hough transform (HoughLinesP) for straight segments
def hough_lines(edges):
    cv2.setRNGSeed(0) # HoughLinesP is randomised; fix it so runs repeat
    found=cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=30,maxLineGap=8)
    return np.zeros((0,4),np.float32) if found is None else found.reshape(-1,4).astype(np.float32)
# Angle of each segment from horizontal, 0..180 degrees
def line_angles(lines):
    return np.degrees(np.arctan2(lines[:,3]-lines[:,1],lines[:,2]-lines[:,0]))%180.0
# Edge precision, recall and F1 against a reference edge map, matched within a pixel tolerance
def edge_f1(edges,ref_edges,tol=2):
    # Share of edge pixels of a that have an edge of b within tol
    def near(a,b): # share of edge pixels of a that have an edge of b within tol
        if not a.any():
            return 0.0
        # Distance transform: distance from every pixel to the nearest edge pixel of b
        dist=cv2.distanceTransform((b==0).astype(np.uint8),cv2.DIST_L2,3)
        return float((dist[a>0]<=tol).mean())
    pr,rc=near(edges,ref_edges) if ref_edges.any() else 0.0,near(ref_edges,edges)
    return pr,rc,(2*pr*rc/(pr+rc) if pr+rc else 0.0)
# Share of reference segments re-found by a segment with similar angle passing close to the reference midpoint
def line_recall(lines,ref_lines,tol=4.0,max_angle=5.0):
    if len(ref_lines)==0:
        return float("nan")
    if len(lines)==0:
        return 0.0
    mid=(ref_lines[:,:2]+ref_lines[:,2:])*0.5
    ang_d=np.abs(line_angles(ref_lines)[:,None]-line_angles(lines)[None,:])
    ang_d=np.minimum(ang_d,180.0-ang_d)
    p,q=lines[:,:2],lines[:,2:]
    d=q-p
    norm=np.maximum(np.hypot(d[:,0],d[:,1]),1e-6)
    # Perpendicular distance of the reference midpoint to the found line (cross product / length)
    off=np.abs((mid[:,None,0]-p[None,:,0])*d[None,:,1]
        -(mid[:,None,1]-p[None,:,1])*d[None,:,0])/norm[None,:]
    # Position of the midpoint along the found segment (dot product / length)
    along=((mid[:,None,0]-p[None,:,0])*d[None,:,0]+(mid[:,None,1]-p[None,:,1])*d[None,:,1])/norm[None,:]
    ok=(ang_d<=max_angle)&(off<=tol)&(along>=-tol)&(along<=norm[None,:]+tol)
    return float(ok.any(axis=1).mean())
# Downstream test: Canny and Hough counts (vertical, oblique lines), plus F1 and line recall against a clear reference
def downstream_metrics(img,ref=None):
    edges=canny_edges(img)
    lines=hough_lines(edges)
    ang=line_angles(lines)
    out={"canny_pct":float(100.0*(edges>0).mean()),"lines":len(lines),
        "lines_vertical":int((np.abs(ang-90.0)<=15.0).sum()),
        "lines_oblique":int(((np.minimum(ang,180.0-ang)>=20.0)&(np.minimum(ang,180.0-ang)<=70.0)).sum())}
    if ref is not None:
        ref_edges=canny_edges(ref)
        out["edge_f1"]=edge_f1(edges,ref_edges)[2]
        out["edge_f1_auto"]=edge_f1(canny_edges(img,True),canny_edges(ref,True))[2]
        out["line_recall"]=line_recall(lines,hough_lines(ref_edges))
    return out
# Colour-maps (INFERNO) a non-negative float map, scaled so vmax becomes white-yellow
def heat_image(m,vmax=None):
    vmax=float(vmax) if vmax else max(float(m.max()),1e-6)
    g=np.clip(m/vmax*255.0+0.5,0,255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)
# Log-magnitude spectrum of the luminance: Hann window against edge leakage, 2-D FFT, zero frequency centred
def _log_spectrum(img):
    y=_to_ycc(img)[1]
    h,w=y.shape
    win=np.outer(np.hanning(h),np.hanning(w)).astype(np.float32)
    return np.log1p(np.abs(fftshift(fft2((y-y.mean())*win))))
# Spectrum picture (log magnitude, INFERNO) and the scale used, so two images can share one scale
def spectrum_image(img,vmax=None):
    s=_log_spectrum(img)
    vmax=float(vmax) if vmax else float(np.percentile(s,99.9))
    return heat_image(s,vmax),vmax
# Radially averaged log spectrum: mean log magnitude per frequency ring, 0 = DC and 1 = Nyquist along the short side
def radial_spectrum(img,bins=64):
    s=_log_spectrum(img)
    h,w=s.shape
    r=np.hypot(np.arange(w)[None,:]-w//2,np.arange(h)[:,None]-h//2)/(min(h,w)/2.0)
    m=r<1.0
    idx=(r[m]*bins).astype(int)
    return (np.arange(bins)+0.5)/bins,np.bincount(idx,s[m],bins)/np.maximum(np.bincount(idx,minlength=bins),1)
# Response of the frequency-domain filter along the same radius axis as radial_spectrum: 1 + (gain - 1) * Gaussian high-pass
def freq_response_curve(strength,cutoff=FREQ_CUTOFF,shape=(400,640),bins=64):
    h,w=shape
    pad=max(0,min(48,h-1,w-1))
    short=min(h,w)+2*pad # the filter works on the mirror-padded image
    r=(np.arange(bins)+0.5)/bins
    d=r*short/2.0
    return r,1.0+(freq_gain(strength)-1.0)*(1.0-np.exp(-d**2/(2.0*cutoff**2)))
# Per-pixel sharpening gain (amount * weight) that the adaptive unsharp mask applies
def sharpen_weight_map(img,strength):
    return edge_gain(strength)*_sharpen_parts(_to_ycc(img)[1])[1]
# Mean absolute change per pixel made by the adaptive (bilateral) smoothing
def smoothing_change_map(img,strength):
    return np.abs(adaptive_smoothing(img,strength).astype(np.float32)-img.astype(np.float32)).mean(2)
# Global histogram equalisation of the luminance (YCrCb), the classic baseline
def hist_equalise(img):
    ycc=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
    ycc[:,:,0]=cv2.equalizeHist(ycc[:,:,0])
    return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB)
# Simple one-technique baselines that the full pipeline is compared with
def baseline_methods(img):
    dcp_only=dict(P_BALANCED,hist=0,freq=0,smooth=0,sharpen=0,clahe=0,restore=0,gamma=1.0,balance=0,far=0,sat=1.0)
    return {"Histogram equalisation":hist_equalise(img),"CLAHE only":clahe_enhance(img,60,8),
        "Gamma only (1.5)":tone_adjust(img,1.5),
        # Plain (non-adaptive) unsharp mask: 2 * image - Gaussian blur (sigma 1.5)
        "Plain unsharp mask":cv2.addWeighted(img,2.0,cv2.GaussianBlur(img,(0,0),1.5),-1.0,0),
        "Dark channel prior only":dehaze_pipeline(img,dcp_only)}
# Draws detection boxes on a copy of the image: red cars and trucks, orange motorcycles and people, with small filled class labels
def draw_boxes(img,dets,color=(255,64,64)):
    out=img.copy()
    t=max(1,round(max(out.shape[:2])/700))
    for x0,y0,x1,y1,sc,name in dets:
        col=(255,150,0) if name in ("motorcycle","bicycle","person") else color
        cv2.rectangle(out,(int(x0),int(y0)),(int(x1),int(y1)),col,t)
        label=f"{name} {sc:.2f}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
        ty=max(th+3,int(y0))
        # Filled label background keeps the text readable over a busy scene
        cv2.rectangle(out,(int(x0),ty-th-3),(int(x0)+tw+3,ty),col,-1)
        cv2.putText(out,label,(int(x0)+2,ty-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,cv2.LINE_AA)
    return out
# Construction-inspection overlay: dimmed image with Canny edges (red) and Hough line segments (green)
def edge_overlay(img):
    edges=canny_edges(img)
    out=(img.astype(np.float32)*0.55).astype(np.uint8)
    out[cv2.dilate(edges,np.ones((2,2),np.uint8))>0]=(255,64,64)
    t=max(1,round(max(out.shape[:2])/500))
    for x0,y0,x1,y1 in hough_lines(edges).astype(int):
        cv2.line(out,(x0,y0),(x1,y1),(64,255,96),t)
    return out
