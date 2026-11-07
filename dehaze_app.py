"""
Dehaze App — Dehazing Console
=============================
Streamlit console around the image-processing pipeline in `dehaze_core.py`:
upload a hazy photo, tune the filters, download the result.

Folder layout expected next to this file:
    dehaze_app.py
    assets/cars/*.png
    assets/clouds/*.png
    .streamlit/config.toml

Run with:
    streamlit run dehaze_app.py
"""

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
from PIL import Image, ImageOps, UnidentifiedImageError

from dehaze_video import process_video

from dehaze_core import (  # the image-processing code lives in dehaze_core.py
    P_BALANCED, P_MODULES, P_TRAFFIC, STAGE_ORDER, apply_stage, auto_strength_params, dehaze_pipeline,
    downstream_metrics, edge_maps, estimate_blur as _estimate_blur, haze_level as _haze_level, strength_scale, make_sample_scene, psnr, quality_metrics, simulate_blur as _simulate_blur,
    simulate_fog as _simulate_fog, ssim, stage_active,
)

# Optional extra for iPhone HEIC/HEIF photos (it also gives an older Pillow
# AVIF support):   pip install pillow-heif
# The app runs fine without it; such files then get a clear message instead.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

APP_DIR = Path(__file__).parent

# Streamlit-cached versions of the fog / blur simulations
simulate_fog = st.cache_data(show_spinner=False)(_simulate_fog)
simulate_blur = st.cache_data(show_spinner=False)(_simulate_blur)
estimate_blur = st.cache_data(show_spinner=False, max_entries=6)(_estimate_blur)
downstream = st.cache_data(show_spinner=False, max_entries=8)(downstream_metrics)   # Canny + Hough counts
haze_of = st.cache_data(show_spinner=False, max_entries=6)(_haze_level)
MAX_DIM = 1200  # images larger than this (on the long edge) are downscaled


# ============================================================
# CORE PIPELINE
# ============================================================
# Every stage below returns its input untouched when its slider is at 0, so
# Reset really does switch the three filters off. The Dark Channel Prior
# fog-removal step (step 1) is the app's base and always runs.
#
# Colour handling: the smoothing / sharpening / stretch / frequency stages work
# on LUMINANCE only (the Y of YCrCb). Treating R, G and B separately shifts hues
# and paints coloured noise, which is where the purple/blue blotches came from.





# ============================================================
# SAMPLE SCENE, FOG SIMULATION, MEASUREMENTS, TEST IMAGES, PRESETS
# ============================================================


def compare_composite(a, b, pos):
    """Original on the left of the divider, dehazed on the right."""
    h, w = a.shape[:2]
    x = int(round(w * pos / 100))
    out = b.copy()
    out[:, :x] = a[:, :x]
    if 0 < x < w:
        out[:, max(0, x - 1):min(w, x + 1)] = (211, 47, 47)
    return out


def hist_svg(series, w=520, h=150, label="Histogram"):
    """series: [(256-bin histogram, colour, dashed?)] drawn as overlaid lines."""
    m = max(max(s[0].max() for s in series), 1.0)
    lines = ""
    for hh, colour, dashed in series:
        pts = " ".join(f"{i * w / 255:.1f},{h - 4 - (v / m) * (h - 10):.1f}" for i, v in enumerate(hh))
        lines += (f'<polyline fill="none" stroke="{colour}" stroke-width="2" '
                  f'{"stroke-dasharray=\"6 4\" " if dashed else ""}points="{pts}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{label}" '
            f'style="background:#fafafa;border:1px solid rgba(0,0,0,.08);border-radius:.35rem">{lines}</svg>')


TEST_DIR = APP_DIR / "assets" / "test images"


@st.cache_data(show_spinner=False)
def list_test_images():
    """{label: (hazy_path, clear_reference_path_or_None)} for every photo in assets/test images."""
    found = {}
    for f in sorted(glob.glob(str(TEST_DIR / "**" / "*.*"), recursive=True)):
        pth = Path(f)
        if (pth.suffix.lower() not in (".jpg", ".jpeg", ".png") or "clear_reference" in pth.name
                or "_depth" in pth.name or pth.parent.name == "urban_real_clear"):
            continue
        ref = pth.with_suffix("").with_name(pth.stem.replace("_hazy", "_clear_reference")).with_suffix(pth.suffix)
        label = pth.stem.replace("_hazy", "").replace("_", " ").capitalize()
        found[label + (" (clear reference)" if ref.exists() else "")] = (pth, ref if ref.exists() else None)
    return found


@st.cache_data(show_spinner=False)
def list_clear_photos():
    """{label: path} of clear real street photos (python get_street_photos.py): fog is simulated on them."""
    return {"Street photo " + Path(f).stem.split("_")[0][6:] + " (clear; fog simulated)": Path(f)
            for f in sorted(glob.glob(str(TEST_DIR / "urban_real_clear" / "*.jpg")))}


def _read_rgb(path, size=None):
    with Image.open(path) as im:
        arr = np.array(ImageOps.exif_transpose(im).convert("RGB"))
    if size is not None:
        return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return resize_max(arr)


# Balanced (default) comes from a parameter search on the outdoor test scenes that have a clear
# reference (see evaluate.py and README); the numbers live in dehaze_core.P_BALANCED. The other
# presets are hand-set around it.
def _ui(P):
    """Core settings -> the names used by the app's widgets."""
    return dict(hist_strength=P["hist"], freq_strength=P["freq"], smooth_strength=P["smooth"],
                sharpen_strength=P["sharpen"], dcp_omega=P["omega"], dcp_t0=P["t0"], dcp_patch=P["patch"],
                freq_cutoff=P["cutoff"], tone_gamma=P["gamma"], tone_sat=P["sat"], tone_balance=P["balance"],
                deblur_amount=P["deblur"], deblur_kind=P["deblur_kind"], deblur_k=P["deblur_k"], deblur_auto=False,
                deblur_angle=0.0, strength_auto=False, use_dcp=P["use_dcp"], modules_only=False)


_BASE = _ui(P_BALANCED)
PRESETS = {
    "Balanced (default)": dict(_BASE),
    "Light haze": {**_BASE, "hist_strength": 15, "freq_strength": 15, "smooth_strength": 20,
                   "sharpen_strength": 25, "dcp_omega": 0.80, "dcp_t0": 0.25},
    "Dense fog": {**_BASE, "hist_strength": 35, "freq_strength": 30, "smooth_strength": 30,
                  "sharpen_strength": 45, "dcp_omega": 0.95, "dcp_t0": 0.12},
    "Maximum detail": {**_BASE, "hist_strength": 45, "freq_strength": 40, "smooth_strength": 35,
                       "sharpen_strength": 55, "dcp_omega": 0.95, "dcp_t0": 0.12},
    "Fog removal only": {**_BASE, "hist_strength": 0, "freq_strength": 0, "smooth_strength": 0, "sharpen_strength": 0},
    # for vehicle and pedestrian detection: the frequency, sharpening and colour-balance stages lowered detection recall
    "Traffic monitoring": {**_ui(P_TRAFFIC), "strength_auto": True},
    "Course modules only": {**_ui(P_MODULES), "use_dcp": True, "modules_only": True},
}


@st.cache_data(show_spinner=False, max_entries=6)
def run_pipeline(img, P):
    """(dehazed image, seconds it took)"""
    t = time.perf_counter()
    out = dehaze_pipeline(img, P)
    return out, time.perf_counter() - t


@st.cache_data(show_spinner=False, max_entries=3)
def run_stages(img, P):
    """The image after every stage, cumulatively: [(name, image, stage_is_active)]."""
    steps, r = [("Original", img, True)], img
    for name in STAGE_ORDER:
        r = apply_stage(name, r, P)
        steps.append((name, r, stage_active(name, P)))
    return steps


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        # max(1, ...) keeps a very thin image (say 1 x 5000 px) from rounding to 0
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def to_png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _readable_formats():
    """Every format Pillow can decode here, minus EPS. Reading an EPS means
    handing the file to Ghostscript (a PostScript interpreter): pointless for
    a photo tool and not something an upload box that takes any file should do."""
    Image.init()
    return [name for name in Image.OPEN if name != "EPS"]


def load_photo(file):
    """Open any image Pillow can read and return it as an 8-bit RGB array no
    bigger than MAX_DIM on its long edge.

    The uploader accepts every file type, so this has to cope with whatever
    real-world images throw at it: WebP, GIF, BMP, TIFF, AVIF, JPEG 2000, ...
    (plus HEIC with the optional pillow-heif), the camera's rotation flag,
    transparency (flattened onto white), 16-bit and float data, CMYK and
    grayscale files, and animated files (the first frame is used).
    Raises if the file isn't an image it can read — see unreadable_message().
    """
    img = Image.open(file, formats=_readable_formats())
    # Camera JPEGs can be 40+ megapixels. Let the decoder shrink them while it
    # reads (a no-op for every other format) — they are downscaled anyway.
    img.draft("RGB", (MAX_DIM, MAX_DIM))
    try:
        img = ImageOps.exif_transpose(img)  # honour the camera's rotation flag
    except Exception:
        pass  # broken EXIF block: show the picture as stored

    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        # 16-bit / 32-bit / float data. Pillow's own convert() would clip it
        # at 255 and give a white image, so scale it to 8 bits ourselves.
        data = np.asarray(img, dtype=np.float32)
        top = float(data.max()) if data.size else 0.0
        if img.mode == "F" and top <= 1.0:
            gain = 255.0               # float pixels in the 0..1 range
        elif top <= 255.0:
            gain = 1.0                 # already 8-bit values in a bigger container
        else:
            # "16-bit" files often hold 10-, 12- or 14-bit data: the brightest
            # value tells which full scale to use (65535 for true 16-bit data).
            full_scale = next((2 ** bits - 1 for bits in (10, 12, 14, 16)
                               if top <= 2 ** bits - 1), top)
            gain = 255.0 / full_scale
        gray = np.clip(data * gain, 0, 255).astype(np.uint8)
        arr = np.stack([gray] * 3, axis=-1)
    elif img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        # Transparent areas would turn black in a plain RGB conversion;
        # flatten onto white (the colour of fog) instead.
        rgba = img.convert("RGBA")
        page = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        arr = np.array(Image.alpha_composite(page, rgba).convert("RGB"))
    else:
        arr = np.array(img.convert("RGB"))
    return resize_max(arr)


_RAW_EXTENSIONS = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng",
                   ".raf", ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".erf", ".mrw", ".raw"}


def unreadable_message(name, exc):
    """A plain-language explanation for a file load_photo() could not open."""
    Image.init()  # fills Image.OPEN, which tells us if the optional codecs are installed
    ext = Path(name).suffix.lower()
    if isinstance(exc, Image.DecompressionBombError):
        return f"`{name}` is too large to open safely. Try a smaller copy of the photo."
    if ext in (".heic", ".heif", ".hif") and "HEIF" not in Image.OPEN:
        return (f"`{name}` is a HEIC/HEIF photo (the iPhone format), which needs one extra "
                "package: run `pip install pillow-heif`, restart the app and upload it again. "
                "Or save the photo as a JPG first.")
    if ext == ".avif" and "AVIF" not in Image.OPEN:
        return (f"`{name}` is an AVIF image, which needs a newer Pillow: run "
                "`pip install --upgrade pillow` (or `pip install pillow-heif`), restart the app "
                "and upload it again. Or save the image as a JPG first.")
    if ext in _RAW_EXTENSIONS:
        return (f"`{name}` is a camera RAW file, which this app can't develop. "
                "Export the photo as a JPG or TIFF first and upload that.")
    if isinstance(exc, UnidentifiedImageError):
        return (f"`{name}` doesn't look like an image this app can open. "
                "JPG, PNG, WebP, GIF, BMP and TIFF files all work.")
    reason = " ".join(str(exc).split())[:120]
    return (f"Couldn't read `{name}`" + (f" ({reason})" if reason else "") + ". The file may be "
            "damaged or incomplete — try saving it again as a JPG or PNG.")


# (label, Pillow's name for it) of the formats worth naming in the uploader's hint.
_HINT_FORMATS = [("JPG", "JPEG"), ("PNG", "PNG"), ("WebP", "WEBP"), ("GIF", "GIF"),
                 ("BMP", "BMP"), ("TIFF", "TIFF"), ("AVIF", "AVIF"), ("HEIC", "HEIF")]


def supported_formats_hint():
    """e.g. 'JPG, PNG, WebP, GIF, BMP, TIFF, AVIF and more' — only formats this
    install can really open (AVIF and HEIC depend on optional codecs)."""
    Image.init()
    return ", ".join(label for label, key in _HINT_FORMATS if key in Image.OPEN) + " and more"


# ============================================================
# PIXEL-ART HEADER (cars & clouds strip, animated left-to-right)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sprite_meta(kind):
    """Return [(base64_png, aspect_ratio), ...] for every sprite in assets/<kind>.

    The aspect ratio lets _build_sequence compute each row's *exact*
    rendered pixel width so the lane can size its own loop count safely
    (see MIN_TRACK_WIDTH below) instead of guessing.
    """
    folder = APP_DIR / "assets" / kind
    files = sorted(glob.glob(str(folder / "*.png")))
    out = []
    for f in files:
        with open(f, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        with Image.open(f) as im:
            w, h = im.size
        out.append((b64, w / h))
    return out


def _build_sequence(sprite_meta, count, height_range, gap_range, seed, top_range=None):
    """Build one randomized strip of <img> tags: random sprite, random
    height, random trailing gap (and optional vertical jitter) per image,
    so the pattern doesn't look mechanically repeated.

    Returns (html, row_width) in authored px — the caller needs the exact
    width to know how many times this row must repeat to safely outrun any
    viewport (see `_lane_html`).
    """
    rnd = random.Random(seed)
    n = len(sprite_meta)
    order = [rnd.randrange(n) for _ in range(count)]
    rnd.shuffle(order)

    parts = []
    total_w = 0.0
    for idx in order:
        b64, aspect = sprite_meta[idx]
        h = rnd.randint(*height_range)
        gap = rnd.randint(*gap_range)
        top = f"--t:{rnd.randint(*top_range)};" if top_range else ""
        # Sizes are authored in "px of a 200px-tall strip" and stored as bare
        # numbers; the iframe CSS multiplies them by --vu (1/200 of the strip's
        # real height), so the whole strip scales with the page (see render_header).
        parts.append(
            f'<img class="sprite" style="--h:{h};--g:{gap};{top}" '
            f'src="data:image/png;base64,{b64}">'
        )
        total_w += h * aspect + gap
    return "".join(parts), total_w


MIN_TRACK_WIDTH = 8000  # authored px a lane must cover before it's safe on any realistic viewport
# All sprite sizes and gaps in this file are authored for a 200px-tall strip;
# the page then scales the strip with the window (see render_header) — about
# 0.7x on a small laptop window, 2-3x on a large or zoomed-out screen — and
# every lane's real width scales with it. The strip runs full-bleed to the true
# browser edges (see inject_css), so a lane has to cover the *entire* window
# width: 8000 authored px is still ~5500 real px at the smallest scale, which
# is wider than even a 5K screen.


def _lane_html(sprite_meta, count, height_range, gap_range, seed, duration,
               top_range=None, opacity=None):
    """One infinitely-scrolling lane: a randomized row (see _build_sequence)
    repeated enough times that it's always wider than the browser window,
    then slid left by exactly one row-width on a seamless linear loop.

    Looping a *single fixed* copy-count (as an earlier version of this did)
    only stays seamless on viewports narrower than that fixed count implies;
    wider monitors run out of content before the far edge and show bare
    background. Sizing the repeat count from each row's own measured width
    instead makes every lane safe regardless of window size.
    """
    row_html, row_w = _build_sequence(sprite_meta, count, height_range, gap_range, seed, top_range)
    loops = max(3, math.ceil(MIN_TRACK_WIDTH / row_w) + 1)
    style = f"animation-duration:{duration:.2f}s;--shift:{-100 / loops:.4f}%;"
    if opacity is not None:
        style += f"opacity:{opacity};"
    layer_class = "cloud-layer" if top_range else "car-layer"
    return f'<div class="track {layer_class}" style="{style}">' + row_html * loops + "</div>"


def _embed_html(html, height):
    # st.iframe is the supported call on current Streamlit; st.components.v1.html
    # is deprecated (it prints a warning on every rerun), so it is only a fallback
    # for older installs that don't have st.iframe yet.
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        st.components.v1.html(html, height=height, scrolling=False)


def _spread_speeds(seed, lo, hi, n):
    """`n` random speeds stratified across [lo, hi] (one drawn from the
    middle of each equal sub-band) so a run of bad luck can't draw similar
    numbers and leave every lane moving at nearly the same pace — while
    still being a genuine random draw each time the seed changes, not
    fixed values."""
    rnd = random.Random(seed)
    band = (hi - lo) / n
    return [rnd.uniform(lo + (i + 0.15) * band, lo + (i + 0.85) * band) for i in range(n)]


# Column splits shared by the title and the cards. The title is rendered inside
# the same PAGE_SPLIT as the cards, so its left edge lines up with the cards'
# left edge at any window size or zoom level without hand-tuned offsets.
PAGE_SPLIT = [1, 16, 1]   # [left margin, content, right margin]
CARD_SPLIT = [5, 2]      # [Preview card, Enhancement Pipeline card]


def render_title():
    """The page title. It sits in Streamlit's own top bar (next to the Deploy
    button) rather than in a row of its own, which hands that height to the
    cards and so to the preview images — see inject_css."""
    _, title_col, _ = st.columns(PAGE_SPLIT, gap="large")
    title_col.markdown('<div class="ufc-title">Dehaze App</div>', unsafe_allow_html=True)


def render_header():
    cars = load_sprite_meta("cars")
    clouds = load_sprite_meta("clouds")

    if not cars or not clouds:
        # Graceful fallback if the assets folder isn't next to the script:
        # just skip the animated strip (the title is rendered separately).
        return

    # Two light car lanes — deliberately sparse (fewer cars than the
    # original had), with wide random gaps for real spacing. Each lane
    # samples many more cars than are ever on screen at once (count is
    # high relative to how few are visible) purely so the repeating tile
    # cycles through a long, varied sequence before it's ever seen to
    # repeat — with only 1-2 cars per lane the same pair of colors would
    # loop right in front of you. Gap range/density is unchanged, so this
    # doesn't add visible traffic, just variety. Wide gaps mean an edge is
    # more often briefly bare, so both lanes run fast enough that any such
    # gap clears in a couple of seconds rather than sitting there.
    # The cars are ~18% smaller than in the first version (height 54-58 ->
    # 44-48); the gaps grew by the same amount, so each lane's total length
    # — and with it the number of cars on screen and how fast they move — is
    # unchanged. Cars stand on the strip's bottom edge (.car-layer's
    # padding-bottom), i.e. they span roughly 74-122 of the 200 units.
    car_speeds = _spread_speeds(seed=101, lo=7, hi=11, n=2)
    car_lane_specs = [(6, 13), (5, 31)]  # (cars in lane, seed)
    car_layers_html = "".join(
        _lane_html(cars, count=count, height_range=(44, 48), gap_range=(160, 270),
                   seed=seed, duration=duration)
        for (count, seed), duration in zip(car_lane_specs, car_speeds)
    )

    # Two cloud depth-layers (mid/far) — the biggest near layer (70-98 units
    # tall) was dropped: those clouds dominated the strip. The clouds start
    # right at the strip's top edge (the header's end line) and hang low
    # enough that their bottoms still overlap the cars (cloud bottoms land
    # around 51-101, the cars span ~74-122) so the fog reads as drifting
    # behind/around the cars. Clouds are emitted before cars in
    # the HTML, so cars paint on top of any cloud they overlap — that is what
    # makes the fog read as drifting behind/around the cars.
    cloud_layers_html = (
        _lane_html(clouds, count=9, height_range=(51, 75), gap_range=(31, 71),
                   seed=51, duration=61, top_range=(0, 24), opacity=0.80)
        + _lane_html(clouds, count=9, height_range=(37, 53), gap_range=(23, 55),
                     seed=83, duration=79, top_range=(20, 46), opacity=0.62)
    )

    html = f"""
    <div class="ufc-strip">
      {cloud_layers_html}
      {car_layers_html}
    </div>
    <style>
      * {{ box-sizing:border-box; }}
      html, body {{ margin:0; background:#f5f5f5; overflow:hidden; }}
      /* The page scales with the window (see inject_css) and sets this
         iframe's height in rem, so the strip's own height is the one thing
         that tracks that scale. Every sprite size, gap and offset below is
         authored in "px of a 200px-tall strip" (a bare number in the inline
         style) and multiplied by --vu, 1/200 of the real strip height — so
         cars and clouds grow and shrink in step with the cards underneath
         instead of staying a fixed size on a very large or zoomed-out
         screen. */
      :root {{ --vu: 0.5vh; }}
      .ufc-strip {{
        position:relative; height:100vh; overflow:hidden;
        border-top:1px solid rgba(0,0,0,.07);
      }}
      /* Each lane is N copies of one randomized row, slid left by exactly
         one row-width (--shift, a per-lane % set inline by _lane_html) on
         a seamless linear loop — copy N+1 always sits exactly where copy N
         started, so the wrap-around is invisible. --shift is computed from
         each row's own measured width, not a fixed guess, which is what
         keeps this gap-free on any window size (see MIN_TRACK_WIDTH). */
      .track {{ position:absolute; left:0; top:0; height:100%; display:flex;
        animation-name: scrollLTR; animation-timing-function: linear;
        animation-iteration-count: infinite; will-change: transform; }}
      .cloud-layer {{ align-items:flex-start; padding-top:0; }}
      .car-layer {{ align-items:flex-end; padding-bottom:calc(78 * var(--vu)); }}
      img.sprite {{
        image-rendering:pixelated; display:block; flex-shrink:0;
        height:calc(var(--h) * var(--vu));
        margin-right:calc(var(--g) * var(--vu));
        margin-top:calc(var(--t, 0) * var(--vu));
      }}
      @keyframes scrollLTR {{
        from {{ transform:translateX(var(--shift)); }}
        to   {{ transform:translateX(0%); }}
      }}
    </style>
    """
    _embed_html(html, height=200)


# ============================================================
# PAGE-WIDE CSS (fonts + card typography, no widget-overlay hacks)
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        /* ------------------------------------------------------------
           ONE SCALE FOR THE WHOLE PAGE
           Streamlit sizes almost everything (padding, buttons, sliders,
           toolbar, gaps) in rem, and everything below is written in rem
           too, so setting the root font size once scales the entire UI
           together. It follows the window — whichever of width/height
           is the tighter fit — so the page looks the same whether it is
           on a laptop, a big monitor, or a browser that is zoomed out,
           instead of being a fixed-pixel layout that turns tiny on big
           screens. ~17.6px at ~1820x940, floored so text stays legible on
           small windows and capped on huge ones.
           ------------------------------------------------------------ */
        html {
            font-size: clamp(11px, min(0.968vw, 1.84vh), 48px) !important;

            /* Height of the two cards: whatever is left of the window under
               the top bar, the strip and the gap below it (3.75 + 7.5 + 1.2
               = 12.45rem), minus the page's bottom padding (1.2rem) and a
               0.15rem allowance for rounding — so the page ends just inside
               the bottom of the window and never scrolls — but no more than
               40rem on a very tall window. The preview images are sized from
               this (see the preview rules below). */
            --ufc-card-h: min(calc(100vh - 13.8rem), 40rem);
        }

        [data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]) {
            font-family:'IBM Plex Sans', sans-serif;
        }

        /* Page frame. The title lives in Streamlit's own 3.75rem top bar, so
           the content starts right at the top (no top padding) and the title
           block is exactly as tall as the bar. The bottom padding is kept
           small so the whole app fits one screen. The side padding is
           Streamlit's own 5rem; max-width only matters on very wide windows,
           where it stops the cards stretching into long thin bars. */
        [data-testid="stMain"] { overflow-x: hidden; }
        [data-testid="stMainBlockContainer"] {
            padding: 0 2rem 1.2rem 2rem;
            max-width: 112rem;
            margin-left: auto;
            margin-right: auto;
        }
        /* Vertical spacing between the title, strip and cards is set
           explicitly below instead of by Streamlit's default 1rem gap
           between every element. */
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0; }

        /* The blanket IBM Plex rule above has specificity (0,2,0), which beats
           a lone class selector, so every rule that wants a different face
           (display headings, monospace numbers) is written with the same
           prefix, further down, so it wins on order. */
        [data-testid="stAppViewContainer"] .ufc-title,
        [data-testid="stAppViewContainer"] .pipeline-heading,
        [data-testid="stAppViewContainer"] .card-heading {
            font-family:'Big Shoulders Display', sans-serif;
        }
        [data-testid="stAppViewContainer"] .stage-num,
        [data-testid="stAppViewContainer"] .stage-value {
            font-family:'IBM Plex Mono', monospace;
        }

        /* ---------------- title ---------------- */
        /* 2.3rem of text + .725rem above and below = 3.75rem, the height of
           Streamlit's top bar, so the title is centred in that bar on the
           same line as the Deploy button. The bar is opaque and paints over
           the page, so the title's container is lifted just above it. */
        .ufc-title {
            font-weight:800;
            font-size:2.3rem; line-height:1; text-transform:uppercase;
            letter-spacing:.03em; color:#111; margin:.725rem 0;
        }
        [data-testid="stMarkdownContainer"]:has(> .ufc-title) {
            position: relative;
            z-index: 999991;
        }
        /* st.markdown wraps text in a container with a -1rem bottom margin
           (it expects the paragraph's own bottom margin to cancel it). The
           title and the readouts below have no such margin, so cancel it. */
        [data-testid="stMarkdownContainer"]:has(> .ufc-title),
        .st-key-pipeline_card [data-testid="stMarkdownContainer"] {
            margin-bottom: 0 !important;
        }

        /* ---------------- animated strip ----------------
           Break the header iframe out of Streamlit's centered content
           column so the cars/clouds strip runs edge-to-edge of the actual
           browser window. !important is needed because Streamlit sets an
           explicit width/height on these elements. Its height is in rem so
           it scales with everything else (the sprites inside scale with
           this height, see render_header). */
        div[data-testid="stElementContainer"]:has(iframe[data-testid="stIFrame"]) {
            width: 100vw !important;
            max-width: 100vw !important;
            /* Streamlit gives this wrapper `flex: 0 0 200px`, and flex-basis
               wins over height — so without resetting it the wrapper could
               never be shorter than 200px on a small window. */
            flex: 0 0 auto !important;
            height: 7.5rem !important;
            position: relative;
            left: 50%;
            right: 50%;
            margin-left: -50vw !important;
            margin-right: -50vw !important;
            margin-bottom: 1.2rem;
        }
        iframe[data-testid="stIFrame"] {
            height: 7.5rem !important;
            display: block;
        }

        /* ---------------- cards ---------------- */
        /* Both cards always have the same, window-filling height (see
           --ufc-card-h), whether or not a photo is loaded, so nothing jumps
           when one is. position:relative anchors the "Processing…" pill. */
        .st-key-upload_card, .st-key-pipeline_card {
            padding: 1.5rem 1.75rem;
            min-height: var(--ufc-card-h);
            position: relative;
        }
        /* Buttons scale with the text instead of staying at Streamlit's
           fixed small size. (Not the little fullscreen button that appears
           on an image when it is hovered — that one is meant to stay small.) */
        .st-key-upload_card button:not([data-testid="stBaseButton-elementToolbar"]),
        .st-key-pipeline_card button:not([data-testid="stBaseButton-elementToolbar"]) {
            font-size: 1.05rem;
            min-height: 2.8rem;
        }

        /* Equal-height cards. Streamlit already stretches both columns to
           the taller one, but the wrapper around each card doesn't grow to
           fill its column, so the shorter card just stops early. Letting
           that wrapper grow makes the card fill the column; the upload
           area then grows to fill its card, so the dropzone simply gets
           roomier instead of leaving dead space under it. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-upload_card),
        [data-testid="stLayoutWrapper"]:has(> .st-key-pipeline_card) {
            flex: 1 1 auto;
        }
        .st-key-upload_card > [data-testid="stElementContainer"]:has([data-testid="stFileUploader"]) {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }
        .st-key-upload_card [data-testid="stFileUploader"] {
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
        }

        /* ---------------- upload dropzone ----------------
           One big drop target with its contents centered: an upload icon,
           the prompt, the red Upload button and the size hint. The icon and
           prompt are drawn with ::before / ::after (the real label is
           hidden in the uploader call and kept as the dropzone's aria-label)
           and `order` slots them around the button that Streamlit renders. */
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] {
            flex: 1 1 auto;
            min-height: 15rem;
            padding: 1.5rem 1.75rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: .85rem;
            border: 2px dashed rgba(0,0,0,.18);
            border-radius: .9rem;
            background: #fafafa;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"]::before {
            content: "";
            order: 1;
            width: 3.6rem;
            height: 3.6rem;
            opacity: .6;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23666' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 15.5V4.5'/%3E%3Cpath d='M7.5 9 12 4.5 16.5 9'/%3E%3Cpath d='M4.5 14.5v3a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3'/%3E%3C/svg%3E") center / contain no-repeat;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"]::after {
            content: "Drop a foggy photo here";
            order: 2;
            font-size: 1.45rem;
            font-weight: 600;
            line-height: 1.2;
            color: #111;
            margin-bottom: .4rem;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] > span { order: 3; }
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] {
            order: 4;
            flex: 0 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] span,
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"] small {
            font-size: .92rem;
        }
        /* The uploader no longer filters by file type, so Streamlit's own hint
           shows only the size limit; this adds the formats this install opens. */
        .st-key-upload_card [data-testid="stFileUploaderDropzoneInstructions"]::after {
            content: "__FORMAT_HINT__";
            font-size: .92rem;
            color: rgba(49, 51, 63, .6);
            margin-top: .15rem;
        }
        /* "can't open this file" message: the file name is set in code style,
           which Streamlit shrinks to 75% — keep it in step with the sentence. */
        .st-key-upload_card [data-testid="stAlert"] code {
            font-size: .92em;
        }
        /* Red, chunkier "Upload" button inside the dropzone */
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
            background-color: #d32f2f;
            border-color: #d32f2f;
            color: #ffffff;
            padding: .7rem 2.1rem;
            font-size: 1.1rem;
            border-radius: .4rem;
        }
        .st-key-upload_card [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #b71c1c;
            border-color: #b71c1c;
            color: #ffffff;
        }

        /* ---------------- preview images ----------------
           After an upload the pair is as big as the card allows, whatever
           the photo's shape. Python publishes the photo's width/height ratio
           as --pv-ar (see the app section). --pv-maxh is the tallest a
           picture can be before the card (--ufc-card-h) runs out of room:
           card height minus its padding (3rem), the heading row (2.8rem),
           the gap under that (1rem), the caption (1.5rem) and the card's
           border plus some slack (0.2rem). The block is then sized to
           exactly two pictures of that height plus the 1rem between them,
           or to the card's inner width if that is smaller. The two columns
           inside it, and the images filling them, therefore fit their
           pictures exactly: no letterboxing, no max-height clipping, and
           Streamlit's own hover-fullscreen button stays at the corner of
           the picture itself. */
        .st-key-upload_card {
            --pv-maxh: max(12rem, calc(var(--ufc-card-h) - 8.5rem));
        }
        .st-key-preview_images {
            flex: 0 1 auto !important;
            width: min(100%, calc(2 * var(--pv-maxh) * var(--pv-ar, 1.25) + 1rem)) !important;
            margin-left: auto;
            margin-right: auto;
        }
        .st-key-preview_images [data-testid="stImage"] img {
            width: 100% !important;
            height: auto;
        }
        .st-key-preview_images [data-testid="stImageCaption"] {
            margin-top: .3rem;
            padding: 0;
        }
        .st-key-preview_images [data-testid="stImageCaption"] p {
            font-size: .9rem;
            line-height: 1.3;
        }
        /* Any height the pair doesn't need (a wide, short photo) is split
           above and below it instead of leaving a blank strip at the bottom. */
        [data-testid="stLayoutWrapper"]:has(> .st-key-preview_images) {
            margin-top: auto;
            margin-bottom: auto;
        }
        /* st.spinner adds a row to the card (plus a gap) while it works,
           which would make the card — and the page — jump. Float it instead,
           in the free space of the heading row so it never covers a photo. */
        .st-key-upload_card > [data-testid="stElementContainer"]:has(> [data-testid="stSpinner"]) {
            position: absolute;
            top: 2.1rem;
            left: 10.5rem;
            z-index: 3;
            width: auto;
            padding: .3rem .9rem;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,.1);
            border-radius: 999px;
            box-shadow: 0 .1rem .5rem rgba(0,0,0,.12);
        }

        /* ---------------- pipeline card ---------------- */
        /* The card is as tall as the window allows, so any height beyond what
           the controls need is shared out between the sections (the margins
           below are just the minimum between them). */
        .st-key-pipeline_card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 0;
        }
        /* Each stage (label row + slider + readout) sits in its own keyed
           container so its internal spacing is exact and the space BETWEEN
           stages is clearly bigger than the space inside one — that is
           what makes each readout read as belonging to its own slider. */
        .st-key-stage_hist, .st-key-stage_freq, .st-key-stage_smooth, .st-key-stage_sharpen {
            gap: .15rem;
            margin-top: .9rem;
        }
        .st-key-pipeline_actions {
            gap: .7rem;
            margin-top: 1.2rem;
        }

        .pipeline-heading {
            text-transform:uppercase;
            font-weight:800; font-size:2rem; line-height:1.1; letter-spacing:.02em; color:#111;
            border-bottom:1px solid rgba(0,0,0,.08); padding-bottom:.8rem;
        }
        .card-heading {
            text-transform:uppercase;
            font-weight:800; font-size:2rem; line-height:1.1; letter-spacing:.02em; color:#111;
        }
        .stage-row {
            display:flex; justify-content:space-between; align-items:baseline;
            line-height:1.3;
        }
        .stage-num {
            color:#d32f2f; font-size:.85rem;
            margin-right:.55rem;
        }
        .stage-label { font-weight:600; font-size:1.08rem; color:#111; }
        .stage-value {
            color:#d32f2f; font-weight:600; font-size:1.08rem;
        }
        /* Specific enough (p + container) to beat Streamlit's own
           `.st-emotion-cache-xxxx p { font-size:inherit; margin:0 }` rule,
           which otherwise silently overrides a plain class selector. */
        [data-testid="stMarkdownContainer"] p.stage-readout {
            font-family:'IBM Plex Mono', monospace; font-size:.84rem; line-height:1.3;
            color:#777777; margin:0;
        }

        /* Sliders: a compact track with a comfortably sized thumb. The big
           (20px) padding Streamlit puts above and below the track is only
           there to hold the value bubble and the min/max ticks; neither is
           shown (the live value is in each stage's row), so it is trimmed. */
        .st-key-pipeline_card [data-testid="stSlider"] > div > div {
            padding: .75rem 0;
        }
        .st-key-pipeline_card [data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) {
            width: 1.1rem !important;
            height: 1.1rem !important;
        }
        [data-testid="stSliderTickBar"] { display: none; }
        /* The little value bubble above the thumb is only shown while the
           slider is hovered or being dragged. At rest it would sit on top
           of the label row above it (the live value is always shown at the
           right end of that row anyway). :focus-visible is deliberately not
           used — after a mouse drag the browser keeps the hidden range input
           "focus-visible", which would leave the bubble stuck on screen. */
        [data-testid="stSliderThumbValue"] {
            opacity: 0;
            pointer-events: none;
            transition: opacity .12s ease;
            font-size: .88rem;
            font-weight: 600;
            background: #ffffff;
            border: 1px solid rgba(0,0,0,.12);
            border-radius: .35rem;
            padding: .05rem .4rem;
            box-shadow: 0 .1rem .45rem rgba(0,0,0,.14);
            z-index: 5;
        }
        [data-testid="stSlider"]:hover [data-testid="stSliderThumbValue"],
        [data-testid="stSlider"]:active [data-testid="stSliderThumbValue"] {
            opacity: 1;
        }
        </style>
        """.replace("__FORMAT_HINT__", supported_formats_hint()),
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================

st.set_page_config(page_title="Dehaze App", page_icon="🌫️", layout="wide")
inject_css()
st.markdown(
    """
    <style>
    /* extra rows in the pipeline card + the measurements block, in the same style */
    .st-key-stage_input, .st-key-stage_dcp, .st-key-stage_deblur, .st-key-stage_tone { gap: .15rem; margin-top: .9rem; }
    .st-key-upload_card { --pv-maxh: max(12rem, calc(var(--ufc-card-h) - 12.5rem)); }
    .st-key-pipeline_card [data-testid="stSelectbox"] label p,
    .st-key-pipeline_card [data-testid="stToggle"] label p,
    .st-key-pipeline_card [data-testid="stCheckbox"] label p { font-size: .95rem; }
    .st-key-upload_card [data-testid="stSegmentedControl"] { margin-bottom: .2rem; }
    .st-key-method_card, .st-key-batch_card { padding: 1.5rem 1.75rem; margin-top: 1.2rem; }
    table.ufc-m { width: 100%; border-collapse: collapse; }
    table.ufc-m th, table.ufc-m td { text-align: right; padding: .4rem .3rem; border-bottom: 1px solid rgba(0,0,0,.08); }
    table.ufc-m th { font-weight: 600; font-size: .85rem; color: #777; }
    table.ufc-m th:first-child, table.ufc-m td:first-child { text-align: left; }
    [data-testid="stAppViewContainer"] table.ufc-m td:not(:first-child) { font-family: 'IBM Plex Mono', monospace; font-size: .9rem; }
    table.ufc-m td b { color: #d32f2f; font-weight: 600; }
    .ufc-cap { font-size: .85rem; color: #777; margin: 0 0 .4rem; }
    .ufc-eq { font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; }
    @media (max-width: 1000px) {
        .pipeline-heading { font-size: 1.45rem !important; word-break: normal !important;
                            overflow-wrap: normal !important; hyphens: none !important; }
    }
    /* Method card: both columns start on the same line, same line height, indented paragraphs,
       hanging indent on the references */
    .st-key-method_card [data-testid="stMarkdownContainer"] p { margin: 0 0 .8rem; line-height: 1.65;
        text-align: justify; hyphens: auto; text-indent: 1.6rem; }
    .st-key-method_card [data-testid="stMarkdownContainer"] p.ufc-sub { text-indent: 0; text-align: left;
        font-weight: 600; font-size: 1.05rem; margin: .2rem 0 .7rem; }
    .st-key-method_card [data-testid="stMarkdownContainer"] p.ufc-eq { text-indent: 0; text-align: left; margin-bottom: 1rem; }
    .st-key-method_card [data-testid="stMarkdownContainer"] ol { margin: 0; padding-left: 1.9rem; }
    .st-key-method_card [data-testid="stMarkdownContainer"] ol li { margin: 0 0 .7rem; padding-left: .5rem;
        line-height: 1.65; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)
render_title()
render_header()

SAMPLE_LABEL, PLACEHOLDER = "Sample scene (synthetic city)", "Or try a sample…"
VIEWS = ["Side by side", "Slider", "Original", "Dehazed", "Stages", "Edges"]
STAGE_LABELS = ["01 Dark Channel", "02 Deblur", "05 Smoothing", "06 Sharpening", "03 Histogram", "04 Frequency", "07 Tone"]  # run order

for _k, _v in {"source_key": None, "source_img": None, "source_depth": None, "reference": None,
               "pre_foggy": True, "fog_density": 1.6, "upload_round": 0, "upload_error": None,
               "use_dcp": True, "source_choice": PLACEHOLDER, "blur_amount": 0.0, "blur_kind": "Gaussian",
               "preset_choice": "Balanced (default)", "blur_angle": 0.0,
               **PRESETS["Balanced (default)"]}.items():
    st.session_state.setdefault(_k, _v)
if st.session_state.pop("_reset_source_choice", False):     # must happen before the selectbox is created
    st.session_state.source_choice = PLACEHOLDER


def _apply_auto_enhance():
    for k, v in PRESETS["Balanced (default)"].items():
        st.session_state[k] = v
    st.session_state.use_dcp = True
    st.session_state.strength_auto = True       # Auto-Enhance = Balanced, strength from the haze level, blur measured
    st.session_state.deblur_auto = True
    st.session_state.preset_choice = "Balanced (default)"


def _apply_reset():
    st.session_state.hist_strength = 0
    st.session_state.freq_strength = 0
    st.session_state.smooth_strength = 0
    st.session_state.sharpen_strength = 0
    st.session_state.tone_gamma = 1.0
    st.session_state.tone_sat = 1.0
    st.session_state.tone_balance = 0
    st.session_state.deblur_amount = 0.0
    st.session_state.deblur_auto = False
    st.session_state.strength_auto = False


def _apply_preset():
    for k, v in PRESETS[st.session_state.preset_choice].items():
        st.session_state[k] = v


def _choose_source():
    choice = st.session_state.source_choice
    if choice == PLACEHOLDER:
        return
    if choice == SAMPLE_LABEL:
        img, depth = make_sample_scene()
        st.session_state.update(source_img=img, source_key="sample", source_depth=depth,
                                reference=("sample", img), pre_foggy=False)
        return
    if choice in list_clear_photos():          # a clear street photo: fog is simulated on it, the photo is the reference
        img = _read_rgb(list_clear_photos()[choice])
        st.session_state.update(source_img=img, source_key=f"street:{choice}", source_depth=None,
                                reference=(f"street:{choice}", img), pre_foggy=False)
        return
    path, ref = list_test_images()[choice]
    img, key = _read_rgb(path), f"test:{choice}"
    st.session_state.update(
        source_img=img, source_key=key, source_depth=None, pre_foggy=True,
        reference=(key, _read_rgb(ref, (img.shape[1], img.shape[0]))) if ref else None)


def current_params():
    """Pipeline settings from the sliders. 'Course modules only' keeps just the project's three
    modules: frequency-domain enhancement, histogram stretching, adaptive smoothing and sharpening."""
    ss = st.session_state
    only = ss.modules_only
    return dict(hist=ss.hist_strength, freq=ss.freq_strength, smooth=ss.smooth_strength,
                sharpen=ss.sharpen_strength, omega=ss.dcp_omega, t0=ss.dcp_t0, patch=ss.dcp_patch,
                use_dcp=bool(ss.use_dcp and not only), deblur=0.0 if only else ss.deblur_amount,
                deblur_kind=ss.deblur_kind, deblur_k=ss.deblur_k, deblur_angle=ss.deblur_angle,
                gamma=1.0 if only else ss.tone_gamma, sat=1.0 if only else ss.tone_sat,
                balance=0 if only else ss.tone_balance, cutoff=ss.freq_cutoff)


def effective_params(img, P):
    """P with the automatic options applied to this photo: the blur measured by estimate_blur and the strengths scaled
    to the haze level. Returns (settings, info) with info["blur"] / info["haze"] set for the options that are on."""
    ss, info = st.session_state, {}
    if ss.deblur_auto and not ss.modules_only:
        est = estimate_blur(img)
        info["blur"] = est
        if est["kind"] in ("Gaussian", "Motion"):
            P = dict(P, deblur=round(est["amount"] * 2) / 2, deblur_kind=est["kind"], deblur_angle=est["angle"])  # slider step 0.5
        else:
            P = dict(P, deblur=0.0)
    if ss.strength_auto:
        h = haze_of(img)
        info["haze"] = (h, strength_scale(h))
        P = auto_strength_params(img, P, haze=h)[0]
    return P, info


has_image = st.session_state.source_img is not None
pipe_info = None  # filled in by the Preview card once a photo is loaded
hazy = result = None
auto_info = {}    # what the automatic options found for the current photo (see effective_params)

# Both cards live inside a narrower, centered strip rather than stretching
# across the whole wide-mode page — full-width made them read as short,
# flattened bars instead of the more upright cards they're meant to be.
_, center_col, _ = st.columns(PAGE_SPLIT, gap="large")
left_col, right_col = center_col.columns(CARD_SPLIT, gap="medium")

# ------------------------------------------------------------
# LEFT CARD — upload / preview / measurements
# ------------------------------------------------------------
with left_col:
    with st.container(border=True, key="upload_card"):
        if not has_image:
            st.selectbox("Or try a sample", [PLACEHOLDER, SAMPLE_LABEL, *list_clear_photos(), *list_test_images()],
                         key="source_choice", on_change=_choose_source, label_visibility="collapsed")
            # No `type=` filter: every image format is accepted (WebP, GIF, BMP,
            # TIFF, ... — see load_photo). A file that isn't a readable image gets
            # an explanation under the box, and the box starts over (a new widget
            # key means a fresh, empty uploader) so the next try is one click away.
            uploaded_file = st.file_uploader(
                "Drop a foggy photo here",
                label_visibility="collapsed",
                key=f"photo_uploader_{st.session_state.upload_round}",
            )
            if uploaded_file is not None:
                try:
                    photo = load_photo(uploaded_file)
                except Exception as exc:
                    st.session_state.upload_error = unreadable_message(uploaded_file.name, exc)
                    st.session_state.upload_round += 1
                else:
                    st.session_state.upload_error = None
                    st.session_state.source_img = photo
                    st.session_state.source_key = f"{uploaded_file.name}-{uploaded_file.size}"
                    st.session_state.source_depth = None
                    st.session_state.reference = None
                    st.session_state.pre_foggy = True
                    st.session_state._reset_source_choice = True
                st.rerun()
            if st.session_state.upload_error:
                st.error(st.session_state.upload_error)
        else:
            raw = st.session_state.source_img
            hazy = raw if st.session_state.pre_foggy else simulate_blur(
                simulate_fog(raw, st.session_state.source_depth, st.session_state.fog_density),
                st.session_state.blur_amount, st.session_state.blur_kind, st.session_state.blur_angle)
            P, auto_info = effective_params(hazy, current_params())
            src_h, src_w = hazy.shape[:2]
            view = st.session_state.get("view_mode") or "Side by side"
            # Only for a sliver of a picture (taller than 5:1): see the preview CSS.
            sliver_cap = (
                '.st-key-preview_images { width: min(100%, max('
                'calc(2 * var(--pv-maxh) * var(--pv-ar) + 1rem), 9rem)) !important; }'
                '.st-key-preview_images [data-testid="stImage"] img '
                '{ max-height: var(--pv-maxh); object-fit: contain; }'
                if src_w / src_h < 0.2 else ""
            )
            single_cap = (
                '.st-key-preview_images { width: min(100%, calc(var(--pv-maxh) * var(--pv-ar, 1.25))) !important; }'
                if view not in ("Side by side", "Stages", "Edges") else ""
            )
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown(
                    f'<style>.st-key-preview_images {{ --pv-ar: {src_w / src_h:.4f}; }}{sliver_cap}{single_cap}</style>'
                    '<style>.st-key-upload_card { min-height: 0 !important; }'
                    '[data-testid="stLayoutWrapper"]:has(> .st-key-upload_card) '
                    '{ flex: 0 0 auto !important; align-self: flex-start; width: 100%; position: sticky; top: 1rem; }'
                    '[data-testid="stLayoutWrapper"]:has(> .st-key-preview_images) '
                    '{ margin-top: 0 !important; margin-bottom: 0 !important; }</style>'
                    '<div class="card-heading">Preview</div>',
                    unsafe_allow_html=True,
                )
            with top_r:
                if st.button("Change Image", width="stretch"):
                    st.session_state.source_img = None
                    st.session_state.source_key = None
                    st.session_state._reset_source_choice = True
                    st.rerun()

            st.segmented_control("View", VIEWS, default="Side by side", key="view_mode",
                                 label_visibility="collapsed")
            if view == "Slider":
                st.slider("Compare position", 0, 100, 50, key="compare_pos", label_visibility="collapsed")
            elif view == "Stages":
                st.pills("Stage", STAGE_LABELS, default=STAGE_LABELS[0], key="stage_pick",
                         label_visibility="collapsed")
                st.segmented_control("Stage mode", ["Cumulative", "This stage only"], default="Cumulative",
                                     key="stage_mode", label_visibility="collapsed")
                st.caption("Cumulative follows the order the stages run. This stage only applies the one stage "
                           "to the original.")

            with st.spinner("Processing…"):
                result, proc_s = run_pipeline(hazy, P)

            with st.container(key="preview_images"):
                if view == "Side by side":
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(hazy, caption="Original", width="stretch")
                    with col_after:
                        st.image(result, caption="Dehazed", width="stretch")
                elif view == "Slider":
                    pos = st.session_state.get("compare_pos", 50)
                    st.image(compare_composite(hazy, result, pos),
                             caption=f"Original (left) · Dehazed (right) · divider at {pos}%", width="stretch")
                elif view == "Stages":
                    pick = st.session_state.get("stage_pick") or STAGE_LABELS[0]
                    idx = STAGE_LABELS.index(pick)
                    name = STAGE_ORDER[idx]
                    if st.session_state.get("stage_mode") == "This stage only":
                        before, after, cap_b = hazy, apply_stage(name, hazy, P), "Original"
                    else:
                        steps = run_stages(hazy, P)
                        before, after = steps[idx][1], steps[idx + 1][1]
                        cap_b = "Original" if idx == 0 else f"After the stage before {pick}"
                    col_b, col_a = st.columns(2)
                    with col_b:
                        st.image(before, caption=cap_b, width="stretch")
                    with col_a:
                        st.image(after, width="stretch",
                                 caption=f"After {pick}" if stage_active(name, P) else f"{pick} is off, so nothing changes")
                elif view == "Edges":
                    e0, e1 = edge_maps(hazy, result)
                    col_b, col_a = st.columns(2)
                    with col_b:
                        st.image(e0, caption="Original edges", width="stretch")
                    with col_a:
                        st.image(e1, caption="Dehazed edges (same scale)", width="stretch")
                elif view == "Original":
                    st.image(hazy, caption="Original", width="stretch")
                else:
                    st.image(result, caption="Dehazed", width="stretch")

            # ---------------- measurements and histogram comparison, below the images ----------------
            st.markdown('<div class="card-heading" style="font-size:1.5rem;margin-top:1.4rem">Measurements</div>',
                        unsafe_allow_html=True)
            m0, m1 = quality_metrics(hazy), quality_metrics(result)
            # Only when fog (and blur) were simulated is there a clear reference: the photo itself.
            ref = raw if not st.session_state.pre_foggy else None
            if ref is None and st.session_state.reference is not None:      # a bundled photo that comes with its clear version
                ref_key, ref_img = st.session_state.reference
                if ref_key == st.session_state.source_key and ref_img.shape == hazy.shape:
                    ref = ref_img
            d0, d1 = downstream(hazy, ref), downstream(result, ref)

            def _row(name, a, b, d):
                return f"<tr><td>{name}</td><td>{a:.{d}f}</td><td><b>{b:.{d}f}</b></td></tr>"

            rows = (_row("Contrast (std. dev.)", m0["contrast"], m1["contrast"], 1)
                    + _row("Entropy (bits)", m0["entropy"], m1["entropy"], 2)
                    + _row("Mean gradient", m0["gradient"], m1["gradient"], 2)
                    + _row("Edge pixels (%)", m0["edges"], m1["edges"], 1)
                    + _row("Canny edges (% px)", d0["canny_pct"], d1["canny_pct"], 1)
                    + _row("Lines found (Hough)", d0["lines"], d1["lines"], 0)
                    + _row("&nbsp;&nbsp;vertical (buildings)", d0["lines_vertical"], d1["lines_vertical"], 0)
                    + _row("&nbsp;&nbsp;oblique (lanes)", d0["lines_oblique"], d1["lines_oblique"], 0))
            if ref is not None:
                rows += (_row("PSNR vs. clear (dB)", psnr(hazy, ref), psnr(result, ref), 1)
                         + _row("SSIM vs. clear", ssim(hazy, ref), ssim(result, ref), 3)
                         + _row("Edge F1 vs. clear", d0["edge_f1"], d1["edge_f1"], 3)
                         + _row("Line recall vs. clear", d0["line_recall"], d1["line_recall"], 3))
            rows += f"<tr><td>Processing time (s)</td><td>&ndash;</td><td><b>{proc_s:.2f}</b></td></tr>"
            st.markdown(
                '<p class="ufc-cap">Structure visibility: edge pixels '
                f'{m0["edges"]:.1f}% &rarr; {m1["edges"]:.1f}% &middot; mean gradient &times;'
                f'{m1["gradient"] / max(m0["gradient"], 1e-3):.1f} &middot; contrast &times;'
                f'{m1["contrast"] / max(m0["contrast"], 1e-3):.1f}</p>',
                unsafe_allow_html=True)
            c_tab, c_h = st.columns([1, 1.1], gap="medium")
            c_tab.markdown(f'<table class="ufc-m"><tr><th></th><th>Original</th><th>Dehazed</th></tr>{rows}</table>',
                           unsafe_allow_html=True)
            c_h.markdown(
                '<p class="ufc-cap">Brightness histogram: original (grey), dehazed (red)</p>'
                + hist_svg([(m0["hist"], "#9a9a9a", False), (m1["hist"], "#d32f2f", False)], label="Brightness histograms"),
                unsafe_allow_html=True)
            chan = ["#e53935", "#43a047", "#1e88e5"]
            c_h.markdown(
                '<p class="ufc-cap">R, G, B histograms: original (dashed), dehazed (solid)</p>'
                + hist_svg([(m0["chans"][c], chan[c], True) for c in range(3)]
                           + [(m1["chans"][c], chan[c], False) for c in range(3)], label="Colour channel histograms"),
                unsafe_allow_html=True)


# ------------------------------------------------------------
# RIGHT CARD — enhancement pipeline
# ------------------------------------------------------------
def _stage_row(num, label, value):
    return (
        f'<div class="stage-row"><span><span class="stage-num">{num}</span>'
        f'<span class="stage-label">{label}</span></span>'
        f'<span class="stage-value">{value}</span></div>'
    )


def _param_row(label, value):
    return (
        '<div class="stage-row" style="margin-top:.45rem"><span class="stage-label" '
        f'style="font-weight:500;font-size:.95rem">{label}</span>'
        f'<span class="stage-value" style="font-size:.95rem">{value}</span></div>'
    )


with right_col:
    with st.container(border=True, key="pipeline_card"):
        st.markdown('<div class="pipeline-heading">Enhancement Pipeline</div>', unsafe_allow_html=True)

        # Input — where the photo comes from, and (optionally) simulated fog on it
        with st.container(key="stage_input"):
            foggy = st.session_state.pre_foggy
            st.markdown(_stage_row("00", "Input", "photo as is" if foggy else f"β {st.session_state.fog_density:.2f}"),
                        unsafe_allow_html=True)
            st.checkbox("Photo is already foggy (skip the fog simulation)", key="pre_foggy")
            st.markdown(_param_row("Simulated fog density", f"{st.session_state.fog_density:.2f}"),
                        unsafe_allow_html=True)
            st.slider("Fog density", 0.0, 4.0, step=0.05, key="fog_density",
                      disabled=st.session_state.pre_foggy, label_visibility="collapsed")
            blur_txt = "off" if st.session_state.blur_amount == 0 else f"{st.session_state.blur_amount:.1f}px"
            st.markdown(_param_row("Simulated blur", blur_txt), unsafe_allow_html=True)
            st.slider("Blur amount", 0.0, 10.0, step=0.5, key="blur_amount",
                      disabled=st.session_state.pre_foggy, label_visibility="collapsed")
            st.selectbox("Blur type", ["Gaussian", "Motion"], key="blur_kind", disabled=st.session_state.pre_foggy)
            if st.session_state.blur_kind == "Motion":
                st.markdown(_param_row("Motion direction", f"{st.session_state.blur_angle:.0f}°"), unsafe_allow_html=True)
                st.slider("Motion direction", 0.0, 179.0, step=5.0, key="blur_angle",
                          disabled=st.session_state.pre_foggy, label_visibility="collapsed")
            st.toggle("Course modules only (skip dark channel, deblur and tone)", key="modules_only")

        # Stage 01 — Dark Channel Prior
        with st.container(key="stage_dcp"):
            st.markdown(_stage_row("01", "Dark Channel Prior",
                                  "On" if st.session_state.use_dcp and not st.session_state.modules_only else "Off"),
                        unsafe_allow_html=True)
            st.toggle("Remove fog before the other stages", key="use_dcp")
            st.markdown(_param_row("Haze removed (ω)", f"{st.session_state.dcp_omega:.2f}"), unsafe_allow_html=True)
            st.slider("Haze removed", 0.50, 1.00, step=0.01, key="dcp_omega", label_visibility="collapsed")
            st.markdown(_param_row("Minimum transmission (t₀)", f"{st.session_state.dcp_t0:.2f}"), unsafe_allow_html=True)
            st.slider("Minimum transmission", 0.05, 0.50, step=0.01, key="dcp_t0", label_visibility="collapsed")
            patch = st.session_state.dcp_patch
            st.markdown(_param_row("Patch size", "auto" if patch == 0 else f"{patch}px"), unsafe_allow_html=True)
            st.slider("Patch size", 0, 61, step=1, key="dcp_patch", label_visibility="collapsed")

        # Stage 02 — Deblur
        with st.container(key="stage_deblur"):
            blur_est = auto_info.get("blur")
            auto_on = bool(st.session_state.deblur_auto and blur_est is not None)
            amt = st.session_state.deblur_amount
            if auto_on:
                amt = round(blur_est["amount"] * 2) / 2 if blur_est["kind"] in ("Gaussian", "Motion") else 0.0
            st.markdown(_stage_row("02", "Deblur (Wiener)", "off" if amt == 0 else f"{amt:.1f}px"),
                        unsafe_allow_html=True)
            st.toggle("Detect the blur automatically", key="deblur_auto", disabled=st.session_state.modules_only)
            if auto_on:
                kind = blur_est["kind"]
                st.caption("Detected: the photo looks sharp, nothing to undo." if kind == "none" else
                           f"Detected: {kind.lower()} blur of about {blur_est['amount']:.1f} px"
                           + (f" at {blur_est['angle']:.0f}° (0° = horizontal)" if kind == "Motion" else "") + ", undone below.")
            st.markdown(_param_row("Blur to undo", "off" if amt == 0 else f"{amt:.1f}px"), unsafe_allow_html=True)
            st.slider("Blur to undo", 0.0, 10.0, step=0.5, key="deblur_amount", label_visibility="collapsed",
                      disabled=auto_on)
            st.selectbox("Assumed blur type", ["Gaussian", "Motion"], key="deblur_kind", disabled=auto_on)
            if st.session_state.deblur_kind == "Motion" or (auto_on and blur_est["kind"] == "Motion"):
                ang = blur_est["angle"] if auto_on and blur_est["kind"] == "Motion" else st.session_state.deblur_angle
                st.markdown(_param_row("Motion direction", f"{ang:.0f}°"), unsafe_allow_html=True)
                st.slider("Motion direction to undo", 0.0, 179.0, step=5.0, key="deblur_angle",
                          label_visibility="collapsed", disabled=auto_on)
            st.markdown(_param_row("Noise damping (K)", f"{st.session_state.deblur_k:.3f}"), unsafe_allow_html=True)
            st.slider("Noise damping", 0.001, 0.100, step=0.001, format="%.3f", key="deblur_k",
                      label_visibility="collapsed")

        # Stage 03 — Histogram Stretch
        with st.container(key="stage_hist"):
            st.markdown(_stage_row("03", "Histogram Stretch", st.session_state.hist_strength), unsafe_allow_html=True)
            hist_strength = st.slider("Histogram strength", 0, 100, key="hist_strength", label_visibility="collapsed")

        # Stage 03 — Frequency-Domain Filter
        with st.container(key="stage_freq"):
            st.markdown(_stage_row("04", "Frequency-Domain Filter", st.session_state.freq_strength),
                        unsafe_allow_html=True)
            freq_strength = st.slider("Frequency strength", 0, 100, key="freq_strength", label_visibility="collapsed")
            st.markdown(_param_row("Cutoff radius", f"{st.session_state.freq_cutoff}px"), unsafe_allow_html=True)
            st.slider("Cutoff radius", 5, 80, key="freq_cutoff", label_visibility="collapsed")

        # Stage 04 — Adaptive Smoothing
        with st.container(key="stage_smooth"):
            st.markdown(_stage_row("05", "Adaptive Smoothing", st.session_state.smooth_strength),
                        unsafe_allow_html=True)
            smooth_strength = st.slider("Smoothing strength", 0, 100, key="smooth_strength", label_visibility="collapsed")

        # Stage 05 — Adaptive Sharpening
        with st.container(key="stage_sharpen"):
            st.markdown(_stage_row("06", "Adaptive Sharpening", st.session_state.sharpen_strength),
                        unsafe_allow_html=True)
            sharpen_strength = st.slider("Sharpening strength", 0, 100, key="sharpen_strength",
                                         label_visibility="collapsed")

        # Stage 06 — Final Tone
        with st.container(key="stage_tone"):
            st.markdown(_stage_row("07", "Final Tone", ""), unsafe_allow_html=True)
            st.markdown(_param_row("Gamma", f"{st.session_state.tone_gamma:.2f}"), unsafe_allow_html=True)
            st.slider("Gamma", 0.60, 1.60, step=0.05, key="tone_gamma", label_visibility="collapsed")
            st.markdown(_param_row("Saturation", f"{st.session_state.tone_sat:.2f}"), unsafe_allow_html=True)
            st.slider("Saturation", 0.50, 2.00, step=0.05, key="tone_sat", label_visibility="collapsed")
            st.markdown(_param_row("Colour balance", f"{st.session_state.tone_balance}"), unsafe_allow_html=True)
            st.slider("Colour balance", 0, 100, key="tone_balance", label_visibility="collapsed")

        with st.container(key="pipeline_actions"):
            st.selectbox("Preset", list(PRESETS), key="preset_choice", on_change=_apply_preset)
            st.toggle("Set the strength from the haze level", key="strength_auto")
            if "haze" in auto_info:
                st.caption(f"Haze level {auto_info['haze'][0]:.2f}: the enhancement runs at "
                           f"{auto_info['haze'][1] * 100:.0f}% of the sliders' strength.")
            b1, b2 = st.columns(2)
            with b1:
                st.button("Auto-Enhance", width="stretch", on_click=_apply_auto_enhance)
            with b2:
                st.button("Reset", width="stretch", on_click=_apply_reset)

            if has_image and result is not None:
                st.download_button("Download Result", data=to_png_bytes(result), file_name="dehazed_result.png",
                                   mime="image/png", type="primary", width="stretch")
            else:
                st.download_button("Download Result", data=b"", file_name="dehazed_result.png",
                                   mime="image/png", type="primary", width="stretch", disabled=True)


# ------------------------------------------------------------
# BELOW THE CARDS — method and references
# ------------------------------------------------------------
_, lower_col, _ = st.columns(PAGE_SPLIT, gap="large")
with lower_col:
    with st.container(border=True, key="batch_card"):
        st.markdown('<div class="card-heading">Batch Processing</div>', unsafe_allow_html=True)
        st.markdown('<p class="ufc-cap">Run several photos through the pipeline with the current settings, '
                    'for example frames from a traffic camera or site-inspection photos. Every photo is '
                    'treated as already foggy.</p>', unsafe_allow_html=True)
        def _clear_batch():                      # a new uploader key drops every uploaded file at once
            st.session_state.batch_round = st.session_state.get("batch_round", 0) + 1
            st.session_state.batch_results = []
        batch_files = st.file_uploader("Photos to process", accept_multiple_files=True,
                                       key=f"batch_up_{st.session_state.get('batch_round', 0)}",
                                       label_visibility="collapsed")
        if batch_files or st.session_state.get("batch_results"):
            run_col, clear_col, _sp = st.columns([1, 1, 3])
            run = run_col.button("Process batch", type="primary", disabled=not batch_files)
            clear_col.button("Clear all", on_click=_clear_batch, help="Remove every uploaded photo and the results")
        else:
            run = False
        if batch_files and run:
            P_b, results, bar = current_params(), [], st.progress(0.0)
            for i, f in enumerate(batch_files):
                try:
                    img_b = load_photo(f)
                except Exception as exc:
                    results.append({"name": f.name, "error": unreadable_message(f.name, exc)})
                else:
                    t_b = time.perf_counter()
                    out_b = dehaze_pipeline(img_b, effective_params(img_b, P_b)[0])
                    sec = time.perf_counter() - t_b
                    q0, q1 = quality_metrics(img_b), quality_metrics(out_b)
                    results.append({"name": f.name, "size": f"{img_b.shape[1]}x{img_b.shape[0]}", "sec": sec,
                                    "contrast": q1["contrast"] / max(q0["contrast"], 1e-3),
                                    "edges": q1["edges"] / max(q0["edges"], 1e-3), "png": to_png_bytes(out_b)})
                bar.progress((i + 1) / len(batch_files))
            bar.empty()
            st.session_state.batch_results = results
        done = [r for r in st.session_state.get("batch_results", []) if "png" in r]
        for r in st.session_state.get("batch_results", []):
            if "error" in r:
                st.error(r["error"])
        if done:
            body = "".join(f"<tr><td>{r['name']}</td><td>{r['size']}</td><td>{r['sec']:.2f}</td>"
                           f"<td>{r['contrast']:.2f}</td><td>{r['edges']:.2f}</td></tr>" for r in done)
            st.markdown('<table class="ufc-m"><tr><th>Photo</th><th>Size</th><th>Time (s)</th>'
                        f'<th>Contrast &times;</th><th>Edge pixels &times;</th></tr>{body}</table>',
                        unsafe_allow_html=True)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in done:
                    zf.writestr(f"dehazed_{Path(r['name']).stem}.png", r["png"])
                zf.writestr("batch_results.csv", "photo,size,seconds,contrast_x,edge_pixels_x\n" + "".join(
                    f"{r['name']},{r['size']},{r['sec']:.3f},{r['contrast']:.3f},{r['edges']:.3f}\n" for r in done))
            st.download_button("Download all as ZIP", data=buf.getvalue(), file_name="dehazed_batch.zip",
                               mime="application/zip")

    with st.container(border=True, key="video_card"):
        st.markdown('<div class="card-heading">Video</div>', unsafe_allow_html=True)
        st.markdown('<p class="ufc-cap">Dehaze a short clip from a fixed camera with the current settings, including the '
                    'automatic options. The fog, colour and contrast estimates are smoothed over time and static pixels are '
                    'averaged, so the result does not flicker; moving vehicles are left alone. Frames are scaled to at most '
                    '640 px.</p>', unsafe_allow_html=True)
        vid = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"], key="video_up", label_visibility="collapsed")
        v1, v2 = st.columns(2)
        v_frames = v1.slider("Frames to process", 10, 300, 90, key="video_frames")
        v_smooth = v2.toggle("Smooth over time", value=True, key="video_temporal")
        if vid is not None and st.button("Process video", type="primary", key="video_go"):
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / ("in" + Path(vid.name).suffix), Path(td) / "out.mp4"
                src.write_bytes(vid.getvalue())
                bar = st.progress(0.0)
                try:
                    n_fr, spf = process_video(
                        src, dst, P=current_params(), auto=st.session_state.strength_auto, temporal=v_smooth,
                        max_dim=640, max_frames=v_frames, auto_blur=st.session_state.deblur_auto and not st.session_state.modules_only,
                        progress=lambda i, total: bar.progress(min(1.0, i / max(total, 1))))
                    st.session_state.video_result = (dst.read_bytes(), n_fr, spf) if n_fr else None
                    st.session_state.video_error = None if n_fr else "No frames could be read from that file."
                except Exception as exc:
                    st.session_state.video_result, st.session_state.video_error = None, f"Could not process that video: {exc}"
            bar.empty()
        if st.session_state.get("video_error"):
            st.error(st.session_state.video_error)
        if st.session_state.get("video_result"):
            vdata, n_fr, spf = st.session_state.video_result
            st.video(vdata)
            st.caption(f"{n_fr} frames, {spf:.2f} s per frame.")
            st.download_button("Download video", data=vdata, file_name="dehazed_video.mp4", mime="video/mp4")

    with st.container(border=True, key="method_card"):
        st.markdown('<div class="card-heading">Method</div>', unsafe_allow_html=True)
        st.markdown('<p class="ufc-eq">I = J·t + A·(1 − t),&nbsp; t = e<sup>−βd</sup></p>', unsafe_allow_html=True)
        m_l, m_r = st.columns(2, gap="large")
        m_l.markdown('<p class="ufc-sub">Approach</p>', unsafe_allow_html=True)
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
            "motion blur (at any angle) whose size you set, or that the app measures from the width of the strongest "
            "edges along eight directions (*Detect the blur automatically*). *Course modules only* runs just the three project modules: "
            "frequency-domain enhancement, histogram stretching, and adaptive smoothing and sharpening. "
            "The Edges view shows the Sobel gradient of both images on one scale, so you can see how much "
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
        m_r.markdown('<p class="ufc-sub">References</p>', unsafe_allow_html=True)
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
