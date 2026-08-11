"""
GeoGuessr local inference server — pairs with geoguessr.user.js

The userscript sends only a Street View panorama ID. This server fetches
the panorama tiles from Google directly (no browser CORS limits), splits
it into four 90-degree heading views, runs your GeoCLIP model on each,
and returns fused country probabilities.

Why panorama IDs instead of screenshots: the image comes back clean (no
GeoGuessr UI), at full resolution, and can be split by compass heading —
which matches how the model was trained (multi-direction, sky cropped).

SETUP
  1. Download from Drive GeoGussrCheat/models/geoclip_v61/ :
       best_model.pth, class_to_idx.json
     into ./models/geoclip_v61/ next to this file.
     (Falls back to models/geoclip_v6/ if v6.1 isn't there yet.)
  2. pip install fastapi uvicorn open_clip_torch torch torchvision pillow requests
  3. python geoguessr_server.py
  4. Install geoguessr.user.js in Tampermonkey, then open GeoGuessr.

Fair play note: intended for solo play and for evaluating your own model.
"""

import os
import io
import json
import math
import concurrent.futures

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import open_clip
import uvicorn

# ── Config ──────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [os.path.join(HERE, 'models', 'geoclip_v61'),
              os.path.join(HERE, 'models', 'geoclip_v6')]

CLIP_MODEL    = 'ViT-B-16'
CLIP_PRETRAIN = 'laion2b_s34b_b88k'

PANO_ZOOM   = 2      # 2 → 4x2 tiles of 512px = 2048x1024 equirectangular
TILE_PX     = 512
N_HEADINGS  = 4      # 0/90/180/270 degree views
VIEW_FOV    = 90.0   # degrees, roughly GeoGuessr's default field of view
VIEW_PITCH  = -5.0   # tilt slightly down: less sky, more road/verge detail
VIEW_PX     = 512

# Google's tile endpoint returns 403 without a browser User-Agent.
TILE_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Referer': 'https://www.google.com/maps/',
}

PORT   = 8000
DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

# ── GeoTips captions — MUST match the training notebook ─────
COUNTRY_HINTS = {
    'Albania': 'mediterranean scrub, older camera quality, mixed road lines',
    'Andorra': 'pyrenees mountain roads, stone buildings',
    'Austria': 'alpine scenery, tidy central european houses, white road lines',
    'Belgium': 'flat terrain, brick row houses, dense signage',
    'Bosnia and Herzegovina': 'yellow road lines, balkan hills, red-roofed houses',
    'Bulgaria': 'balkan countryside, cyrillic signs, striped kerbs',
    'Croatia': 'adriatic coast, white stone, red tiled roofs',
    'Cyprus': 'left-hand driving, yellow road lines, dry mediterranean hills',
    'Czech Republic': 'central european villages, orange roofs, striped bollards',
    'Denmark': 'flat farmland, brick houses, cycle lanes, white lines',
    'Estonia': 'baltic forests, wooden houses, boreal conifers',
    'Finland': 'boreal forest, lakes, wide gravel shoulders',
    'France': 'plane trees, red-topped white bollards, kilometre stones',
    'Germany': 'autobahn style roads, dense signage, half-timbered houses',
    'Greece': 'olive trees, whitewashed buildings, greek signs, dry hills',
    'Hungary': 'flat plains, pannonian villages, striped kerbstones',
    'Iceland': 'treeless volcanic landscape, yellow road lines, gravel roads',
    'Ireland': 'left-hand driving, green hedgerows, yellow lines, stone walls',
    'Italy': 'terracotta roofs, cypress trees, black-banded bollards',
    'Latvia': 'baltic pine forest, soviet era blocks, wooden farmhouses',
    'Liechtenstein': 'alpine valley, swiss style chalets',
    'Lithuania': 'baltic flatlands, wooden villages, conifer forest',
    'Luxembourg': 'wooded hills, tidy prosperous villages',
    'Malta': 'left-hand driving, limestone buildings, dry rocky terrain',
    'Monaco': 'dense high-rise mediterranean coast, luxury cars',
    'Montenegro': 'yellow road lines, dramatic mountains, adriatic coast',
    'Netherlands': 'flat land, canals, brick houses, many bicycles',
    'North Macedonia': 'balkan valleys, cyrillic signs, older imagery',
    'Norway': 'fjords, yellow road lines, red wooden houses, tunnels',
    'Poland': 'flat farmland, striped bollards, concrete utility poles',
    'Portugal': 'azulejo tiles, orange roofs, cobbled roads, atlantic coast',
    'Romania': 'carpathian hills, horse carts, orthodox churches, older camera',
    'Serbia': 'balkan plains, cyrillic and latin signs, striped kerbs',
    'Slovakia': 'carpathian villages, painted houses, striped bollards',
    'Slovenia': 'alpine and mediterranean mix, tidy villages, church spires',
    'Spain': 'dry meseta, orange bollards, white villages, olive groves',
    'Sweden': 'boreal forest, falu red houses, wide roads with white lines',
    'Switzerland': 'alps, chalets, immaculate roads, white lines',
    'Turkey': 'red crescent flags, mosques, dry anatolian hills',
    'Ukraine': 'wide plains, cyrillic signs, soviet era infrastructure',
    'United Kingdom': 'left-hand driving, hedgerows, brick terraces, roundabouts',
}
CAPTION_TEMPLATES = [
    'a street-level photo taken in {c}',
    'a street view photo in {c}, {h}',
    'a road in {c} with {h}',
    'a photo of a street scene in {c}',
]


def country_captions(c):
    h = COUNTRY_HINTS.get(c, 'typical local scenery')
    return [t.format(c=c, h=h) for t in CAPTION_TEMPLATES]


# ── Load model ──────────────────────────────────────────────
def _find_model():
    for d in MODEL_DIRS:
        p = os.path.join(d, 'best_model.pth')
        if os.path.exists(p):
            return d, p
    raise SystemExit(
        'No model found. Download best_model.pth and class_to_idx.json from\n'
        '  Drive/GeoGussrCheat/models/geoclip_v61/  (or geoclip_v6/)\n'
        f'into {MODEL_DIRS[0]}/')


MODEL_DIR, MODEL_PATH = _find_model()
print(f'Loading {os.path.basename(MODEL_DIR)} on {DEVICE} ...')

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN)
tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

_ckpt = torch.load(MODEL_PATH, map_location='cpu')
_state = _ckpt.get('model', _ckpt)
clip_model.load_state_dict(_state)
clip_model = clip_model.to(DEVICE).eval()

if 'classes' in _ckpt:
    CLASSES = list(_ckpt['classes'])
else:
    with open(os.path.join(MODEL_DIR, 'class_to_idx.json')) as f:
        CLASSES = sorted(json.load(f).keys())
LOGIT_SCALE = float(_ckpt.get('logit_scale', math.log(1 / 0.07)))
if LOGIT_SCALE < 10:                      # stored as log — exponentiate
    LOGIT_SCALE = math.exp(LOGIT_SCALE)
LOGIT_SCALE = min(LOGIT_SCALE, 100.0)


@torch.no_grad()
def build_text_anchors():
    out = []
    for c in CLASSES:
        emb = clip_model.encode_text(tokenizer(country_captions(c)).to(DEVICE))
        emb = F.normalize(emb, dim=-1).mean(0)
        out.append(F.normalize(emb, dim=-1))
    return torch.stack(out)


TEXT_ANCHORS = build_text_anchors()
print(f'Ready — {len(CLASSES)} countries, logit scale {LOGIT_SCALE:.1f}')


# ── Panorama fetching ───────────────────────────────────────
TILE_URL = ('https://streetviewpixels-pa.googleapis.com/v1/tile'
            '?cb_client=maps_sv.tactile&panoid={pid}&x={x}&y={y}&zoom={z}')


def _fetch_tile(args):
    pid, x, y, z = args
    try:
        r = requests.get(TILE_URL.format(pid=pid, x=x, y=y, z=z),
                         headers=TILE_HEADERS, timeout=10)
        if r.status_code != 200 or not r.content:
            return x, y, None
        return x, y, Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return x, y, None


def fetch_panorama(pano_id, zoom=PANO_ZOOM):
    """Stitch the equirectangular panorama for a pano id."""
    nx, ny = 2 ** zoom, 2 ** (zoom - 1)
    canvas = Image.new('RGB', (nx * TILE_PX, ny * TILE_PX))
    tasks = [(pano_id, x, y, zoom) for x in range(nx) for y in range(ny)]
    got = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for x, y, tile in pool.map(_fetch_tile, tasks):
            if tile is not None:
                canvas.paste(tile, (x * TILE_PX, y * TILE_PX))
                got += 1
    if got == 0:
        return None
    return canvas


def equirect_to_perspective(pano, yaw_deg, pitch_deg=VIEW_PITCH,
                            fov_deg=VIEW_FOV, out=VIEW_PX):
    """Reproject the equirectangular panorama into a normal camera view.

    Cropping the equirect directly leaves heavy barrel distortion (straight
    lines bend), which is nothing like the photos the model trained on.
    This casts a ray per output pixel and samples the sphere properly.
    """
    src = np.asarray(pano)
    H, W = src.shape[:2]
    f = 0.5 * out / np.tan(np.radians(fov_deg) / 2.0)

    i, j = np.meshgrid(np.arange(out), np.arange(out), indexing='ij')
    x = j - out / 2.0
    y = i - out / 2.0
    z = np.full_like(x, f, dtype=np.float64)

    p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
    y2 = y * np.cos(p) - z * np.sin(p)          # pitch about X
    z2 = y * np.sin(p) + z * np.cos(p)
    x3 = x * np.cos(yw) + z2 * np.sin(yw)       # yaw about Y
    z3 = -x * np.sin(yw) + z2 * np.cos(yw)

    r = np.sqrt(x3 ** 2 + y2 ** 2 + z3 ** 2)
    lon = np.arctan2(x3, z3)
    lat = np.arcsin(np.clip(y2 / r, -1.0, 1.0))

    u = np.clip((((lon / (2 * np.pi)) + 0.5) * W).astype(np.int32), 0, W - 1)
    v = np.clip((((lat / np.pi) + 0.5) * H).astype(np.int32), 0, H - 1)
    return Image.fromarray(src[v, u])


def heading_views(pano, n=N_HEADINGS):
    """One rectilinear view per compass heading — the model was trained to
    handle looking in every direction, so all n are scored independently."""
    return [equirect_to_perspective(pano, yaw_deg=i * (360.0 / n))
            for i in range(n)]


# ── Inference ───────────────────────────────────────────────
@torch.no_grad()
def predict_views(views):
    """Average per-view softmax (each heading = independent evidence),
    with horizontal-flip TTA on each view."""
    batch = torch.stack([clip_preprocess(v) for v in views]).to(DEVICE)
    e1 = F.normalize(clip_model.encode_image(batch), dim=-1)
    e2 = F.normalize(clip_model.encode_image(torch.flip(batch, dims=[3])), dim=-1)
    emb = F.normalize((e1 + e2) / 2, dim=-1)
    probs = F.softmax(LOGIT_SCALE * emb @ TEXT_ANCHORS.T, dim=1)
    return probs.float().cpu().numpy()          # (n_views, n_classes)


# ── API ─────────────────────────────────────────────────────
app = FastAPI(title='GeoGuessr GeoCLIP server')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

_cache = {}          # pano_id -> response (panoramas are immutable)


class PredictRequest(BaseModel):
    pano_id: str


@app.get('/health')
def health():
    return {'ok': True, 'model': os.path.basename(MODEL_DIR),
            'countries': len(CLASSES), 'device': DEVICE}


@app.post('/predict')
def predict(req: PredictRequest):
    pid = req.pano_id.strip()
    if pid in _cache:
        return _cache[pid]

    pano = fetch_panorama(pid)
    if pano is None:
        return {'error': f'could not fetch panorama {pid}'}

    views = heading_views(pano)
    per_view = predict_views(views)
    mean = per_view.mean(axis=0)
    order = np.argsort(mean)[::-1][:5]

    result = {
        'pano_id': pid,
        'top': [[CLASSES[i], round(float(mean[i]), 4)] for i in order],
        'per_heading': [
            {'heading': i * (360 // N_HEADINGS),
             'country': CLASSES[int(p.argmax())],
             'prob': round(float(p.max()), 4)}
            for i, p in enumerate(per_view)
        ],
    }
    if len(_cache) > 500:
        _cache.clear()
    _cache[pid] = result
    return result


if __name__ == '__main__':
    print(f'\nServing on http://127.0.0.1:{PORT}  —  install '
          'geoguessr.user.js in Tampermonkey and open GeoGuessr.\n')
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
