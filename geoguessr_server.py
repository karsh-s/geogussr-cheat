"""
GeoGuessr local inference server — pairs with geoguessr.user.js

The userscript sends only a Street View panorama ID. This server fetches
the panorama tiles from Google directly (no browser CORS limits), splits
it into four 90-degree heading views, runs your GeoCLIP model on each,
and returns fused country probabilities.

Why panorama IDs instead of screenshots: the image comes back clean (no
GeoGuessr UI) and can be split by compass heading using the same
projection maths streetview_collector.py used to build the training set,
so what the model sees in play is identical in kind to what it trained on.

v7 scores 76.7% top-1 and 96.3% top-5 per location (4 headings averaged)
across 10,038 held-out locations.

SETUP
  1. Download from Drive GeoGussrCheat/models/geoclip_v7/ :
       best_model.pth, class_to_idx.json
     into ./models/geoclip_v7/ next to this file.
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
import torch.nn as nn
import open_clip
import uvicorn

# ── Config ──────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = [os.path.join(HERE, 'models', 'geoclip_v7'),
              os.path.join(HERE, 'models', 'geoclip_v61')]

CLIP_MODEL    = 'ViT-B-16'
CLIP_PRETRAIN = 'laion2b_s34b_b88k'

PANO_ZOOM   = 2      # 2 → 4x2 tiles of 512px = 2048x1024 equirectangular
TILE_PX     = 512
N_HEADINGS  = 4      # 0/90/180/270 degree views
VIEW_FOV    = 90.0   # must match streetview_collector.py
VIEW_PITCH  = -5.0
VIEW_PX     = 320    # same size the training views were saved at

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

# ── Load model ──────────────────────────────────────────────
def _find_model():
    for d in MODEL_DIRS:
        p = os.path.join(d, 'best_model.pth')
        if os.path.exists(p):
            return d, p
    raise SystemExit(
        'No model found. Download best_model.pth and class_to_idx.json from\n'
        '  Drive/GeoGussrCheat/models/geoclip_v7/\n'
        f'into {MODEL_DIRS[0]}/')


MODEL_DIR, MODEL_PATH = _find_model()
print(f'Loading {os.path.basename(MODEL_DIR)} on {DEVICE} ...')

_ckpt = torch.load(MODEL_PATH, map_location='cpu')
if 'mode' not in _ckpt:
    raise SystemExit(
        f'{MODEL_PATH} is not a v7 checkpoint (no "mode" key).\n'
        'Earlier models were trained on data whose labels were wrong and do '
        'not work; retrain with geoclip_v7.ipynb.')

MODE     = _ckpt['mode']                 # 'probe' or 'finetune'
CLASSES  = list(_ckpt['classes'])
CLIP_MODEL    = _ckpt.get('clip_model', 'ViT-B-16')
CLIP_PRETRAIN = _ckpt.get('clip_pretrain', 'laion2b_s34b_b88k')

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN)


class Head(nn.Module):
    """Must match geoclip_v7.ipynb exactly."""
    def __init__(self, dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, n_cls))

    def forward(self, x):
        return self.net(x)


visual = clip_model.visual
if MODE == 'finetune':
    visual.load_state_dict(_ckpt['visual'])
head = Head(visual.output_dim, len(CLASSES))
head.load_state_dict(_ckpt['head'])

visual = visual.to(DEVICE).eval()
head   = head.to(DEVICE).eval()
del clip_model
print(f'Ready — {MODE} model, {len(CLASSES)} countries')


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
    """Per-view softmax with horizontal-flip TTA. Averaging the views is
    done by the caller — that is the 'per-location' metric v7 reports."""
    batch = torch.stack([clip_preprocess(v) for v in views]).to(DEVICE)
    e1 = F.normalize(visual(batch), dim=-1)
    e2 = F.normalize(visual(torch.flip(batch, dims=[3])), dim=-1)
    emb = F.normalize((e1 + e2) / 2, dim=-1)
    return F.softmax(head(emb), dim=1).float().cpu().numpy()


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
