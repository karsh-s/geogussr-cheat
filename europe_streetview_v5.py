# %% [markdown]
# # Europe StreetView CNN v5 — Mass-Data + GeoTips Meta-Cue Training
#
# What's new vs v4:
# - **Streaming Mapillary pipeline — no Drive storage needed**: instead
#   of saving images to Drive, only a lightweight *URL manifest* (image
#   id + URL per country, a few hundred MB of JSON for 550k images) is
#   stored on Drive. Images are downloaded on the fly during training
#   into Colab's free ~200 GB local disk cache. Expired Mapillary URLs
#   are automatically re-fetched by image id. The manifest keeps growing
#   in a background thread *while training runs* — up to 10,000 train +
#   3,000 val images per country — and the dataset rescans it at the
#   start of every epoch.
# - **No OCR** — removed entirely.
# - **GeoTips meta-cue head**: every country is tagged with the
#   "dead giveaway" cues catalogued on geotips.net (driving side,
#   bollard style, camera generation, architecture family, vegetation
#   zone, road-line colour). The model learns to predict these cues as
#   an auxiliary task, forcing it to attend to exactly the features
#   human pros use.
# - **Zoom-robust training**: RandomResizedCrop with scale 0.12–1.0
#   simulates everything from a full zoom-in on a bollard/sign to the
#   widest zoom-out. Random perspective + directional crops simulate
#   looking around in every direction; the sky is randomly cropped off
#   so the model never leans on it.
# - **Architecture & flora sensitivity**: multi-crop consistency —
#   in addition to the full frame, a random *detail crop* (a zoomed
#   patch) must predict the same country, teaching fine textures:
#   brickwork, roof tiles, vegetation, road surface.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install albumentations timm mapbox_vector_tile

# %%
import os, json, math, time, random, shutil, threading, re
import concurrent.futures
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
TRAIN_MODE   = True     # False = inference only, loads saved model
COLLECT_MODE = True     # False = skip Mapillary collection (use existing images)

# ── Data targets ────────────────────────────────────────────
TARGET_TRAIN = 4000     # images per country (train)
TARGET_VAL   = 1000     # images per country (val)
MLY_TOKEN    = "YOUR_MAPILLARY_CLIENT_TOKEN_HERE"   # mapillary.com/app/account/developers

DRIVE_ROOT     = '/content/drive/MyDrive/GeoGussrCheat'
TRAIN_MANIFEST = f'{DRIVE_ROOT}/manifest_train_v5.json'   # id+url per country
VAL_MANIFEST   = f'{DRIVE_ROOT}/manifest_val_v5.json'
IMG_CACHE      = '/content/img_cache'      # free ephemeral local disk (~200 GB)
os.makedirs(IMG_CACHE, exist_ok=True)

MODEL_DIR       = f'{DRIVE_ROOT}/models/efficientnet_b4_v5'
BEST_MODEL_PATH = f'{MODEL_DIR}/best_model.pth'
CKPT_PATH       = f'{MODEL_DIR}/checkpoint_latest.pth'
CLASS_MAP_PATH  = f'{MODEL_DIR}/class_to_idx.json'
HISTORY_PATH    = f'{MODEL_DIR}/history.json'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Training ────────────────────────────────────────────────
IMG_SIZE      = 448
BATCH_SIZE    = 28
NUM_EPOCHS    = 16
FREEZE_EPOCHS = 3
LR_HEAD       = 5e-4
LR_FULL       = 4e-5
WEIGHT_DECAY  = 1e-4
LAMBDA_CUE    = 0.30    # weight of GeoTips meta-cue auxiliary loss
LAMBDA_DETAIL = 0.25    # weight of zoomed detail-crop consistency loss
MIXUP_ALPHA   = 0.3
FOCAL_GAMMA   = 2.0
NUM_WORKERS   = 8

# %% [markdown]
# ## 3 · GeoTips meta-cue encoding
#
# Every country is annotated with the giveaway cues from geotips.net.
# Each cue group becomes a block of the multi-hot target vector for the
# auxiliary cue head:
#
# | block | cue | source |
# |---|---|---|
# | 0 | drives on LEFT | geotips: UK, Ireland, Malta, Cyprus |
# | 1–4 | road centre-line colour: white / yellow / none-common / mixed | geotips road lines |
# | 5–9 | bollard family: FR-red-top / ES-orange / IT-black-band / Nordic-flex / Eastern-striped | geotips bollards |
# | 10–14 | architecture family: Nordic-wood / Mediterranean-stucco / Central-masonry / Balkan-mixed / Baltic-Soviet | geotips architecture |
# | 15–18 | vegetation zone: boreal-conifer / temperate-broadleaf / mediterranean-scrub / alpine | flora giveaways |
# | 19 | Google-car gen-4 blur/low camera common | geotips camera meta |
# | 20 | kilometre stones / red-white guardrail ends | geotips misc |

# %%
NUM_CUES = 21

# cue indices
LEFT_DRIVE = 0
LINE_WHITE, LINE_YELLOW, LINE_NONE, LINE_MIXED = 1, 2, 3, 4
BOL_FR, BOL_ES, BOL_IT, BOL_NORDIC, BOL_EAST = 5, 6, 7, 8, 9
ARCH_NORDIC, ARCH_MED, ARCH_CENTRAL, ARCH_BALKAN, ARCH_BALTIC = 10, 11, 12, 13, 14
VEG_BOREAL, VEG_TEMPERATE, VEG_MED, VEG_ALPINE = 15, 16, 17, 18
CAM_LOWGEN, MISC_KM = 19, 20

COUNTRY_CUES = {
    'Albania':          [LINE_MIXED, ARCH_BALKAN, VEG_MED, CAM_LOWGEN],
    'Andorra':          [LINE_WHITE, ARCH_MED, VEG_ALPINE],
    'Austria':          [LINE_WHITE, ARCH_CENTRAL, VEG_ALPINE, BOL_NORDIC],
    'Belgium':          [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE],
    'Bosnia and Herzegovina': [LINE_YELLOW, ARCH_BALKAN, VEG_TEMPERATE, CAM_LOWGEN],
    'Bulgaria':         [LINE_MIXED, ARCH_BALKAN, VEG_TEMPERATE, BOL_EAST],
    'Croatia':          [LINE_WHITE, ARCH_BALKAN, VEG_MED],
    'Cyprus':           [LEFT_DRIVE, LINE_YELLOW, ARCH_MED, VEG_MED],
    'Czech Republic':   [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE, BOL_EAST],
    'Denmark':          [LINE_WHITE, ARCH_NORDIC, VEG_TEMPERATE, BOL_NORDIC],
    'Estonia':          [LINE_WHITE, ARCH_BALTIC, VEG_BOREAL, BOL_NORDIC],
    'Finland':          [LINE_WHITE, ARCH_NORDIC, VEG_BOREAL, BOL_NORDIC],
    'France':           [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE, BOL_FR, MISC_KM],
    'Germany':          [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE],
    'Greece':           [LINE_MIXED, ARCH_MED, VEG_MED, MISC_KM],
    'Hungary':          [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE, BOL_EAST],
    'Iceland':          [LINE_YELLOW, ARCH_NORDIC, VEG_BOREAL],
    'Ireland':          [LEFT_DRIVE, LINE_YELLOW, ARCH_CENTRAL, VEG_TEMPERATE],
    'Italy':            [LINE_WHITE, ARCH_MED, VEG_MED, BOL_IT, MISC_KM],
    'Latvia':           [LINE_WHITE, ARCH_BALTIC, VEG_BOREAL, BOL_EAST],
    'Liechtenstein':    [LINE_WHITE, ARCH_CENTRAL, VEG_ALPINE],
    'Lithuania':        [LINE_WHITE, ARCH_BALTIC, VEG_BOREAL, BOL_EAST],
    'Luxembourg':       [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE],
    'Malta':            [LEFT_DRIVE, LINE_NONE, ARCH_MED, VEG_MED],
    'Monaco':           [LINE_WHITE, ARCH_MED, VEG_MED],
    'Montenegro':       [LINE_YELLOW, ARCH_BALKAN, VEG_MED, CAM_LOWGEN],
    'Netherlands':      [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE],
    'North Macedonia':  [LINE_MIXED, ARCH_BALKAN, VEG_TEMPERATE, CAM_LOWGEN],
    'Norway':           [LINE_YELLOW, ARCH_NORDIC, VEG_BOREAL, BOL_NORDIC],
    'Poland':           [LINE_WHITE, ARCH_CENTRAL, VEG_TEMPERATE, BOL_EAST],
    'Portugal':         [LINE_WHITE, ARCH_MED, VEG_MED, BOL_ES],
    'Romania':          [LINE_WHITE, ARCH_BALKAN, VEG_TEMPERATE, BOL_EAST, CAM_LOWGEN],
    'Serbia':           [LINE_MIXED, ARCH_BALKAN, VEG_TEMPERATE, BOL_EAST, CAM_LOWGEN],
    'Slovakia':         [LINE_WHITE, ARCH_CENTRAL, VEG_ALPINE, BOL_EAST],
    'Slovenia':         [LINE_WHITE, ARCH_CENTRAL, VEG_ALPINE],
    'Spain':            [LINE_WHITE, ARCH_MED, VEG_MED, BOL_ES, MISC_KM],
    'Sweden':           [LINE_WHITE, ARCH_NORDIC, VEG_BOREAL, BOL_NORDIC],
    'Switzerland':      [LINE_WHITE, ARCH_CENTRAL, VEG_ALPINE],
    'United Kingdom':   [LEFT_DRIVE, LINE_MIXED, ARCH_CENTRAL, VEG_TEMPERATE],
    'Turkey':           [LINE_MIXED, ARCH_BALKAN, VEG_MED, MISC_KM],
    'Ukraine':          [LINE_WHITE, ARCH_BALTIC, VEG_TEMPERATE, BOL_EAST, CAM_LOWGEN],
}

def cue_multihot(country):
    v = np.zeros(NUM_CUES, dtype=np.float32)
    for idx in COUNTRY_CUES.get(country, []):
        v[idx] = 1.0
    return v

# Visually confusable pairs (GeoTips "commonly confused") — oversampled
HARD_PAIRS_EXTRA = {
    'Slovenia': 1.5, 'Slovakia': 1.5, 'Czech Republic': 1.3, 'Austria': 1.3,
    'Latvia': 1.5, 'Lithuania': 1.5, 'Estonia': 1.5,
    'Serbia': 1.4, 'Bosnia and Herzegovina': 1.4, 'North Macedonia': 1.4,
    'Montenegro': 1.4, 'Albania': 1.3,
    'Sweden': 1.2, 'Norway': 1.2, 'Finland': 1.2,
    'Portugal': 1.2, 'Spain': 1.1,
}

# %% [markdown]
# ## 4 · Mapillary manifest collector (URLs only — no Drive image storage)

# %%
COUNTRY_BBOX = {
    'Albania': (19.3, 39.6, 21.1, 42.7), 'Andorra': (1.4, 42.4, 1.8, 42.7),
    'Austria': (9.5, 46.4, 17.2, 49.0), 'Belgium': (2.5, 49.5, 6.4, 51.5),
    'Bosnia and Herzegovina': (15.7, 42.6, 19.6, 45.3),
    'Bulgaria': (22.4, 41.2, 28.7, 44.2), 'Croatia': (13.5, 42.4, 19.4, 46.6),
    'Cyprus': (32.2, 34.6, 34.1, 35.7), 'Czech Republic': (12.1, 48.6, 18.9, 51.1),
    'Denmark': (8.1, 54.6, 15.2, 57.8), 'Estonia': (21.8, 57.5, 28.2, 59.7),
    'Finland': (20.0, 59.8, 31.6, 70.1), 'France': (-5.1, 41.3, 9.6, 51.1),
    'Germany': (5.9, 47.3, 15.0, 55.1), 'Greece': (19.4, 34.8, 28.3, 41.8),
    'Hungary': (16.1, 45.7, 22.9, 48.6), 'Iceland': (-24.6, 63.4, -13.5, 66.6),
    'Ireland': (-10.5, 51.4, -6.0, 55.4), 'Italy': (6.6, 36.6, 18.5, 47.1),
    'Latvia': (21.0, 55.7, 28.2, 57.8), 'Liechtenstein': (9.5, 47.0, 9.7, 47.3),
    'Lithuania': (20.9, 53.9, 26.8, 56.4), 'Luxembourg': (5.7, 49.4, 6.5, 50.2),
    'Malta': (14.3, 35.8, 14.6, 36.1), 'Monaco': (7.38, 43.72, 7.44, 43.76),
    'Montenegro': (18.4, 41.9, 20.4, 43.6), 'Netherlands': (3.4, 50.8, 7.2, 53.6),
    'North Macedonia': (20.5, 40.9, 23.0, 42.4), 'Norway': (4.6, 57.9, 31.1, 71.2),
    'Poland': (14.1, 49.0, 24.2, 54.9), 'Portugal': (-9.5, 36.9, -6.2, 42.2),
    'Romania': (22.0, 43.6, 29.7, 48.3), 'Serbia': (19.0, 42.2, 23.0, 46.2),
    'Slovakia': (16.8, 47.7, 22.6, 49.6), 'Slovenia': (13.4, 45.4, 16.6, 46.9),
    'Spain': (-9.3, 35.9, 4.3, 43.8), 'Sweden': (11.1, 55.3, 24.2, 69.1),
    'Switzerland': (5.9, 45.8, 10.5, 47.8), 'Turkey': (26.0, 35.8, 44.8, 42.1),
    'Ukraine': (22.1, 44.4, 40.2, 52.4),
    'United Kingdom': (-8.2, 49.9, 1.8, 60.9),
}

HEADERS = {'Authorization': f'OAuth {MLY_TOKEN}'}
STOP_COLLECTOR = threading.Event()
_search_err_count = 0

# ── Manifests: {country: {image_id: url, ...}} — the ONLY thing on Drive ──
_manifest_lock = threading.Lock()

def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_manifest(manifest, path):
    with _manifest_lock:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(manifest, f)
        shutil.move(tmp, path)

train_manifest = load_manifest(TRAIN_MANIFEST)
val_manifest   = load_manifest(VAL_MANIFEST)

import mapbox_vector_tile

def _deg2tile(lat, lon, z=14):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def _country_tiles(bbox, z=14):
    """All z14 vector-tile coordinates covering the country bbox,
    shuffled so images spread across the whole country."""
    w, sth, e, nth = bbox
    x0, y1 = _deg2tile(sth, w, z)
    x1, y0 = _deg2tile(nth, e, z)
    tiles = [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)
                    for y in range(min(y0, y1), max(y0, y1) + 1)]
    random.shuffle(tiles)
    return tiles

def _tile_image_ids(x, y, z=14):
    """Image ids inside one vector tile (empty list if none/error)."""
    url = (f'https://tiles.mapillary.com/maps/vtp/mly1_public/2/'
           f'{z}/{x}/{y}?access_token={MLY_TOKEN}')
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or not r.content:
            return []
        layers = mapbox_vector_tile.decode(r.content)
        return [str(f['properties']['id'])
                for f in layers.get('image', {}).get('features', [])]
    except Exception as ex:
        global _search_err_count
        _search_err_count += 1
        if _search_err_count <= 5:
            print(f'  [tile error] {x}/{y}: {ex}')
        return []

def _batch_thumb_urls(ids):
    """Resolve thumb URLs for up to 50 image ids in one API call."""
    try:
        r = requests.get('https://graph.mapillary.com/images',
                         headers=HEADERS, timeout=30,
                         params={'image_ids': ','.join(ids),
                                 'fields': 'id,thumb_2048_url'})
        r.raise_for_status()
        return {d['id']: d['thumb_2048_url'] for d in r.json().get('data', [])
                if 'thumb_2048_url' in d}
    except Exception as ex:
        global _search_err_count
        _search_err_count += 1
        if _search_err_count <= 5:
            print(f'  [url-batch error] {ex}')
        return {}

MAX_PER_TILE = 250   # cap ids taken per tile → geographic diversity

def collect_country_urls(country, manifest, target, tiles_per_side=None):
    """Walk random z14 tiles of the country, harvest image ids from the
    vector-tile API, then batch-resolve thumb URLs. URL-only: fast,
    tiny on Drive."""
    entries = manifest.setdefault(country, {})
    other = val_manifest if manifest is train_manifest else train_manifest
    taken = set(other.get(country, {}))
    if len(entries) >= target:
        return 0
    added = 0
    for x, y in _country_tiles(COUNTRY_BBOX[country]):
        if STOP_COLLECTOR.is_set() or len(entries) >= target:
            break
        ids = [i for i in _tile_image_ids(x, y)
               if i not in entries and i not in taken]
        if not ids:
            continue
        if len(ids) > MAX_PER_TILE:
            ids = random.sample(ids, MAX_PER_TILE)
        for j in range(0, len(ids), 50):
            if len(entries) >= target or STOP_COLLECTOR.is_set():
                break
            for iid, url in _batch_thumb_urls(ids[j:j + 50]).items():
                if len(entries) >= target:
                    break
                entries[iid] = url
                added += 1
        time.sleep(0.2)
    return added

def collector_worker(manifest, manifest_path, target, label):
    """Round-robins countries, growing each manifest toward the target.
    Saves to Drive after every country. Runs until done or stopped."""
    while not STOP_COLLECTOR.is_set():
        all_done = True
        for country in COUNTRY_BBOX:
            if STOP_COLLECTOR.is_set():
                return
            have = len(manifest.get(country, {}))
            if have < target:
                all_done = False
                n = collect_country_urls(country, manifest,
                                         min(have + 2000, target))
                if n:
                    save_manifest(manifest, manifest_path)
                    print(f'[collector/{label}] {country}: +{n} URLs '
                          f'(now {have + n})')
        if all_done:
            print(f'[collector/{label}] all countries at target — done.')
            return

# ── On-demand image fetch with URL-expiry recovery ──────────
def _refresh_url(image_id):
    """Mapillary thumb URLs expire; re-fetch a fresh one by image id."""
    try:
        r = requests.get(f'https://graph.mapillary.com/{image_id}',
                         headers=HEADERS,
                         params={'fields': 'thumb_2048_url'}, timeout=15)
        r.raise_for_status()
        return r.json().get('thumb_2048_url')
    except Exception:
        return None

def fetch_image(image_id, url, manifest, country):
    """Return local cache path for an image, downloading if needed.
    None on permanent failure (image removed from Mapillary)."""
    dst = os.path.join(IMG_CACHE, f'{image_id}.jpg')
    if os.path.exists(dst):
        return dst
    from io import BytesIO
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
            if img.width < 300 or img.height < 200:
                return None
            # pre-shrink so the cache holds ~550k images in ~200 GB
            img.thumbnail((1024, 1024))
            img.save(dst, 'JPEG', quality=87)
            return dst
        except Exception:
            if attempt == 0:                     # URL likely expired
                url = _refresh_url(image_id)
                if url is None:
                    return None
                manifest[country][image_id] = url  # heal the manifest
            else:
                return None
    return None

def prefetch(manifest, per_country=None, workers=16, label=''):
    """Warm the local cache in parallel (optional but speeds up epoch 1)."""
    tasks = []
    for country, entries in manifest.items():
        items = list(entries.items())
        if per_country:
            items = items[:per_country]
        tasks += [(iid, url, manifest, country) for iid, url in items]
    random.shuffle(tasks)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fetch_image, *t) for t in tasks]
        for f in tqdm(concurrent.futures.as_completed(futs),
                      total=len(futs), desc=f'prefetch {label}'):
            if f.result():
                done += 1
    print(f'{label}: {done:,}/{len(tasks):,} images in local cache')

# %%
if COLLECT_MODE and MLY_TOKEN.startswith('MLY|'):
    # Phase A: URL floor for every country (search only — takes minutes)
    MIN_FLOOR = 1500
    print(f'Bootstrapping manifests: {MIN_FLOOR} train / '
          f'{MIN_FLOOR // 4} val URLs per country...')
    for c in tqdm(COUNTRY_BBOX):
        collect_country_urls(c, train_manifest, MIN_FLOOR)
        collect_country_urls(c, val_manifest, MIN_FLOOR // 4)
    save_manifest(train_manifest, TRAIN_MANIFEST)
    save_manifest(val_manifest, VAL_MANIFEST)
    _tot_t = sum(len(e) for e in train_manifest.values())
    _tot_v = sum(len(e) for e in val_manifest.values())
    print(f'Manifest totals after bootstrap: {_tot_t:,} train / '
          f'{_tot_v:,} val URLs')
    if _tot_t == 0:
        raise RuntimeError(
            'Bootstrap collected 0 URLs. Most common causes:\n'
            '  1. Invalid/expired MLY_TOKEN — check the [search error] '
            'messages above\n'
            '  2. No internet access in this Colab session\n'
            'Fix the token at mapillary.com/app/account/developers and '
            're-run this cell.')

    # Phase B: keep growing manifests toward full targets in background
    threading.Thread(target=collector_worker,
                     args=(train_manifest, TRAIN_MANIFEST, TARGET_TRAIN,
                           'train'), daemon=True).start()
    threading.Thread(target=collector_worker,
                     args=(val_manifest, VAL_MANIFEST, TARGET_VAL, 'val'),
                     daemon=True).start()
    print('Background URL collectors running.')

    # Phase C: warm the local cache for the floor set so epoch 1 is fast
    prefetch(train_manifest, per_country=MIN_FLOOR, label='train')
    prefetch(val_manifest, per_country=MIN_FLOOR // 4, label='val')
elif COLLECT_MODE:
    print('!! Set MLY_TOKEN to enable collection. Using existing manifests.')

# %% [markdown]
# ## 5 · Augmentation — zoom-in / zoom-out / all directions, no sky

# %%
# Zoom-robust pipeline:
#  - RandomResizedCrop scale 0.12–1.0: 0.12 ≈ full zoom on a bollard or
#    road sign; 1.0 = widest view. Ratio range covers portrait-ish looks
#    up/down and wide looks left/right.
#  - Random top-crop removes 0–30% of the sky so the model can't cheat.
#  - Perspective + rotate simulate camera pitch/yaw when panning.
class SkyCrop(A.ImageOnlyTransform):
    """Randomly crop 0–30% off the top of the frame (removes sky)."""
    def __init__(self, max_frac=0.30, p=0.5):
        super().__init__(p=p)
        self.max_frac = max_frac
    def apply(self, img, **params):
        h = img.shape[0]
        cut = int(h * random.uniform(0.05, self.max_frac))
        return img[cut:, :, :]

train_tf = A.Compose([
    SkyCrop(max_frac=0.30, p=0.5),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE),
                        scale=(0.12, 1.0), ratio=(0.6, 1.7), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.02, 0.08), p=0.3),
    A.Rotate(limit=8, p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(0.25, 0.25),
        A.RandomGamma(gamma_limit=(70, 130)),
        A.CLAHE(clip_limit=2.0),
    ], p=0.6),
    A.OneOf([
        A.GaussNoise(std_range=(0.02, 0.08)),
        A.ImageCompression(quality_range=(45, 85)),
        A.MotionBlur(blur_limit=5),
    ], p=0.4),
    A.RandomFog(p=0.05), A.RandomRain(p=0.05), A.RandomShadow(p=0.15),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Detail crop: an aggressive zoom (8–25% of the frame) used for the
# multi-crop consistency loss — architecture textures, flora, road paint.
detail_tf = A.Compose([
    SkyCrop(max_frac=0.30, p=0.5),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE),
                        scale=(0.08, 0.25), ratio=(0.7, 1.4), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# %% [markdown]
# ## 6 · Streaming dataset — reads the manifest, downloads on demand
#
# Images live in the local cache (`/content/img_cache`). A cache miss
# triggers a download inside the DataLoader worker; expired URLs are
# refreshed by image id. The manifest is re-read every epoch, so URLs
# added by the background collectors join training automatically.

# %%
def manifest_samples(manifest, class_to_idx):
    return [(iid, url, c) for c, entries in manifest.items()
            if c in class_to_idx for iid, url in entries.items()]

class GeoStreamDataset(Dataset):
    def __init__(self, manifest, class_to_idx, tf, detail=False):
        self.manifest, self.class_to_idx = manifest, class_to_idx
        self.tf, self.detail = tf, detail
        self.rescan()

    def rescan(self):
        self.samples = manifest_samples(self.manifest, self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, url, country = self.samples[idx]
        path = fetch_image(iid, url, self.manifest, country)
        if path is None:                      # dead image → random substitute
            return self[random.randrange(len(self.samples))]
        try:
            img = np.array(Image.open(path).convert('RGB'))
        except Exception:
            try: os.remove(path)              # corrupt cache entry
            except OSError: pass
            return self[random.randrange(len(self.samples))]
        label = self.class_to_idx[country]
        cue_t = torch.from_numpy(cue_multihot(country))
        img_t = self.tf(image=img)['image']
        if self.detail:
            det_t = detail_tf(image=img)['image']
            return img_t, det_t, label, cue_t
        return img_t, label, cue_t

# Fix the class list from the manifest
CLASSES = sorted(c for c, e in train_manifest.items() if len(e) >= 100)
if os.path.exists(CLASS_MAP_PATH) and not TRAIN_MODE:
    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)
    CLASSES = sorted(class_to_idx.keys())
else:
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(class_to_idx)
if NUM_CLASSES == 0:
    raise RuntimeError(
        'No countries with >=100 URLs in the train manifest. Run the '
        'collection cell (section 4) successfully before this one — '
        'check for [search error] messages there.')
_n_train     = sum(len(train_manifest.get(c, {})) for c in CLASSES)
print(f'{NUM_CLASSES} countries | {_n_train:,} train URLs in manifest')

def make_loaders():
    """(Re)build loaders — called at each epoch start so URLs added by
    the background collectors are picked up."""
    train_ds = GeoStreamDataset(train_manifest, class_to_idx, train_tf,
                                detail=True)
    val_ds   = GeoStreamDataset(val_manifest, class_to_idx, val_tf)
    counts = {c: 0 for c in class_to_idx}
    for _, _, c in train_ds.samples:
        counts[c] += 1
    weights = [ (1.0 / max(counts[c], 1)) * HARD_PAIRS_EXTRA.get(c, 1.0)
                for _, _, c in train_ds.samples ]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                    replacement=True)
    lkw = dict(num_workers=NUM_WORKERS, pin_memory=True,
               persistent_workers=False, drop_last=False)
    return (train_ds, val_ds,
            DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **lkw),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **lkw))

# %% [markdown]
# ## 7 · Model — EfficientNet-B4 + GeoTips cue head

# %%
class GeoClassifierV5(nn.Module):
    def __init__(self, num_classes, num_cues=NUM_CUES):
        super().__init__()
        base          = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        feat_dim      = 1792

        self.country_head = nn.Sequential(
            nn.Dropout(p=0.45),
            nn.Linear(feat_dim, 896),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(896, num_classes),
        )
        # GeoTips meta-cue head: predicts driving side, bollard family,
        # architecture family, vegetation zone, line colour, camera gen
        self.cue_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 384),
            nn.SiLU(),
            nn.Linear(384, num_cues),
        )

    def _extract(self, x):
        return self.pool(self.backbone(x)).flatten(1)

    def forward(self, x, detail=None):
        feat = self._extract(x)
        out  = {'country': self.country_head(feat),
                'cues':    self.cue_head(feat)}
        if detail is not None:
            out['country_detail'] = self.country_head(self._extract(detail))
        return out


def set_backbone_grad(model, requires_grad):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    for name, p in m.named_parameters():
        if not name.startswith(('country_head', 'cue_head')):
            p.requires_grad = requires_grad

def get_state_dict(model):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    return m.state_dict()

model = GeoClassifierV5(NUM_CLASSES).to(DEVICE)

# ── Resume from checkpoint (with stale-checkpoint protection) ──
START_EPOCH, best_val_acc, _phase2_started = 0, 0.0, False
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def _ckpt_compatible(ckpt, num_classes):
    w = ckpt['model'].get('country_head.4.weight')
    return w is not None and w.shape[0] == num_classes

if TRAIN_MODE and os.path.exists(CKPT_PATH):
    print(f'Found checkpoint: {CKPT_PATH}')
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if not _ckpt_compatible(ckpt, NUM_CLASSES):
        n = ckpt['model']['country_head.4.weight'].shape[0]
        print(f'WARNING: checkpoint has {n} classes, dataset has '
              f'{NUM_CLASSES}. Discarding stale checkpoint.')
        shutil.move(CKPT_PATH, CKPT_PATH + f'.stale_{n}cls.bak')
    else:
        model.load_state_dict(ckpt['model'])
        START_EPOCH     = ckpt['epoch'] + 1
        best_val_acc    = ckpt['best_val_acc']
        _phase2_started = ckpt.get('phase2_started', False)
        history         = ckpt.get('history', history)
        print(f'Resuming from epoch {START_EPOCH} '
              f'(best val acc {best_val_acc:.4f})')

set_backbone_grad(model, requires_grad=_phase2_started)

# %% [markdown]
# ## 8 · Losses

# %%
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=0.1):
        super().__init__()
        self.gamma, self.label_smoothing = gamma, label_smoothing

    def forward(self, logits, targets):
        n = logits.size(1)
        smooth  = self.label_smoothing / (n - 1)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * smooth
        log_p   = F.log_softmax(logits, dim=1)
        focal   = (1 - log_p.exp()) ** self.gamma
        return -(focal * one_hot * log_p).sum(dim=1).mean()

focal    = FocalLoss()
bce_cues = nn.BCEWithLogitsLoss()

def mixup_batch(imgs, details, labels, cues, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return (lam * imgs + (1 - lam) * imgs[idx],
            lam * details + (1 - lam) * details[idx],
            labels, labels[idx],
            lam * cues + (1 - lam) * cues[idx], lam)

# %% [markdown]
# ## 9 · Training loop — collects data + saves every epoch

# %%
scaler = torch.amp.GradScaler('cuda')

def save_checkpoint(epoch, model, best_val_acc, phase2_started, history):
    torch.save({'epoch': epoch, 'model': get_state_dict(model),
                'best_val_acc': best_val_acc,
                'phase2_started': phase2_started, 'history': history},
               CKPT_PATH)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)

def run_validation(model, val_loader):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, cues in val_loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.autocast('cuda'):
                out  = model(imgs)
                loss = focal(out['country'], labels)
            tot_loss += loss.item() * imgs.size(0)
            correct  += (out['country'].argmax(1) == labels).sum().item()
            n        += imgs.size(0)
    return tot_loss / max(n, 1), correct / max(n, 1)

if TRAIN_MODE:
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        # rescan: pick up newly collected images
        train_ds, val_ds, train_loader, val_loader = make_loaders()
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} — '
              f'{len(train_ds):,} train / {len(val_ds):,} val images')

        # phase switch
        if epoch == FREEZE_EPOCHS and not _phase2_started:
            _phase2_started = True
            set_backbone_grad(model, requires_grad=True)
            print('>> Phase 2: unfreezing backbone')
        lr  = LR_HEAD if not _phase2_started else LR_FULL
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=len(train_loader))

        model.train()
        tot_loss, correct, n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for imgs, details, labels, cues in pbar:
            imgs    = imgs.to(DEVICE, non_blocking=True)
            details = details.to(DEVICE, non_blocking=True)
            labels  = labels.to(DEVICE, non_blocking=True)
            cues    = cues.to(DEVICE, non_blocking=True)

            use_mix = _phase2_started and random.random() < 0.5
            if use_mix:
                imgs, details, la, lb, cues_m, lam = mixup_batch(
                    imgs, details, labels, cues)
            with torch.autocast('cuda'):
                out = model(imgs, detail=details)
                if use_mix:
                    L_country = lam * focal(out['country'], la) \
                              + (1 - lam) * focal(out['country'], lb)
                    L_detail  = lam * focal(out['country_detail'], la) \
                              + (1 - lam) * focal(out['country_detail'], lb)
                    L_cue     = bce_cues(out['cues'], cues_m)
                else:
                    L_country = focal(out['country'], labels)
                    L_detail  = focal(out['country_detail'], labels)
                    L_cue     = bce_cues(out['cues'], cues)
                loss = L_country + LAMBDA_DETAIL * L_detail + LAMBDA_CUE * L_cue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            tot_loss += loss.item() * imgs.size(0)
            if not use_mix:
                correct += (out['country'].argmax(1) == labels).sum().item()
                n       += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}')

        train_loss = tot_loss / max(len(train_ds), 1)
        train_acc  = correct / max(n, 1)
        val_loss, val_acc = run_validation(model, val_loader)
        print(f'train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(get_state_dict(model), BEST_MODEL_PATH)
            print(f'>> New best model saved ({val_acc:.4f})')
        save_checkpoint(epoch, model, best_val_acc, _phase2_started, history)

    STOP_COLLECTOR.set()
    print(f'\nTraining complete. Best val acc: {best_val_acc:.4f}')
else:
    print('TRAIN_MODE = False — loading saved model for inference.')
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

# %% [markdown]
# ## 10 · Evaluation — multi-zoom TTA

# %%
eval_model = model
eval_model.eval()

@torch.no_grad()
def predict_multizoom(img_np, model):
    """TTA across zoom levels + flip: full frame, centre 70%, centre 45%.
    Mirrors how you zoom in-game."""
    h, w = img_np.shape[:2]
    views = [img_np]
    for frac in (0.70, 0.45):
        ch, cw = int(h * frac), int(w * frac)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        views.append(img_np[y0:y0 + ch, x0:x0 + cw])
    probs = None
    for v in views:
        t = val_tf(image=v)['image'].unsqueeze(0).to(DEVICE)
        for x in (t, torch.flip(t, dims=[3])):
            p = F.softmax(model(x)['country'], dim=1)
            probs = p if probs is None else probs + p
    return (probs / (2 * len(views)))[0]

def evaluate_split(model, manifest, max_per_country=None):
    correct, total, per_c = 0, 0, {}
    for country, entries in tqdm(manifest.items()):
        if country not in class_to_idx:
            continue
        items = list(entries.items())
        if max_per_country:
            items = random.sample(items, min(max_per_country, len(items)))
        c_ok, c_n = 0, 0
        for iid, url in items:
            path = fetch_image(iid, url, manifest, country)
            if path is None:
                continue
            img   = np.array(Image.open(path).convert('RGB'))
            probs = predict_multizoom(img, model)
            if idx_to_class[int(probs.argmax())] == country:
                c_ok += 1
            c_n += 1
        if c_n:
            per_c[country] = c_ok / c_n
            correct += c_ok
            total   += c_n
    print(f'\nOverall accuracy: {correct / total:.4f} ({correct}/{total})')
    for c, a in sorted(per_c.items(), key=lambda kv: kv[1]):
        print(f'  {c:28s} {a:.3f}')
    return per_c

# Validate (cap per-country for speed; raise/remove for the full 3k run)
per_country_acc = evaluate_split(eval_model, val_manifest, max_per_country=300)

# %% [markdown]
# ## 11 · Cue-head introspection — what giveaways does the model see?

# %%
CUE_NAMES = ['LEFT drive', 'line:white', 'line:yellow', 'line:none',
             'line:mixed', 'bollard:FR', 'bollard:ES', 'bollard:IT',
             'bollard:Nordic', 'bollard:East', 'arch:Nordic', 'arch:Med',
             'arch:Central', 'arch:Balkan', 'arch:Baltic', 'veg:boreal',
             'veg:temperate', 'veg:mediterranean', 'veg:alpine',
             'camera:low-gen', 'misc:km-stones']

@torch.no_grad()
def explain_image(path, model, topk=3):
    img   = np.array(Image.open(path).convert('RGB'))
    t     = val_tf(image=img)['image'].unsqueeze(0).to(DEVICE)
    out   = model(t)
    probs = F.softmax(out['country'], dim=1)[0]
    cues  = torch.sigmoid(out['cues'])[0]
    top   = probs.topk(topk)
    print(f'{os.path.basename(path)}')
    for p, i in zip(top.values, top.indices):
        print(f'  {idx_to_class[int(i)]:25s} {float(p) * 100:5.1f}%')
    strong = [(CUE_NAMES[i], float(c)) for i, c in enumerate(cues) if c > 0.5]
    print('  giveaway cues:',
          ', '.join(f'{n} ({c:.0%})' for n, c in
                    sorted(strong, key=lambda x: -x[1])) or 'none detected')

_val_items = manifest_samples(val_manifest, class_to_idx)
for iid, url, country in random.sample(_val_items, min(6, len(_val_items))):
    path = fetch_image(iid, url, val_manifest, country)
    if path:
        print(f'[true: {country}]')
        explain_image(path, eval_model)

# %% [markdown]
# ## 12 · Training curves

# %%
if history['train_loss']:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history['train_loss'], label='train')
    ax[0].plot(history['val_loss'], label='val')
    ax[0].set_title('Loss'); ax[0].legend()
    ax[1].plot(history['train_acc'], label='train')
    ax[1].plot(history['val_acc'], label='val')
    ax[1].set_title('Accuracy'); ax[1].legend()
    plt.tight_layout(); plt.show()
