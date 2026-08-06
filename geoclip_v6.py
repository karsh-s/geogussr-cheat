# %% [markdown]
# # GeoCLIP v6 — Your Own StreetCLIP
#
# Builds a CLIP-style geolocation model from your Mapillary manifests:
#
# 1. **CLIP backbone** (ViT-B/16, LAION-2B pretrained via open_clip).
#    Instead of ImageNet features, we start from vision-language
#    features that already understand scenes, signage and text.
# 2. **GeoTips text anchors**: each country is described by several
#    captions built from its geotips.net giveaways ("a street-level
#    photo in France, white road lines, red-topped bollards, ...").
#    The image encoder is fine-tuned so images embed close to their
#    country's caption anchors — exactly how StreetCLIP was trained,
#    but on YOUR data with YOUR cue vocabulary.
# 3. **Data cleaning (#2)**: base CLIP itself filters junk before
#    training — dashboards, interiors, blur, close-ups. Verdicts are
#    cached on Drive so cleaning runs once.
# 4. **Giveaway fusion (#3)**: if the giveaway crop model exists, its
#    per-object evidence is fused at inference.
# 5. **Incremental data (#4)**: background collectors keep growing the
#    same v5 manifests; the dataset rescans each epoch; training
#    resumes from checkpoints — so accuracy keeps climbing session
#    after session without retraining from scratch.
# 6. **Zero-shot before/after**: the notebook measures zero-shot
#    accuracy BEFORE fine-tuning so you can see exactly what your
#    training adds. (#5, evidence accumulation, lives in
#    geoguessr_live.py and multiplies whatever single-frame accuracy
#    this model reaches.)

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install open_clip_torch albumentations mapbox_vector_tile ultralytics

# %%
import os, json, math, time, random, shutil, threading
import concurrent.futures
from pathlib import Path
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import open_clip

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
TRAIN_MODE   = True     # False = inference only (loads saved model)
COLLECT_MODE = True     # False = stop growing the manifests
CLEAN_MODE   = True     # False = skip junk filtering (use all images)

MLY_TOKEN = "YOUR_MAPILLARY_CLIENT_TOKEN_HERE"
HEADERS   = {'Authorization': f'OAuth {MLY_TOKEN}'}

DRIVE_ROOT     = '/content/drive/MyDrive/GeoGussrCheat'
TRAIN_MANIFEST = f'{DRIVE_ROOT}/manifest_train_v5.json'   # shared with v5
VAL_MANIFEST   = f'{DRIVE_ROOT}/manifest_val_v5.json'
CLEAN_INDEX    = f'{DRIVE_ROOT}/clean_index_v6.json'      # junk verdicts
IMG_CACHE      = '/content/img_cache'
os.makedirs(IMG_CACHE, exist_ok=True)

MODEL_DIR       = f'{DRIVE_ROOT}/models/geoclip_v6'
BEST_MODEL_PATH = f'{MODEL_DIR}/best_model.pth'
CKPT_PATH       = f'{MODEL_DIR}/checkpoint_latest.pth'
CLASS_MAP_PATH  = f'{MODEL_DIR}/class_to_idx.json'
HISTORY_PATH    = f'{MODEL_DIR}/history.json'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data ────────────────────────────────────────────────────
TARGET_TRAIN = 6000     # collectors keep growing manifests to this
TARGET_VAL   = 1500
JUNK_MARGIN  = 0.12     # higher = stricter junk filtering

# ── Training ────────────────────────────────────────────────
CLIP_MODEL    = 'ViT-B-16'
CLIP_PRETRAIN = 'laion2b_s34b_b88k'
IMG_SIZE      = 224
BATCH_SIZE    = 256
NUM_EPOCHS    = 10
LR_VISUAL     = 1e-5     # image encoder (gentle — don't wreck pretraining)
LR_HEAD       = 1e-3     # logit scale / projection
WEIGHT_DECAY  = 0.1
LABEL_SMOOTH  = 0.1
NUM_WORKERS   = 8

# %% [markdown]
# ## 3 · Manifests + background collector (same pipeline as v5)

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

STOP_COLLECTOR = threading.Event()
_search_err_count = 0
_manifest_lock = threading.Lock()
import mapbox_vector_tile

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(obj, path):
    with _manifest_lock:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        shutil.move(tmp, path)

train_manifest = load_json(TRAIN_MANIFEST, {})
val_manifest   = load_json(VAL_MANIFEST, {})
assert train_manifest, 'Run the v5 collection first — manifests are shared.'

def _deg2tile(lat, lon, z=14):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def _country_tiles(bbox, z=14):
    w, sth, e, nth = bbox
    x0, y1 = _deg2tile(sth, w, z)
    x1, y0 = _deg2tile(nth, e, z)
    tiles = [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)
                    for y in range(min(y0, y1), max(y0, y1) + 1)]
    random.shuffle(tiles)
    return tiles

def _tile_image_ids(x, y, z=14):
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
    try:
        r = requests.get('https://graph.mapillary.com/images',
                         headers=HEADERS, timeout=30,
                         params={'image_ids': ','.join(ids),
                                 'fields': 'id,thumb_2048_url'})
        r.raise_for_status()
        return {d['id']: d['thumb_2048_url'] for d in r.json().get('data', [])
                if 'thumb_2048_url' in d}
    except Exception:
        return {}

def collect_country_urls(country, manifest, target):
    entries = manifest.setdefault(country, {})
    other = val_manifest if manifest is train_manifest else train_manifest
    taken = set(other.get(country, {}))
    added = 0
    for x, y in _country_tiles(COUNTRY_BBOX[country]):
        if STOP_COLLECTOR.is_set() or len(entries) >= target:
            break
        ids = [i for i in _tile_image_ids(x, y)
               if i not in entries and i not in taken]
        if len(ids) > 250:
            ids = random.sample(ids, 250)
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

def collector_worker(manifest, path, target, label):
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
                    save_json(manifest, path)
                    print(f'[collector/{label}] {country}: +{n} '
                          f'(now {have + n})')
        if all_done:
            return

if COLLECT_MODE and MLY_TOKEN.startswith('MLY|'):
    threading.Thread(target=collector_worker,
                     args=(train_manifest, TRAIN_MANIFEST, TARGET_TRAIN,
                           'train'), daemon=True).start()
    threading.Thread(target=collector_worker,
                     args=(val_manifest, VAL_MANIFEST, TARGET_VAL, 'val'),
                     daemon=True).start()
    print('Background collectors growing manifests toward '
          f'{TARGET_TRAIN}/{TARGET_VAL} per country.')

def _refresh_url(image_id):
    try:
        r = requests.get(f'https://graph.mapillary.com/{image_id}',
                         headers=HEADERS,
                         params={'fields': 'thumb_2048_url'}, timeout=15)
        r.raise_for_status()
        return r.json().get('thumb_2048_url')
    except Exception:
        return None

def fetch_image(image_id, url, manifest, country):
    dst = os.path.join(IMG_CACHE, f'{image_id}.jpg')
    if os.path.exists(dst):
        return dst
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
            if img.width < 300 or img.height < 200:
                return None
            img.thumbnail((1024, 1024))
            img.save(dst, 'JPEG', quality=87)
            return dst
        except Exception:
            if attempt == 0:
                url = _refresh_url(image_id)
                if url is None:
                    return None
                manifest[country][image_id] = url
            else:
                return None
    return None

# %% [markdown]
# ## 4 · Load CLIP

# %%
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN)
tokenizer  = open_clip.get_tokenizer(CLIP_MODEL)
clip_model = clip_model.to(DEVICE)
print(f'{CLIP_MODEL} ({CLIP_PRETRAIN}) loaded — '
      f'{sum(p.numel() for p in clip_model.parameters())/1e6:.0f}M params')

# %% [markdown]
# ## 5 · GeoTips captions — the text anchors
#
# Several caption templates per country, each weaving in that country's
# geotips.net giveaways. The country's anchor embedding is the average
# of its caption embeddings (prompt ensembling, like the CLIP paper).

# %%
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

def country_captions(country):
    h = COUNTRY_HINTS.get(country, 'typical local scenery')
    return [t.format(c=country, h=h) for t in CAPTION_TEMPLATES]

CLASSES = sorted(c for c in train_manifest if len(train_manifest[c]) >= 100)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(CLASSES)
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(class_to_idx, f, indent=2)
print(f'{NUM_CLASSES} countries')

@torch.no_grad()
def build_text_anchors(model):
    """(NUM_CLASSES, dim) matrix of normalised caption-ensemble embeddings."""
    anchors = []
    for c in CLASSES:
        toks = tokenizer(country_captions(c)).to(DEVICE)
        emb  = model.encode_text(toks)
        emb  = F.normalize(emb, dim=-1).mean(0)
        anchors.append(F.normalize(emb, dim=-1))
    return torch.stack(anchors)

text_anchors = build_text_anchors(clip_model)
print('Text anchors built:', tuple(text_anchors.shape))

# %% [markdown]
# ## 6 · Data cleaning (#2) — CLIP filters its own training data
#
# Base CLIP scores every image against "good street view" vs junk
# captions (dashboard, interior, blur, close-up...). Images where a
# junk caption beats the good caption by JUNK_MARGIN are dropped.
# Verdicts cache to Drive — cleaning only ever runs once per image.

# %%
GOOD_PROMPTS = [
    'an outdoor street-level photo of a road, street or countryside',
    'a photo taken from a vehicle showing a road and surroundings',
]
JUNK_PROMPTS = [
    'a car dashboard or windshield interior',
    'an indoor photo inside a building',
    'a blurry unrecognisable photo',
    'a close-up photo of a single object',
    'a completely dark photo taken at night',
    'a photo of the sky',
]

clean_index = load_json(CLEAN_INDEX, {})   # {image_id: 0 junk / 1 clean}

@torch.no_grad()
def _clean_scores(model):
    toks = tokenizer(GOOD_PROMPTS + JUNK_PROMPTS).to(DEVICE)
    emb  = F.normalize(model.encode_text(toks), dim=-1)
    return emb  # (n_good + n_junk, dim)

def clean_manifest(manifest, model, label=''):
    probe = _clean_scores(model)
    n_g   = len(GOOD_PROMPTS)
    todo  = [(iid, url, c) for c, e in manifest.items() if c in class_to_idx
             for iid, url in e.items() if iid not in clean_index]
    if not todo:
        print(f'{label}: nothing new to clean.')
        return
    model.eval()
    B = 512                       # GPU batch
    FETCH_WORKERS = 128           # parallel downloads — the real bottleneck
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS)

    def _load(args):
        iid, url, c = args
        p = fetch_image(iid, url, manifest, c)
        if p is None:
            return iid, None, 'fetch_fail'   # network issue — do NOT blacklist
        try:
            return iid, clip_preprocess(Image.open(p).convert('RGB')), 'ok'
        except Exception:
            return iid, None, 'corrupt'

    for k in tqdm(range(0, len(todo), B), desc=f'clean {label}'):
        batch, metas = [], []
        for iid, tensor, status in pool.map(_load, todo[k:k + B]):
            if status == 'corrupt':
                clean_index[iid] = 0
            elif status == 'ok':
                batch.append(tensor)
                metas.append(iid)
            # fetch_fail: leave unjudged so a later run retries it
        if not batch:
            continue
        with torch.no_grad(), torch.autocast('cuda'):
            emb = F.normalize(clip_model.encode_image(
                torch.stack(batch).to(DEVICE)), dim=-1)
            sims = emb @ probe.T          # (B, n_good+n_junk)
        good = sims[:, :n_g].max(1).values
        junk = sims[:, n_g:].max(1).values
        for iid, g, j in zip(metas, good.tolist(), junk.tolist()):
            clean_index[iid] = int(j - g < JUNK_MARGIN)
        if k % (B * 10) == 0:
            save_json(clean_index, CLEAN_INDEX)
    pool.shutdown(wait=False)
    save_json(clean_index, CLEAN_INDEX)
    kept = sum(clean_index.values())
    print(f'{label}: {kept:,}/{len(clean_index):,} images kept '
          f'({100 * kept / max(len(clean_index), 1):.1f}%)')

if CLEAN_MODE and TRAIN_MODE:
    clean_manifest(train_manifest, clip_model, 'train')
    clean_manifest(val_manifest, clip_model, 'val')

def is_clean(iid):
    return (not CLEAN_MODE) or clean_index.get(iid, 1) == 1

# %% [markdown]
# ## 7 · Dataset

# %%
from torchvision import transforms as T

train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.3, 1.0), ratio=(0.7, 1.4)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.2, 0.02),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
])
val_tf = clip_preprocess

class GeoClipDataset(Dataset):
    def __init__(self, manifest, tf, train=True):
        self.manifest, self.tf, self.train = manifest, tf, train
        self.rescan()

    def rescan(self):
        self.samples = [(iid, url, c) for c, e in self.manifest.items()
                        if c in class_to_idx
                        for iid, url in e.items() if is_clean(iid)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, url, country = self.samples[idx]
        path = fetch_image(iid, url, self.manifest, country)
        if path is None:
            return self[random.randrange(len(self.samples))]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            return self[random.randrange(len(self.samples))]
        return self.tf(img), class_to_idx[country]

def make_loaders():
    train_ds = GeoClipDataset(train_manifest, train_tf)
    val_ds   = GeoClipDataset(val_manifest, val_tf, train=False)
    counts = {}
    for _, _, c in train_ds.samples:
        counts[c] = counts.get(c, 0) + 1
    weights = [1.0 / counts[c] for _, _, c in train_ds.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    lkw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (train_ds, val_ds,
            DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **lkw),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **lkw))

# %% [markdown]
# ## 8 · Baseline: zero-shot accuracy BEFORE fine-tuning
#
# This is what pretrained CLIP + GeoTips captions gets for free.

# %%
@torch.no_grad()
def evaluate(model, anchors, loader, max_batches=None):
    model.eval()
    correct = n = 0
    for bi, (imgs, labels) in enumerate(tqdm(loader, leave=False)):
        if max_batches and bi >= max_batches:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.autocast('cuda'):
            emb = F.normalize(model.encode_image(imgs), dim=-1)
        pred = (emb @ anchors.T).argmax(1)
        correct += (pred == labels).sum().item()
        n += imgs.size(0)
    return correct / max(n, 1)

_, _, _tl, val_loader = make_loaders()
zs_acc = evaluate(clip_model, text_anchors, val_loader, max_batches=40)
print(f'ZERO-SHOT val accuracy (no training): {zs_acc:.4f}')

# %% [markdown]
# ## 9 · Fine-tune the image encoder against the text anchors
#
# Text encoder stays frozen (the anchors are fixed geography knowledge);
# the image encoder learns to project YOUR imagery onto them. Gentle LR
# + high weight decay preserves the pretrained features.

# %%
for p in clip_model.transformer.parameters():        # freeze text tower
    p.requires_grad = False
if hasattr(clip_model, 'token_embedding'):
    clip_model.token_embedding.weight.requires_grad = False

logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=DEVICE))

START_EPOCH, best_val_acc = 0, zs_acc
history = load_json(HISTORY_PATH, {'train_acc': [], 'val_acc': []})

if TRAIN_MODE and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    clip_model.load_state_dict(ckpt['model'])
    logit_scale.data = ckpt['logit_scale']
    START_EPOCH  = ckpt['epoch'] + 1
    best_val_acc = ckpt['best_val_acc']
    history      = ckpt.get('history', history)
    text_anchors = build_text_anchors(clip_model)
    print(f'Resumed from epoch {START_EPOCH} (best {best_val_acc:.4f})')

if TRAIN_MODE:
    scaler = torch.amp.GradScaler('cuda')
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        train_ds, val_ds, train_loader, val_loader = make_loaders()
        print(f'\nEpoch {epoch + 1}/{NUM_EPOCHS} — '
              f'{len(train_ds):,} clean train / {len(val_ds):,} val')
        opt = torch.optim.AdamW([
            {'params': clip_model.visual.parameters(), 'lr': LR_VISUAL},
            {'params': [logit_scale], 'lr': LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[LR_VISUAL, LR_HEAD],
            total_steps=len(train_loader))

        clip_model.train()
        correct = n = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.autocast('cuda'):
                emb    = F.normalize(clip_model.encode_image(imgs), dim=-1)
                logits = logit_scale.exp() * emb @ text_anchors.T
                loss   = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}',
                             acc=f'{correct / n:.3f}')

        val_acc = evaluate(clip_model, text_anchors, val_loader,
                           max_batches=60)
        print(f'train acc {correct / n:.4f} | val acc {val_acc:.4f} '
              f'(zero-shot was {zs_acc:.4f})')
        history['train_acc'].append(correct / n)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': clip_model.state_dict(),
                        'logit_scale': logit_scale.data,
                        'classes': CLASSES}, BEST_MODEL_PATH)
            print(f'>> best saved ({val_acc:.4f})')
        torch.save({'epoch': epoch, 'model': clip_model.state_dict(),
                    'logit_scale': logit_scale.data,
                    'best_val_acc': best_val_acc,
                    'history': history}, CKPT_PATH)
        save_json(history, HISTORY_PATH)
    STOP_COLLECTOR.set()
    print(f'\nDone. Best val acc: {best_val_acc:.4f} '
          f'(vs {zs_acc:.4f} zero-shot)')
else:
    ckpt = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    clip_model.load_state_dict(ckpt['model'])
    logit_scale.data = ckpt['logit_scale']
    clip_model.eval()
    text_anchors = build_text_anchors(clip_model)
    print('Loaded fine-tuned GeoCLIP.')

# %% [markdown]
# ## 10 · Full evaluation — per-country, with flip TTA

# %%
@torch.no_grad()
def predict_probs(img_pil, model=clip_model):
    t = val_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.autocast('cuda'):
        e1 = F.normalize(model.encode_image(t), dim=-1)
        e2 = F.normalize(model.encode_image(torch.flip(t, dims=[3])), dim=-1)
    logits = logit_scale.exp() * ((e1 + e2) / 2) @ text_anchors.T
    return F.softmax(logits, dim=1)[0].float().cpu().numpy()

def evaluate_per_country(manifest, max_per_country=200):
    per_c, correct, total = {}, 0, 0
    for country, entries in tqdm(sorted(manifest.items())):
        if country not in class_to_idx:
            continue
        items = [(i, u) for i, u in entries.items() if is_clean(i)]
        items = random.sample(items, min(max_per_country, len(items)))
        ok = n = 0
        for iid, url in items:
            p = fetch_image(iid, url, manifest, country)
            if p is None:
                continue
            probs = predict_probs(Image.open(p).convert('RGB'))
            ok += idx_to_class[int(probs.argmax())] == country
            n  += 1
        if n:
            per_c[country] = ok / n
            correct += ok; total += n
    print(f'\nOverall: {correct / total:.4f} ({correct}/{total})')
    for c, a in sorted(per_c.items(), key=lambda kv: kv[1]):
        print(f'  {c:28s} {a:.3f}')
    return per_c

per_country = evaluate_per_country(val_manifest)

# %% [markdown]
# ## 11 · Fusion with the giveaway detector (#3)
#
# If you've trained the giveaway crop model (giveaway_detector notebook),
# this cell fuses its per-object evidence with GeoCLIP's scene score.
# Skipped gracefully if that model isn't on Drive yet.

# %%
GIVEAWAY_MODEL = f'{DRIVE_ROOT}/models/giveaway_v1/best_crop_model.pth'
GIVEAWAY_MAP   = f'{DRIVE_ROOT}/models/giveaway_v1/class_to_idx.json'
fusion_ready   = os.path.exists(GIVEAWAY_MODEL) and os.path.exists(GIVEAWAY_MAP)

if fusion_ready:
    from ultralytics import YOLOWorld
    from torchvision import models as tv_models

    GIVEAWAY_PROMPTS = [
        'bollard', 'road sign', 'traffic sign', 'direction signpost',
        'license plate', 'utility pole', 'street lamp', 'guardrail',
        'road marking', 'fire hydrant', 'postbox', 'telephone booth',
        'bus stop', 'chimney', 'roof', 'fence', 'kilometre stone',
        'house facade',
    ]
    OBJ_TO_IDX = {n: i for i, n in enumerate(GIVEAWAY_PROMPTS)}
    yw = YOLOWorld('yolov8l-worldv2.pt')
    yw.set_classes(GIVEAWAY_PROMPTS)

    with open(GIVEAWAY_MAP) as f:
        gw_class_to_idx = json.load(f)

    class CropClassifier(nn.Module):
        def __init__(self, num_classes, num_obj=len(GIVEAWAY_PROMPTS)):
            super().__init__()
            base = tv_models.efficientnet_b0(weights=None)
            self.backbone = base.features
            self.pool     = nn.AdaptiveAvgPool2d(1)
            self.obj_emb  = nn.Embedding(num_obj, 32)
            self.head = nn.Sequential(
                nn.Dropout(0.35), nn.Linear(1280 + 32, 512), nn.SiLU(),
                nn.Dropout(0.25), nn.Linear(512, num_classes))
        def forward(self, x, obj_idx):
            f = self.pool(self.backbone(x)).flatten(1)
            return self.head(torch.cat([f, self.obj_emb(obj_idx)], dim=1))

    crop_model = CropClassifier(len(gw_class_to_idx)).to(DEVICE)
    crop_model.load_state_dict(torch.load(GIVEAWAY_MODEL,
                                          map_location=DEVICE))
    crop_model.eval()
    crop_norm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    print('Giveaway fusion armed.')
else:
    print('Giveaway model not on Drive yet — predict_fused uses '
          'GeoCLIP alone.')

@torch.no_grad()
def predict_fused(img_pil, verbose=True):
    img_np = np.array(img_pil)
    logp = np.log(predict_probs(img_pil) + 1e-12)

    if fusion_ready:
        res = yw.predict(img_np, conf=0.15, verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if x2 - x1 < 40 or y2 - y1 < 40:
                continue
            cname = GIVEAWAY_PROMPTS[int(box.cls[0])]
            h, w = img_np.shape[:2]
            px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
            crop = Image.fromarray(
                img_np[max(0, y1 - py):min(h, y2 + py),
                       max(0, x1 - px):min(w, x2 + px)])
            t = crop_norm(crop).unsqueeze(0).to(DEVICE)
            o = torch.tensor([OBJ_TO_IDX[cname]], device=DEVICE)
            pc = F.softmax(crop_model(t, o), dim=1)[0].cpu().numpy()
            # map giveaway classes onto GeoCLIP class order
            p = np.full(NUM_CLASSES, 1e-9)
            for c, i in gw_class_to_idx.items():
                if c in class_to_idx:
                    p[class_to_idx[c]] = pc[i]
            p /= p.sum()
            ent = -(p * np.log(p + 1e-12)).sum() / math.log(NUM_CLASSES)
            logp += float(box.conf[0]) * (1 - ent) * np.log(p + 1e-12)

    probs = np.exp(logp - logp.max()); probs /= probs.sum()
    top = np.argsort(probs)[::-1][:5]
    if verbose:
        for i in top:
            print(f'  {idx_to_class[i]:25s} {probs[i] * 100:5.1f}%')
    return [(idx_to_class[i], float(probs[i])) for i in top]

# Demo on random val images
val_items = [(i, u, c) for c, e in val_manifest.items()
             if c in class_to_idx for i, u in e.items()]
for iid, url, country in random.sample(val_items, 4):
    p = fetch_image(iid, url, val_manifest, country)
    if p:
        print(f'\n=== true: {country} ===')
        predict_fused(Image.open(p).convert('RGB'))
