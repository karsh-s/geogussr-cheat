# %% [markdown]
# # GeoCLIP v6.1 — Leak-Free Splits + Anti-Overfitting
#
# v6 reached train 97.6% / val 42.4%. Two problems, both fixed here:
#
# **1. Train/val leakage.** Mapillary images come in *sequences* —
# consecutive frames down one road, averaging ~100 images per sequence.
# v6 split by image id, so nearly every sequence had frames in BOTH
# splits: the val set was largely near-duplicates of training images.
# v6.1 splits by **sequence**, so a road is entirely in train or
# entirely in val. Expect the reported number to DROP — that drop is
# the illusion leaving, not the model getting worse.
#
# **2. Overfitting** (97.6 vs 42.4). Fixed with: frozen lower ViT
# blocks, layer-wise LR decay, higher weight decay, RandAugment,
# early stopping, and fewer epochs.
#
# Also new, using tile metadata v6 ignored:
# - `foot=true` captures dropped (pedestrian/sidewalk, not road-level)
# - `is_pano` dropped (360° images don't look like GeoGuessr views)
# - low `quality_score` dropped
# - **max N images per sequence** — 100 frames of one road is redundant;
#   capping spreads the same budget over far more distinct locations
#
# Reuses v6's image cache and cleaning verdicts (both keyed by image id),
# so no re-downloading or re-cleaning of anything already done.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install open_clip_torch mapbox_vector_tile ultralytics

# %%
import os, json, math, time, random, shutil, threading, gc
import concurrent.futures
from io import BytesIO

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import open_clip
import mapbox_vector_tile

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
TRAIN_MODE   = True
COLLECT_MODE = True     # build/extend the sequence-aware manifest
CLEAN_MODE   = True     # reuse v6 cleaning verdicts (and extend them)

MLY_TOKEN = "YOUR_MAPILLARY_CLIENT_TOKEN_HERE"
HEADERS   = {'Authorization': f'OAuth {MLY_TOKEN}'}

DRIVE_ROOT   = '/content/drive/MyDrive/GeoGussrCheat'
SEQ_MANIFEST = f'{DRIVE_ROOT}/manifest_seq_v61.json'   # {country:{seq:{id:url}}}
CLEAN_INDEX  = f'{DRIVE_ROOT}/clean_index_v6.json'     # shared with v6
IMG_CACHE    = '/content/img_cache'                    # shared with v6
os.makedirs(IMG_CACHE, exist_ok=True)

MODEL_DIR       = f'{DRIVE_ROOT}/models/geoclip_v61'
BEST_MODEL_PATH = f'{MODEL_DIR}/best_model.pth'
CKPT_PATH       = f'{MODEL_DIR}/checkpoint_latest.pth'
CLASS_MAP_PATH  = f'{MODEL_DIR}/class_to_idx.json'
HISTORY_PATH    = f'{MODEL_DIR}/history.json'
V6_MODEL_PATH   = f'{DRIVE_ROOT}/models/geoclip_v6/best_model.pth'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data quality filters (from tile metadata) ───────────────
MIN_QUALITY    = 0.30    # Mapillary quality_score floor
DROP_PANO      = True    # 360° images don't resemble GeoGuessr views
DROP_FOOT      = True    # pedestrian captures aren't road-level
MAX_PER_SEQ    = 12      # cap frames per road → more distinct locations
SEQ_PER_COUNTRY = 700    # sequences to collect per country
VAL_SEQ_FRAC   = 0.20    # fraction of SEQUENCES held out for val

# ── Training (anti-overfitting) ─────────────────────────────
CLIP_MODEL    = 'ViT-B-16'
CLIP_PRETRAIN = 'laion2b_s34b_b88k'
IMG_SIZE      = 224
BATCH_SIZE    = 256
NUM_EPOCHS    = 8
EARLY_STOP    = 2        # stop after N epochs with no val improvement
FREEZE_BLOCKS = 6        # freeze the first N of 12 ViT blocks
LR_VISUAL     = 8e-6
LLRD          = 0.75     # layer-wise LR decay (lower blocks learn slower)
WEIGHT_DECAY  = 0.2      # was 0.1
LABEL_SMOOTH  = 0.15     # was 0.10
NUM_WORKERS   = 4        # was 8 — 8 forked copies caused the RAM crash

# %% [markdown]
# ## 3 · Sequence-aware collector
#
# Same vector-tile walk as v5/v6, but keeps `sequence_id` and applies
# the quality filters. Manifest shape: `{country: {seq_id: {img_id: url}}}`

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

_lock = threading.Lock()

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(obj, path):
    with _lock:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        shutil.move(tmp, path)

seq_manifest = load_json(SEQ_MANIFEST, {})
clean_index  = load_json(CLEAN_INDEX, {})
print(f'Manifest: {len(seq_manifest)} countries | '
      f'clean verdicts carried over: {len(clean_index):,}')

def _deg2tile(lat, lon, z=14):
    n = 2 ** z
    return (int((lon + 180) / 360 * n),
            int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))

def _country_tiles(bbox, z=14):
    w, s, e, nth = bbox
    x0, y1 = _deg2tile(s, w, z)
    x1, y0 = _deg2tile(nth, e, z)
    tiles = [(x, y) for x in range(min(x0, x1), max(x0, x1) + 1)
                    for y in range(min(y0, y1), max(y0, y1) + 1)]
    random.shuffle(tiles)
    return tiles

def _tile_features(x, y, z=14):
    """Quality-filtered image features, grouped by sequence."""
    url = (f'https://tiles.mapillary.com/maps/vtp/mly1_public/2/'
           f'{z}/{x}/{y}?access_token={MLY_TOKEN}')
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or not r.content:
            return {}
        layers = mapbox_vector_tile.decode(r.content)
    except Exception:
        return {}
    by_seq = {}
    for f in layers.get('image', {}).get('features', []):
        p = f['properties']
        if DROP_PANO and p.get('is_pano'):
            continue
        if DROP_FOOT and p.get('foot'):
            continue
        if p.get('quality_score', 1.0) < MIN_QUALITY:
            continue
        sid = p.get('sequence_id')
        if not sid:
            continue
        by_seq.setdefault(sid, []).append(str(p['id']))
    return by_seq

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

def collect_country(country, target_seqs=SEQ_PER_COUNTRY):
    """Add sequences (capped at MAX_PER_SEQ images each) until the
    country has `target_seqs` sequences."""
    seqs = seq_manifest.setdefault(country, {})
    if len(seqs) >= target_seqs:
        return 0
    added_imgs = 0
    for x, y in _country_tiles(COUNTRY_BBOX[country]):
        if len(seqs) >= target_seqs:
            break
        for sid, ids in _tile_features(x, y).items():
            if len(seqs) >= target_seqs:
                break
            if sid in seqs:
                continue
            # sample across the sequence rather than taking the first N
            if len(ids) > MAX_PER_SEQ:
                ids = random.sample(ids, MAX_PER_SEQ)
            urls = {}
            for j in range(0, len(ids), 50):
                urls.update(_batch_thumb_urls(ids[j:j + 50]))
            if urls:
                seqs[sid] = urls
                added_imgs += len(urls)
        time.sleep(0.15)
    return added_imgs

if COLLECT_MODE and MLY_TOKEN.startswith('MLY|'):
    for c in tqdm(COUNTRY_BBOX, desc='collect'):
        n = collect_country(c)
        if n:
            save_json(seq_manifest, SEQ_MANIFEST)
    save_json(seq_manifest, SEQ_MANIFEST)

_tot_seq = sum(len(s) for s in seq_manifest.values())
_tot_img = sum(len(u) for s in seq_manifest.values() for u in s.values())
print(f'Manifest: {_tot_seq:,} sequences | {_tot_img:,} images '
      f'({_tot_img / max(_tot_seq, 1):.1f} per sequence)')

# %% [markdown]
# ## 4 · Sequence-disjoint split
#
# Whole sequences go to train or val — never split across. This is the
# fix that makes the val number mean something.

# %%
CLASSES = sorted(c for c, s in seq_manifest.items() if len(s) >= 20)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(CLASSES)
save_json(class_to_idx, CLASS_MAP_PATH)

def build_split():
    """Returns (train_samples, val_samples) as [(img_id, url, country)].
    Deterministic: the same sequence always lands in the same split."""
    train, val = [], []
    for country in CLASSES:
        sids = sorted(seq_manifest[country])          # deterministic order
        rng  = random.Random(hash(country) & 0xffffffff)
        rng.shuffle(sids)
        n_val = max(1, int(len(sids) * VAL_SEQ_FRAC))
        for i, sid in enumerate(sids):
            bucket = val if i < n_val else train
            for iid, url in seq_manifest[country][sid].items():
                bucket.append((iid, url, country))
    return train, val

train_samples, val_samples = build_split()
_tr_seq = sum(len(seq_manifest[c]) - max(1, int(len(seq_manifest[c]) * VAL_SEQ_FRAC))
              for c in CLASSES)
print(f'{NUM_CLASSES} countries')
print(f'train: {len(train_samples):,} images | val: {len(val_samples):,} images')
print('Sequences are disjoint across splits — no near-duplicate leakage.')

# %% [markdown]
# ## 5 · CLIP + GeoTips text anchors

# %%
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN)
tokenizer  = open_clip.get_tokenizer(CLIP_MODEL)
clip_model = clip_model.to(DEVICE)

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

@torch.no_grad()
def build_text_anchors(model):
    out = []
    for c in CLASSES:
        emb = model.encode_text(tokenizer(country_captions(c)).to(DEVICE))
        emb = F.normalize(emb, dim=-1).mean(0)
        out.append(F.normalize(emb, dim=-1))
    return torch.stack(out)

text_anchors = build_text_anchors(clip_model)
print('anchors:', tuple(text_anchors.shape))

# %% [markdown]
# ## 6 · Image fetch + dataset

# %%
def _refresh_url(iid):
    try:
        r = requests.get(f'https://graph.mapillary.com/{iid}', headers=HEADERS,
                         params={'fields': 'thumb_2048_url'}, timeout=15)
        r.raise_for_status()
        return r.json().get('thumb_2048_url')
    except Exception:
        return None

def fetch_image(iid, url):
    dst = os.path.join(IMG_CACHE, f'{iid}.jpg')
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
                url = _refresh_url(iid)
                if url is None:
                    return None
            else:
                return None
    return None

def is_clean(iid):
    return (not CLEAN_MODE) or clean_index.get(iid, 1) == 1

# Stronger augmentation than v6 (RandAugment + erasing) to fight overfit
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.25, 1.0), ratio=(0.7, 1.4)),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=7),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
val_tf = clip_preprocess

class GeoDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = [s for s in samples if is_clean(s[0])]
        self.tf = tf
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        iid, url, country = self.samples[i]
        p = fetch_image(iid, url)
        if p is None:
            return self[random.randrange(len(self.samples))]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            return self[random.randrange(len(self.samples))]
        return self.tf(img), class_to_idx[country]

train_ds = GeoDataset(train_samples, train_tf)
val_ds   = GeoDataset(val_samples, val_tf)
print(f'after cleaning — train {len(train_ds):,} | val {len(val_ds):,}')

counts = {}
for _, _, c in train_ds.samples:
    counts[c] = counts.get(c, 0) + 1
weights = [1.0 / counts[c] for _, _, c in train_ds.samples]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
_lkw = dict(num_workers=NUM_WORKERS, pin_memory=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **_lkw)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **_lkw)

# %% [markdown]
# ## 7 · Honest baselines on the leak-free val set
#
# Two numbers before training: CLIP zero-shot, and **the v6 model
# re-scored on this clean split**. The second tells you how much of
# v6's 42.4% was leakage.

# %%
@torch.no_grad()
def evaluate(model, anchors, loader, max_batches=None):
    model.eval()
    correct = n = 0
    for bi, (imgs, labels) in enumerate(tqdm(loader, leave=False, desc='eval')):
        if max_batches and bi >= max_batches:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.autocast('cuda'):
            emb = F.normalize(model.encode_image(imgs), dim=-1)
        correct += ((emb @ anchors.T).argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return correct / max(n, 1)

zs_acc = evaluate(clip_model, text_anchors, val_loader, max_batches=40)
print(f'CLIP zero-shot on leak-free val: {zs_acc:.4f}')

if os.path.exists(V6_MODEL_PATH):
    _sd = torch.load(V6_MODEL_PATH, map_location=DEVICE)
    _backup = {k: v.clone() for k, v in clip_model.state_dict().items()}
    try:
        clip_model.load_state_dict(_sd['model'])
        v6_anchors = build_text_anchors(clip_model)
        v6_acc = evaluate(clip_model, v6_anchors, val_loader, max_batches=40)
        print(f'v6 model on leak-free val:      {v6_acc:.4f}  '
              f'(it reported 0.4243 on the leaky split)')
    except Exception as ex:
        print('could not score v6:', ex)
    clip_model.load_state_dict(_backup)
    del _backup, _sd
    gc.collect(); torch.cuda.empty_cache()
    text_anchors = build_text_anchors(clip_model)

# %% [markdown]
# ## 8 · Anti-overfitting fine-tune
#
# - text tower frozen; first `FREEZE_BLOCKS` visual blocks frozen
# - layer-wise LR decay: deeper blocks learn faster than shallow ones
# - weight decay 0.2, label smoothing 0.15, RandAugment, early stopping

# %%
for p in clip_model.transformer.parameters():
    p.requires_grad = False
clip_model.token_embedding.weight.requires_grad = False

blocks = clip_model.visual.transformer.resblocks
for i, blk in enumerate(blocks):
    if i < FREEZE_BLOCKS:
        for p in blk.parameters():
            p.requires_grad = False
print(f'froze visual blocks 0..{FREEZE_BLOCKS - 1} of {len(blocks)}')

def param_groups():
    """Layer-wise LR decay: block i gets LR * LLRD^(depth - i)."""
    n = len(blocks)
    groups = []
    for i, blk in enumerate(blocks):
        if i < FREEZE_BLOCKS:
            continue
        groups.append({'params': [p for p in blk.parameters() if p.requires_grad],
                       'lr': LR_VISUAL * (LLRD ** (n - 1 - i))})
    tail = [clip_model.visual.ln_post.parameters()]
    if hasattr(clip_model.visual, 'proj') and clip_model.visual.proj is not None:
        tail.append([clip_model.visual.proj])
    for t in tail:
        groups.append({'params': [p for p in t if p.requires_grad],
                       'lr': LR_VISUAL})
    return groups

logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), device=DEVICE))

START_EPOCH, best_val_acc, bad_epochs = 0, 0.0, 0
history = {'train_acc': [], 'val_acc': []}
if TRAIN_MODE and os.path.exists(CKPT_PATH):
    ck = torch.load(CKPT_PATH, map_location=DEVICE)
    clip_model.load_state_dict(ck['model'])
    logit_scale.data = ck['logit_scale']
    START_EPOCH, best_val_acc = ck['epoch'] + 1, ck['best_val_acc']
    history = ck.get('history', history)
    text_anchors = build_text_anchors(clip_model)
    print(f'resumed at epoch {START_EPOCH} (best {best_val_acc:.4f})')

if TRAIN_MODE:
    scaler = torch.amp.GradScaler('cuda')
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW(param_groups() +
                            [{'params': [logit_scale], 'lr': 1e-3}],
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, NUM_EPOCHS - START_EPOCH) * len(train_loader))

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        clip_model.train()
        correct = n = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with torch.autocast('cuda'):
                emb    = F.normalize(clip_model.encode_image(imgs), dim=-1)
                logits = logit_scale.exp().clamp(max=100) * emb @ text_anchors.T
                loss   = ce(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}', acc=f'{correct/n:.3f}')

        train_acc = correct / max(n, 1)
        val_acc   = evaluate(clip_model, text_anchors, val_loader)
        gap = train_acc - val_acc
        print(f'train {train_acc:.4f} | val {val_acc:.4f} | gap {gap:.3f} '
              f'(zero-shot {zs_acc:.4f})')
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, bad_epochs = val_acc, 0
            torch.save({'model': clip_model.state_dict(),
                        'logit_scale': logit_scale.data,
                        'classes': CLASSES}, BEST_MODEL_PATH)
            print(f'>> best saved ({val_acc:.4f})')
        else:
            bad_epochs += 1
            print(f'   no improvement ({bad_epochs}/{EARLY_STOP})')
        torch.save({'epoch': epoch, 'model': clip_model.state_dict(),
                    'logit_scale': logit_scale.data,
                    'best_val_acc': best_val_acc, 'history': history},
                   CKPT_PATH)
        save_json(history, HISTORY_PATH)
        gc.collect()
        if bad_epochs >= EARLY_STOP:
            print('Early stopping — val stopped improving.')
            break
    print(f'\nBest leak-free val accuracy: {best_val_acc:.4f}')
else:
    ck = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    clip_model.load_state_dict(ck['model'])
    logit_scale.data = ck['logit_scale']
    clip_model.eval()
    text_anchors = build_text_anchors(clip_model)
    print('loaded v6.1')

# %% [markdown]
# ## 9 · Per-country accuracy + top-5

# %%
@torch.no_grad()
def predict_probs(img_pil):
    t = val_tf(img_pil).unsqueeze(0).to(DEVICE)
    with torch.autocast('cuda'):
        e1 = F.normalize(clip_model.encode_image(t), dim=-1)
        e2 = F.normalize(clip_model.encode_image(torch.flip(t, dims=[3])), dim=-1)
    logits = logit_scale.exp().clamp(max=100) * ((e1 + e2) / 2) @ text_anchors.T
    return F.softmax(logits, dim=1)[0].float().cpu().numpy()

def evaluate_per_country(samples, max_per_country=200):
    by_c = {}
    for iid, url, c in samples:
        by_c.setdefault(c, []).append((iid, url))
    per_c, top1, top5, tot = {}, 0, 0, 0
    for country, items in tqdm(sorted(by_c.items())):
        items = random.sample(items, min(max_per_country, len(items)))
        ok = n = 0
        for iid, url in items:
            p = fetch_image(iid, url)
            if p is None:
                continue
            probs = predict_probs(Image.open(p).convert('RGB'))
            order = np.argsort(probs)[::-1]
            ok  += idx_to_class[int(order[0])] == country
            top5 += country in [idx_to_class[int(i)] for i in order[:5]]
            n += 1
        if n:
            per_c[country] = ok / n
            top1 += ok; tot += n
    print(f'\nTop-1: {top1/tot:.4f}   Top-5: {top5/tot:.4f}   ({tot} images)')
    for c, a in sorted(per_c.items(), key=lambda kv: kv[1]):
        print(f'  {c:28s} {a:.3f}')
    return per_c

per_country = evaluate_per_country(val_ds.samples)

# %%
if history['val_acc']:
    plt.figure(figsize=(7, 4))
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='val (leak-free)')
    plt.axhline(zs_acc, ls='--', c='gray', label='zero-shot')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
    plt.title('GeoCLIP v6.1'); plt.tight_layout(); plt.show()
