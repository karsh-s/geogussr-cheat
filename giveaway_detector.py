# %% [markdown]
# # GeoTips Giveaway Detector — Pipeline Model 2
#
# A second model that works WITH the v5 whole-scene CNN:
#
# 1. **Detector (zero-shot, no training needed)**: YOLO-World detects
#    the objects GeoTips says are dead giveaways — bollards, road signs,
#    signposts, license plates, utility poles, guardrails, road markings,
#    chimneys/roofs. Open-vocabulary detection means we just *name* the
#    objects; no bounding-box labelling required.
# 2. **Crop classifier (trained here)**: every detected object is
#    cropped and fed to an EfficientNet-B0 that learns which country
#    that object style belongs to. Labels are *weak*: a crop inherits
#    the country of the street-view image it came from — exactly how
#    GeoTips knowledge works (French bollard ⇒ France).
# 3. **Fusion**: at inference, whole-scene probabilities (v5 model) and
#    per-object probabilities (this model) are combined. Objects the
#    crop model finds distinctive (low entropy) get more weight.
#
# Uses the SAME Drive manifests as v5 — no extra collection needed.
# Run the v5 notebook's collection cell at least once first.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install ultralytics albumentations

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
TRAIN_MODE = True          # False = load saved crop classifier

MLY_TOKEN  = "YOUR_MAPILLARY_CLIENT_TOKEN_HERE"
HEADERS    = {'Authorization': f'OAuth {MLY_TOKEN}'}

DRIVE_ROOT     = '/content/drive/MyDrive/GeoGussrCheat'
TRAIN_MANIFEST = f'{DRIVE_ROOT}/manifest_train_v5.json'
VAL_MANIFEST   = f'{DRIVE_ROOT}/manifest_val_v5.json'
IMG_CACHE      = '/content/img_cache'
CROP_CACHE     = '/content/crop_cache'          # detected object crops
CROP_INDEX     = f'{DRIVE_ROOT}/crop_index_v1.json'  # crop metadata on Drive
os.makedirs(IMG_CACHE, exist_ok=True)
os.makedirs(CROP_CACHE, exist_ok=True)

MODEL_DIR       = f'{DRIVE_ROOT}/models/giveaway_v1'
BEST_MODEL_PATH = f'{MODEL_DIR}/best_crop_model.pth'
CKPT_PATH       = f'{MODEL_DIR}/checkpoint_latest.pth'
CLASS_MAP_PATH  = f'{MODEL_DIR}/class_to_idx.json'
os.makedirs(MODEL_DIR, exist_ok=True)

# How many source images per country to mine for crops
MINE_PER_COUNTRY = 800
# Crop classifier training
CROP_SIZE     = 224
BATCH_SIZE    = 96
NUM_EPOCHS    = 12
FREEZE_EPOCHS = 2
LR_HEAD       = 1e-3
LR_FULL       = 1e-4
FOCAL_GAMMA   = 2.0
NUM_WORKERS   = 8

# %% [markdown]
# ## 3 · The GeoTips giveaway vocabulary
#
# These are the object classes YOLO-World is prompted with. Each maps
# to the GeoTips categories that make countries identifiable.

# %%
GIVEAWAY_PROMPTS = [
    'bollard',                    # THE classic giveaway (style differs per country)
    'road sign',                  # font, shape, colour conventions
    'traffic sign',
    'direction signpost',
    'license plate',              # EU strip colour, format
    'utility pole',               # wood vs concrete, insulator style
    'street lamp',
    'guardrail',                  # end-cap style is country-specific
    'road marking',               # centre-line colour/pattern
    'fire hydrant',
    'postbox',
    'telephone booth',
    'bus stop',
    'chimney',                    # architecture giveaway
    'roof',                       # tile style: Nordic metal vs Med terracotta
    'fence',
    'kilometre stone',
    'house facade',
]
# Detection confidence floor and crop-size floor
DET_CONF   = 0.15
MIN_CROP   = 40          # px — ignore tiny detections

# %% [markdown]
# ## 4 · Zero-shot detector — YOLO-World

# %%
from ultralytics import YOLOWorld

detector = YOLOWorld('yolov8l-worldv2.pt')
detector.set_classes(GIVEAWAY_PROMPTS)
print('YOLO-World ready — open-vocabulary detection, no training needed.')

def detect_giveaways(img_np, conf=DET_CONF):
    """Return [(class_name, conf, (x1,y1,x2,y2)), ...] for one image."""
    res = detector.predict(img_np, conf=conf, verbose=False)[0]
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        if x2 - x1 < MIN_CROP or y2 - y1 < MIN_CROP:
            continue
        out.append((GIVEAWAY_PROMPTS[int(box.cls[0])],
                    float(box.conf[0]), (x1, y1, x2, y2)))
    return out

# %% [markdown]
# ## 5 · Mine crops from the v5 manifests
#
# Streams images (same on-demand download + URL-healing as v5), runs the
# detector, saves crops locally, and records metadata in a Drive index —
# so mining is resumable and never repeats an image.

# %%
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

train_manifest = load_json(TRAIN_MANIFEST, {})
val_manifest   = load_json(VAL_MANIFEST, {})
assert train_manifest, ('Train manifest missing/empty — run the v5 '
                        'notebook collection cell first.')

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

# crop_index: {image_id: [[crop_file, class_name, conf, country], ...]}
crop_index = load_json(CROP_INDEX, {})

def mine_crops(manifest, per_country=MINE_PER_COUNTRY):
    n_new = 0
    for country in sorted(manifest):
        items = list(manifest[country].items())[:per_country]
        todo  = [(iid, url) for iid, url in items if iid not in crop_index]
        if not todo:
            continue
        pbar = tqdm(todo, desc=country[:22], leave=False)
        for iid, url in pbar:
            path = fetch_image(iid, url, manifest, country)
            if path is None:
                crop_index[iid] = []          # mark done (dead image)
                continue
            img_np = np.array(Image.open(path).convert('RGB'))
            crops  = []
            for cname, conf, (x1, y1, x2, y2) in detect_giveaways(img_np):
                # pad the box 15% for context
                h, w = img_np.shape[:2]
                px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
                x1, y1 = max(0, x1 - px), max(0, y1 - py)
                x2, y2 = min(w, x2 + px), min(h, y2 + py)
                cfile = f'{iid}_{len(crops)}.jpg'
                Image.fromarray(img_np[y1:y2, x1:x2]).save(
                    os.path.join(CROP_CACHE, cfile), 'JPEG', quality=90)
                crops.append([cfile, cname, round(conf, 3), country])
                n_new += 1
            crop_index[iid] = crops
        # save index to Drive after each country (resumable)
        with open(CROP_INDEX, 'w') as f:
            json.dump(crop_index, f)
    print(f'Mining done: +{n_new:,} new crops '
          f'({sum(len(v) for v in crop_index.values()):,} total)')

if TRAIN_MODE:
    mine_crops(train_manifest)

# %% [markdown]
# ## 6 · Crop dataset

# %%
countries = sorted({c[3] for crops in crop_index.values() for c in crops})
if TRAIN_MODE or not os.path.exists(CLASS_MAP_PATH):
    class_to_idx = {c: i for i, c in enumerate(countries)}
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
else:
    with open(CLASS_MAP_PATH) as f:
        class_to_idx = json.load(f)
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(class_to_idx)

OBJ_TO_IDX = {name: i for i, name in enumerate(GIVEAWAY_PROMPTS)}

all_crops = [c for crops in crop_index.values() for c in crops
             if c[3] in class_to_idx
             and os.path.exists(os.path.join(CROP_CACHE, c[0]))]
random.shuffle(all_crops)
n_val = max(1, int(0.12 * len(all_crops)))
val_crops, train_crops = all_crops[:n_val], all_crops[n_val:]
print(f'{len(train_crops):,} train / {len(val_crops):,} val crops, '
      f'{NUM_CLASSES} countries')

crop_train_tf = A.Compose([
    A.RandomResizedCrop(size=(CROP_SIZE, CROP_SIZE), scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
    A.ImageCompression(quality_range=(50, 90), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
crop_val_tf = A.Compose([
    A.Resize(CROP_SIZE, CROP_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class CropDataset(Dataset):
    def __init__(self, crops, tf):
        self.crops, self.tf = crops, tf
    def __len__(self):
        return len(self.crops)
    def __getitem__(self, idx):
        cfile, cname, conf, country = self.crops[idx]
        try:
            img = np.array(Image.open(
                os.path.join(CROP_CACHE, cfile)).convert('RGB'))
        except Exception:
            return self[random.randrange(len(self.crops))]
        return (self.tf(image=img)['image'],
                class_to_idx[country],
                OBJ_TO_IDX.get(cname, 0))

def make_crop_loaders():
    train_ds = CropDataset(train_crops, crop_train_tf)
    val_ds   = CropDataset(val_crops, crop_val_tf)
    counts = {}
    for _, _, _, c in train_crops:
        counts[c] = counts.get(c, 0) + 1
    weights = [1.0 / counts[c[3]] for c in train_crops]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    lkw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **lkw),
            DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **lkw))

# %% [markdown]
# ## 7 · Crop classifier — EfficientNet-B0 + object-type embedding
#
# Small and fast: crops are simple objects, B0 @ 224 is plenty. The
# object type (bollard vs roof vs plate) is embedded and concatenated so
# the head can learn per-object-type country signatures.

# %%
class CropClassifier(nn.Module):
    def __init__(self, num_classes, num_obj=len(GIVEAWAY_PROMPTS)):
        super().__init__()
        base          = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.obj_emb  = nn.Embedding(num_obj, 32)
        self.head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(1280 + 32, 512),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, obj_idx):
        feat = self.pool(self.backbone(x)).flatten(1)
        return self.head(torch.cat([feat, self.obj_emb(obj_idx)], dim=1))

model = CropClassifier(NUM_CLASSES).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=0.1):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing
    def forward(self, logits, targets):
        n = logits.size(1)
        sm = self.ls / (n - 1)
        oh = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        oh = oh * (1 - self.ls) + (1 - oh) * sm
        lp = F.log_softmax(logits, dim=1)
        return -(((1 - lp.exp()) ** self.gamma) * oh * lp).sum(1).mean()

focal = FocalLoss()

# %% [markdown]
# ## 8 · Train (checkpointed to Drive every epoch)

# %%
START_EPOCH, best_val_acc = 0, 0.0
if TRAIN_MODE and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if ckpt['model']['head.4.weight'].shape[0] == NUM_CLASSES:
        model.load_state_dict(ckpt['model'])
        START_EPOCH  = ckpt['epoch'] + 1
        best_val_acc = ckpt['best_val_acc']
        print(f'Resuming from epoch {START_EPOCH}')
    else:
        print('Stale checkpoint (class mismatch) — starting fresh.')

def set_frozen(model, frozen):
    for name, p in model.named_parameters():
        if name.startswith('backbone'):
            p.requires_grad = not frozen

if TRAIN_MODE and len(train_crops) > 0:
    scaler = torch.amp.GradScaler('cuda')
    train_loader, val_loader = make_crop_loaders()
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        frozen = epoch < FREEZE_EPOCHS
        set_frozen(model, frozen)
        lr  = LR_HEAD if frozen else LR_FULL
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4)

        model.train()
        correct = n = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        for imgs, labels, objs in pbar:
            imgs, labels, objs = (imgs.to(DEVICE), labels.to(DEVICE),
                                  objs.to(DEVICE))
            with torch.autocast('cuda'):
                logits = model(imgs, objs)
                loss   = focal(logits, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            correct += (logits.argmax(1) == labels).sum().item()
            n += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.3f}',
                             acc=f'{correct / n:.3f}')

        model.eval()
        v_correct = v_n = 0
        with torch.no_grad():
            for imgs, labels, objs in val_loader:
                imgs, labels, objs = (imgs.to(DEVICE), labels.to(DEVICE),
                                      objs.to(DEVICE))
                with torch.autocast('cuda'):
                    logits = model(imgs, objs)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_n += imgs.size(0)
        val_acc = v_correct / max(v_n, 1)
        print(f'val acc: {val_acc:.4f}')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f'>> best saved ({val_acc:.4f})')
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'best_val_acc': best_val_acc}, CKPT_PATH)
    print(f'Done. Best crop-classifier val acc: {best_val_acc:.4f}')
else:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print('Loaded saved crop classifier.')

# %% [markdown]
# ## 9 · Fusion inference — detector + crop classifier + v5 scene model
#
# `predict_fused(img)` returns combined country probabilities:
# - v5 whole-scene softmax (weight 1.0)
# - each detected giveaway crop's softmax, weighted by detector
#   confidence × (1 − normalised entropy): a crop the classifier finds
#   distinctive counts more; an ambiguous one counts less.

# %%
# Load the v5 scene model if available
V5_MODEL_PATH = f'{DRIVE_ROOT}/models/efficientnet_b4_v5/best_model.pth'
V5_CLASS_MAP  = f'{DRIVE_ROOT}/models/efficientnet_b4_v5/class_to_idx.json'
scene_model = None
if os.path.exists(V5_MODEL_PATH) and os.path.exists(V5_CLASS_MAP):
    with open(V5_CLASS_MAP) as f:
        v5_class_to_idx = json.load(f)
    NUM_CUES = 21
    class GeoClassifierV5(nn.Module):
        def __init__(self, num_classes, num_cues=NUM_CUES):
            super().__init__()
            base = models.efficientnet_b4(weights=None)
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.country_head = nn.Sequential(
                nn.Dropout(0.45), nn.Linear(1792, 896), nn.SiLU(),
                nn.Dropout(0.3), nn.Linear(896, num_classes))
            self.cue_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(1792, 384), nn.SiLU(),
                nn.Linear(384, num_cues))
        def forward(self, x):
            f = self.pool(self.backbone(x)).flatten(1)
            return {'country': self.country_head(f),
                    'cues': self.cue_head(f)}
    scene_model = GeoClassifierV5(len(v5_class_to_idx)).to(DEVICE)
    sd = torch.load(V5_MODEL_PATH, map_location=DEVICE)
    scene_model.load_state_dict(sd)
    scene_model.eval()
    print('v5 scene model loaded for fusion.')
else:
    print('v5 model not found — fusion will use crop evidence only.')

scene_tf = A.Compose([
    A.Resize(448, 448),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

@torch.no_grad()
def predict_fused(img_np, verbose=True):
    logp_total = np.zeros(NUM_CLASSES)

    # 1 · whole-scene model
    if scene_model is not None:
        t = scene_tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)
        p_scene = F.softmax(scene_model(t)['country'], dim=1)[0].cpu().numpy()
        # align v5 classes to crop-model class order
        p = np.full(NUM_CLASSES, 1e-9)
        for c, i in v5_class_to_idx.items():
            if c in class_to_idx:
                p[class_to_idx[c]] = p_scene[i]
        logp_total += np.log(p / p.sum())

    # 2 · giveaway crops
    detections = detect_giveaways(img_np)
    evidence = []
    for cname, conf, (x1, y1, x2, y2) in detections:
        h, w = img_np.shape[:2]
        px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
        crop = img_np[max(0, y1 - py):min(h, y2 + py),
                      max(0, x1 - px):min(w, x2 + px)]
        t   = crop_val_tf(image=crop)['image'].unsqueeze(0).to(DEVICE)
        o   = torch.tensor([OBJ_TO_IDX[cname]], device=DEVICE)
        p   = F.softmax(model(t, o), dim=1)[0].cpu().numpy()
        ent = -(p * np.log(p + 1e-12)).sum() / math.log(NUM_CLASSES)
        wgt = conf * (1 - ent)          # confident + distinctive ⇒ heavy
        logp_total += wgt * np.log(p + 1e-12)
        evidence.append((cname, conf, 1 - ent,
                         idx_to_class[int(p.argmax())], float(p.max())))

    probs = np.exp(logp_total - logp_total.max())
    probs = probs / probs.sum()
    top   = np.argsort(probs)[::-1][:5]
    if verbose:
        print('Giveaway evidence:')
        for cname, conf, distinct, guess, gp in sorted(
                evidence, key=lambda e: -e[2]):
            print(f'  {cname:20s} det {conf:.2f}  distinctive {distinct:.2f}'
                  f'  → {guess} ({gp:.0%})')
        print('Fused prediction:')
        for i in top:
            print(f'  {idx_to_class[i]:25s} {probs[i] * 100:5.1f}%')
    return [(idx_to_class[i], float(probs[i])) for i in top]

# %% [markdown]
# ## 10 · Try it on validation images

# %%
val_items = [(iid, url, c) for c, e in val_manifest.items()
             if c in class_to_idx for iid, url in e.items()]
for iid, url, country in random.sample(val_items, min(4, len(val_items))):
    path = fetch_image(iid, url, val_manifest, country)
    if not path:
        continue
    img_np = np.array(Image.open(path).convert('RGB'))
    print(f'\n=== true country: {country} ===')
    predict_fused(img_np)
