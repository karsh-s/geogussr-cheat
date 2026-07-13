# ============================================================
# European Streetview CNN — v4
# Upload to Colab and run all cells.
#
# SESSION MODES:
#   TRAIN_MODE = True   → trains from scratch (or resumes checkpoint)
#   TRAIN_MODE = False  → skips all training, loads best model from Drive
#                         Use this every time you just want to test/predict.
#
# Checkpoint auto-saves to Drive every epoch — if the session dies,
# re-run all cells and it picks up from the last completed epoch.
# ============================================================

# %% [markdown]
# ## 0 · Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))

# %%
import shutil, time, os

_DRIVE_TRAIN = '/content/drive/MyDrive/GeoGussrCheat/trainEurope'
_DRIVE_TEST  = '/content/drive/MyDrive/GeoGussrCheat/testEurope'
_LOCAL_TRAIN = '/content/trainEurope'
_LOCAL_TEST  = '/content/testEurope'

def _copy_if_needed(src, dst):
    if os.path.exists(dst):
        print(f'  Already exists, skipping: {dst}')
        return
    print(f'  Copying {src} → {dst} ...', end=' ', flush=True)
    t0 = time.time()
    shutil.copytree(src, dst)
    n = sum(len(f) for _, _, f in os.walk(dst))
    print(f'{n} files copied in {time.time()-t0:.0f}s')

print('Copying dataset to local SSD...')
_copy_if_needed(_DRIVE_TRAIN, _LOCAL_TRAIN)
_copy_if_needed(_DRIVE_TEST,  _LOCAL_TEST)
print('Done.')

# %% [markdown]
# ## 1 · Install dependencies

# %%
!pip install -q albumentations easyocr grad-cam tqdm
print('Done.')

# %% [markdown]
# ## 2 · Imports & reproducibility

# %%
import os, random, json, time, warnings
from collections import Counter
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  GPU : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# %% [markdown]
# ## 3 · Configuration
#
# **Set TRAIN_MODE = False to skip training and jump straight to inference.**

# %%
# ── Toggle this ──────────────────────────────────────────────
TRAIN_MODE = True   # False = load saved model and go straight to testing
# ─────────────────────────────────────────────────────────────

TRAIN_DIR = '/content/trainEurope'
TEST_DIR  = '/content/testEurope'

# Checkpoint / model output directory on Drive (persists forever)
MODEL_DIR  = '/content/drive/MyDrive/GeoGussrCheat/models/efficientnet_b4_v4'
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH  = f'{MODEL_DIR}/best_model.pth'
CKPT_PATH        = f'{MODEL_DIR}/checkpoint_latest.pth'   # resume point
CLASS_MAP_PATH   = f'{MODEL_DIR}/class_to_idx.json'
HISTORY_PATH     = f'{MODEL_DIR}/history.json'

IMG_SIZE        = 448
BATCH_SIZE      = 28
NUM_EPOCHS      = 20
FREEZE_EPOCHS   = 4
LR_HEAD         = 5e-4
LR_FULL         = 4e-5
VAL_SPLIT       = 0.15
NUM_WORKERS     = min(os.cpu_count() or 2, 4)

LAMBDA_SIGN     = 0.25
LAMBDA_LANG     = 0.20
MIXUP_ALPHA     = 0.3
FOCAL_GAMMA     = 2.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_LANG_FEATURES = 16

COUNTRY_LANG_FEATURES = {
    'Bulgaria':               [0],
    'Serbia':                 [0],
    'Montenegro':             [0],
    'North Macedonia':        [0],
    'Ukraine':                [0],
    'Belarus':                [0],
    'Russia':                 [0],
    'Bosnia and Herzegovina': [0, 15],
    'Kosovo':                 [0, 15],
    'Greece':                 [1],
    'Cyprus':                 [1],
    'Georgia':                [2],
    'Armenia':                [3],
    'Turkey':                 [12],
    'Poland':                 [6],
    'Hungary':                [7],
    'Romania':                [8],
    'Czech Republic':         [9],
    'Czechia':                [9],
    'Slovakia':               [9],
    'Sweden':                 [10],
    'Norway':                 [10],
    'Denmark':                [10],
    'Finland':                [10],
    'Iceland':                [11],
    'Faroe Islands':          [11],
    'Latvia':                 [13],
    'Lithuania':              [13],
    'Estonia':                [13],
    'Malta':                  [14],
}

def lang_multihot(country_name):
    v = np.zeros(NUM_LANG_FEATURES, dtype=np.float32)
    for idx in COUNTRY_LANG_FEATURES.get(country_name, [15]):
        v[idx] = 1.0
    return v

FEATURE_NAMES = [
    'Cyrillic', 'Greek', 'Georgian', 'Armenian', 'Arabic', 'Hebrew',
    'Polish', 'Hungarian', 'Romanian', 'Czech/Slovak',
    'Scandinavian', 'Icelandic/Faroese', 'Turkish', 'Baltic', 'Maltese',
    'Plain Latin',
]

print(f'TRAIN_MODE={TRAIN_MODE}  IMG_SIZE={IMG_SIZE}  BATCH={BATCH_SIZE}  EPOCHS={NUM_EPOCHS}')

# %% [markdown]
# ## 4 · Augmentation pipelines

# %%
def build_train_transform(img_size=448):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Perspective(scale=(0.02, 0.08), p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
            A.OpticalDistortion(distort_limit=0.12, p=1.0),
        ], p=0.45),
        A.Affine(translate_percent=(0.0, 0.06), scale=(0.88, 1.12),
                 rotate=(-12, 12), border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=25),
            A.CLAHE(clip_limit=4.0),
            A.RandomGamma(gamma_limit=(70, 130)),
        ], p=0.65),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MotionBlur(blur_limit=9),
            A.MedianBlur(blur_limit=5),
            A.Sharpen(alpha=(0.2, 0.5)),
        ], p=0.35),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.1, 0.3)),
            A.RandomRain(slant_range=(-10, 10), drop_length=10, blur_value=3),
            A.RandomSunFlare(src_radius=80, num_flare_circles_range=(3, 6)),
        ], p=0.15),
        A.ImageCompression(quality_range=(55, 100), p=0.25),
        A.GaussNoise(std_range=(0.03, 0.15), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(8, img_size // 8),
                        hole_width_range=(8, img_size // 8),
                        fill=0, p=0.35),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def build_val_transform(img_size=448):
    return A.Compose([
        A.Resize(height=int(img_size * 1.14), width=int(img_size * 1.14)),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def build_sign_transform(size=192):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

train_tf = build_train_transform(IMG_SIZE)
val_tf   = build_val_transform(IMG_SIZE)
sign_tf  = build_sign_transform(192)
print('Transforms built.')

# %% [markdown]
# ## 5 · Sign-crop helper

# %%
def extract_sign_crops(imgs_np, sign_transform, n_crops=2):
    crops = []
    for img in imgs_np:
        h, w = img.shape[:2]
        upper_h = max(int(h * 0.42), 80)
        for _ in range(n_crops):
            y0 = random.randint(0, max(0, upper_h - 80))
            x0 = random.randint(0, max(0, w - 192))
            y1 = min(y0 + random.randint(80, upper_h), h)
            x1 = min(x0 + random.randint(140, w // 2), w)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                crop = img[:upper_h, :w//2]
            crops.append(sign_transform(image=crop)['image'])
    return torch.stack(crops)

# %% [markdown]
# ## 6 · EasyOCR script-detection

# %%
import easyocr
import unicodedata, re

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print('Initialising EasyOCR...')
        _ocr_reader = easyocr.Reader(
            ['en', 'ru', 'bg', 'uk', 'el', 'ka', 'hy',
             'pl', 'hu', 'ro', 'cs', 'sk', 'sv', 'no',
             'da', 'fi', 'lv', 'lt', 'et', 'tr', 'mt'],
            gpu=True, verbose=False
        )
        print('EasyOCR ready.')
    return _ocr_reader

_RANGES = {
    0: re.compile(r'[Ѐ-ӿ]'),
    1: re.compile(r'[Ͱ-Ͽ]'),
    2: re.compile(r'[Ⴀ-ჿ]'),
    3: re.compile(r'[԰-֏]'),
    4: re.compile(r'[؀-ۿ]'),
    5: re.compile(r'[֐-׿]'),
}
_DIACRITICS = {
    6:  set('ąęóźżńćśĄĘÓŹŻŃĆŚ'),
    7:  set('őűŐŰáéíóöúüÁÉÍÓÖÚÜ'),
    8:  set('șțâîȘȚÂÎ'),
    9:  set('čšžřěďťČŠŽŘĚĎŤ'),
    10: set('åøæöÅØÆÖ'),
    11: set('þðÞÐ'),
    12: set('ğşıİĞŞ'),
    13: set('āēīūčšžģķļņĀĒĪŪČŠŽĢĶĻŅ'),
    14: set('ħġĦĠ'),
}

def _score_text(all_text):
    counts = np.zeros(NUM_LANG_FEATURES, dtype=np.float32)
    for feat_idx, pattern in _RANGES.items():
        counts[feat_idx] = len(pattern.findall(all_text))
    for feat_idx, char_set in _DIACRITICS.items():
        counts[feat_idx] = sum(1 for c in all_text if c in char_set)
    all_diac = set().union(*_DIACRITICS.values())
    counts[15] = sum(
        1 for c in all_text
        if unicodedata.category(c).startswith('L')
        and not any(p.search(c) for p in _RANGES.values())
        and c not in all_diac
    )
    return counts

def detect_script_from_image(img_input, confidence_threshold=0.45):
    reader = get_ocr_reader()
    if isinstance(img_input, str):
        result = reader.readtext(img_input, detail=1)
    else:
        img_bgr = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        result  = reader.readtext(img_bgr, detail=1)
    all_text = ' '.join(text for _, text, conf in result if conf >= confidence_threshold)
    if not all_text.strip():
        v = np.zeros(NUM_LANG_FEATURES, dtype=np.float32); v[15] = 1.0
        return v
    counts = _score_text(all_text)
    return counts / (counts.sum() + 1e-8)

print('EasyOCR script detection ready.')

# %% [markdown]
# ## 7 · Dataset

# %%
class StreetviewDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None, return_numpy=False):
        self.root_dir     = root_dir
        self.transform    = transform
        self.return_numpy = return_numpy
        self.samples      = []

        countries = sorted([d for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(countries)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes      = countries

        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
        for country in countries:
            if country not in self.class_to_idx:
                continue
            label  = self.class_to_idx[country]
            lv     = lang_multihot(country)
            folder = os.path.join(root_dir, country)
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                    self.samples.append((os.path.join(folder, fname), label, lv))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, lang_vec = self.samples[idx]
        img_np = np.array(Image.open(path).convert('RGB'))
        img_t  = self.transform(image=img_np)['image'] if self.transform \
                 else torch.from_numpy(img_np).permute(2,0,1).float() / 255.
        lang_t = torch.tensor(lang_vec, dtype=torch.float32)
        if self.return_numpy:
            return img_t, label, lang_t, img_np
        return img_t, label, lang_t


def stratified_split(dataset, val_fraction=0.15, seed=42):
    labels  = [s[1] for s in dataset.samples]
    indices = list(range(len(labels)))
    return train_test_split(indices, test_size=val_fraction,
                            stratify=labels, random_state=seed)


full_train_ds = StreetviewDataset(TRAIN_DIR)
NUM_CLASSES   = len(full_train_ds.class_to_idx)
print(f'Countries: {NUM_CLASSES}  |  Train imgs: {len(full_train_ds):,}')
print('Classes:', sorted(full_train_ds.class_to_idx.keys()))

# Save class map to Drive now (needed for inference even without retraining)
with open(CLASS_MAP_PATH, 'w') as f:
    json.dump(full_train_ds.class_to_idx, f, indent=2)
print(f'Class map saved → {CLASS_MAP_PATH}')

# %% [markdown]
# ## 8 · Weighted sampler

# %%
label_counts = Counter(s[1] for s in full_train_ds.samples)

sample_weights = [1.0 / label_counts[s[1]] for s in full_train_ds.samples]

HARD_PAIRS_EXTRA = {
    'Austria': 1.6, 'Germany': 1.6,
    'Slovakia': 1.6, 'Czech Republic': 1.6, 'Czechia': 1.6,
    'Serbia': 1.6, 'Bulgaria': 1.6, 'Montenegro': 1.6,
    'Norway': 1.5, 'Sweden': 1.5, 'Finland': 1.5,
    'Liechtenstein': 2.0, 'Monaco': 2.0, 'Andorra': 2.0,
    'Luxembourg': 1.8, 'Kosovo': 1.8,
}
for i, (_, label, _) in enumerate(full_train_ds.samples):
    cname = full_train_ds.idx_to_class[label]
    if cname in HARD_PAIRS_EXTRA:
        sample_weights[i] *= HARD_PAIRS_EXTRA[cname]

print(f'Sample weight range: {min(sample_weights):.6f} – {max(sample_weights):.4f}')

# %% [markdown]
# ## 9 · Train/val loaders

# %%
class TransformSubset(Dataset):
    def __init__(self, subset, transform, return_numpy=False):
        self.subset       = subset
        self.transform    = transform
        self.return_numpy = return_numpy

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label, lang_vec = self.subset.dataset.samples[self.subset.indices[idx]]
        img_np = np.array(Image.open(path).convert('RGB'))
        img_t  = self.transform(image=img_np)['image'] if self.transform \
                 else torch.from_numpy(img_np).permute(2,0,1).float() / 255.
        lang_t = torch.tensor(lang_vec, dtype=torch.float32)
        if self.return_numpy:
            return img_t, label, lang_t, img_np
        return img_t, label, lang_t


train_idx, val_idx = stratified_split(full_train_ds, val_fraction=VAL_SPLIT, seed=SEED)
train_ds = TransformSubset(Subset(full_train_ds, train_idx), train_tf, return_numpy=True)
val_ds   = TransformSubset(Subset(full_train_ds, val_idx),   val_tf,   return_numpy=False)

sampler = WeightedRandomSampler(
    weights=[sample_weights[i] for i in train_idx],
    num_samples=len(train_idx), replacement=True
)

_lkw = dict(num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **_lkw)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   **_lkw)

steps_per_epoch = len(train_loader)
print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}  Steps/epoch: {steps_per_epoch}')

# %% [markdown]
# ## 10 · Model — EfficientNet-B4

# %%
class GeoClassifier(nn.Module):
    def __init__(self, num_classes, num_lang_features=NUM_LANG_FEATURES):
        super().__init__()
        base          = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        feat_dim      = 1792

        self.country_head = nn.Sequential(
            nn.Dropout(p=0.45),
            nn.Linear(feat_dim + num_lang_features, 896),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(896, num_classes),
        )
        self.sign_head = nn.Sequential(
            nn.Dropout(p=0.45),
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )
        self.lang_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, num_lang_features),
        )

    def _extract(self, x):
        return self.pool(self.backbone(x)).flatten(1)

    def forward(self, x, lang_vec, sign_crops=None):
        feat           = self._extract(x)
        country_logits = self.country_head(torch.cat([feat, lang_vec], dim=1))
        lang_logits    = self.lang_head(feat)
        out = {'country': country_logits, 'lang': lang_logits}
        if sign_crops is not None:
            out['sign'] = self.sign_head(self._extract(sign_crops))
        return out


def set_backbone_grad(model, requires_grad):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    for name, param in m.named_parameters():
        if not name.startswith(('country_head', 'sign_head', 'lang_head')):
            param.requires_grad = requires_grad

def get_state_dict(model):
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    return m.state_dict()


model = GeoClassifier(NUM_CLASSES).to(DEVICE)

# ── Check for existing checkpoint and resume if found ─────────
START_EPOCH     = 1
best_val_acc    = 0.0
_phase2_started = False
history         = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

if TRAIN_MODE and os.path.exists(CKPT_PATH):
    print(f'Found checkpoint: {CKPT_PATH}')
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    m.load_state_dict(ckpt['model'])
    START_EPOCH     = ckpt['epoch'] + 1
    best_val_acc    = ckpt['best_val_acc']
    _phase2_started = ckpt.get('phase2_started', False)
    history         = ckpt.get('history', history)
    print(f'Resuming from epoch {START_EPOCH}  |  best_val_acc so far: {best_val_acc:.2%}')
    if _phase2_started:
        set_backbone_grad(model, requires_grad=True)
        print('Phase 2 already started — backbone unfrozen.')
    else:
        set_backbone_grad(model, requires_grad=False)
        print('Still in phase 1 — backbone frozen.')
elif TRAIN_MODE:
    set_backbone_grad(model, requires_grad=False)
    print('No checkpoint found — starting fresh.')
    print('Backbone frozen for warm-up.')

if TRAIN_MODE:
    try:
        model = torch.compile(model)
        print('torch.compile() applied.')
    except Exception as e:
        print(f'torch.compile skipped: {e}')

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {total:,}  |  Trainable: {trainable:,}')

# %% [markdown]
# ## 11 · Loss, MixUp, optimiser

# %%
bce_lang = nn.BCEWithLogitsLoss()
ce_sign  = nn.CrossEntropyLoss(label_smoothing=0.1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        smooth      = self.label_smoothing / (num_classes - 1)
        one_hot     = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        one_hot     = one_hot * (1 - self.label_smoothing) + (1 - one_hot) * smooth
        log_p       = F.log_softmax(logits, dim=1)
        focal       = (1 - log_p.exp()) ** self.gamma
        return -(focal * one_hot * log_p).sum(dim=1).mean()


focal_country = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=0.1)


def mixup_batch(imgs, labels, lang_vecs, alpha=MIXUP_ALPHA):
    lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha))
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return (lam * imgs + (1-lam) * imgs[idx],
            labels, labels[idx],
            lam * lang_vecs + (1-lam) * lang_vecs[idx],
            lam)


def multi_task_loss(out, labels_a, labels_b, lang_targets, lam):
    L_country = (lam * focal_country(out['country'], labels_a)
                 + (1-lam) * focal_country(out['country'], labels_b))
    L_lang    = bce_lang(out['lang'], lang_targets)
    loss      = L_country + LAMBDA_LANG * L_lang
    if 'sign' in out:
        n_crops  = out['sign'].size(0) // labels_a.size(0)
        loss    += LAMBDA_SIGN * ce_sign(out['sign'], labels_a.repeat_interleave(n_crops))
    return loss, L_country


def make_optimizer_and_scheduler(model, lr, epochs, steps_per_epoch, phase=1):
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = optim.AdamW(params, lr=lr, weight_decay=1.5e-4)
    sched  = (optim.lr_scheduler.OneCycleLR(
                  opt, max_lr=lr, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, pct_start=0.3)
              if phase == 1 else
              optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=steps_per_epoch * 10, T_mult=1, eta_min=lr / 50))
    return opt, sched


# Build initial optimiser (may be replaced at phase-2 transition)
if TRAIN_MODE:
    if _phase2_started:
        remaining = NUM_EPOCHS - max(START_EPOCH - 1, FREEZE_EPOCHS)
        optimizer, scheduler = make_optimizer_and_scheduler(
            model, LR_FULL, remaining, steps_per_epoch, phase=2)
        print(f'Phase-2 optimiser: CosineWarmRestarts, lr={LR_FULL}')
    else:
        remaining_freeze = FREEZE_EPOCHS - (START_EPOCH - 1)
        optimizer, scheduler = make_optimizer_and_scheduler(
            model, LR_HEAD, max(remaining_freeze, 1), steps_per_epoch, phase=1)
        print(f'Phase-1 optimiser: OneCycleLR, lr={LR_HEAD}')

# %% [markdown]
# ## 12 · Training loop with per-epoch Drive checkpoint

# %%
def save_checkpoint(epoch, model, best_val_acc, phase2_started, history):
    """Save full training state to Drive so any session can resume."""
    torch.save({
        'epoch':          epoch,
        'model':          get_state_dict(model),
        'best_val_acc':   best_val_acc,
        'phase2_started': phase2_started,
        'history':        history,
    }, CKPT_PATH)
    # Also keep history JSON for quick inspection without loading the checkpoint
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f)


if TRAIN_MODE:
    scaler   = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    LOG_STEPS = 25

    try:
        from tqdm.notebook import tqdm as tqdm_nb
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False

    def run_train_epoch(loader, epoch):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        steps = len(loader)
        pbar  = tqdm_nb(loader, desc=f'Ep {epoch:>3} train', leave=False,
                        unit='batch') if HAS_TQDM else loader

        for step, batch in enumerate(pbar):
            if len(batch) == 4:
                imgs, labels, lang_vecs, imgs_np = batch
                imgs_np_list = [
                    imgs_np[i].numpy() if torch.is_tensor(imgs_np[i]) else imgs_np[i]
                    for i in range(len(imgs_np))
                ]
                sign_crops = extract_sign_crops(imgs_np_list, sign_tf, n_crops=2).to(DEVICE, non_blocking=True)
            else:
                imgs, labels, lang_vecs = batch
                sign_crops = None

            imgs      = imgs.to(DEVICE, non_blocking=True)
            labels    = labels.to(DEVICE, non_blocking=True)
            lang_vecs = lang_vecs.to(DEVICE, non_blocking=True)

            do_mixup = _phase2_started and random.random() < 0.5
            if do_mixup:
                imgs, labels_a, labels_b, lang_vecs, lam = mixup_batch(imgs, labels, lang_vecs)
                sign_crops = None
            else:
                labels_a, labels_b, lam = labels, labels, 1.0

            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                out  = model(imgs, lang_vecs, sign_crops=sign_crops)
                loss, L_country = multi_task_loss(out, labels_a, labels_b, lang_vecs, lam)
                loss = loss / 2  # grad accum = 2

            scaler.scale(loss).backward()
            if (step + 1) % 2 == 0 or (step + 1) == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += L_country.item() * imgs.size(0)
            correct    += (out['country'].argmax(1) == labels).sum().item()
            total      += imgs.size(0)

            if HAS_TQDM:
                pbar.set_postfix(loss=f'{L_country.item():.4f}',
                                 acc=f'{(out["country"].argmax(1)==labels).float().mean().item():.2%}',
                                 lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
            elif (step + 1) % LOG_STEPS == 0:
                print(f'  step {step+1:>4}/{steps}  '
                      f'loss={total_loss/total:.4f}  acc={correct/total:.2%}  '
                      f'lr={optimizer.param_groups[0]["lr"]:.2e}')

        return total_loss / total, correct / total

    def run_val_epoch(loader, epoch):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm_nb(loader, desc=f'Ep {epoch:>3} val  ', leave=False,
                       unit='batch') if HAS_TQDM else loader
        with torch.no_grad():
            for batch in pbar:
                imgs, labels, lang_vecs = batch[:3]
                imgs      = imgs.to(DEVICE, non_blocking=True)
                labels    = labels.to(DEVICE, non_blocking=True)
                lang_vecs = lang_vecs.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                    out  = model(imgs, lang_vecs)
                    loss = focal_country(out['country'], labels)
                total_loss += loss.item() * imgs.size(0)
                correct    += (out['country'].argmax(1) == labels).sum().item()
                total      += imgs.size(0)
                if HAS_TQDM:
                    pbar.set_postfix(loss=f'{loss.item():.4f}',
                                     acc=f'{(out["country"].argmax(1)==labels).float().mean().item():.2%}')
        return total_loss / total, correct / total

    # ── Epoch loop ────────────────────────────────────────────
    print(f"{'Ep':>4} | {'Phase':>6} | {'TrLoss':>8} | {'TrAcc':>7} | "
          f"{'VlLoss':>8} | {'VlAcc':>7} | {'LR':>9} | {'Min':>5} | {'Best':>7}")
    print('-' * 82)

    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):

        if epoch == FREEZE_EPOCHS + 1 and not _phase2_started:
            _phase2_started = True
            set_backbone_grad(model, requires_grad=True)
            remaining = NUM_EPOCHS - FREEZE_EPOCHS
            optimizer, scheduler = make_optimizer_and_scheduler(
                model, LR_FULL, remaining, steps_per_epoch, phase=2)
            print(f'  → Backbone unfrozen. Phase-2: CosineWarmRestarts, lr={LR_FULL}')

        phase = 'freeze' if epoch <= FREEZE_EPOCHS else 'full'
        t0    = time.time()

        tr_loss, tr_acc = run_train_epoch(train_loader, epoch)
        vl_loss, vl_acc = run_val_epoch(val_loader,   epoch)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        improved = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(get_state_dict(model), BEST_MODEL_PATH)
            improved = '✓ saved'

        # ── Save full checkpoint every epoch (enables resume after any crash)
        save_checkpoint(epoch, model, best_val_acc, _phase2_started, history)

        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f'{epoch:>4} | {phase:>6} | {tr_loss:>8.4f} | {tr_acc:>6.2%} | '
              f'{vl_loss:>8.4f} | {vl_acc:>6.2%} | {lr_now:>9.2e} | '
              f'{elapsed/60:>4.1f} | {best_val_acc:>6.2%} {improved}')

    print(f'\nTraining complete. Best val accuracy: {best_val_acc:.2%}')
    print(f'Model saved → {BEST_MODEL_PATH}')

else:
    print('TRAIN_MODE=False — skipping training.')

# %% [markdown]
# ## 13 · Training curves

# %%
# Load history from Drive if we're in inference-only mode
if not TRAIN_MODE and os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH) as f:
        history = json.load(f)

if history['train_loss']:
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', lw=2)
    axes[0].plot(epochs_range, history['val_loss'],   label='Val Loss',   lw=2, ls='--')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_range, [a*100 for a in history['train_acc']], label='Train Acc', lw=2)
    axes[1].plot(epochs_range, [a*100 for a in history['val_acc']],   label='Val Acc',   lw=2, ls='--')
    axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle('EfficientNet-B4 v4 — European Streetview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    shutil.copy('training_curves.png', f'{MODEL_DIR}/training_curves.png')
    plt.show()
else:
    print('No history yet.')

# %% [markdown]
# ## 14 · Load model for inference
#
# This cell always runs — it loads best_model.pth from Drive.
# In TRAIN_MODE=False, this is the first time the model weights are set.

# %%
# Always reload from the best saved checkpoint (not the last epoch)
eval_model = GeoClassifier(NUM_CLASSES).to(DEVICE)

if os.path.exists(BEST_MODEL_PATH):
    eval_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    print(f'Loaded best model from {BEST_MODEL_PATH}')
else:
    print(f'WARNING: No model found at {BEST_MODEL_PATH}')
    print('Train first (TRAIN_MODE=True), then set TRAIN_MODE=False for inference.')

eval_model.eval()

# %% [markdown]
# ## 15 · EasyOCR cache for test images

# %%
_CACHE_DIR  = '/content/drive/MyDrive/geoguessr_outputs/ocr_cache'
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_FILE = f'{_CACHE_DIR}/script_vectors_v4_easyocr.json'

def _cache_key(path):
    parts = path.replace('\\', '/').split('/')
    return '/'.join(parts[-2:])

def load_script_cache():
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            raw = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    return {}

def save_script_cache(cache):
    tmp = _CACHE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({k: v.tolist() for k, v in cache.items()}, f)
    os.replace(tmp, _CACHE_FILE)

def precompute_script_vectors(image_paths, save_every=300):
    from tqdm.notebook import tqdm as tqdm_nb
    cache = load_script_cache()
    todo  = [p for p in image_paths if _cache_key(p) not in cache]
    print(f'OCR cache: {len(cache):,} existing  |  {len(todo):,} to process')
    if not todo:
        print('Cache complete — nothing to do.')
        return cache
    get_ocr_reader()
    t0 = time.time()
    for i, path in enumerate(tqdm_nb(todo, desc='OCR', unit='img')):
        try:
            vec = detect_script_from_image(path)
        except Exception:
            vec = np.zeros(NUM_LANG_FEATURES, dtype=np.float32); vec[15] = 1.0
        cache[_cache_key(path)] = vec
        if (i + 1) % save_every == 0:
            save_script_cache(cache)
    save_script_cache(cache)
    print(f'Done — {len(todo):,} images OCR\'d in {(time.time()-t0)/60:.1f} min.')
    return cache

def cached_script_vec(path, cache, fallback_country=None):
    key = _cache_key(path)
    if key in cache:
        return cache[key]
    if fallback_country:
        return lang_multihot(fallback_country)
    v = np.zeros(NUM_LANG_FEATURES, dtype=np.float32); v[15] = 1.0
    return v

test_ds = StreetviewDataset(TEST_DIR, transform=val_tf,
                             class_to_idx=full_train_ds.class_to_idx)
_test_paths = [p for p, _, _ in test_ds.samples]
print(f'Test images: {len(_test_paths):,}')
script_cache = precompute_script_vectors(_test_paths)

# %% [markdown]
# ## 16 · TTA evaluation

# %%
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',
    persistent_workers=NUM_WORKERS > 0,
)

def tta_predict(model, loader, dataset, use_cached_ocr=True):
    all_probs, all_labels = [], []
    sample_ptr = 0
    samples    = dataset.samples
    with torch.no_grad():
        for batch in loader:
            imgs, labels, gt_lang_vecs = batch[:3]
            bs = imgs.size(0)
            if use_cached_ocr:
                lang_vecs = torch.tensor(
                    np.stack([cached_script_vec(samples[sample_ptr+j][0], script_cache)
                              for j in range(bs)]),
                    dtype=torch.float32
                )
            else:
                lang_vecs = gt_lang_vecs
            sample_ptr += bs
            imgs      = imgs.to(DEVICE, non_blocking=True)
            lang_vecs = lang_vecs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                p1 = torch.softmax(model(imgs,          lang_vecs)['country'], 1)
                p2 = torch.softmax(model(imgs.flip(-1), lang_vecs)['country'], 1)
            all_probs.append(((p1 + p2) / 2).cpu())
            all_labels.extend(labels.numpy())
    all_probs = torch.cat(all_probs, 0)
    return all_probs.argmax(1).numpy(), np.array(all_labels), all_probs.numpy()


all_preds, all_labels, all_probs = tta_predict(eval_model, test_loader, test_ds)
test_acc = np.mean(all_preds == all_labels)
print(f'\nTest Accuracy (TTA + EasyOCR): {test_acc:.2%}')
top3 = sum(all_labels[i] in all_probs[i].argsort()[-3:] for i in range(len(all_labels)))
print(f'Top-3 Accuracy               : {top3/len(all_labels):.2%}')

# %% [markdown]
# ## 17 · Per-country report

# %%
class_names = [full_train_ds.idx_to_class[i] for i in range(NUM_CLASSES)]
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

# %% [markdown]
# ## 18 · Confusion matrix

# %%
cm = confusion_matrix(all_labels, all_preds)
fig_h = max(12, NUM_CLASSES // 2)
plt.figure(figsize=(fig_h + 2, fig_h))
sns.heatmap(cm, annot=(NUM_CLASSES <= 25), fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — testEurope v4', fontsize=14)
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
shutil.copy('confusion_matrix.png', f'{MODEL_DIR}/confusion_matrix.png')
plt.show()

# %% [markdown]
# ## 19 · Hard-pair analysis

# %%
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
pairs = sorted(
    [(cm_copy[i,j], class_names[i], class_names[j])
     for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
     if i != j and cm_copy[i,j] > 0],
    reverse=True
)
print('Top-20 confused pairs:')
print(f'  {"True":25s}  {"Predicted":25s}  Errors')
print('  ' + '-'*60)
for count, tc, pc in pairs[:20]:
    print(f'  {tc:25s}  {pc:25s}  {count:5d}')

# %% [markdown]
# ## 20 · GradCAM

# %%
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class GradCAMWrapper(nn.Module):
    def __init__(self, model, lang_vec):
        super().__init__()
        self.model    = model
        self.lang_vec = lang_vec   # (1, NUM_LANG_FEATURES)

    def forward(self, x):
        lv = self.lang_vec.expand(x.size(0), -1).to(x.device)
        return self.model(x, lv)['country']


def visualise_gradcam(image_paths, model, class_to_idx, idx_to_class,
                      n_images=6, country_filter=None):
    paths = [p for p in image_paths
             if country_filter and country_filter.lower() in p.lower()] \
            if country_filter else image_paths
    paths = random.sample(paths, min(n_images, len(paths)))

    lang_vec = torch.zeros(1, NUM_LANG_FEATURES); lang_vec[0, 15] = 1.0
    wrapped  = GradCAMWrapper(model, lang_vec)
    m        = model._orig_mod if hasattr(model, '_orig_mod') else model
    cam      = GradCAM(model=wrapped, target_layers=[m.backbone[-1]])

    fig, axes = plt.subplots(len(paths), 2, figsize=(10, 4 * len(paths)))
    if len(paths) == 1:
        axes = [axes]
    for ax_row, path in zip(axes, paths):
        img_pil   = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        img_np    = np.array(img_pil) / 255.0
        img_t     = val_tf(image=(img_np * 255).astype(np.uint8))['image'].unsqueeze(0)
        country   = os.path.basename(os.path.dirname(path))
        cam_img   = show_cam_on_image(img_np.astype(np.float32),
                                      cam(input_tensor=img_t,
                                          targets=[ClassifierOutputTarget(class_to_idx.get(country, 0))])[0])
        ax_row[0].imshow(img_pil); ax_row[0].axis('off'); ax_row[0].set_title(f'Input: {country}')
        ax_row[1].imshow(cam_img); ax_row[1].axis('off'); ax_row[1].set_title('GradCAM')
    plt.tight_layout(); plt.savefig('gradcam.png', dpi=120); plt.show()
    cam.remove_hooks()


visualise_gradcam(
    [p for p, _, _ in test_ds.samples], eval_model,
    full_train_ds.class_to_idx, full_train_ds.idx_to_class, n_images=6
)

# %% [markdown]
# ## 21 · Single-image inference
#
# Use this to test any screenshot from GeoGuessr.
# Upload the image to Drive and call predict_image() with the path.
#
# Example:
#   predict_image('/content/drive/MyDrive/GeoGussrCheat/screenshots/round1.jpg')

# %%
def predict_image(image_path, top_k=5, use_ocr=True, explain=True):
    img_pil = Image.open(image_path).convert('RGB')
    img_np  = np.array(img_pil)

    if use_ocr:
        cached = script_cache.get(_cache_key(image_path)) \
                 if 'script_cache' in globals() else None
        if cached is not None:
            script_vec = cached
            if explain:
                print('Script (cached):')
                for name, score in zip(FEATURE_NAMES, script_vec):
                    if score > 0.01:
                        print(f'  {name:20s} {score:.3f}  {"█" * int(score*30)}')
        else:
            script_vec = detect_script_from_image(img_np)
            if explain:
                print('Script (live OCR):')
                for name, score in zip(FEATURE_NAMES, script_vec):
                    if score > 0.01:
                        print(f'  {name:20s} {score:.3f}  {"█" * int(score*30)}')
    else:
        script_vec = np.zeros(NUM_LANG_FEATURES, dtype=np.float32)
        script_vec[15] = 1.0

    dominant = FEATURE_NAMES[script_vec.argmax()]
    print(f'\nDominant script: {dominant} ({script_vec.max()*100:.0f}%)')

    lv    = torch.tensor(script_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    img_t = val_tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)

    eval_model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
        p1 = torch.softmax(eval_model(img_t,          lv)['country'], 1)
        p2 = torch.softmax(eval_model(img_t.flip(-1), lv)['country'], 1)
    probs = (p1 + p2) / 2

    top_probs, top_idx = torch.topk(probs[0], k=top_k)
    results = [(full_train_ds.idx_to_class[i.item()], p.item())
               for i, p in zip(top_idx, top_probs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                              gridspec_kw={'width_ratios': [1, 2]})
    axes[0].imshow(img_pil); axes[0].axis('off')
    axes[0].set_title(f'Script: {dominant}', fontsize=10)
    axes[1].barh([r[0] for r in results][::-1],
                 [r[1]*100 for r in results][::-1], color='steelblue')
    axes[1].set_xlabel('Confidence (%)'); axes[1].set_xlim(0, 100)
    axes[1].set_title(f'Prediction: {results[0][0]} ({results[0][1]:.1%})')
    plt.tight_layout(); plt.show()
    return results

# Quick test — comment out if you don't have a screenshot ready
# predict_image('/content/drive/MyDrive/GeoGussrCheat/screenshots/test.jpg')
print('predict_image() ready. Call it with any image path.')
