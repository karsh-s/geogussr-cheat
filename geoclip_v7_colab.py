# %% [markdown]
# # GeoCLIP v7 — trained on real Google Street View imagery
#
# **Why this replaces v5 / v6 / v6.1.** Those trained on Mapillary, where
# a handful of volunteers upload most of each country's photos. The model
# learned to recognise *their cameras* rather than the countries: it
# scored 76% on its own validation set, 10% on Mapillary photos from
# unseen contributors, and **0/18 on real Google Street View**.
#
# `trainEurope/` has none of that structure:
#
# | | Mapillary | trainEurope |
# |---|---|---|
# | camera | thousands of volunteer dashcams | one uniform Google rig worldwide |
# | photographer shortcut | available — the model took it | **impossible** |
# | matches GeoGuessr | no | **it is the game's imagery** |
#
# Each image is an independent random location at a random heading, 90°
# FOV — the same format the live server feeds the model at inference.
#
# **Method.** Two stages:
# 1. **Linear probe** — freeze CLIP, extract features once, fit a small
#    head. Fast, and it cannot wreck CLIP's general knowledge the way
#    v6.1's full fine-tune did.
# 2. **Light fine-tune** — unfreeze the top few ViT blocks and train with
#    the head. Usually a few points better; skip it if stage 1 suffices.
#
# `testEurope/` is never trained on and is only scored at the very end.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install open_clip_torch

# %%
import os, json, glob, math, random, shutil, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import open_clip

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, '|', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
DRIVE_ROOT = '/content/drive/MyDrive/GeoGussrCheat'

# Where the Street View images live. The next cell searches these in
# order, so you don't need to edit anything if they're in the usual spot.
TRAIN_CANDIDATES = [
    f'{DRIVE_ROOT}/trainEurope',
    '/content/trainEurope',
    f'{DRIVE_ROOT}/trainCountries',
]
TEST_CANDIDATES = [
    f'{DRIVE_ROOT}/testEurope',
    '/content/testEurope',
    f'{DRIVE_ROOT}/testCountries',
]

LOCAL_TRAIN = '/content/sv_train'      # fast local copy (Drive reads are slow)
LOCAL_TEST  = '/content/sv_test'

OUT_DIR   = f'{DRIVE_ROOT}/models/geoclip_v7'
FEAT_DIR  = f'{DRIVE_ROOT}/features_v7'      # cached CLIP features
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

CLIP_MODEL    = 'ViT-B-16'
CLIP_PRETRAIN = 'laion2b_s34b_b88k'
IMG_SIZE      = 224

# Stage 1 — linear probe
PROBE_EPOCHS  = 80
PROBE_LR      = 1e-3
PROBE_WD      = 1e-3

# Stage 2 — light fine-tune (set DO_FINETUNE=False to stop after stage 1)
DO_FINETUNE   = True
FT_EPOCHS     = 8
FT_UNFREEZE   = 4         # unfreeze the top N of 12 ViT blocks
FT_LR         = 1e-5
FT_WD         = 0.1
FT_BATCH      = 128
EARLY_STOP    = 2

VAL_FRAC      = 0.12      # carved out of train, for early stopping
LABEL_SMOOTH  = 0.1
EXTRACT_BATCH = 256
NUM_WORKERS   = 4

# %% [markdown]
# ## 3 · Locate the dataset
#
# If nothing is found, upload `trainEurope`/`testEurope` to
# `MyDrive/GeoGussrCheat/`. Zipping them first makes the upload far
# faster — the cell below unzips automatically if it finds a .zip.

# %%
def _resolve(cands, name):
    for p in cands:
        if os.path.isdir(p) and glob.glob(os.path.join(p, '*', '*.jpg')):
            return p
        if os.path.exists(p + '.zip'):
            print(f'unzipping {p}.zip ...')
            dest = '/content/' + os.path.basename(p)
            with zipfile.ZipFile(p + '.zip') as z:
                z.extractall('/content/')
            if glob.glob(os.path.join(dest, '*', '*.jpg')):
                return dest
    raise SystemExit(
        f'Could not find the {name} images. Looked in:\n  ' +
        '\n  '.join(cands) +
        f'\nUpload the {name} folder (or {name}.zip) to {DRIVE_ROOT}/ '
        'and re-run this cell.')

TRAIN_SRC = _resolve(TRAIN_CANDIDATES, 'training')
TEST_SRC  = _resolve(TEST_CANDIDATES, 'test')
print('train source:', TRAIN_SRC)
print('test  source:', TEST_SRC)

def _copy_local(src, dst):
    """Get the images onto local disk — per-file Drive reads are ~100x
    slower than local, and Colab wipes /content between sessions.

    Fast path: a single .zip on Drive. Reading one big file off Drive is
    far quicker than 33k small ones, so the first run builds the zip and
    every later session unzips it in a couple of minutes.
    """
    if not src.startswith('/content/drive'):
        return src
    if os.path.isdir(dst) and glob.glob(os.path.join(dst, '*', '*.jpg')):
        return dst

    zip_path = src.rstrip('/') + '.zip'
    if os.path.exists(zip_path):
        print(f'unzipping {zip_path} -> {dst} ...')
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dst)
        inner = glob.glob(os.path.join(dst, '*'))
        # tolerate zips that contain a single wrapping folder
        if len(inner) == 1 and os.path.isdir(inner[0]) and \
                not glob.glob(os.path.join(dst, '*', '*.jpg')):
            for p in glob.glob(os.path.join(inner[0], '*')):
                shutil.move(p, dst)
        return dst

    print(f'copying {src} -> {dst} (slow, one time) ...')
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    try:
        print(f'building {zip_path} so future sessions start fast ...')
        shutil.make_archive(src.rstrip('/'), 'zip', dst)
    except Exception as ex:
        print('  (could not write zip, not fatal):', ex)
    return dst

TRAIN_DIR = _copy_local(TRAIN_SRC, LOCAL_TRAIN)
TEST_DIR  = _copy_local(TEST_SRC, LOCAL_TEST)

def list_split(root, classes=None):
    items, found = [], set()
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        country = os.path.basename(d)
        if classes is not None and country not in classes:
            continue
        paths = [p for p in glob.glob(os.path.join(d, '*'))
                 if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if paths:
            found.add(country)
            items += [(p, country) for p in paths]
    return items, sorted(found)

train_items, CLASSES = list_split(TRAIN_DIR)
test_items, _        = list_split(TEST_DIR, set(CLASSES))
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(CLASSES)

# Held-out val split for early stopping (locations are independent random
# samples, so a plain random split is safe here — unlike Mapillary)
rng = random.Random(SEED)
rng.shuffle(train_items)
n_val = int(len(train_items) * VAL_FRAC)
val_items, fit_items = train_items[:n_val], train_items[n_val:]

print(f'{NUM_CLASSES} countries')
print(f'fit {len(fit_items):,} | val {len(val_items):,} | test {len(test_items):,}')
with open(f'{OUT_DIR}/class_to_idx.json', 'w') as f:
    json.dump(class_to_idx, f, indent=2)

# %% [markdown]
# ## 4 · Load CLIP

# %%
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    CLIP_MODEL, pretrained=CLIP_PRETRAIN)
tokenizer  = open_clip.get_tokenizer(CLIP_MODEL)
clip_model = clip_model.to(DEVICE).eval()
FEAT_DIM = clip_model.visual.output_dim
print(f'{CLIP_MODEL} loaded — feature dim {FEAT_DIM}')

# %% [markdown]
# ## 5 · Extract frozen CLIP features (cached to Drive)

# %%
class PathDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, c = self.items[i]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        return self.tf(img), class_to_idx[c]

@torch.no_grad()
def extract(items, tag):
    fx, fy = f'{FEAT_DIR}/{tag}_X.npy', f'{FEAT_DIR}/{tag}_y.npy'
    if os.path.exists(fx) and os.path.exists(fy):
        X, y = np.load(fx), np.load(fy)
        if len(y) == len(items):
            print(f'{tag}: cached ({len(y):,})')
            return X, y
    loader = DataLoader(PathDataset(items, clip_preprocess),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    for imgs, ys in tqdm(loader, desc=f'extract {tag}'):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast('cuda'):
            e1 = F.normalize(clip_model.encode_image(imgs), dim=-1)
            e2 = F.normalize(clip_model.encode_image(
                torch.flip(imgs, dims=[3])), dim=-1)
        feats.append(F.normalize((e1 + e2) / 2, dim=-1).float().cpu().numpy())
        labels.append(ys.numpy())
    X = np.concatenate(feats); y = np.concatenate(labels)
    np.save(fx, X); np.save(fy, y)
    print(f'{tag}: {X.shape}')
    return X, y

Xfit, yfit = extract(fit_items, 'fit')
Xval, yval = extract(val_items, 'val')
Xtest, ytest = extract(test_items, 'test')

# %% [markdown]
# ## 6 · Zero-shot baseline
#
# What plain CLIP scores with no training — the bar v7 has to clear.

# %%
@torch.no_grad()
def zero_shot_anchors():
    prompts = [f'a Google Street View photo taken in {c}' for c in CLASSES]
    A = F.normalize(clip_model.encode_text(tokenizer(prompts).to(DEVICE)), dim=-1)
    return A.float().cpu().numpy()

A0 = zero_shot_anchors()
_s = Xtest @ A0.T
zs1 = float((_s.argmax(1) == ytest).mean())
zs5 = float(np.mean([y in r for y, r in zip(ytest, np.argsort(_s, 1)[:, -5:])]))
print(f'ZERO-SHOT   top-1 {zs1:.4f}   top-5 {zs5:.4f}')

# %% [markdown]
# ## 7 · Stage 1 — linear probe on frozen features

# %%
class Head(nn.Module):
    def __init__(self, dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, n_cls))
    def forward(self, x):
        return self.net(x)

def evaluate_feats(head, X, y):
    head.eval()
    with torch.no_grad():
        lg = head(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
        top1 = (lg.argmax(1) == yt).float().mean().item()
        top5 = (lg.topk(5, 1).indices == yt[:, None]).any(1).float().mean().item()
    return top1, top5

head = Head(FEAT_DIM, NUM_CLASSES).to(DEVICE)
opt  = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)
ce   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

counts = np.bincount(yfit, minlength=NUM_CLASSES).astype(np.float32)
w = torch.tensor((1.0 / np.maximum(counts, 1))[yfit], dtype=torch.float)
Xfit_t = torch.tensor(Xfit, dtype=torch.float32)
yfit_t = torch.tensor(yfit, dtype=torch.long)

best_probe, best_state = 0.0, None
for ep in range(PROBE_EPOCHS):
    head.train()
    idx = torch.multinomial(w, len(yfit), replacement=True)
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        loss = ce(head(Xfit_t[b].to(DEVICE)), yfit_t[b].to(DEVICE))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sch.step()
    v1, _ = evaluate_feats(head, Xval, yval)
    if v1 > best_probe:
        best_probe = v1
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
    if ep % 20 == 0 or ep == PROBE_EPOCHS - 1:
        print(f'  epoch {ep:3d}  val top-1 {v1:.4f}')

head.load_state_dict(best_state)
p1, p5 = evaluate_feats(head, Xtest, ytest)
print(f'\nLINEAR PROBE  test top-1 {p1:.4f}  top-5 {p5:.4f}   '
      f'(zero-shot was {zs1:.4f})')
torch.save({'mode': 'probe', 'head': best_state, 'classes': CLASSES,
            'clip_model': CLIP_MODEL, 'clip_pretrain': CLIP_PRETRAIN},
           f'{OUT_DIR}/best_model.pth')
print(f'saved -> {OUT_DIR}/best_model.pth')

# %% [markdown]
# ## 8 · Stage 2 — light fine-tune of the top ViT blocks
#
# Trains the images end-to-end (not cached features), so this is the slow
# part — but only the top `FT_UNFREEZE` blocks move, and early stopping
# watches the val split. Only overwrites the saved model if it beats
# stage 1 on **test**.

# %%
ft_train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.25, 0.25, 0.2, 0.02),
    T.ToTensor(),
    T.Normalize((0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
    T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

class FTModel(nn.Module):
    def __init__(self, clip, head):
        super().__init__()
        self.visual = clip.visual
        self.head   = head
    def forward(self, x):
        return self.head(F.normalize(self.visual(x), dim=-1))

if DO_FINETUNE:
    ft = FTModel(clip_model, Head(FEAT_DIM, NUM_CLASSES).to(DEVICE)).to(DEVICE)
    ft.head.load_state_dict(best_state)

    for p in ft.visual.parameters():
        p.requires_grad = False
    blocks = ft.visual.transformer.resblocks
    for blk in blocks[-FT_UNFREEZE:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in ft.visual.ln_post.parameters():
        p.requires_grad = True
    print(f'unfroze top {FT_UNFREEZE} of {len(blocks)} blocks')

    fit_loader = DataLoader(
        PathDataset(fit_items, ft_train_tf), batch_size=FT_BATCH,
        sampler=WeightedRandomSampler(w, len(w), replacement=True),
        num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(PathDataset(val_items, clip_preprocess),
                            batch_size=FT_BATCH, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(PathDataset(test_items, clip_preprocess),
                             batch_size=FT_BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    @torch.no_grad()
    def eval_loader(model, loader):
        model.eval()
        c1 = c5 = n = 0
        for imgs, ys in loader:
            imgs, ys = imgs.to(DEVICE), ys.to(DEVICE)
            with torch.autocast('cuda'):
                lg = model(imgs)
            c1 += (lg.argmax(1) == ys).sum().item()
            c5 += (lg.topk(5, 1).indices == ys[:, None]).any(1).sum().item()
            n  += ys.size(0)
        return c1 / n, c5 / n

    opt = torch.optim.AdamW(
        [p for p in ft.parameters() if p.requires_grad],
        lr=FT_LR, weight_decay=FT_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=FT_EPOCHS * len(fit_loader))
    scaler = torch.amp.GradScaler('cuda')

    best_ft, best_ft_state, bad = 0.0, None, 0
    for ep in range(FT_EPOCHS):
        ft.train()
        corr = n = 0
        pbar = tqdm(fit_loader, desc=f'finetune {ep + 1}/{FT_EPOCHS}')
        for imgs, ys in pbar:
            imgs, ys = imgs.to(DEVICE, non_blocking=True), ys.to(DEVICE)
            with torch.autocast('cuda'):
                lg = ft(imgs)
                loss = ce(lg, ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
            corr += (lg.argmax(1) == ys).sum().item(); n += ys.size(0)
            pbar.set_postfix(acc=f'{corr / n:.3f}')
        v1, v5 = eval_loader(ft, val_loader)
        print(f'  train {corr / n:.4f} | val top-1 {v1:.4f} top-5 {v5:.4f}')
        if v1 > best_ft:
            best_ft, bad = v1, 0
            best_ft_state = {k: v.detach().clone()
                             for k, v in ft.state_dict().items()}
        else:
            bad += 1
            print(f'   no improvement ({bad}/{EARLY_STOP})')
            if bad >= EARLY_STOP:
                print('early stop'); break

    ft.load_state_dict(best_ft_state)
    f1, f5 = eval_loader(ft, test_loader)
    print(f'\nFINE-TUNED   test top-1 {f1:.4f}  top-5 {f5:.4f}')
    if f1 > p1:
        torch.save({'mode': 'finetune',
                    'visual': ft.visual.state_dict(),
                    'head': ft.head.state_dict(), 'classes': CLASSES,
                    'clip_model': CLIP_MODEL, 'clip_pretrain': CLIP_PRETRAIN},
                   f'{OUT_DIR}/best_model.pth')
        print(f'fine-tune wins — saved -> {OUT_DIR}/best_model.pth')
    else:
        print(f'probe ({p1:.4f}) still better — keeping it.')

# %% [markdown]
# ## 9 · Results

# %%
print(f'{"zero-shot":16s} top-1 {zs1:.4f}  top-5 {zs5:.4f}')
print(f'{"linear probe":16s} top-1 {p1:.4f}  top-5 {p5:.4f}')
if DO_FINETUNE:
    print(f'{"fine-tuned":16s} top-1 {f1:.4f}  top-5 {f5:.4f}')

# per-country breakdown of the best model
if DO_FINETUNE and f1 > p1:
    preds = []
    ft.eval()
    with torch.no_grad():
        for imgs, _ in DataLoader(PathDataset(test_items, clip_preprocess),
                                  batch_size=FT_BATCH, num_workers=NUM_WORKERS):
            with torch.autocast('cuda'):
                preds.append(ft(imgs.to(DEVICE)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
else:
    with torch.no_grad():
        pred = head(torch.tensor(Xtest, dtype=torch.float32,
                                 device=DEVICE)).argmax(1).cpu().numpy()

print('\nper-country test accuracy (worst first):')
rows = []
for i, c in enumerate(CLASSES):
    m = ytest == i
    if m.sum():
        rows.append((float((pred[m] == i).mean()), c, int(m.sum())))
for a, c, n in sorted(rows):
    print(f'  {c:28s} {a:.3f}  ({n} images)')

# %%
plt.figure(figsize=(9, 5))
names = [c for _, c, _ in sorted(rows)]
vals  = [a for a, _, _ in sorted(rows)]
plt.barh(names, vals, color='#0a84ff')
plt.axvline(zs1, ls='--', c='gray', label=f'zero-shot overall ({zs1:.2f})')
plt.xlabel('test accuracy'); plt.legend(); plt.tight_layout(); plt.show()
