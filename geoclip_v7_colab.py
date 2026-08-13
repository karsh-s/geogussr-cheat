# %% [markdown]
# # GeoCLIP v7 — trained on polygon-verified Google Street View imagery
#
# **What went wrong before, and what this fixes.**
#
# * **v5 / v6 / v6.1 (Mapillary).** A handful of volunteers upload most of
#   each country's photos, so the model learned to recognise *their
#   cameras*. It scored 76% on its own validation set, 10% on Mapillary
#   photos from unseen contributors, and **0/18 on real Street View**.
# * **`trainEurope/` (the old Street View set).** Sampled from country
#   *bounding boxes*, which overlap neighbours and sea. Measured against
#   real borders, only **~37% of those images carry the correct label** —
#   Norway is 8%, because its box covers most of Sweden and Finland. The
#   coordinates were never saved, so it cannot be relabelled.
#
# This notebook trains on `trainStreetView/` from `streetview_collector`,
# where every panorama's own coordinates were checked against the true
# country polygon before saving.
#
# **Three things that keep the number honest:**
#
# 1. **Split by panorama, never by image.** Each panorama yields 4
#    headings of the *same spot*; splitting by image would put heading 0
#    in train and heading 90 in test — the same leakage that inflated v6
#    and v6.1.
# 2. **Test set carved from the clean data.** `testEurope/` shares the
#    63% mislabelling, so it cannot serve as an answer key.
# 3. **Two metrics.** Per-image accuracy, and **per-location** accuracy
#    (the 4 headings averaged) — the latter is what the live server
#    actually computes, so it is the number that predicts in-game
#    performance.
#
# **Method:** linear probe on frozen CLIP features first (fast, and it
# cannot destroy CLIP's knowledge the way v6.1's full fine-tune did),
# then an optional light fine-tune of the top ViT blocks. Whichever wins
# on the held-out test panoramas is what gets saved.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install open_clip_torch

# %%
import os, json, glob, math, random, shutil, zipfile, re, time
from collections import defaultdict

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
print('Device:', DEVICE,
      '|', torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '')

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
DRIVE_ROOT = '/content/drive/MyDrive/GeoGussrCheat'

# Clean, polygon-verified data from streetview_collector.ipynb.
# Several candidates because GeoGussrCheat may be a *shortcut* to a
# shared folder, and Colab's Drive mount traverses shortcuts unreliably
# (os.path.isdir can pass while os.scandir fails on the same path).
SV_CANDIDATES = [
    '/content/drive/MyDrive/trainStreetView',        # moved out of the share
    f'{DRIVE_ROOT}/trainStreetView',                 # via the shortcut
    '/content/sv_data',                              # already local
]
LOCAL_SV   = '/content/sv_data'

# Set True only if you understand the labels are ~63% wrong; kept so the
# old data can be used as a rough sanity check, never as a real result.
ALLOW_DIRTY_FALLBACK = False
DIRTY_DIR  = f'{DRIVE_ROOT}/trainEurope'

def _writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        t = os.path.join(path, '.w')
        open(t, 'w').close(); os.remove(t)
        return True
    except Exception:
        return False

OUT_DIR  = f'{DRIVE_ROOT}/models/geoclip_v7'
FEAT_DIR = f'{DRIVE_ROOT}/features_v7'
if not _writable(OUT_DIR):
    OUT_DIR = '/content/drive/MyDrive/geoclip_v7'
    print('!! Drive shortcut not writable — saving model to', OUT_DIR)
    _writable(OUT_DIR)
if not _writable(FEAT_DIR):
    FEAT_DIR = '/content/features_v7'
    print('!! caching features locally at', FEAT_DIR, '(lost on disconnect)')
    _writable(FEAT_DIR)

LOCAL_OUT = '/content/geoclip_v7_out'      # always writable
os.makedirs(LOCAL_OUT, exist_ok=True)


def save_model(obj, name='best_model.pth'):
    """Save locally, mirror to Drive, and verify it actually landed.

    A ~350MB write straight to a Drive *shortcut* can report success and
    silently not persist - that is how the first fine-tuned model was
    lost. Local write is the source of truth; Drive is a copy that gets
    size-checked.
    """
    local = os.path.join(LOCAL_OUT, name)
    torch.save(obj, local)
    size = os.path.getsize(local)
    torch.load(local, map_location='cpu')      # prove it reloads
    print(f'saved {local}  ({size/1e6:.1f} MB, verified)')

    try:
        dst = os.path.join(OUT_DIR, name)
        shutil.copyfile(local, dst)
        if os.path.getsize(dst) != size:
            raise IOError(f'Drive copy is {os.path.getsize(dst)/1e6:.1f} MB, '
                          f'expected {size/1e6:.1f} MB')
        print(f'  mirrored to {dst}')
    except Exception as ex:
        print(f'  !! could not mirror to Drive: {ex}')
        print(f'  !! DOWNLOAD IT DIRECTLY:')
        print(f'       from google.colab import files')
        print(f"       files.download('{local}')")


CLIP_MODEL    = 'ViT-B-16'      # try 'ViT-L-14' for ~+5-10 pts, ~3x slower
CLIP_PRETRAIN = 'laion2b_s34b_b88k'
IMG_SIZE      = 224

# Splits — by PANORAMA, not by image
VAL_FRAC   = 0.10
TEST_FRAC  = 0.20

# Stage 1 — linear probe on frozen features
PROBE_EPOCHS = 80
PROBE_LR     = 1e-3
PROBE_WD     = 1e-3

# Stage 2 — light fine-tune
DO_FINETUNE  = True
FT_EPOCHS    = 8
FT_UNFREEZE  = 4          # top N of 12 ViT blocks
FT_LR        = 1e-5
FT_WD        = 0.1
FT_BATCH     = 128
EARLY_STOP   = 2

LABEL_SMOOTH  = 0.1
EXTRACT_BATCH = 256
NUM_WORKERS   = 4
IMG_EXT = ('.webp', '.jpg', '.jpeg', '.png')

# %% [markdown]
# ## 3 · Load the dataset and split by panorama
#
# Collector filenames are `<panoid>_h<k>.webp`, so the panorama id is
# recoverable from the filename — that is what the split groups on.

# %%
def _copy_local(src, dst):
    """Images onto local disk; Drive per-file reads are ~100x slower.
    Uses a zip on Drive if present, and builds one the first time."""
    if not src.startswith('/content/drive'):
        return src
    if os.path.isdir(dst) and glob.glob(os.path.join(dst, '*', '*')):
        return dst
    zip_path = src.rstrip('/') + '.zip'
    if os.path.exists(zip_path):
        print(f'unzipping {zip_path} ...')
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dst)
        inner = glob.glob(os.path.join(dst, '*'))
        if len(inner) == 1 and os.path.isdir(inner[0]) and \
                not glob.glob(os.path.join(dst, '*', '*')):
            for p in glob.glob(os.path.join(inner[0], '*')):
                shutil.move(p, dst)
        return dst
    # Copy country by country with retries and resume. One giant
    # copytree over ~185k Drive files fails if the mount hiccups even
    # once, and throws away everything already copied.
    print(f'copying {src} -> {dst} (one time; Drive is slow for many files)')
    os.makedirs(dst, exist_ok=True)
    countries = sorted(os.listdir(src))
    for c in tqdm(countries, desc='copy'):
        s_dir, d_dir = os.path.join(src, c), os.path.join(dst, c)
        if not os.path.isdir(s_dir):
            continue
        for attempt in range(3):
            try:
                have = len(os.listdir(d_dir)) if os.path.isdir(d_dir) else 0
                want = len(os.listdir(s_dir))
                if have >= want:
                    break                      # already copied
                shutil.copytree(s_dir, d_dir, dirs_exist_ok=True)
                break
            except Exception as ex:
                if attempt == 2:
                    print(f'  {c}: giving up after 3 tries ({ex})')
                else:
                    time.sleep(3)
    try:
        print('building zip so future sessions start fast ...')
        shutil.make_archive(src.rstrip('/'), 'zip', dst)
    except Exception as ex:
        print('  (zip failed, not fatal):', ex)
    return dst


def _usable(p):
    """Readable AND listable — a shortcut can pass isdir but fail scandir."""
    try:
        if not os.path.isdir(p):
            return False
        subs = os.listdir(p)
        for c in subs[:3]:
            os.listdir(os.path.join(p, c))
        return bool(subs)
    except Exception as ex:
        print(f'  {p}: not usable ({type(ex).__name__})')
        return False

SRC = None
for cand in SV_CANDIDATES:
    if _usable(cand):
        SRC = cand
        break
    if os.path.exists(cand + '.zip'):
        SRC = cand
        break

if SRC is not None:
    DIRTY = False
    print('data source:', SRC)
elif ALLOW_DIRTY_FALLBACK and os.path.isdir(DIRTY_DIR):
    SRC = DIRTY_DIR
    DIRTY = True
    print('!! WARNING: using trainEurope — roughly 63% of its labels are\n'
          '!! wrong (bounding-box sampling). Any number below is not real.')
else:
    raise SystemExit(
        'Could not read the Street View data. Tried:\n  ' +
        '\n  '.join(SV_CANDIDATES) + '\n\n'
        'If the folder exists in Drive but is unreadable here, it is almost\n'
        'certainly behind a SHORTCUT (GeoGussrCheat shows a shortcut badge).\n'
        'Colab cannot reliably read through those. Fix in Drive:\n'
        '  right-click trainStreetView -> Organise -> Move -> My Drive\n'
        'then re-run. Otherwise run streetview_collector.ipynb first.')

DATA_DIR = _copy_local(SRC, LOCAL_SV)

_PANO_RE = re.compile(r'^(.*)_h\d+$')

def load_items(root):
    """[(path, country, pano_id)] — pano_id falls back to the filename
    for datasets without heading structure."""
    items, classes = [], set()
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        country = os.path.basename(d)
        n_before = len(items)
        for p in glob.glob(os.path.join(d, '*')):
            if not p.lower().endswith(IMG_EXT):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            m = _PANO_RE.match(stem)
            items.append((p, country, m.group(1) if m else stem))
        if len(items) > n_before:
            classes.add(country)
    return items, sorted(classes)


all_items, CLASSES = load_items(DATA_DIR)
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
NUM_CLASSES  = len(CLASSES)

# group panoramas per country, then split whole panoramas
by_country = defaultdict(lambda: defaultdict(list))
for p, c, pid in all_items:
    by_country[c][pid].append((p, c, pid))

fit_items, val_items, test_items = [], [], []
for c in CLASSES:
    pids = sorted(by_country[c])
    rng = random.Random(hash(c) & 0xffffffff)      # deterministic per country
    rng.shuffle(pids)
    n = len(pids)
    n_test = max(1, int(n * TEST_FRAC))
    n_val  = max(1, int(n * VAL_FRAC))
    for i, pid in enumerate(pids):
        bucket = (test_items if i < n_test else
                  val_items if i < n_test + n_val else fit_items)
        bucket += by_country[c][pid]

# integrity check — no panorama may appear in two splits
_s = [set((t[1], t[2]) for t in s) for s in (fit_items, val_items, test_items)]
assert not (_s[0] & _s[1]) and not (_s[0] & _s[2]) and not (_s[1] & _s[2]), \
    'panorama leaked across splits'

print(f'{NUM_CLASSES} countries | {len(all_items):,} images from '
      f'{sum(len(v) for v in by_country.values()):,} panoramas')
print(f'fit {len(fit_items):,} | val {len(val_items):,} | test {len(test_items):,}'
      f'   ({len(_s[2]):,} test panoramas, disjoint)')
for _d in (OUT_DIR, '/content/geoclip_v7_out'):
    try:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'class_to_idx.json'), 'w') as f:
            json.dump(class_to_idx, f, indent=2)
    except Exception:
        pass

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
        p, c, _ = self.items[i]
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        return self.tf(img), class_to_idx[c]

TAG = f'{CLIP_MODEL}_{"dirty" if DIRTY else "sv"}'

@torch.no_grad()
def extract(items, name):
    fx, fy = f'{FEAT_DIR}/{TAG}_{name}_X.npy', f'{FEAT_DIR}/{TAG}_{name}_y.npy'
    if os.path.exists(fx) and os.path.exists(fy):
        X, y = np.load(fx), np.load(fy)
        if len(y) == len(items):
            print(f'{name}: cached ({len(y):,})')
            return X, y
    loader = DataLoader(PathDataset(items, clip_preprocess),
                        batch_size=EXTRACT_BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    feats, labels = [], []
    for imgs, ys in tqdm(loader, desc=f'extract {name}'):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast('cuda'):
            e1 = F.normalize(clip_model.encode_image(imgs), dim=-1)
            e2 = F.normalize(clip_model.encode_image(
                torch.flip(imgs, dims=[3])), dim=-1)
        feats.append(F.normalize((e1 + e2) / 2, dim=-1).float().cpu().numpy())
        labels.append(ys.numpy())
    X = np.concatenate(feats); y = np.concatenate(labels)
    np.save(fx, X); np.save(fy, y)
    print(f'{name}: {X.shape}')
    return X, y

Xfit, yfit   = extract(fit_items, 'fit')
Xval, yval   = extract(val_items, 'val')
Xtest, ytest = extract(test_items, 'test')

# %% [markdown]
# ## 6 · Metrics
#
# `per-image` scores one view. `per-location` averages the 4 headings of
# a panorama — exactly what geoguessr_server.py does, so it is the number
# that predicts in-game behaviour.

# %%
test_panos = [t[2] for t in test_items]

def location_scores(probs, panos, y):
    """Average probabilities across all views of each panorama."""
    agg, truth = defaultdict(list), {}
    for i, pid in enumerate(panos):
        agg[pid].append(probs[i]); truth[pid] = y[i]
    top1 = top5 = 0
    for pid, ps in agg.items():
        m = np.mean(ps, axis=0)
        order = np.argsort(m)[::-1]
        top1 += int(order[0] == truth[pid])
        top5 += int(truth[pid] in order[:5])
    n = len(agg)
    return top1 / n, top5 / n, n

def report(probs, tag):
    order = np.argsort(probs, axis=1)[:, ::-1]
    i1 = float((order[:, 0] == ytest).mean())
    i5 = float(np.mean([y in r for y, r in zip(ytest, order[:, :5])]))
    l1, l5, n = location_scores(probs, test_panos, ytest)
    print(f'{tag:16s} per-image  top-1 {i1:.4f}  top-5 {i5:.4f}')
    print(f'{"":16s} per-location top-1 {l1:.4f}  top-5 {l5:.4f}  ({n} locations)')
    return i1, i5, l1, l5

# %% [markdown]
# ## 7 · Zero-shot baseline

# %%
with torch.no_grad():
    prompts = [f'a Google Street View photo taken in {c}' for c in CLASSES]
    A0 = F.normalize(clip_model.encode_text(
        tokenizer(prompts).to(DEVICE)), dim=-1).float().cpu().numpy()
_logits = Xtest @ A0.T
_e = np.exp(_logits * 100 - (_logits * 100).max(1, keepdims=True))
zs_probs = _e / _e.sum(1, keepdims=True)
zs = report(zs_probs, 'ZERO-SHOT')

# %% [markdown]
# ## 8 · Stage 1 — linear probe

# %%
class Head(nn.Module):
    def __init__(self, dim, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, n_cls))
    def forward(self, x):
        return self.net(x)

head = Head(FEAT_DIM, NUM_CLASSES).to(DEVICE)
opt  = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)
ce   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

counts = np.bincount(yfit, minlength=NUM_CLASSES).astype(np.float32)
w = torch.tensor((1.0 / np.maximum(counts, 1))[yfit], dtype=torch.float)
Xfit_t = torch.tensor(Xfit, dtype=torch.float32)
yfit_t = torch.tensor(yfit, dtype=torch.long)
Xval_t = torch.tensor(Xval, dtype=torch.float32, device=DEVICE)
yval_t = torch.tensor(yval, dtype=torch.long, device=DEVICE)

best_val, best_state = 0.0, None
for ep in range(PROBE_EPOCHS):
    head.train()
    idx = torch.multinomial(w, len(yfit), replacement=True)
    for i in range(0, len(idx), 512):
        b = idx[i:i + 512]
        loss = ce(head(Xfit_t[b].to(DEVICE)), yfit_t[b].to(DEVICE))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sch.step()
    head.eval()
    with torch.no_grad():
        v = (head(Xval_t).argmax(1) == yval_t).float().mean().item()
    if v > best_val:
        best_val = v
        best_state = {k: t.clone() for k, t in head.state_dict().items()}
    if ep % 20 == 0 or ep == PROBE_EPOCHS - 1:
        print(f'  epoch {ep:3d}  val top-1 {v:.4f}')

head.load_state_dict(best_state); head.eval()
with torch.no_grad():
    probe_probs = F.softmax(head(torch.tensor(Xtest, dtype=torch.float32,
                                              device=DEVICE)), dim=1).cpu().numpy()
print()
probe = report(probe_probs, 'LINEAR PROBE')
save_model({'mode': 'probe', 'head': best_state, 'classes': CLASSES,
            'clip_model': CLIP_MODEL, 'clip_pretrain': CLIP_PRETRAIN})

# %% [markdown]
# ## 9 · Stage 2 — light fine-tune of the top ViT blocks

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
    def __init__(self, visual, head):
        super().__init__()
        self.visual, self.head = visual, head
    def forward(self, x):
        return self.head(F.normalize(self.visual(x), dim=-1))

if DO_FINETUNE:
    ft = FTModel(clip_model.visual, Head(FEAT_DIM, NUM_CLASSES).to(DEVICE)).to(DEVICE)
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
    def probs_of(model, loader):
        model.eval(); out = []
        for imgs, _ in loader:
            with torch.autocast('cuda'):
                lg = model(imgs.to(DEVICE))
            out.append(F.softmax(lg.float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    opt = torch.optim.AdamW([p for p in ft.parameters() if p.requires_grad],
                            lr=FT_LR, weight_decay=FT_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=FT_EPOCHS * len(fit_loader))
    scaler = torch.amp.GradScaler('cuda')

    best_ft, best_ft_state, bad = 0.0, None, 0
    for ep in range(FT_EPOCHS):
        ft.train(); corr = n = 0
        pbar = tqdm(fit_loader, desc=f'finetune {ep + 1}/{FT_EPOCHS}')
        for imgs, ys in pbar:
            imgs, ys = imgs.to(DEVICE, non_blocking=True), ys.to(DEVICE)
            with torch.autocast('cuda'):
                lg = ft(imgs); loss = ce(lg, ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
            corr += (lg.argmax(1) == ys).sum().item(); n += ys.size(0)
            pbar.set_postfix(acc=f'{corr / n:.3f}')
        vp = probs_of(ft, val_loader)
        v = float((vp.argmax(1) == yval).mean())
        print(f'  train {corr / n:.4f} | val top-1 {v:.4f}')
        if v > best_ft:
            best_ft, bad = v, 0
            best_ft_state = {k: t.detach().clone()
                             for k, t in ft.state_dict().items()}
        else:
            bad += 1
            print(f'   no improvement ({bad}/{EARLY_STOP})')
            if bad >= EARLY_STOP:
                print('early stop'); break

    ft.load_state_dict(best_ft_state)
    ft_probs = probs_of(ft, test_loader)
    print()
    ftres = report(ft_probs, 'FINE-TUNED')

    if ftres[2] > probe[2]:      # compare on per-location top-1
        save_model({'mode': 'finetune', 'visual': ft.visual.state_dict(),
                    'head': ft.head.state_dict(), 'classes': CLASSES,
                    'clip_model': CLIP_MODEL, 'clip_pretrain': CLIP_PRETRAIN})
        print('fine-tune wins')
        final_probs = ft_probs
    else:
        print(f'probe still better ({probe[2]:.4f} vs {ftres[2]:.4f}) — keeping it')
        final_probs = probe_probs
else:
    final_probs = probe_probs

# %% [markdown]
# ## 10 · Results

# %%
print(f'{"":16s} {"img top-1":>10} {"img top-5":>10} {"loc top-1":>10} {"loc top-5":>10}')
print(f'{"zero-shot":16s} {zs[0]:10.4f} {zs[1]:10.4f} {zs[2]:10.4f} {zs[3]:10.4f}')
print(f'{"linear probe":16s} {probe[0]:10.4f} {probe[1]:10.4f} {probe[2]:10.4f} {probe[3]:10.4f}')
if DO_FINETUNE:
    print(f'{"fine-tuned":16s} {ftres[0]:10.4f} {ftres[1]:10.4f} {ftres[2]:10.4f} {ftres[3]:10.4f}')
print('\n"loc" = the 4 headings of a panorama averaged — what the live '
      'server computes, so it is the number that predicts in-game accuracy.')

pred = final_probs.argmax(1)
rows = []
for i, c in enumerate(CLASSES):
    m = ytest == i
    if m.sum():
        rows.append((float((pred[m] == i).mean()), c, int(m.sum())))
print('\nper-country (worst first):')
for a, c, n in sorted(rows):
    print(f'  {c:28s} {a:.3f}  ({n})')

plt.figure(figsize=(9, 8))
plt.barh([c for _, c, _ in sorted(rows)], [a for a, _, _ in sorted(rows)],
         color='#0a84ff')
plt.axvline(zs[0], ls='--', c='gray', label=f'zero-shot ({zs[0]:.2f})')
plt.xlabel('test accuracy'); plt.legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11 · Most-confused pairs
#
# Where to spend the next round of effort — these are the countries the
# model actually mixes up, rather than the ones you'd guess.

# %%
conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for t, p in zip(ytest, pred):
    conf[t, p] += 1
pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and conf[i, j]:
            pairs.append((conf[i, j] / max(conf[i].sum(), 1),
                          idx_to_class[i], idx_to_class[j], conf[i, j]))
print('true -> predicted (rate, count):')
for r, a, b, n in sorted(pairs, reverse=True)[:20]:
    print(f'  {a:24s} -> {b:24s} {r:.2f}  ({n})')


# %% [markdown]
# ## 12 · Download the model
#
# Do this now — a runtime restart wipes everything, and a large write to
# a Drive *shortcut* can silently fail to persist.

# %%
from google.colab import files
_p = os.path.join(LOCAL_OUT, 'best_model.pth')
print(f'{_p}  ({os.path.getsize(_p)/1e6:.1f} MB)')
print('mode:', torch.load(_p, map_location='cpu')['mode'])
files.download(_p)
files.download(f'{OUT_DIR}/class_to_idx.json'
               if os.path.exists(f'{OUT_DIR}/class_to_idx.json')
               else f'{LOCAL_OUT}/class_to_idx.json')
