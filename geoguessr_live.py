# ============================================================
# GeoGuessr Live Assistant
#
# Runs on your laptop while you play GeoGuessr in a browser.
# Every few seconds it screenshots the game, feeds the frame to
# your trained EfficientNet-B4 model, and shows an always-on-top
# popup with the predicted country + confidence. Evidence from
# multiple frames is accumulated, so the guess sharpens as you
# move around. Press "New Round" when a new location loads.
#
# SETUP (one-time):
#   1. Download from your Google Drive folder
#      GeoGussrCheat/models/efficientnet_b4_v4/ :
#        - best_model.pth
#        - class_to_idx.json
#      and put them next to this script (or edit paths below).
#   2. pip install torch torchvision mss pillow numpy
#      (optional, for sign/text detection: pip install easyocr)
#   3. python geoguessr_live.py
#   4. macOS will ask for Screen Recording permission the first
#      time — grant it to Terminal/Python in System Settings →
#      Privacy & Security → Screen Recording, then rerun.
#
# USAGE:
#   - Put your GeoGuessr browser window on your main display.
#   - The popup floats on top. It shows the top-5 countries.
#   - "New Round" resets accumulated evidence.
#   - "Pause" stops capturing (e.g. between games).
# ============================================================

import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import mss

# ── Config ──────────────────────────────────────────────────
MODEL_PATH     = os.path.join(os.path.dirname(__file__), 'best_model.pth')
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'class_to_idx.json')

IMG_SIZE          = 448
NUM_LANG_FEATURES = 16
CAPTURE_INTERVAL  = 2.5    # seconds between frames
EMA_DECAY         = 0.80   # evidence memory: higher = slower to change
MIN_FRAME_DIFF    = 8.0    # mean-abs-pixel-diff below this = you haven't moved, skip
USE_OCR           = False  # True = also run EasyOCR on each frame (slower, better on signs)
MONITOR_INDEX     = 1      # 1 = primary display (mss numbering)

# Optionally capture only part of the screen, e.g. the browser window:
#   CAPTURE_REGION = {'left': 100, 'top': 80, 'width': 1400, 'height': 900}
CAPTURE_REGION = None      # None = whole primary monitor

DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

# ── Model (must match the training notebook exactly) ────────
class GeoClassifier(nn.Module):
    def __init__(self, num_classes, num_lang_features=NUM_LANG_FEATURES):
        super().__init__()
        base          = models.efficientnet_b4(weights=None)
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

    def forward(self, x, lang_vec):
        feat           = self._extract(x)
        country_logits = self.country_head(torch.cat([feat, lang_vec], dim=1))
        lang_logits    = self.lang_head(feat)
        return {'country': country_logits, 'lang': lang_logits}


# ── Optional OCR script detection (same features as training) ──
_ocr_readers = None

def _get_ocr_readers():
    global _ocr_readers
    if _ocr_readers is None:
        import easyocr
        print('Loading EasyOCR readers (first run downloads models)...')
        _ocr_readers = (
            easyocr.Reader(['en', 'ru', 'bg', 'uk', 'rs_cyrillic', 'be'],
                           gpu=False, verbose=False),
            easyocr.Reader(['en', 'pl', 'hu', 'ro', 'cs', 'sk', 'sv', 'no',
                            'da', 'lv', 'lt', 'et', 'tr', 'mt', 'bs', 'hr'],
                           gpu=False, verbose=False),
        )
    return _ocr_readers

import re, unicodedata
_RANGES = {
    0: re.compile(r'[Ѐ-ӿ]'),  # Cyrillic
    1: re.compile(r'[Ͱ-Ͽ]'),  # Greek
    2: re.compile(r'[Ⴀ-ჿ]'),  # Georgian
    3: re.compile(r'[԰-֏]'),  # Armenian
    4: re.compile(r'[؀-ۿ]'),  # Arabic
    5: re.compile(r'[֐-׿]'),  # Hebrew
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

def script_vector_from_frame(img_np):
    """OCR the frame and return the 16-dim script feature vector."""
    r_cyr, r_lat = _get_ocr_readers()
    result = r_cyr.readtext(img_np, detail=1) + r_lat.readtext(img_np, detail=1)
    text = ' '.join(t for _, t, conf in result if conf >= 0.45)
    v = np.zeros(NUM_LANG_FEATURES, dtype=np.float32)
    if not text.strip():
        v[15] = 1.0
        return v
    for idx, pat in _RANGES.items():
        v[idx] = len(pat.findall(text))
    for idx, chars in _DIACRITICS.items():
        v[idx] = sum(1 for c in text if c in chars)
    all_diac = set().union(*_DIACRITICS.values())
    v[15] = sum(1 for c in text
                if unicodedata.category(c).startswith('L')
                and not any(p.search(c) for p in _RANGES.values())
                and c not in all_diac)
    total = v.sum()
    return v / total if total > 0 else v


# ── Preprocessing ───────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img_pil):
    img = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


# ── Prediction engine with evidence accumulation ────────────
class Predictor:
    def __init__(self):
        with open(CLASS_MAP_PATH) as f:
            class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes  = len(class_to_idx)

        print(f'Loading model on {DEVICE} ({self.num_classes} countries)...')
        self.model = GeoClassifier(self.num_classes)
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        print('Model ready.')

        self.reset()

    def reset(self):
        self.evidence  = None   # EMA of log-probs
        self.n_frames  = 0
        self.lang_ema  = None

    @torch.no_grad()
    def update(self, img_pil, img_np):
        # Language vector: OCR if enabled, else plain-Latin default,
        # refined by the model's own lang head estimate from prior frames.
        if USE_OCR:
            lang_vec = script_vector_from_frame(img_np)
        elif self.lang_ema is not None:
            lang_vec = self.lang_ema
        else:
            lang_vec = np.zeros(NUM_LANG_FEATURES, dtype=np.float32)
            lang_vec[15] = 1.0
        lv = torch.from_numpy(lang_vec).unsqueeze(0).to(DEVICE)

        x   = preprocess(img_pil).to(DEVICE)
        out = self.model(x, lv)

        # TTA: also run the horizontal flip and average
        out_f  = self.model(torch.flip(x, dims=[3]), lv)
        logits = (out['country'] + out_f['country']) / 2
        logp   = F.log_softmax(logits, dim=1)[0].cpu().numpy()

        # Feed the model's own script prediction back in as a prior
        lang_pred = torch.sigmoid(out['lang'])[0].cpu().numpy()
        lang_pred = lang_pred / max(lang_pred.sum(), 1e-6)
        self.lang_ema = (lang_pred if self.lang_ema is None
                         else 0.7 * self.lang_ema + 0.3 * lang_pred)

        # Accumulate evidence across frames (EMA of log-probs)
        self.evidence = (logp if self.evidence is None
                         else EMA_DECAY * self.evidence + (1 - EMA_DECAY) * logp)
        self.n_frames += 1

        probs = np.exp(self.evidence - self.evidence.max())
        probs = probs / probs.sum()
        top   = np.argsort(probs)[::-1][:5]
        return [(self.idx_to_class[i], float(probs[i])) for i in top]


# ── Overlay UI ──────────────────────────────────────────────
class Overlay:
    BAR_W = 220

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('GeoGuessr AI')
        self.root.attributes('-topmost', True)
        self.root.geometry('+40+60')
        self.root.configure(bg='#1c1c1e')
        self.root.resizable(False, False)

        self.header = tk.Label(self.root, text='Starting...', font=('Helvetica', 22, 'bold'),
                               fg='#ffffff', bg='#1c1c1e', padx=16, pady=8)
        self.header.pack(anchor='w')

        self.conf_label = tk.Label(self.root, text='', font=('Helvetica', 13),
                                   fg='#a0a0a5', bg='#1c1c1e', padx=16)
        self.conf_label.pack(anchor='w')

        self.rows = []
        for _ in range(5):
            frame = tk.Frame(self.root, bg='#1c1c1e')
            frame.pack(fill='x', padx=16, pady=2)
            name = tk.Label(frame, text='', width=18, anchor='w',
                            font=('Helvetica', 12), fg='#e5e5ea', bg='#1c1c1e')
            name.pack(side='left')
            canvas = tk.Canvas(frame, width=self.BAR_W, height=14,
                               bg='#2c2c2e', highlightthickness=0)
            canvas.pack(side='left', padx=(4, 6))
            pct = tk.Label(frame, text='', width=6, anchor='e',
                           font=('Helvetica', 12), fg='#e5e5ea', bg='#1c1c1e')
            pct.pack(side='left')
            self.rows.append((name, canvas, pct))

        btns = tk.Frame(self.root, bg='#1c1c1e')
        btns.pack(fill='x', padx=16, pady=(8, 12))
        self.paused = False
        self.reset_requested = False
        tk.Button(btns, text='New Round', command=self._reset).pack(side='left', padx=(0, 8))
        self.pause_btn = tk.Button(btns, text='Pause', command=self._toggle_pause)
        self.pause_btn.pack(side='left')
        self.status = tk.Label(btns, text='', font=('Helvetica', 11),
                               fg='#8e8e93', bg='#1c1c1e')
        self.status.pack(side='right')

    def _reset(self):
        self.reset_requested = True

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text='Resume' if self.paused else 'Pause')

    def show(self, ranked, n_frames):
        best, conf = ranked[0]
        self.header.config(text=best)
        if conf >= 0.75:   level, color = 'HIGH confidence',   '#30d158'
        elif conf >= 0.45: level, color = 'MEDIUM confidence', '#ffd60a'
        else:              level, color = 'LOW confidence',    '#ff453a'
        self.conf_label.config(text=f'{conf * 100:.1f}% — {level}', fg=color)
        for (name, canvas, pct), (country, p) in zip(self.rows, ranked):
            name.config(text=country)
            pct.config(text=f'{p * 100:.1f}%')
            canvas.delete('all')
            canvas.create_rectangle(0, 0, max(2, int(self.BAR_W * p)), 14,
                                    fill=color if country == best else '#0a84ff',
                                    outline='')
        self.status.config(text=f'{n_frames} frames')


# ── Capture loop ────────────────────────────────────────────
def capture_loop(predictor, overlay):
    prev_small = None
    with mss.mss() as sct:
        region = CAPTURE_REGION or sct.monitors[MONITOR_INDEX]
        while True:
            time.sleep(CAPTURE_INTERVAL)
            if overlay.paused:
                continue
            if overlay.reset_requested:
                predictor.reset()
                overlay.reset_requested = False
                prev_small = None

            shot   = sct.grab(region)
            img    = Image.frombytes('RGB', shot.size, shot.bgra, 'raw', 'BGRX')
            img_np = np.asarray(img)

            # Skip frames where nothing changed (player hasn't moved)
            small = np.asarray(img.resize((64, 64)), dtype=np.float32)
            if prev_small is not None and np.abs(small - prev_small).mean() < MIN_FRAME_DIFF:
                continue
            prev_small = small

            try:
                ranked = predictor.update(img, img_np)
                overlay.root.after(0, overlay.show, ranked, predictor.n_frames)
            except Exception as ex:
                print(f'Prediction error: {ex}')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    for path in (MODEL_PATH, CLASS_MAP_PATH):
        if not os.path.exists(path):
            raise SystemExit(
                f'Missing {path}\n'
                'Download best_model.pth and class_to_idx.json from your Drive '
                'folder GeoGussrCheat/models/efficientnet_b4_v4/ and place them '
                'next to this script.'
            )
    predictor = Predictor()
    overlay   = Overlay()
    threading.Thread(target=capture_loop, args=(predictor, overlay), daemon=True).start()
    overlay.root.mainloop()
