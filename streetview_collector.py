# %% [markdown]
# # Street View collector — polygon-checked, 4 headings per panorama
#
# Builds training data in **exactly** the form the live inference server
# produces, so there is no train/play mismatch by construction.
#
# How it works, per image:
#
# 1. **Pick a point genuinely inside the country.** Random lat/lng in the
#    bounding box, then tested against the real country polygon (Natural
#    Earth, offline). Your old collector sampled bounding boxes only —
#    France's box includes Spain, Germany, Italy and open sea, so a slice
#    of every country's images carried the wrong label.
# 2. **Find the nearest panorama.** Google's `SingleImageSearch` endpoint
#    — the one the Maps JS API itself calls. No API key, no billing.
# 3. **Download the panorama tiles** (a browser User-Agent header is
#    mandatory; without it every request 403s).
# 4. **Reproject to 4 rectilinear views** at headings 0/90/180/270,
#    90° FOV, pitched slightly down. One download → four training images.
# 5. **Save small.** 320px WebP ≈ 20 KB each, so 185k images ≈ 3.7 GB
#    instead of ~14 GB at the old 640px JPEG.
#
# Fully resumable: every accepted panorama id is written to an index on
# Drive, so re-runs skip what exists and the dataset can be rebuilt from
# the index alone if you ever need the space back.
#
# Network-bound, not GPU-bound — run it on a free CPU runtime and save
# your A100 credits for training.
#
# Note: bulk-downloading Street View imagery is against Google Maps'
# terms of service. Personal use only; don't redistribute the data.

# %% [markdown]
# ## 1 · Setup

# %%
!pip -q install shapely pyproj

# %%
import os, io, json, math, random, time, threading, zipfile, urllib.request
import concurrent.futures

import numpy as np
import requests
from PIL import Image
from tqdm.auto import tqdm
from shapely.geometry import shape, Point, box
from shapely.prepared import prep

from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 2 · Configuration

# %%
DRIVE_ROOT = '/content/drive/MyDrive/GeoGussrCheat'
OUT_DIR    = f'{DRIVE_ROOT}/trainStreetView'      # images land here
INDEX_PATH = f'{DRIVE_ROOT}/sv_index.json'        # {country: [pano_id, ...]}
os.makedirs(OUT_DIR, exist_ok=True)

# ── How much, and how big ───────────────────────────────────
TARGET_PER_COUNTRY = 5000     # images (4 per panorama → 1250 panoramas)
SAVE_SIZE          = 320      # px; CLIP needs 224, this leaves crop headroom
SAVE_FORMAT        = 'WEBP'   # 'WEBP' ≈ 30% smaller than 'JPEG'
SAVE_QUALITY       = 80

# ── View geometry — must match geoguessr_server.py ──────────
N_HEADINGS = 4
VIEW_FOV   = 90.0
VIEW_PITCH = -5.0
PANO_ZOOM  = 2                # 2 → 2048x1024 equirectangular
TILE_PX    = 512

# ── Sampling behaviour ──────────────────────────────────────
SEARCH_RADIUS_M = 2000        # how far from the random point to look
MIN_SEP_M       = 150         # reject panos this close to one already kept
OFFICIAL_ONLY   = True        # drop user-uploaded photospheres
WORKERS         = 8
MAX_TRIES_MULT  = 40          # give up on a country after this many misses
SAVE_EVERY      = 25          # write the index to Drive this often

TILE_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0.0.0 Safari/537.36'),
    'Referer': 'https://www.google.com/maps/',
}

# %% [markdown]
# ## 3 · Country polygons (Natural Earth — offline, free)
#
# This is the fix for label noise: a point only counts if it falls inside
# the country's actual borders, not merely its bounding box.

# %%
NE_URL = ('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
          'master/geojson/ne_50m_admin_0_countries.geojson')
NE_LOCAL = '/content/ne_countries.geojson'

if not os.path.exists(NE_LOCAL):
    print('downloading country polygons ...')
    urllib.request.urlretrieve(NE_URL, NE_LOCAL)

with open(NE_LOCAL) as f:
    _ne = json.load(f)

# Natural Earth names -> the folder names your dataset already uses
NAME_FIXES = {
    'Czechia': 'Czech Republic',
    'Bosnia and Herz.': 'Bosnia and Herzegovina',
    'North Macedonia': 'North Macedonia',
    'Republic of Serbia': 'Serbia',
    'United Kingdom': 'United Kingdom',
}

# Clip to Europe first: Natural Earth's "France" includes French Guiana
# and Reunion, "Netherlands" includes the Caribbean, "Norway" includes
# Svalbard. Without this their bounding boxes span the globe and almost
# no random sample lands on the mainland.
EUROPE = box(-25, 34, 45, 72)

COUNTRIES = {}
for feat in _ne['features']:
    props = feat['properties']
    name = props.get('NAME') or props.get('ADMIN')
    name = NAME_FIXES.get(name, name)
    geom = shape(feat['geometry']).intersection(EUROPE)
    if geom.is_empty:
        continue
    COUNTRIES[name] = {'geom': geom, 'prep': prep(geom),
                       'bounds': geom.bounds}   # (minx, miny, maxx, maxy)

WANTED = [
    'Albania', 'Andorra', 'Austria', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland',
    'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
    'Malta', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia',
    'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'Ukraine',
    'United Kingdom',
]
missing = [c for c in WANTED if c not in COUNTRIES]
print(f'{len(WANTED) - len(missing)}/{len(WANTED)} country polygons loaded')
if missing:
    print('  missing (will be skipped):', missing)
WANTED = [c for c in WANTED if c in COUNTRIES]


def random_point_in(country):
    """Uniform random lat/lng actually inside the country's borders."""
    minx, miny, maxx, maxy = COUNTRIES[country]['bounds']
    pre = COUNTRIES[country]['prep']
    for _ in range(400):
        lng = random.uniform(minx, maxx)
        lat = random.uniform(miny, maxy)
        if pre.contains(Point(lng, lat)):
            return lat, lng
    return None


def haversine_m(a, b):
    (lat1, lon1), (lat2, lon2) = a, b
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

# %% [markdown]
# ## 4 · Panorama lookup and download

# %%
SEARCH_URL = 'https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch'
TILE_URL   = ('https://streetviewpixels-pa.googleapis.com/v1/tile'
              '?cb_client=maps_sv.tactile&panoid={pid}&x={x}&y={y}&zoom={z}')

_PANO_RE = None
import re as _re
_PANO_RE = _re.compile(r'"([A-Za-z0-9_\-]{22})"')
_LATLNG_RE = _re.compile(r'\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)\]')


def find_pano(lat, lng, radius=SEARCH_RADIUS_M):
    """Nearest panorama to a coordinate. Returns (pano_id, lat, lng) or None.

    `!1e2` in the request selects official Google car coverage; dropping
    the photosphere flags keeps user-uploaded 360s out of the dataset.
    """
    pb = (f'!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lng}'
          f'!2d{radius}!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2'
          '!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2')
    if OFFICIAL_ONLY:
        pb = pb.replace('!4m10!1e1!1e2!1e3!1e4!1e8!1e6', '!4m6!1e1!1e8!1e6')
    try:
        r = requests.get(SEARCH_URL, params={'pb': pb, 'callback': 'cb'},
                         timeout=20)
        if r.status_code != 200:
            return None
        ids = _PANO_RE.findall(r.text)
        if not ids:
            return None
        coords = _LATLNG_RE.search(r.text)
        plat, plng = ((float(coords.group(1)), float(coords.group(2)))
                      if coords else (lat, lng))
        return ids[0], plat, plng
    except Exception:
        return None


def _tile(args):
    pid, x, y, z = args
    try:
        r = requests.get(TILE_URL.format(pid=pid, x=x, y=y, z=z),
                         headers=TILE_HEADERS, timeout=15)
        if r.status_code != 200 or not r.content:
            return x, y, None
        return x, y, Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return x, y, None


def fetch_panorama(pid, zoom=PANO_ZOOM):
    nx, ny = 2 ** zoom, 2 ** (zoom - 1)
    canvas = Image.new('RGB', (nx * TILE_PX, ny * TILE_PX))
    got = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for x, y, t in pool.map(_tile, [(pid, x, y, zoom)
                                        for x in range(nx) for y in range(ny)]):
            if t is not None:
                canvas.paste(t, (x * TILE_PX, y * TILE_PX))
                got += 1
    return canvas if got >= nx * ny * 0.75 else None


def equirect_to_perspective(pano, yaw_deg, pitch_deg=VIEW_PITCH,
                            fov_deg=VIEW_FOV, out=SAVE_SIZE):
    """Sphere -> normal camera view. Identical maths to the inference
    server, so training images and in-game views match exactly."""
    src = np.asarray(pano)
    H, W = src.shape[:2]
    f = 0.5 * out / np.tan(np.radians(fov_deg) / 2.0)
    i, j = np.meshgrid(np.arange(out), np.arange(out), indexing='ij')
    x = j - out / 2.0
    y = i - out / 2.0
    z = np.full_like(x, f, dtype=np.float64)
    p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
    y2 = y * np.cos(p) - z * np.sin(p)
    z2 = y * np.sin(p) + z * np.cos(p)
    x3 = x * np.cos(yw) + z2 * np.sin(yw)
    z3 = -x * np.sin(yw) + z2 * np.cos(yw)
    r = np.sqrt(x3 ** 2 + y2 ** 2 + z3 ** 2)
    lon = np.arctan2(x3, z3)
    lat = np.arcsin(np.clip(y2 / r, -1.0, 1.0))
    u = np.clip((((lon / (2 * np.pi)) + 0.5) * W).astype(np.int32), 0, W - 1)
    v = np.clip((((lat / np.pi) + 0.5) * H).astype(np.int32), 0, H - 1)
    return Image.fromarray(src[v, u])

# %% [markdown]
# ## 5 · Collect

# %%
index = json.load(open(INDEX_PATH)) if os.path.exists(INDEX_PATH) else {}
_lock = threading.Lock()
EXT = 'webp' if SAVE_FORMAT.upper() == 'WEBP' else 'jpg'


def save_index():
    with _lock:
        tmp = INDEX_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(index, f)
        os.replace(tmp, INDEX_PATH)


def collect_country(country, target=TARGET_PER_COUNTRY):
    out = os.path.join(OUT_DIR, country)
    os.makedirs(out, exist_ok=True)
    entry = index.setdefault(country, {'panos': [], 'coords': []})
    kept = {p for p in entry['panos']}
    coords = [tuple(c) for c in entry['coords']]

    have = len(kept) * N_HEADINGS
    if have >= target:
        return 0

    need_panos = math.ceil((target - have) / N_HEADINGS)
    added = tries = 0
    max_tries = need_panos * MAX_TRIES_MULT
    pbar = tqdm(total=need_panos, desc=f'{country[:20]:20s}', leave=False)

    def one():
        pt = random_point_in(country)
        if pt is None:
            return None
        found = find_pano(*pt)
        if found is None:
            return None
        pid, plat, plng = found
        if pid in kept:
            return None
        # reject panoramas too close to one we already have
        for c in coords[-4000:]:
            if haversine_m((plat, plng), c) < MIN_SEP_M:
                return None
        # the panorama can sit up to SEARCH_RADIUS_M away, possibly over a
        # border - verify the pano's OWN coordinates, not the sample point
        if not COUNTRIES[country]['prep'].contains(Point(plng, plat)):
            return None
        pano = fetch_panorama(pid)
        if pano is None:
            return None
        return pid, plat, plng, pano

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while added < need_panos and tries < max_tries:
            batch = [pool.submit(one) for _ in range(WORKERS)]
            for fut in batch:
                tries += 1
                res = fut.result()
                if res is None:
                    continue
                pid, plat, plng, pano = res
                if pid in kept:
                    continue
                for k in range(N_HEADINGS):
                    view = equirect_to_perspective(
                        pano, yaw_deg=k * (360.0 / N_HEADINGS))
                    view.save(os.path.join(out, f'{pid}_h{k}.{EXT}'),
                              SAVE_FORMAT, quality=SAVE_QUALITY)
                kept.add(pid)
                coords.append((plat, plng))
                entry['panos'].append(pid)
                entry['coords'].append([plat, plng])
                added += 1
                pbar.update(1)
                if added % SAVE_EVERY == 0:
                    save_index()
                if added >= need_panos:
                    break
    pbar.close()
    save_index()
    return added * N_HEADINGS


print(f'Target {TARGET_PER_COUNTRY:,} images/country '
      f'({TARGET_PER_COUNTRY // N_HEADINGS:,} panoramas)\n')
for country in WANTED:
    n = collect_country(country)
    have = len(index.get(country, {}).get('panos', [])) * N_HEADINGS
    print(f'{country:26s} +{n:5,}  (total {have:,})')

save_index()

# %% [markdown]
# ## 6 · Summary + size check

# %%
total_img = total_bytes = 0
print(f'{"country":26s} {"images":>8} {"size":>9}')
print('-' * 46)
for country in WANTED:
    d = os.path.join(OUT_DIR, country)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    b = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    total_img += len(files); total_bytes += b
    print(f'{country:26s} {len(files):8,} {b / 1e6:8.1f}M')
print('-' * 46)
print(f'{"TOTAL":26s} {total_img:8,} {total_bytes / 1e9:8.2f}G')
if total_img:
    print(f'\naverage {total_bytes / total_img / 1024:.1f} KB per image')
print(f'index: {os.path.getsize(INDEX_PATH) / 1e6:.1f} MB '
      f'— the dataset can be rebuilt from this alone')

# %% [markdown]
# ## 7 · Preview

# %%
import matplotlib.pyplot as plt
picks = []
for country in random.sample(WANTED, min(6, len(WANTED))):
    d = os.path.join(OUT_DIR, country)
    fs = os.listdir(d) if os.path.isdir(d) else []
    if fs:
        picks.append((country, os.path.join(d, random.choice(fs))))
if picks:
    fig, axes = plt.subplots(1, len(picks), figsize=(3 * len(picks), 3.2))
    if len(picks) == 1:
        axes = [axes]
    for ax, (c, p) in zip(axes, picks):
        ax.imshow(Image.open(p)); ax.set_title(c, fontsize=9); ax.axis('off')
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8 · Next
#
# Point `geoclip_v7.ipynb` at this folder by changing its config to:
#
# ```python
# TRAIN_CANDIDATES = [f'{DRIVE_ROOT}/trainStreetView', ...]
# ```
#
# Keep `testEurope` as the test set — it is separate imagery the model
# has never seen, and holding it fixed makes v7 and v8 directly
# comparable.
