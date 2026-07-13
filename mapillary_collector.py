# ============================================================
# Mapillary → Google Drive bulk image collector
# Run this in Google Colab (GPU not needed).
#
# HOW TO USE:
#   1. Go to https://www.mapillary.com/app/account/developers
#      → "Register Application" → copy the CLIENT TOKEN (not secret)
#   2. Paste it into MLY_TOKEN below
#   3. Upload this file to Colab: File → Upload notebook (or paste cells)
#   4. Run all cells — images land directly in your Drive folder
#
# Target: 500 NEW images per country on top of what you already have.
# ============================================================

# ── Cell 0: Mount Drive ─────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 1: Config ──────────────────────────────────────────
import os, time, random, requests, concurrent.futures
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm.notebook import tqdm

MLY_TOKEN    = "YOUR_MAPILLARY_CLIENT_TOKEN_HERE"   # ← paste yours
DRIVE_TRAIN  = "/content/drive/MyDrive/GeoGussrCheat/trainEurope"
DRIVE_TEST   = "/content/drive/MyDrive/GeoGussrCheat/testEurope"

# How many NEW images to collect per country in each split
NEW_TRAIN    = 600   # added on top of existing
NEW_TEST     = 100   # added on top of existing

# Parallel download workers — Colab handles 8 well
WORKERS      = 8
# Mapillary rate limit: ~50 search req/min on free tier; sleep between pages
SEARCH_SLEEP = 0.5

HEADERS = {"Authorization": f"OAuth {MLY_TOKEN}"}

# ── Cell 2: Country bounding boxes (lon_min, lat_min, lon_max, lat_max) ──
COUNTRY_BBOX = {
    "Albania":          (19.3,  39.6,  21.1,  42.7),
    "Andorra":          ( 1.4,  42.4,   1.8,  42.7),
    "Austria":          ( 9.5,  46.4,  17.2,  49.0),
    "Belgium":          ( 2.5,  49.5,   6.4,  51.5),
    "Bosnia and Herzegovina": (15.7, 42.6, 19.6, 45.3),
    "Bulgaria":         (22.4,  41.2,  28.7,  44.2),
    "Croatia":          (13.5,  42.4,  19.4,  46.6),
    "Cyprus":           (32.2,  34.6,  34.1,  35.7),
    "Czech Republic":   (12.1,  48.6,  18.9,  51.1),
    "Denmark":          ( 8.1,  54.6,  15.2,  57.8),
    "Estonia":          (21.8,  57.5,  28.2,  59.7),
    "Finland":          (20.0,  59.8,  31.6,  70.1),
    "France":           (-5.1,  41.3,   9.6,  51.1),
    "Germany":          ( 5.9,  47.3,  15.0,  55.1),
    "Greece":           (19.4,  34.8,  28.3,  41.8),
    "Hungary":          (16.1,  45.7,  22.9,  48.6),
    "Iceland":          (-24.6, 63.4, -13.5,  66.6),
    "Ireland":          (-10.5, 51.4,  -6.0,  55.4),
    "Italy":            ( 6.6,  36.6,  18.5,  47.1),
    "Latvia":           (21.0,  55.7,  28.2,  57.8),
    "Liechtenstein":    ( 9.5,  47.0,   9.7,  47.3),
    "Lithuania":        (20.9,  53.9,  26.8,  56.4),
    "Luxembourg":       ( 5.7,  49.4,   6.5,  50.2),
    "Malta":            (14.3,  35.8,  14.6,  36.1),
    "Monaco":           ( 7.38, 43.72,  7.44, 43.76),
    "Montenegro":       (18.4,  41.9,  20.4,  43.6),
    "Netherlands":      ( 3.4,  50.8,   7.2,  53.6),
    "North Macedonia":  (20.5,  40.9,  23.0,  42.4),
    "Norway":           ( 4.6,  57.9,  31.1,  71.2),
    "Poland":           (14.1,  49.0,  24.2,  54.9),
    "Portugal":         (-9.5,  36.9,  -6.2,  42.2),
    "Romania":          (22.0,  43.6,  29.7,  48.3),
    "Serbia":           (19.0,  42.2,  23.0,  46.2),
    "Slovakia":         (16.8,  47.7,  22.6,  49.6),
    "Slovenia":         (13.4,  45.4,  16.6,  46.9),
    "Spain":            (-9.3,  35.9,   4.3,  43.8),
    "Sweden":           (11.1,  55.3,  24.2,  69.1),
    "Switzerland":      ( 5.9,  45.8,  10.5,  47.8),
    "United Kingdom":   (-8.2,  49.9,   1.8,  60.9),
    # Extra countries not yet in dataset but in the lang feature map
    "Armenia":          (43.4,  38.8,  46.6,  41.3),
    "Belarus":          (23.2,  51.3,  32.8,  56.2),
    "Georgia":          (40.0,  41.0,  46.7,  43.6),
    "Kosovo":           (20.0,  41.9,  21.8,  43.3),
    "Turkey":           (26.0,  35.8,  44.8,  42.1),
    "Ukraine":          (22.1,  44.4,  40.2,  52.4),
}

# ── Cell 3: Helpers ─────────────────────────────────────────

def search_images(bbox, limit=2000, after=None):
    """One page of Mapillary image search within bbox."""
    w, s, e, n = bbox
    params = {
        "fields": "id,thumb_2048_url",
        "bbox":   f"{w},{s},{e},{n}",
        "limit":  limit,
    }
    if after:
        params["after"] = after
    r = requests.get(
        "https://graph.mapillary.com/images",
        headers=HEADERS, params=params, timeout=30
    )
    r.raise_for_status()
    j = r.json()
    data   = j.get("data", [])
    cursor = j.get("paging", {}).get("cursors", {}).get("after")
    return data, cursor


def collect_image_urls(country, bbox, need, existing_ids=None):
    """
    Paginate Mapillary until we have `need` unique image URLs
    not already downloaded (identified by image id).
    """
    existing_ids = existing_ids or set()
    urls, ids = [], []
    cursor = None
    while len(urls) < need:
        try:
            data, cursor = search_images(bbox, limit=2000, after=cursor)
        except Exception as ex:
            print(f"  [{country}] search error: {ex}")
            break
        for item in data:
            if item["id"] not in existing_ids and "thumb_2048_url" in item:
                urls.append(item["thumb_2048_url"])
                ids.append(item["id"])
                if len(urls) >= need:
                    break
        if not cursor or not data:
            break
        time.sleep(SEARCH_SLEEP)
    return urls[:need], ids[:need]


def download_one(args):
    url, dst_path = args
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        # Reject tiny / broken images
        if img.width < 300 or img.height < 200:
            return False
        img.save(dst_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def download_country(country, bbox, out_dir, need, split_label):
    out_dir = Path(out_dir) / country
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = {f.stem for f in out_dir.glob("*.jpg")}
    already  = len(existing)
    if already >= need:
        print(f"  [{split_label}] {country}: already has {already} images, skipping.")
        return 0

    still_need = need - already
    print(f"  [{split_label}] {country}: {already} existing → need {still_need} more ...")

    # Existing Mapillary ids (filenames are mly_<id>.jpg)
    existing_ids = {f.replace("mly_", "") for f in existing if f.startswith("mly_")}

    urls, ids = collect_image_urls(country, bbox, still_need * 2, existing_ids)
    if not urls:
        print(f"    No URLs found for {country}!")
        return 0

    tasks = []
    for url, img_id in zip(urls, ids):
        dst = out_dir / f"mly_{img_id}.jpg"
        if not dst.exists():
            tasks.append((url, str(dst)))

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(tqdm(
            pool.map(download_one, tasks),
            total=len(tasks),
            desc=f"{country[:20]:20s}",
            leave=False,
        ))
    downloaded = sum(results)
    print(f"    → {downloaded}/{len(tasks)} saved  (total now: {already + downloaded})")
    return downloaded


# ── Cell 4: Run train collection ────────────────────────────
print("=" * 60)
print("COLLECTING TRAINING IMAGES")
print("=" * 60)

total_dl = 0
for country, bbox in COUNTRY_BBOX.items():
    existing_path = Path(DRIVE_TRAIN) / country
    target = NEW_TRAIN + len(list(existing_path.glob("*.jpg"))) if existing_path.exists() else NEW_TRAIN
    n = download_country(country, bbox, DRIVE_TRAIN, target, "train")
    total_dl += n

print(f"\nTrain collection done — {total_dl:,} new images downloaded.")


# ── Cell 5: Run test collection ─────────────────────────────
print("=" * 60)
print("COLLECTING TEST IMAGES")
print("=" * 60)

total_dl = 0
for country, bbox in COUNTRY_BBOX.items():
    existing_path = Path(DRIVE_TEST) / country
    target = NEW_TEST + len(list(existing_path.glob("*.jpg"))) if existing_path.exists() else NEW_TEST
    n = download_country(country, bbox, DRIVE_TEST, target, "test")
    total_dl += n

print(f"\nTest collection done — {total_dl:,} new images downloaded.")


# ── Cell 6: Summary ─────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COUNTS")
print("=" * 60)
print(f"{'Country':30s}  {'Train':>6}  {'Test':>5}")
print("-" * 46)
for country in sorted(COUNTRY_BBOX.keys()):
    tr = len(list((Path(DRIVE_TRAIN) / country).glob("*.jpg"))) if (Path(DRIVE_TRAIN) / country).exists() else 0
    te = len(list((Path(DRIVE_TEST)  / country).glob("*.jpg"))) if (Path(DRIVE_TEST)  / country).exists() else 0
    print(f"{country:30s}  {tr:>6}  {te:>5}")
