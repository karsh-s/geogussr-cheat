import os
import csv
import time
import random
import requests
import googlemaps

API_KEY = 'AIzaSyBJnACLUXMKDxC6hb-ufer3JokopWwxpfk' # set this in your env
gmaps = googlemaps.Client(key=API_KEY)

STREETVIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
STREETVIEW_IMG_URL  = "https://maps.googleapis.com/maps/api/streetview"

def safe_slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")

def get_country_bounds(country_name: str):
    """Geocode a country name and return (min_lat, max_lat, min_lng, max_lng)."""
    res = gmaps.geocode(country_name)
    if not res:
        raise RuntimeError(f"Geocoding failed for: {country_name}")
    viewport = res[0]["geometry"]["viewport"]
    ne = viewport["northeast"]
    sw = viewport["southwest"]
    return sw["lat"], ne["lat"], sw["lng"], ne["lng"]

def streetview_metadata(lat: float, lng: float):
    params = {"location": f"{lat},{lng}", "key": API_KEY}
    r = requests.get(STREETVIEW_META_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def download_streetview(lat: float, lng: float, out_path: str, heading: int = 0, fov: int = 90, pitch: int = 0):
    """
    Downloads a Street View image. Output is JPEG by default; saving as .jpg is correct.
    """
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": API_KEY
    }
    r = requests.get(STREETVIEW_IMG_URL, params=params, timeout=60)
    r.raise_for_status()

    # Basic sanity check: make sure we actually received a JPEG
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype:
        raise RuntimeError(f"Unexpected content type: {ctype}")

    with open(out_path, "wb") as f:
        f.write(r.content)

def generate_streetview_images_for_country(
    country: str,
    n_images: int = 50,
    heading: int = 0,          # ONE photo per pano (simpler: exactly 50 JPGs)
    max_tries: int = 50000,
    sleep_s: float = 0.05,
    root_dir: str = "streetview_dataset"
):
    """
    Saves exactly n_images JPGs for one country (unique pano_id).
    """
    min_lat, max_lat, min_lng, max_lng = get_country_bounds(country)

    country_dir = os.path.join(root_dir, safe_slug(country))
    os.makedirs(country_dir, exist_ok=True)

    seen_panos = set()
    saved = 0
    tries = 0

    while saved < n_images and tries < max_tries:
        tries += 1

        # Random point in the country viewport (simple, but may be inefficient)
        lat = random.uniform(min_lat, max_lat)
        lng = random.uniform(min_lng, max_lng)

        meta = streetview_metadata(lat, lng)
        if meta.get("status") != "OK":
            continue

        pano_id = meta.get("pano_id")
        if not pano_id or pano_id in seen_panos:
            continue

        pano_loc = meta.get("location", {})
        pano_lat = pano_loc.get("lat")
        pano_lng = pano_loc.get("lng")
        if pano_lat is None or pano_lng is None:
            continue

        filename = f"{safe_slug(country)}_{saved:05d}.jpg"
        out_path = os.path.join(country_dir, filename)

        try:
            download_streetview(pano_lat, pano_lng, out_path, heading=heading)
        except Exception:
            # occasionally metadata says OK but image fetch fails; just skip
            continue

        seen_panos.add(pano_id)
        saved += 1
        time.sleep(sleep_s)

        if saved % 10 == 0:
            print(f"{country}: saved {saved}/{n_images} (tries={tries})")

    if saved < n_images:
        print(f"WARNING: {country}: only saved {saved}/{n_images} after {tries} tries")

    return country_dir

def generate_dataset(
    countries,
    n_images_per_country: int = 50,
    root_dir: str = "streetview_dataset",
    metadata_csv: str = "metadata.csv"
):
    """
    Generates n_images_per_country JPGs for each country and writes ONE metadata.csv.
    """
    os.makedirs(root_dir, exist_ok=True)
    csv_path = os.path.join(root_dir, metadata_csv)

    fieldnames = [
        "country", "file",
        "query_lat", "query_lng",
        "pano_lat", "pano_lng",
        "pano_id", "heading"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for country in countries:
            print(f"\n=== Generating {n_images_per_country} images for {country} ===")

            # bounds for this country
            min_lat, max_lat, min_lng, max_lng = get_country_bounds(country)
            country_dir = os.path.join(root_dir, safe_slug(country))
            os.makedirs(country_dir, exist_ok=True)

            seen_panos = set()
            saved = 0
            tries = 0
            heading = 0

            while saved < n_images_per_country and tries < 50000:
                tries += 1
                qlat = random.uniform(min_lat, max_lat)
                qlng = random.uniform(min_lng, max_lng)

                meta = streetview_metadata(qlat, qlng)
                if meta.get("status") != "OK":
                    continue

                pano_id = meta.get("pano_id")
                if not pano_id or pano_id in seen_panos:
                    continue

                pano_loc = meta.get("location", {})
                plat = pano_loc.get("lat")
                plng = pano_loc.get("lng")
                if plat is None or plng is None:
                    continue

                filename = f"{safe_slug(country)}_{saved:05d}.jpg"
                out_path = os.path.join(country_dir, filename)

                try:
                    download_streetview(plat, plng, out_path, heading=heading)
                except Exception:
                    continue

                writer.writerow({
                    "country": country,
                    "file": os.path.join(safe_slug(country), filename),
                    "query_lat": qlat,
                    "query_lng": qlng,
                    "pano_lat": plat,
                    "pano_lng": plng,
                    "pano_id": pano_id,
                    "heading": heading
                })

                seen_panos.add(pano_id)
                saved += 1
                time.sleep(0.05)

            if saved < n_images_per_country:
                print(f"WARNING: {country}: only saved {saved}/{n_images_per_country} after {tries} tries")
            else:
                print(f"Done: {country}: saved {saved}/{n_images_per_country}")

    print(f"\nAll done. JPGs saved under: {root_dir}/ and metadata at {root_dir}/{metadata_csv}")

if __name__ == "__main__":
    # Put any countries you want here. For “every country”, you need a list of country names.
    countries = ["Australia"]  # add more, e.g. ["Australia", "Japan", "Brazil"]
    generate_dataset(countries, n_images_per_country=50)