#!/usr/bin/env bash
# Builds the Chrome Web Store upload package.
# Ships only what the extension loads at runtime: no README, no model
# metadata (unused by the localhost build), no build script itself.
set -euo pipefail
cd "$(dirname "$0")"

VER=$(python3 -c "import json;print(json.load(open('manifest.json'))['version'])")
OUT="geoclip-assistant-${VER}.zip"
rm -f "$OUT"

zip -r -X "$OUT" \
  manifest.json background.js content.js europe_paths.js \
  popup.html popup.js icons \
  -x '.*' -x '__MACOSX/*' >/dev/null

echo "$OUT  ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT"
