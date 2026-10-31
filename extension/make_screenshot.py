#!/usr/bin/env python3
"""Pad or scale a screenshot to the exact 1280x800 the Web Store requires.

    python3 make_screenshot.py shot.png [more.png ...]

Writes store_screenshot_1.png, _2.png ... beside this script. The image is
scaled down to fit (never up, so nothing goes soft) and centred on a tile
matching the extension's own background.
"""
import sys, os
from PIL import Image

W, H = 1280, 800
BG = (17, 19, 28)

def main(paths):
    for i, src in enumerate(paths, 1):
        img = Image.open(src).convert('RGB')
        scale = min(W / img.width, H / img.height, 1.0)
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)),
                             Image.LANCZOS)
        canvas = Image.new('RGB', (W, H), BG)
        canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'store_screenshot_{i}.png')
        canvas.save(out)
        print(f'{out}  {W}x{H}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
