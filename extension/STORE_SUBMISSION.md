# Chrome Web Store submission — GeoCLIP Assistant

Everything below is copy-paste ready for
<https://chrome.google.com/webstore/devconsole>.
Build the upload with `./build.sh` → `geoclip-assistant-1.0.0.zip`.

---

## 1. Before you can submit

- **One-time US$5 registration fee**, paid in the Developer Dashboard. Not
  refundable, covers the account forever.
- **Publisher display name.** Set this under *Account → Publisher settings*
  before publishing. If you leave it blank, the listing shows the Google
  account name. Pick something you're willing to have attached to this
  publicly — it appears on the listing and cannot be hidden.
- **A verified contact email**, shown on the listing. Use one you don't mind
  being public.

---

## 2. Package

`geoclip-assistant-1.0.0.zip` — 32 KB, 11 files, no remote code,
no third-party libraries.

---

## 3. Listing fields

**Name**
```
GeoCLIP Assistant
```

**Short description** (132 char limit; this is 128)
```
Predicts the country from Street View imagery using your own trained CLIP model, and shows the visual evidence behind the guess.
```

**Category:** Just for Fun
**Language:** English (United States)

**Detailed description**
```
GeoCLIP Assistant shows a country prediction for the Street View panorama
currently on screen, together with the visual cues the model used to reach it.

IMPORTANT: THIS EXTENSION REQUIRES A SEPARATE LOCAL SERVER

The extension is a front-end only. It contains no model. All prediction is
done by an open-source Python server you run yourself on your own computer:

  https://github.com/karsh-s/geogussr-cheat

Without that server running, the extension will display "server offline" and
do nothing else. If you are not prepared to install and run Python locally,
this extension will not be useful to you.

HOW IT WORKS

The page loads Street View image tiles, and those tile URLs contain a
panorama identifier. The extension reads that identifier, passes it to your
local server, and the server fetches the panorama itself, reprojects it into
four compass-heading views, and runs a CLIP-based classifier trained on
polygon-verified Street View coverage across 41 European countries.

The result is shown in a draggable panel: the top five countries with
confidence bars, a map highlighting the leading guess, and a list of the
visual cues — road markings, bollards, licence plates, architecture,
vegetation — that pushed the model toward it. Predictions accumulate across
every panorama you visit, so the estimate sharpens as you move.

SOLO PLAY ONLY

The extension deliberately disables itself when the page URL indicates a
multiplayer mode — duels, team duels, battle royale, parties, lobbies,
ranked and similar. In those modes the panel shows "solo play only" and no
prediction is made.

Using any assistance tool may still breach the terms of service of the site
you use it on, including in single-player. That is between you and that
site; please read their rules.

PRIVACY

No analytics, no telemetry, no accounts, no developer-operated server. The
only thing that leaves the page is the panorama identifier, and it goes to
127.0.0.1 — your own machine. The extension can reach no other host.

SUPPORTED SITES

geoguessr.com, openguessr.com, worldguessr.com, freeguessr.com

OPEN SOURCE

Extension and server: https://github.com/karsh-s/geogussr-cheat
```

---

## 4. Privacy tab

**Single purpose description**
```
Display a country prediction, and the visual evidence behind it, for the
Street View panorama currently shown on a geography-guessing website.
```

**Permission justifications**

`storage`
```
Stores three user preferences: the address of the user's local prediction
server, the keyboard shortcut that shows and hides the panel, and whether to
analyse automatically or only on request. No other data is stored.
```

Host permission `http://127.0.0.1/*` and `http://localhost/*`
```
The extension performs no inference itself. It sends the panorama identifier
to a prediction server that the user installs and runs on their own machine,
and receives the result. The whole loopback host is requested rather than one
port because the user can change the port in the extension's options, and the
server's default port may conflict with other local software. No non-loopback
host is requested, so the extension cannot contact any remote server.
```

Content script host access (the four guessing sites)
```
To read the Street View panorama identifier out of the URLs of image tiles
the page is already loading, and to draw the prediction panel over the page.
The extension reads no other page content: no account data, no page text, no
cookies, no form fields.
```

**Remote code:** No, I am not using remote code.

**Data usage — tick nothing.** The extension collects none of the listed
categories. Then certify:
- [x] I do not sell or transfer user data to third parties, outside of the approved use cases
- [x] I do not use or transfer user data for purposes unrelated to my item's single purpose
- [x] I do not use or transfer user data to determine creditworthiness or for lending purposes

**Privacy policy URL**
```
https://github.com/karsh-s/geogussr-cheat/blob/main/PRIVACY.md
```

---

## 5. Graphics you still need to make

| Asset | Size | Required |
|---|---|---|
| Store icon | 128×128 PNG | Yes — `icons/icon128.png` works |
| Screenshot | 1280×800 or 640×400 PNG | Yes, at least 1, up to 5 |
| Small promo tile | 440×280 PNG | Only if you want to be featured |
| Marquee promo tile | 1400×560 PNG | Only if you want to be featured |

Take the screenshot with the panel open and the server running, then run
`python3 make_screenshot.py <your.png>` to pad it to exactly 1280×800.

---

## 6. Distribution

- **Visibility:** Public, or Unlisted if you'd rather share by link only.
  Unlisted still goes through the same review.
- **Regions:** all, unless you have a reason to narrow.

---

## 7. What to expect from review

Typical turnaround is a few hours to a few business days. First submissions
and anything touching host permissions often sit at the longer end.

Realistic grounds for rejection on this one, and what to do:

- **"Does not function" / broken on install.** The most likely rejection. A
  reviewer installs it, opens a supported site, sees "server offline", and
  marks it non-functional. The local-server requirement is already stated in
  the first lines of the description, which is the main defence. If rejected
  on this ground, reply with a link to the server repo and setup steps.
- **Host permission scope.** If they object to the whole loopback host,
  narrow `host_permissions` to the single default port `http://127.0.0.1:8010/*`
  and drop the configurable-port field from the popup.
- **A complaint from one of the four sites.** A site operator can ask Google
  to remove an extension that interferes with their service. If that happens
  the listing is taken down; it is not appealable in any practical sense.
