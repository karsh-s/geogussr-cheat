# GeoCLIP Assistant — Chrome extension

Shows what country your trained model thinks a Street View panorama is
in, with a map and the visual evidence behind the guess.

It predicts from pixels. It does not read the game's answer, place pins,
or submit guesses.

## Install

1. Start the model server:

       python3 geoguessr_server.py

2. Open `chrome://extensions`, turn on **Developer mode** (top right).
3. **Load unpacked** → select this `extension/` folder.
4. Open a supported site and press **G**.

## Supported sites

GeoGuessr, OpenGuessr, WorldGuessr, FreeGuessr — all serve Google Street
View panoramas, so the same model works on each.

## How it finds the imagery

Every Street View tile the page loads carries the panorama id in its
URL. The content script reads it from the page's own network activity,
and sends only that id to the local server, which fetches the panorama
and reprojects it into four compass views using the same maths that
built the training set. Nothing is screenshotted, so the game interface
never reaches the model.

## The panel

* **Map** — every country shaded by probability, brightest for the leader
* **Most likely** — top five with confidence bars
* **Why** — the GeoTips cues the image most supports (bollards, road
  lines, architecture, vegetation…)

Evidence is scored against the image only when the server runs a *probe*
model, where CLIP's image and text spaces still align. With a fine-tuned
model the image encoder has moved, so the panel lists the country's
typical clues and labels them as such instead of inventing a score.

Predictions accumulate across panoramas as you move. **New round**
clears them.

## Settings

Click the extension icon to set the server URL, change the hotkey, or
turn on automatic analysis for each new panorama.
