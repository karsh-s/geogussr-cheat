# Privacy Policy — GeoCLIP Assistant

**Last updated: 31 October 2025**

GeoCLIP Assistant is a browser extension that displays a country prediction for
the Street View imagery currently on screen. The prediction is produced by a
model running on **your own computer**. This policy describes every piece of
data the extension touches.

## What the extension collects

**Nothing is collected, transmitted to us, or stored on any server we control.**

The developer operates no backend, no analytics, no telemetry, no error
reporting, and no advertising. There is no account, no sign-in, and no
identifier of any kind.

## What the extension stores on your device

Three preferences, held in `chrome.storage.sync`:

| Setting | Purpose |
|---|---|
| Server address | Where your local prediction server is listening (default `http://127.0.0.1:8010`) |
| Hotkey | The key that shows and hides the panel |
| Auto-analyse | Whether to run on every new panorama, or only when you ask |

Because this uses `storage.sync`, Chrome itself may replicate these three
values across browsers signed in to your Google account. That replication is
performed by Chrome, under Google's privacy policy, not by this extension.
Switching to local-only storage is possible by signing out of Chrome sync.

Clearing these is done by removing the extension.

## What the extension reads from the page

On the supported sites listed below, the extension watches the page's own
network activity for Street View **panorama identifiers** — short opaque IDs
such as `CAoSLEFGMVFpcE...` that appear in the URLs of image tiles the page is
already loading. It reads nothing else: not your account, not your game
history, not page text, not cookies, not form fields, not other tabs.

## Where that identifier goes

The panorama identifier is sent to **the address you configured**, which by
default is `127.0.0.1` — your own machine. Nothing leaves your computer. If
you deliberately change the server address to a remote host, the identifier
will go there instead; that is your choice and your responsibility.

The extension has permission to contact `127.0.0.1` and `localhost` only. It
cannot reach any other host.

## Sites the extension runs on

`geoguessr.com`, `openguessr.com`, `worldguessr.com`, `freeguessr.com`.

It is inactive on every other site. It additionally disables itself when the
page URL indicates a multiplayer mode (duels, team duels, battle royale,
parties, lobbies, ranked and similar).

## Permissions, and why each is needed

- **`storage`** — to remember the three preferences above.
- **Access to `127.0.0.1` / `localhost`** — to send the panorama identifier to
  your local prediction server and receive the result. The whole host is
  requested rather than a single port because the port is user-configurable.
- **Access to the four guessing sites** — to read the panorama identifier from
  the page's own resource loads and to draw the on-screen panel.

The extension requests no other permissions. It does not use `tabs`,
`activeTab`, `webRequest`, `cookies`, `history`, or `nativeMessaging`, and it
loads no remote code.

## Children

The extension is not directed at children and collects no data from anyone.

## Changes

Material changes to this policy will be published at this URL with an updated
date, and shipped alongside a new extension version.

## Contact

Open an issue at <https://github.com/karsh-s/geogussr-cheat/issues>.
