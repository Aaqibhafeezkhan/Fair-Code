# brag-output asset sources and licences

`brag-output/` holds the generated launch-video project (a HyperFrames composition made with the `/brag`
plugin, its snapshots and the rendered `brag.mp4` / `brag.jpg`). It is not part of the Fair Code library or
its docs. This file records where each bundled third-party asset came from and the licence evidence for it.
A test (`tests/test_brag_assets.py`) fails if a file is added under `composition/assets/` without a row here.

**Status legend:** `known` = the licence is in this repo or was read from the publisher's own page (with the
date); `check` = a caveat is listed under "Open points".

## Credits

- Music: "Happy Beats / Business Moves Vol. 12" by [ende.app](https://ende.app/), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit given here although the site says
  attribution is optional.
- Sound effects: [Kenney](https://kenney.nl/), public domain
  ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)).
- Fonts: Instrument Serif, Archivo and IBM Plex Mono under the SIL Open Font License 1.1.
- Video generation: [HyperFrames](https://hyperframes.heygen.com/) via the MIT-licensed `/brag` plugin
  (its README credits the music to ende.app and the sound effects to Kenney, and its `sfx-analysis.md`
  names Kenney as the source of the whole bundled sound-effect set).

## Assets

| Asset (under `composition/assets/`) | Source | Licence | Status |
|-------------------------------------|--------|---------|--------|
| `fonts/instrumentserif-400.woff2`, `instrumentserif-400italic.woff2` | Instrument Serif, same family as `assets/fonts/InstrumentSerif-Italic.ttf` | SIL OFL 1.1, text in `assets/fonts/OFL-InstrumentSerif.txt` | known |
| `fonts/archivo-variable.woff2` | Archivo, same family as `assets/fonts/Archivo[wdth,wght].ttf` | SIL OFL 1.1, `assets/fonts/OFL-Archivo.txt` | known |
| `fonts/ibmplexmono-400/500/600/700.woff2` | IBM Plex Mono, same family as `assets/fonts/IBMPlexMono-*.ttf` | SIL OFL 1.1, `assets/fonts/OFL-IBMPlexMono.txt` | known |
| `music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (and its `cues/` analysis files) | ende.app, "Happy Beats / Business Moves", bundled with the `/brag` plugin | CC BY 4.0, commercial use allowed (ende.app homepage, read 2026-10-14) | check (see open points) |
| `sfx/impact/impactBell_heavy_000.ogg`, `impactBell_heavy_003.ogg`, `impactSoft_medium_000.ogg` | Kenney, via the `/brag` plugin's bundled set | CC0 (kenney.nl/support, read 2026-10-14) | known |
| `sfx/interface/bong_001.ogg`, `click_002.ogg` | Kenney, same set | CC0 | known |
| `sfx/keyboard/keypress-004.wav`, `keypress-006.wav`, `keypress-011.wav` | Kenney per the plugin's `sfx-analysis.md` ("Source: Kenney") | CC0 | known |

## Open points

1. The ende.app homepage states "Everything is under CC BY 4.0 ... commercial use is allowed", with credit
   described as optional; the individual page of Vol. 12 was not checked. If it says otherwise, follow it.
2. kenney.nl/support says "all game assets ... are public domain licensed (CC0)" without naming audio
   explicitly; the sound packs are listed on the same site as CC0.
3. `brag.mp4` (1.3 MB) and the music (2.7 MB) are most of the folder's 5.4 MB; Git LFS or a release
   attachment would keep clones small. Not required by the licences.
