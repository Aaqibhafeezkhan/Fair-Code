# brag-output asset sources

`brag-output/` holds the generated launch-video project (a HyperFrames composition, its snapshots and the
rendered `brag.mp4` / `brag.jpg`). It is not part of the Fair Code library or its docs. This file records
where each bundled third-party asset came from and what is, and is not, known about its licence, so the
folder can be kept, trimmed or removed on facts. A test (`tests/test_brag_assets.py`) fails if a file is
added under `composition/assets/` without a row here.

**Status legend:** `known` = the licence is in this repo; `unverified` = the source is recorded but the
licence terms are not in the repo and must be confirmed before anyone relies on redistribution.

| Asset (under `composition/assets/`) | Source recorded in the repo | Licence | Status |
|-------------------------------------|-----------------------------|---------|--------|
| `fonts/instrumentserif-400.woff2`, `instrumentserif-400italic.woff2` | Instrument Serif, same family as `assets/fonts/InstrumentSerif-Italic.ttf` | SIL OFL 1.1, text in `assets/fonts/OFL-InstrumentSerif.txt` | known |
| `fonts/archivo-variable.woff2` | Archivo, same family as `assets/fonts/Archivo[wdth,wght].ttf` | SIL OFL 1.1, `assets/fonts/OFL-Archivo.txt` | known |
| `fonts/ibmplexmono-400/500/600/700.woff2` | IBM Plex Mono, same family as `assets/fonts/IBMPlexMono-*.ttf` | SIL OFL 1.1, `assets/fonts/OFL-IBMPlexMono.txt` | known |
| `music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (and its `cues/` analysis files) | named in `../composition-brief.md` as the bundled preset track "by ende.app" from the video tool's catalogue | not recorded here | unverified |
| `sfx/impact/impactBell_heavy_000.ogg`, `impactBell_heavy_003.ogg`, `impactSoft_medium_000.ogg` | "the copied" sound effects chosen in `../composition-brief.md` from the video tool's bundled set | not recorded here | unverified |
| `sfx/interface/bong_001.ogg`, `click_002.ogg` | same bundled set | not recorded here | unverified |
| `sfx/keyboard/keypress-004.wav`, `keypress-006.wav`, `keypress-011.wav` | same bundled set | not recorded here | unverified |

The rendered `brag.mp4` mixes the music and sound effects above, so it carries the same open question.

## To resolve

1. Find the licence of the bundled music and sound-effect set in the video tool's own documentation and
   paste it (and any attribution wording) into the rows above, changing the status to `known`.
2. If it does not allow redistribution in a public MIT repository, remove the `music/` and `sfx/` files and
   re-render without them, or move the media to a release asset; keep this file as the record.
3. Optionally move `brag.mp4` (1.3 MB) and the music (2.7 MB) out of git history's way with Git LFS or a
   release attachment, since they are most of the folder's 5.4 MB.
