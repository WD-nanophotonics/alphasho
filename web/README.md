# AlphaSho Web

Static, serverless browser shogi built with TypeScript and Vite.

## Features

- Local two-player shogi.
- CPU-only heuristic alpha-beta player in a dedicated Web Worker.
- Easy (100 ms/depth 2), Normal (500 ms/depth 6), and Hard (2 s/depth 64).
- PeerJS/WebRTC rooms with a host-authoritative casual protocol.
- Japanese, Simplified Chinese, and English UI.
- Responsive board, installable PWA, and offline local/CPU play.

No Python runtime, cshogi binary, PyTorch checkpoint, or neural-network model is
included in the deployment. `cshogi` is used only by the development parity
generator in `../benchmarks/web_rule_parity.py`.

## Development

```powershell
npm.cmd install
npm.cmd test
npm.cmd run build
npm.cmd run dev
```

The production base URL is `/alphasho/`. To run browser tests, install Playwright's
browsers once with `npx playwright install`, then use `npm.cmd run test:e2e`.

## Protocol and privacy

Room codes are generated from `crypto.getRandomValues()`. PeerJS's public PeerServer
is used for signaling; moves and state synchronization travel over a WebRTC data
channel. The host validates sequence, position hash, turn, and move legality.
Rooms support two players only and have no accounts, chat, ratings, spectators, or
cloud persistence.

## Licenses

The files in this `web/` project are MIT licensed. Piece artwork is from
[Sunfish Shogi Images](https://sunfish-shogi.github.io/shogi-images/) and is CC0 1.0.
PeerJS is bundled from its locked npm dependency and retains its own license.
