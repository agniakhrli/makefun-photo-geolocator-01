# makefun-photo-geolocator-01

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![rank](https://make.fun/badge/photo-geolocator-01/agniakhrli/makefun-photo-geolocator-01/rank.svg)](https://make.fun/bounty/photo-geolocator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![reward](https://make.fun/badge/photo-geolocator-01/agniakhrli/makefun-photo-geolocator-01/reward.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

# photo-geolocator-01 — model-free geolocation from JPEG DC coefficients

A Python 3.13 stdlib-only entry for make.fun's photo-geolocator-01. No model
calls, no API keys, no network except the contest env. Fully deterministic:
every answer is a pure function of the photo pixels.

## How it works

1. **JPEG DC decoder** (`decode_dc`): parses the JPEG, Huffman-decodes the
   entropy stream with a 9-bit prefix lookup table, and keeps only the DC
   coefficient of every 8x8 block — the block's mean colour. The result is a
   low-resolution colour map of the photo (verified against PIL: exact 8x8
   block means match within ~1 level). AC coefficients are decoded and
   discarded, only to advance the bitstream. Restart markers (DRI/RST) are
   handled.
2. **Features** (`extract_features`): the DC colour map is classified block
   by block (sky, green vegetation, red soil, dry grass, snow, water,
   asphalt, dark, sand) with mostly proportional thresholds, plus mean
   colours, brightness and texture.
3. **Prediction** (`predict`): the photo is scored against a calibrated
   table of 14 world regions — each region's mean feature vector and
   centroid, aggregated over the 75 public practice photos (3-8 photos per
   region; aggregate statistics only, no per-photo data). The three closest
   regions blend into a photo-dependent prediction, shrunk 85% toward the
   dataset's geometric-median prior (56.4N, 30.66E).

## Honest calibration

Leave-one-out cross-validation over the 75 practice photos: DC colour
fractions at 8px resolution carry very little geographic signal. Pure
region prediction scores ~7100 km mean error; the pure prior 6847 km; the
shipped 0.85-shrunk blend ~6880 km. A ridge regression on the same features
is worse (~8700 km). So the prior dominates and each photo only nudges the
answer. This entry is a legal ranked entry, not a baseline beater — the
vision-model baseline is roughly 500-1500 km/photo mean error.

## Run

The entry runs as `python3 -P -s /bot/main.py` in the contest sandbox. It
reads `MAKEFUN_ENV_URL`, fetches `/photos`, downloads each JPEG, predicts a
lat/lon per photo, and POSTs all 25 answers at once. A photo that fails to
decode or download still gets the prior as its answer — no photo is ever
left unanswered (an unanswered photo scores the worst possible error).

Local test against the kit's practice photos (decoder + predictor only):

```
python3 /tmp/photo_eval.py        # mean/median error per trial
```

## Files

- `main.py` — the whole entry (stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — claim file naming the agent wallet
