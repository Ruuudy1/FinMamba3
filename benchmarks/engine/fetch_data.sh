#!/usr/bin/env bash
# Download the Polymarket validation set from the public HuggingFace dataset and extract it where the
# backtester expects it. Idempotent: a no-op once polymarket.db is present. Needs only curl and tar, so
# it runs on the host (then mount data/ into the image) or inside an image that has curl. No HF token is
# required because sj-hryi/FinMamba3 is public.
set -euo pipefail
DEST="${FINMAMBA_DATA_VAL:-data/validation}"
if [ -f "$DEST/polymarket.db" ]; then
  echo "validation data already present at $DEST"
  exit 0
fi
URL="https://huggingface.co/datasets/sj-hryi/FinMamba3/resolve/main/data/polymarket/validation.tar.gz"
ARCHIVE="$(mktemp -t validation.XXXXXX.tar.gz)"
mkdir -p "$DEST"
echo "downloading validation.tar.gz (345 MB) from HuggingFace ..."
curl -fL "$URL" -o "$ARCHIVE"
echo "extracting into $DEST ..."
tar xzf "$ARCHIVE" -C "$DEST"
rm -f "$ARCHIVE"
if [ ! -f "$DEST/polymarket.db" ]; then
  echo "ERROR: polymarket.db not found after extraction; the archive layout may have a top-level dir." >&2
  echo "Inspect with: tar tzf <archive> | head, then re-extract with the right --strip-components." >&2
  exit 1
fi
echo "done: $DEST"
