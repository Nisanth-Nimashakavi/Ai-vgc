#!/usr/bin/env bash
# Download the published checkpoints into data/models (GitHub release "models-v1").
#   scripts/get_models.sh            # models in use (~170 MB)
#   scripts/get_models.sh --all      # plus the archive (~250 MB more)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${REPO:-nimnim111/ai-vgc}
TAG=${TAG:-models-v1}
assets=(models-current.tar)
[[ ${1:-} == --all ]] && assets+=(models-archive.tar)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for a in "${assets[@]}" SHA256SUMS; do
  curl -fL --retry 3 -o "$tmp/$a" "https://github.com/$REPO/releases/download/$TAG/$a"
done
(cd "$tmp" && grep -F -f <(printf '%s\n' "${assets[@]}") SHA256SUMS | sha256sum -c -)
for a in "${assets[@]}"; do tar xf "$tmp/$a"; done
echo "models in data/models:"; ls data/models
