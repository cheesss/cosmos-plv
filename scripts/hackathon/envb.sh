#!/usr/bin/env bash
# Environment B (Cosmos): Python 3.11, torch 2.8 cu128, transformers >= 5.15.
# Source this file: `source scripts/hackathon/envb.sh`.
export COSMOS_ENV=${COSMOS_ENV:-$HOME/.venvs/cosmos-plv}
export PATH="$COSMOS_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
