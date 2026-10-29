#!/usr/bin/env bash
# Shared environment for the Cosmos-PLV hackathon scripts (Environment A: PLV).
# Source this file: `source scripts/hackathon/env.sh`.
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}
# shellcheck disable=SC1091
_plv_nounset=0; [[ $- == *u* ]] && _plv_nounset=1 && set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${PLV_ENV:-robodiff}"
(( _plv_nounset )) && set -u; unset _plv_nounset
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export WANDB_MODE=${WANDB_MODE:-offline}
# robomimic 0.3.0 imports mujoco_py, whose cymj extension links the system
# libLLVM-12 against libffi 7. The conda env ships libffi.so.7 as a symlink to
# libffi 8, so preload the system copy when it exists.
if [[ -f /usr/lib/x86_64-linux-gnu/libffi.so.7 ]]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}
fi
