"""
Usage:
Training:
python train.py --config-name=son_train_diffusion_transformer_real_hybrid_workspace task=son_push_image_abs
"""

import sys

# print every line immediately, without buffering (no batched output)
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# lets configs use expressions
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    # path: ~/diffusion_policy/diffusion_policy/config  <-- the yaml config data seems to be loaded from here
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # get the __target__ class in cfg as cls
    cls = hydra.utils.get_class(cfg._target_)
    # create a workspace instance from that class, then call run
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
