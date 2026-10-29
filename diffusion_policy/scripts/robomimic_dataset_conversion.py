if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter

_worker_converter = None
_worker_do_eval = False


def init_worker(path, do_eval):
    global _worker_converter, _worker_do_eval
    _worker_converter = RobomimicAbsoluteActionConverter(path)
    _worker_do_eval = do_eval


def worker(idx):
    converter = _worker_converter
    do_eval = _worker_do_eval
    if do_eval:
        abs_actions, info = converter.convert_and_eval_idx(idx)
    else:
        abs_actions = converter.convert_idx(idx)
        info = dict()
    return abs_actions, info

@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
@click.option('-e', '--eval_dir', default=None, help='directory to output evaluation metrics')
@click.option('-n', '--num_workers', default=None, type=int)
def main(input, output, eval_dir, num_workers):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.parent.is_dir()
    assert not output.is_dir()

    do_eval = False
    if eval_dir is not None:
        eval_dir = pathlib.Path(eval_dir).expanduser()
        assert eval_dir.parent.exists()
        do_eval = True
    
    converter = RobomimicAbsoluteActionConverter(input)
    demo_keys = converter.demo_keys

    # run
    with multiprocessing.Pool(
            num_workers,
            initializer=init_worker,
            initargs=(input, do_eval)) as pool:
        results = list(tqdm(
            pool.imap(worker, range(len(converter))),
            total=len(converter),
            desc="Converting actions"))
    
    # save output
    print('Copying hdf5')
    shutil.copy(str(input), str(output))

    # modify action
    with h5py.File(output, 'r+') as out_file:
        for i in tqdm(range(len(converter)), desc="Writing to output"):
            abs_actions, info = results[i]
            demo = out_file[f'data/{demo_keys[i]}']
            demo['actions'][:] = abs_actions
    
    # save eval
    if do_eval:
        eval_dir.mkdir(parents=False, exist_ok=True)

        print("Writing error_stats.pkl")
        infos = [info for _, info in results]
        pickle.dump(infos, eval_dir.joinpath('error_stats.pkl').open('wb'))

        print("Generating visualization")
        metrics = ['pos', 'rot']
        metrics_dicts = dict()
        for m in metrics:
            metrics_dicts[m] = collections.defaultdict(list)

        for i in range(len(infos)):
            info = infos[i]
            for k, v in info.items():
                for m in metrics:
                    metrics_dicts[m][k].append(v[m])

        from matplotlib import pyplot as plt
        plt.switch_backend('PDF')

        fig, ax = plt.subplots(1, len(metrics))
        for i in range(len(metrics)):
            axis = ax[i]
            data = metrics_dicts[metrics[i]]
            for key, value in data.items():
                axis.plot(value, label=key)
            axis.legend()
            axis.set_title(metrics[i])
        fig.set_size_inches(10,4)
        fig.savefig(str(eval_dir.joinpath('error_stats.pdf')))
        fig.savefig(str(eval_dir.joinpath('error_stats.png')))


if __name__ == "__main__":
    main()
