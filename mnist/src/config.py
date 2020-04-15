import inspect
import torch

from pathlib import Path
from typing import Union


def get_local_time():
    import datetime
    import pytz
    timezone = pytz.timezone('Europe/Rome')
    return datetime.datetime.now(tz=timezone).strftime('%d-%m-%H-%M-%S')


def get_last_modified(dirname: Union[Path, str]) -> Union[str, Path]:
    dirname = Path(dirname) if isinstance(dirname, str) else dirname
    if len(list(dirname.iterdir())) == 0:
        return ""
    files = [(file, file.stat().st_mtime) for file in dirname.iterdir()]
    return sorted(files, key=lambda x: x[1])[-1][0]


project_dir = Path(inspect.getsourcefile(lambda: 0)).resolve().parent.parent
data_dir = project_dir / 'data'

local_time = get_local_time()
log_dir = project_dir / 'runs'
log_dir.mkdir(exist_ok=True)
log_dir_local_time = log_dir / local_time
log_dir_last_modified = get_last_modified(log_dir)
checkpoints_dir = project_dir / 'checkpoints'
checkpoints_dir.mkdir(exist_ok=True)
checkpoints_dir_local_time = checkpoints_dir / f"{local_time}.pt"
checkpoints_dir_last_modified = get_last_modified(checkpoints_dir)

interpolations_dir = project_dir / 'interpolations'
interpolations_dir.mkdir(exist_ok=True)

knobs = dict()
if torch.cuda.is_available():
    knobs['device'] = torch.device('cuda')
    # print('Running on GPU (device).')
else:
    knobs['device'] = torch.device('cpu')
    print('The program is running on CPU: CUDA not available.')
knobs['device'] = torch.device('cuda')
knobs['num_epochs'] = 100
knobs['batch_size'] = 100
knobs['lr_encoder'] = 1e-4
knobs['lr_decoder'] = 1e-4
knobs['lr_discriminator'] = knobs['lr_encoder']
knobs['hidden_dim'] = 8
knobs['lambda_reconstruction'] = 1.
knobs['lambda_penalty'] = 1.
knobs['wasserstein_penalty'] = knobs['lambda_penalty']
knobs['lambda_fooling_term'] = 1.
knobs['gamma'] = 0.2
knobs['sigma'] = 1.
knobs['time_to_collect'] = 100
knobs['max_norm_encoder'] = 5.  # (average norm of gradients: 50)
knobs['max_norm_decoder'] = 15.  # (average norm of gradients: 150)
knobs['max_norm_discriminator'] = knobs['max_norm_encoder']
knobs['clip_gradient'] = False
knobs['fast_models'] = True
knobs['resume'] = True

"""
----------------------------------------------------------------------------------------
Configurations
----------------------------------------------------------------------------------------
Configurations #1
Model: acwwai
lambda_reconstruction: 0.05
clip_gradient: False


Configurations #2
max_norm_*: 
----------------------------------------------------------------------------------------
"""
