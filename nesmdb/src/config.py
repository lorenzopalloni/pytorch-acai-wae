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
nesmdb_wav_dir = data_dir / 'nesmdb_wav'
valid_wav_dir = nesmdb_wav_dir / 'valid_wav'
test_wav_dir = nesmdb_wav_dir / 'test_wav'
resources_out_dir = project_dir / 'tests' / 'resources_out'
audio_pool_dir = data_dir / 'audio_pool'
audio_pool_csv = audio_pool_dir / 'audio_pool.csv'

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

torch.manual_seed(42)

knobs = dict()
if torch.cuda.is_available():
    knobs['device'] = torch.device('cuda')
    # print('Running on GPU (device).')
else:
    knobs['device'] = torch.device('cpu')
    print('The program is running on CPU: CUDA not available.')
knobs['device'] = torch.device('cuda')
knobs['num_epochs'] = 10_000
knobs['batch_size'] = 100
knobs['lr_encoder'] = 1e-3
knobs['lr_decoder'] = 1e-3
knobs['lr_discriminator'] = knobs['lr_encoder']
knobs['hidden_dim'] = 8
knobs['lambda_reconstruction'] = 1.
knobs['lambda_penalty'] = 1.
knobs['lambda_fooling_term'] = 1.
knobs['gamma'] = 0.2
knobs['sigma'] = 1.
knobs['n_jobs_loader'] = 6
knobs['max_norm_encoder'] = 35.
knobs['max_norm_decoder'] = 35.
knobs['max_norm_discriminator'] = 35.
knobs['clip_gradient'] = False
knobs['time_to_collect'] = 250
knobs['fast_models'] = True
knobs['resume'] = True

"""
-----------------------------------------------------------------------------------------
Configurations
-----------------------------------------------------------------------------------------
Configuration #1
- LeakyReLU, slope 0.1
- lr, 1e-4
- hidden_dim, 64
- batch_size 64

Epoch #1 loss: 203
Epoch #2 loss: 160
Epoch #3 loss: 140
Epoch #4 loss: 134

Same parameters, but hidden_dim 64 -> 16
Epoch #1 loss: 1600
Epoch #2 loss: 600
Epoch #3 loss: 291
Epoch #4 loss: 159
Epoch #5 loss: 145
-----------------------------------------------------------------------------------------
Configuration #2
- LeakyReLu 0.2
- lr, 1e-4
- hidden_dim, 2
- batch_size, 128
- lambda penalty, 1.

Epoch #1 loss: 1900
Epoch #2 loss: 800

Same configuration, but batch_size 128 -> 64
Epoch #1 loss: 1600
Epoch #2 loss: 748
Epoch #3 losS: 348
Epoch #4 loss: 139
Epoch #5 loss: 132
-----------------------------------------------------------------------------------------
Configuration #3
- LeakyReLU, 0.2
- lr, 1e-4
- hidden_dim, 4
- batch_size, 64
- lambda penalty, 10.
- FC at the end of the Encoder and at the beginning of the Decoder

Epoch #1 loss: 718
Epoch #2 loss: 282
Epoch #3 loss: 161
Epoch #4 loss: 129
Epoch #5 loss: 116
Epoch #6 loss: 109

Same configuration, but batch_size 64 -> 100 and hidden_dim 4 -> 8
Epoch #1 loss: 726
Epoch #2 loss: 365
Epoch #3 loss: 267
Epoch #4 loss: 196
Epoch #5 loss: 150
Epoch #6 loss: 118
...
Epoch #32 loss: 100
-----------------------------------------------------------------------------------------
Notes:
    - try to remove FC layer in the end of Encoder and at the beginning of the Decoder
        and put a Conv1d (1x1) kernel size at the beginning of the Encoder and at the
        end of the Decoder.
"""

