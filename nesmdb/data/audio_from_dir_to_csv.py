import sys
import librosa
import pandas as pd
import numpy as np

from pathlib import Path


def audio_from_dir_to_csv(in_dir, out_file, sr=8820):
    in_dir = Path(in_dir) if isinstance(in_dir, str) else in_dir
    out_file = Path(out_file) if isinstance(out_file, str) else out_file
    filenames = list(in_dir.iterdir())
    df = pd.DataFrame(np.stack([librosa.load(filename, sr)[0] for filename in filenames]))
    df.to_csv(out_file, index=None)


if __name__ == '__main__':
    argc = len(sys.argv)
    usage_message = "Usage: audio_from_dir_to_csv.py <in_dir> <out_file> [<sr>]"
    if sys.argv[1] == '-h' or sys.argv[1] == '--help':
        print(usage_message)
    elif argc == 3:
        audio_from_dir_to_csv(sys.argv[1], sys.argv[2])
    elif argc == 4:
        audio_from_dir_to_csv(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print(usage_message)
