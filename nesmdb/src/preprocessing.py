import torch
import librosa
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Union, Generator
from multiprocessing import Pool
from itertools import chain

from config import knobs
from utils import standardize


class BruteAudioLoader:
    def __init__(self, in_file: Union[Path, str], batch_size, sr=8820, audio_len=17640):
        self.values = pd.read_csv(
            in_file, dtype={k: np.float32 for k in range(audio_len)}
            # , nrows=100
        )
        self.batch_size = batch_size
        self.sr = sr
        self.size = self.values.shape[0]
        self.audio_len = audio_len

    def __len__(self):
        return self.size

    def get_random_batches_indexes(self) -> Generator[List[int], None, None]:
        indexes_permuted = np.random.permutation(range(self.size))
        return (
            indexes_permuted[idx: idx + self.batch_size]
            for idx in range(0, self.size - self.batch_size + 1, self.batch_size)
        )

    def pipeline(self, x):
        return (
            torch.as_tensor(
                # standardize(
                self.values.iloc[x].values
                # )
            ).view(-1, 1, self.audio_len).to(knobs["device"])
        )

    def __iter__(self):
        random_batches_indexes = self.get_random_batches_indexes()

        for batch in random_batches_indexes:
            yield self.pipeline(batch)


class AudioLoader:
    def __init__(
        self,
        dirname: Union[Path, str],
        batch_size: int,
        sr=44100,
        new_sr=8820,
        audio_len=17640,
        n_jobs=6,
    ):
        self.dirname = Path(dirname) if isinstance(dirname, str) else dirname
        self.filenames = [fn for fn in self.dirname.iterdir() if ".wav" in str(fn)]
        self.num_filenames = len(self.filenames)
        self.batch_size = batch_size
        self.sr = sr
        self.new_sr = new_sr
        self.audio_len = audio_len
        self.n_jobs = n_jobs

    def _preprocess(self, filenames):
        return [librosa.load(filename, self.new_sr)[0] for filename in filenames]

    def parallel_loading(self, filenames):
        with Pool(processes=self.n_jobs) as pool:
            chunks = np.array_split(filenames, self.n_jobs)
            multiple_results = [
                pool.apply_async(self._preprocess, [chunk]) for chunk in chunks
            ]
            result = list(chain.from_iterable([res.get() for res in multiple_results]))
        return result

    def pipeline(self, filenames: List[Path]):
        # >>> Parallel Loading one audio / one thread
        # result = np.stack(
        #     Parallel(n_jobs=-1)(
        #         delayed(lambda x: librosa.load(x, self.new_sr)[0])(filename)
        #         for filename in filenames
        #     )
        # )
        # >>> Sequential Loading
        if self.n_jobs == 1:
            result = np.stack(
                [librosa.load(filename, sr=self.new_sr)[0] for filename in filenames]
            )
        else:
            result = np.stack(self.parallel_loading(filenames))
        # result = utils.standardize(result)
        return torch.as_tensor(result).view(-1, 1, self.audio_len).to(knobs["device"])

    @staticmethod
    def list_splitter(x, batch_size):
        len_x = len(x)
        num_batches = len_x // batch_size
        indexes = [
            (idx * batch_size, idx * batch_size + batch_size)
            for idx in range(num_batches)
        ]
        return [x[start:end] for start, end in indexes]

    def get_batches(self):
        filenames_permutated = np.random.permutation(self.filenames)
        return self.list_splitter(filenames_permutated, self.batch_size)

    def __len__(self):
        return self.num_filenames

    def __iter__(self):
        batches = self.get_batches()
        for batch in batches:
            yield self.pipeline(batch)


if __name__ == "__main__":
    # average loading time sequential: about 6s / batch (batch_size = 100)
    # average loading time parallel (12 threads): about 2.3s / batch (batch_size = 100)
    # average loading time parallel (10 threads): about 2.24 / batch
    # average loading time parallel (8 threads): about 2.24 / batch
    # average loading time parallel (6 threads): about 2.00 / batch
    # average loading time parallel (5 threads): about 2.07 / batch
    # average loading time parallel (4 threads): about 2.20 / batch

    from config import project_dir, audio_pool_dir

    import time

    resources_dir = project_dir / "tests" / "resources"
    out_dir = project_dir / "tests" / "resources_out"
#    out_dir.mkdir(exist_ok=True)

    # loader = AudioLoader(audio_pool_dir, 100, n_jobs=6)
    # collector = []
    # counter = 0
    # t0 = time.time()
    # for xb in loader:
    #     print(xb.shape)
    #     collector.append(time.time() - t0)
    #     t0 = time.time()
    #     if counter == 16:
    #         break
    #     counter += 1
    # print('average time for a batch:\t', sum(collector) / len(collector))

    # current_dir = Path()
    # in_file = current_dir / 'audio_pool.csv'
    # if not in_file.exists():
    #     num_obs = 10
    #     num_vars = 17640
    #     data = [np.random.randn(num_vars) for _ in range(num_obs)]
    #     pd.DataFrame(np.stack(data)).to_csv(in_file, index=None)
    # df = pd.read_csv(in_file)
    # loader = BruteAudioLoader(in_file, 2)
    # for batch in loader:
    #     print(batch.shape)

