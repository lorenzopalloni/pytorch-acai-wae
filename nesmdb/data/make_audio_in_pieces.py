import librosa
import sys

from pathlib import Path
from tqdm import tqdm
from typing import Union


def make_audio_in_pieces(in_dir: Union[Path, str], out_dir: Union[Path, str], sr=44100):
    if isinstance(in_dir, str):
        in_dir = Path(in_dir)
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

    counter = len(list(out_dir.iterdir()))
    for audio_filename in tqdm(in_dir.iterdir()):
        for piece in get_pieces(audio_filename):
            librosa.output.write_wav(str(out_dir / f"{counter}.wav"), piece, sr)
            counter += 1


def get_pieces(audio_filename, sr=44100, piece_len=88200):
    audio = librosa.load(audio_filename, sr=sr)[0]
    indexes = get_indexes(audio, piece_len)
    return [audio[start:end] for start, end in indexes]


def get_indexes(audio, piece_len=88200):
    audio_len = len(audio)
    if audio_len < piece_len:
        return []
    num_pieces = audio_len // piece_len
    return [(idx * piece_len, idx * piece_len + piece_len) for idx in range(num_pieces)]


if __name__ == "__main__":
    argc = len(sys.argv)
    if argc > 1 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        print("Usage: make_audio_in_pieces.py <in_dir> <out_dir> [<sr>]\n")
    elif argc == 3:
        make_audio_in_pieces(Path(sys.argv[1]), sys.argv[2])
    elif argc == 4:
        make_audio_in_pieces(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print("Usage: make_audio_in_pieces.py <in_dir> <out_dir> [<sr>]\n")
