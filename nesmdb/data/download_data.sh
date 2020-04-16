#!/usr/bin/sh

mkdir -p nesmdb_wav
mkdir -p audio_pool

docker build -t midi-converter .
docker run -it --rm --name running-midi-converter \
  --mount type=bind,source="$(pwd)"/nesmdb_wav,target=/opt/src/data/nesmdb_wav \
  midi-converter

python make_audio_in_pieces.py nesmdb_wav audio_pool
python audio_from_dir_to_csv.py audio_pool audio_pool/audio_pool.csv
