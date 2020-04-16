FROM python:2

WORKDIR ./opt/src/data/

RUN pip install --no-cache-dir nesmdb pretty_midi
RUN wget http://deepyeti.ucsd.edu/cdonahue/nesmdb/nesmdb_midi.tar.gz
RUN tar xvfz nesmdb_midi.tar.gz
RUN mkdir nesmdb_wav
RUN nesmdb_convert midi_to_wav \
  --out_dir nesmdb_wav \
  --midi_to_wav_rate 100 ./nesmdb_midi/valid/*.mid
RUN nesmdb_convert midi_to_wav \
  --out_dir nesmdb_wav \
  --midi_to_wav_rate 100 ./nesmdb_midi/test/*.mid

CMD [ "bash" ]
