#!/usr/bin/sh

set -o errexit

if [ $# -ne 1 ]; then
  echo -e "Usage:\tget_configuration <out_dir>"
  exit 1
fi

OUT_DIR=$1
PATTERN="([0-9][0-9][-\.]*){5}"
CKPT_DIR=$(ls ../checkpoints -lt | head -n 2 | grep -E -o "${PATTERN}pt")
RUNS_DIR=$(ls ../runs -lt | head -n 2 | grep -E -o "${PATTERN}")

cp "../checkpoints/$CKPT_DIR" -t "${OUT_DIR}"
cp -r "../runs/$RUNS_DIR" -t "${OUT_DIR}"
cp -r "../interpolations/" -t "${OUT_DIR}"

grep knobs ../src/config.py > "${OUT_DIR}/configuration.txt"
echo -e "\n-----\nmodels.py\n-----\n" >> "${OUT_DIR}/configuration.txt"
cat "../src/models.py" >> "${OUT_DIR}/configuration.txt"

echo "Done."
