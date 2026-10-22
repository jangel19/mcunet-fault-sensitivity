# model file

use the official mcunet-in3 int8 export, `mcunet-320kb-1mb_imagenet.tflite`.
the evaluator reads a local file; it does not download it. obtain it from the
[han lab release](https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-320kb-1mb_imagenet.tflite)
or supply your existing copy. verify it before running:

```sh
shasum -a 256 "$MCUNET_MODEL"
```

expected sha256: `780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1`.

all python tflite tools accept `--model /path/to/model.tflite`. otherwise they
use `MCUNET_MODEL`, then `~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite`.
this covers `eval_int8_tflite.py`, `make_subset.py`, `c/ref_logits.py`,
`layer_inventory.py`, `counting.py` and `get_tendor_data.py`. inventory also
accepts `--out` so inspection need not overwrite the checked-in csv.

```sh
export MCUNET_MODEL="/absolute/path with spaces/mcunet-320kb-1mb_imagenet.tflite"
python eval_int8_tflite.py
python make_subset.py --n 200 --out subset
python layer_inventory.py --out /tmp/layer_inventory.csv
```

the c executables take an explicit `--model`; the make targets also accept
`MODEL=...` and use `MCUNET_MODEL` before the same default path.
