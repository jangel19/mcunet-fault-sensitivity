# python environment

use python 3.11.9 on an apple silicon mac, with `venv311/` at the repo root:

```sh
python3.11 -m venv venv311
source venv311/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` is the runtime package snapshot from the working
venv, including tensorflow 2.21.0, tflite 2.18.0 (the flatbuffer schema), numpy
2.4.6, torch 2.14.0 and torchvision 0.29.0. these are the observed versions,
not a claim that other platforms or python versions work. installation from
the index into an empty temporary venv succeeded, `pip check` passed, and
the unchanged c/python suite passed all 99 checks with that fresh environment.
see `results/setup_fresh_environment.log` and
`results/setup_fresh_environment_test.log`. pip build tooling is not pinned
by this runtime snapshot; the original venv has pip 24.0, setuptools 78.1.0
and wheel 0.48.0.

## optional mcunet source dependency

`baseline.py`, `export.py`, `modelsize.py`, `quantize.py` and `int8_eval.py`
import `mcunet`. it was missing from the venv, so the package snapshot alone
does not provide those scripts. use this separate, pinned source dependency:

```sh
git clone https://github.com/mit-han-lab/mcunet.git ../mcunet-source
git -C ../mcunet-source checkout 9c164f8483003a5c6445871d94d30720aecc5918
export PYTHONPATH="$(cd ../mcunet-source && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from mcunet.model_zoo import build_model; print("mcunet import ok")'
```

the source root contains the `mcunet/` package. `PYTHONPATH` adds that root to
the interpreter's import path; cloning a sibling repo by itself does not.
this exact revision passed the import check in the working venv. full float,
export and eager quantization runs were not repeated in this pass.

`eval_int8_tflite.py`, `make_subset.py` and `c/ref_logits.py` do not need
mcunet. the first two currently import torchvision, datasets and tensorflow;
the reference generator needs only numpy, tensorflow and the shared path
helper. `layer_inventory.py` and `counting.py` need the tflite schema package;
`get_tendor_data.py` also imports numpy. the built c campaign needs no python,
mcunet or tinyengine. python is used by `make test` to regenerate references.
