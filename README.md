# mcunet fault sensitivity

this is a working host-side c prototype for persistent faults in stored int8
convolution weights in one official mcunet-in3 tflite artifact. it runs on a
mac through the tflite c api. no board is needed. it does not run tinyengine on
a microcontroller or inject hardware compute faults.

the runner loads a mutable model copy. each trial flips one bit in one weight,
evaluates all 200 fixed images while that fault persists, restores the byte,
and checks the whole model buffer against the clean copy. only convolution and depthwise
weights are injected, excluding biases, metadata, activations and compute.
all 55 weight tensors have per-channel scale metadata: convolution uses
dimension 0, depthwise uses dimension 3, and weight zero points are 0. the
classifier scales happen to be equal. `layer_inventory.csv` currently records
only scale entry 0, so it cannot describe every real-valued fault magnitude.
that limitation does not change the xor.

## mac setup

use an apple silicon mac, python 3.11.9, git, xcode command line tools with a c
compiler, cmake and ninja. licensed imagenet access and the model file are
external prerequisites. the tested project venv is `venv311/`.

```sh
git clone https://github.com/jangel19/mcunet-fault-sensitivity.git
cd mcunet-fault-sensitivity
python3.11 -m venv venv311
source venv311/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export MCUNET_MODEL="/absolute/path/to/mcunet-320kb-1mb_imagenet.tflite"
shasum -a 256 "$MCUNET_MODEL"
```

expected model sha256:
`780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1`.
see [model setup](MODEL.md) for the official release link and path options.
the int8 evaluator reads this local file; it does not download it.

[environment setup](ENVIRONMENT.md) explains the exact installed-package
snapshot and the optional, pinned mcunet source dependency. the c campaign,
subset builder, int8 evaluator and schema inspection scripts do not need
mcunet. the float baseline, export, model-size and eager-pytorch quantization
scripts do. tinyengine is not required for this host prototype.

## imagenet and the fixed subset

accept the license and request access on the
[imagenet dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
create a read token in your account, then authenticate locally:

```sh
hf auth login
python make_subset.py --model "$MCUNET_MODEL" --n 200 --out subset
```

this streams validation examples and keeps the first example of each new
class until it has 200 images. it is a fixed class-selection rule, not a
random or representative sample of all imagenet classes. the saved campaign
subset has 200 distinct labels and ends at stream index 225. the builder
saves raw int8 nhwc images, little-endian int32 labels and a hash manifest.

preprocessing converts to rgb, resizes the short edge to `int(176*256/224)`,
center crops to 176, computes `pixel/255*255 - 128` in float32, casts to int8
and reshapes to `[1,176,176,3]`. it is hardcoded, not derived from the input
quantization metadata. equivalence to quantizing the float baseline input
has not been established. do not change it when reproducing these results.

campaign subset hash, defined as sha256 of the exact `subset/manifest.json`
bytes: `9c6d5d1e0e20c551cdcb01d9894a58743bc377b9fa4cb82a1750cd0c3dedea7a`.
labels sha256: `0ee160010336676b4577833fe8051d7107969675b752d2551bb856076acba149`.
a copy of the manifest is in [the campaign artifacts](results/random_k30_subset_manifest.json).
verify every image and label hash, not just the manifest hash.

images and labels are not distributed in git. the original dataset revision
was not recorded and the streaming loader is unpinned. a newly built subset
can run the prototype, but exact reproduction of these results requires the
original bytes or a rebuild matching every saved hash. keep an approved local
copy of that subset. rebuilding from a changing upstream stream alone is not
a guarantee of identical selection or decoding.

## build, test and run c

first follow [the full c library build and installation steps](c/BUILD_TFLITE.md).
they include header installation, cmake/ninja/compiler prerequisites and mac
dylib install-name/rpath setup. the default external install is
`~/.local/tflite-2.21.0`; override `TFLITE` for another location.

from the project root, with the venv active, model configured and subset ready:

```sh
make -C c all
make -C c test
make -C c smoke
mkdir -p results
c/run_trials --model "$MCUNET_MODEL" --subset subset --mode random \
  --k 30 --seed 20261021 --images 200 --out results/my_random_k30.csv \
  > results/my_random_k30.log 2>&1
```

use a new output name each time: the c runner overwrites the named csv.
`make test` regenerates python reference files inside `subset/`, then checks
byte flips, restoration, all 55 layer locations, clean repeatability,
persistent-versus-fresh interpreters and python/c parity. the unchanged suite
passes 99 checks, zero failures and zero skips with the installed library.
a scratch-source build and test also passed with spaces in the repository,
model, subset, python and library paths; see [the test log](results/setup_space_paths_test.log).
a separate empty venv was also installed from the index using the pinned
requirements, passed `pip check`, and passed the same 99-check suite. see
[the fresh environment test](results/setup_fresh_environment_test.log). the
first spaces check reuses the original local c library. a separate fresh
v2.21.0 source-library build, header installation and mac signing also passed
99 checks and the smoke run with the fresh venv. see
[the recorded fresh build](results/setup_fresh_tflite_provenance.json).

custom paths work without source edits, for example:

```sh
make -C c test TFLITE="/absolute/path with spaces/tflite" \
  MODEL="$MCUNET_MODEL" SUBSET="/absolute/path with spaces/subset" \
  PYTHON="/absolute/path with spaces/venv311/bin/python"
```

paths passed to make are relative to `c/`; use absolute paths for overrides.
python entry points accept `--model`, then use `MCUNET_MODEL`, then the shared
`~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite` default. make uses `MODEL`,
then `MCUNET_MODEL`, then that default. direct c commands require `--model`.

## runtime and baseline comparisons

the c runner uses one thread and the library's default resolver. the installed
library reports `2.19.0`; the directory name does not establish its source tag.
its binary hash is recorded in the random campaign provenance. in-place weight
mutation was verified for this library; repeat the tests after changing it,
especially with delegates or weight packing.

on the same 200 images, c gets 130 correct (65%), python default 131, python
without delegates 133, and python reference kernels 130. c and python reference
have identical top-1 on all 200 images, 70 bit-exact output vectors and maximum
absolute output difference 1. c versus python default has 10 top-1 differences
and maximum absolute difference 8. this establishes configuration-dependent
predictions; the exact kernel rounding cause has not been traced.

`python eval_int8_tflite.py --model "$MCUNET_MODEL"` uses the python default
interpreter on 50,000 streamed validation images, with progress printed at
indices 0, 100, 200 and so on. its saved log reports 61.82% top-1. the float
baseline `python baseline.py` also evaluates 50,000 images, prints every 50,
and makes a second stream pass to count classes. its historical reported
accuracy is 62.14%; the int8 script's printed float number is a hardcoded
string. these full-dataset runs were not repeated in this pass. see
[the baseline experiment](baseline_experiment_week3.md) and
[the saved int8 log](results/eval50k_refactored.log).

the 61.82% python result and 65% c subset result differ in both images and
runtime. agreement with a paper's rounded accuracy does not establish
calibration or isolate quantization as the only difference between baselines.

## historical corner campaign

`results/first_pass_corner.csv` and its log are unchanged. this design uses
14 selected layers, eight bits and five deterministic rules: largest absolute
weight, smallest nonzero absolute weight, first zero, first element and last
element. it has 560 trial rows but 528 distinct `(op, element, bit)` faults,
because some rules select the same element. means weight rules equally; they
are not unbiased fault probabilities or random samples for confidence intervals.

all trials use 200 images: 112,000 trial-image evaluations in total. masked
means an identical final quantized output vector; no pre-quantization output
comparison is available. subtle means a changed vector
with the same first-maximum argmax. catastrophic means a different argmax,
including wrong-to-right and wrong-to-wrong changes. returned c-api errors
are counted as due, a mismatch and an incorrect prediction.

keep these measures separate:

- any output-vector change: 504/560 trials (90%) change at least one image's
  vector; 79,710/112,000 trial-image evaluations (71.170%) change vectors.
- any top-1 change: 411/560 trials (73.393%) change at least one prediction.
- top-1 mismatch rate: `(catastrophic + due) / images`, compared with clean c
  on the same subset. bit 7's mean is 12.50%, bit 6's is 10.114%.
- accuracy drop: clean accuracy minus faulted accuracy, in percentage points.
  bit 7's mean drop is 4.407 points, bit 6's is 3.086 points.

op 3 has 26.675% mean mismatch and a 10.275-point mean accuracy drop; the stem
(op 1) has 17.3625% and 4.675 points. the worst trial is 79: op 3, last element
143, bit 7, -2 to 126. accuracy falls from 130/200 (65%) to 1/200 (0.5%), with
199/200 top-1 changes. bits 0, 1 and 2 still cause 2.15%, 2.657% and 2.907%
pooled top-1 mismatch; their pooled accuracy drops are slightly negative.
low-bit mismatch cannot be treated as accuracy loss or dismissed altogether.

op 120 has zero catastrophic outcomes across its 40 tested faults and 8,000
trial-image evaluations, independently checked with fresh python reference
interpreters. that does not cover all 160,000 classifier weights or other
images. early-layer and high-bit effects are observations of this design;
they do not establish a causal depthwise/shortcut explanation.

## seeded random campaign

the unchanged c runner ran k=30 with seed `20261021` across 2026-10-21/22
(america/new_york):
14 default layers, all eight bits, 200 fixed images per trial. that is 3,360
trial rows, 3,322 distinct physical faults and 672,000 trial-image evaluations.
the first 40 completed k=1 pilot trials took 37.3 seconds, giving the
initial 52.2-minute estimate before launching k=30. the completed 112-trial
pilot took 105.2 seconds, consistent with a 52.6-minute projection. the full
runner loop took 3136.9 seconds (52.28 minutes),
excluding model loading and the two clean baseline passes.

the pooled top-1 mismatch rate is **5.316%**, with denominator
672,000 trial-image evaluations. mean accuracy drop is **0.819
percentage points**, relative to clean c accuracy of 130/200 (65%). these
are separate measures. bit 7 has 11.710% mean mismatch and a
3.598-point mean accuracy drop; bit 0 has 2.532% and
-0.256 points. each bit summary uses 420 trials and 84,000 trial-image
evaluations, weighting the 14 selected layers equally. there were 0 returned
api errors; crash/hang detection remains unimplemented.

[the full report](results/random_k30_report.md) gives all per-bit and per-layer
means, explicit denominators and conditional uncertainty intervals, including
an allowance for unseen rare effects in zero-observation groups. these
summaries weight selected cells equally, rather than weights across the whole
model. the images are fixed and reused; they do not supply independent
imagenet uncertainty. this is a different sampling design from the historical
560 corner-rule rows and their 528 distinct faults.

op 120 had two top-1 changes across 240 random faults and 48,000 trial-image
evaluations, both at bit 6. one trial lost one correct prediction; the other
kept the same accuracy. the earlier zero-change corner result therefore stays
limited to its 40 tested faults. op 3 and bit 7 have the highest observed
group means in this random design under both mismatch and accuracy drop.

all 112 first draws were repeated across the k=1 pilot and k=30 campaign:
every csv field matched except the sequential trial id, including selected
weights and aggregate outcomes on all 200 images. every campaign seed,
selected element and original/xor byte was independently recomputed through
the python schema. the runner repeated its clean baseline and checked exact
full-buffer restoration after every fault. the full 3,360 rows were not run
twice.

artifacts: [csv](results/random_k30.csv), [log](results/random_k30.log),
[exact commands and provenance](results/random_k30_provenance.json),
[pilot csv](results/random_pilot_k1.csv), [pilot log](results/random_pilot_k1.log)
and [verification log](results/random_k30_verification.log). the campaign used
the original installed library, reported version `2.19.0`, with its binary
hash recorded in provenance; the fresh setup-validation library was separate.

before the k=30 command above, time a pilot on your machine with a new filename:

```sh
c/run_trials --model "$MCUNET_MODEL" --subset subset --mode random \
  --k 1 --seed 20261021 --images 200 --out results/my_random_pilot_k1.csv \
  > results/my_random_pilot_k1.log 2>&1
```

then verify and summarize your completed campaign:

```sh
python scripts/summarize_random.py --csv results/my_random_k30.csv \
  --pilot results/my_random_pilot_k1.csv --model "$MCUNET_MODEL" \
  --subset subset --seed 20261021 --k 30 --out results/my_random_report.md
```

## status and remaining work

week 4 inventory and manual stage map exist, week 5 fault design exists, and
the week 6 host prototype and corner campaign passed the audit. seeded random
sampling is now exercised separately from the corner design. the five corner
rules still need advisor approval. stages are manual, and the architecture
notes contain an early block-count table that conflicts with their later table.

current due handling detects returned api errors only. a crash ends the
process and a hang stalls it; there is no worker isolation, timeout or durable
failed-trial row. zero recorded due does not prove crash/hang freedom. input
validation and trusted-flatbuffer bounds issues remain open. full per-channel
scale metadata is also deferred. these fixes need a separate pass because
this pass preserves injector logic and historical experiments.

stronger population claims need an approved sampling design and broader
image/fault coverage. microcontroller claims need an identified board,
deployment runtime and separate on-device validation. the existing campaign
library still needs its original checkout/build record for historical source
provenance. a separate fresh build is now recorded and validated for future
setup; it was not substituted into the campaign.

## file guide

- `c/fault_inject.c`, `c/run_trials.c`, `c/test_fi.c`: injector, campaign and tests.
- `c/ref_logits.py`, `make_subset.py`: references and subset construction.
- `layer_inventory.py`, `counting.py`, `get_tendor_data.py`: schema inspection.
  the `tendor` filename is intentional for compatibility in this pass.
- `fault_model_week5.md`, `architecture_week4.md`: design and manual mapping.
- `results/`: intentional experiment csvs, logs and reports remain tracked.
- `results/audit_20261021/audit_report.md`: the separately supplied local audit,
  with detailed verification and corrected results. that directory is not
  tracked in git and was left untouched.
