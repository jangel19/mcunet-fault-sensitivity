# week 6 audit

audited 2026-10-21, tracked source at `6fd39f0`. the existing untracked `results/week6_handoff.md` was left alone. source, historical results, and original subset files were not changed. no fixes or commits were applied, and nothing was pushed.

## what exists

this is a working host-side c prototype for persistent faults in stored int8 convolution weights. it loads a mutable copy of a tflite flatbuffer, finds convolution weights, flips one stored bit, evaluates a fixed image subset, restores the byte, verifies the whole buffer, and writes aggregate outcome counts. it uses the tflite c api on this mac; it does not run tinyengine on a microcontroller or inject hardware compute faults.

| files | actual job |
|---|---|
| `c/fault_inject.h`, `c/fault_inject.c` | flatbuffer locator, model copies, xor/restore, interpreter, subset loading, scoring, splitmix64 |
| `c/run_trials.c` | 14 default layers; five corner rules or seeded random elements; eight bits; one csv row per trial |
| `c/test_fi.c` | value/byte/restore tests, locator against inventory, clean repeatability, persistent versus fresh interpreter, python parity |
| `c/ref_logits.py` | generates clean int8 outputs using three python kernel configurations |
| `c/Makefile`, `c/BUILD_TFLITE.md` | local build/test/smoke commands and mac-specific c-library build notes |
| `eval_int8_tflite.py` | shared preprocessing and validation stream; python default-interpreter 50,000-image evaluation |
| `make_subset.py` | first example of each new class in stream order; raw int8 images, little-endian labels, hash manifest |
| `layer_inventory.py`, `layer_inventory.csv` | raw schema inventory of 122 operators; 55 convolution/depthwise candidates, 721,032 weight elements |
| `counting.py`, `get_tendor_data.py` | graph listing and tensor inspection; the latter filename really contains `tendor` |
| `baseline.py` | pretrained pytorch float evaluation, 50,000 images, plus a second stream pass to count represented classes |
| `export.py`, `modelsize.py` | onnx export and parameter/nominal storage estimates |
| `quantize.py`, `int8_eval.py` | separate eager-pytorch quantization experiments, different preprocessing, unrelated to the official tflite fault campaign |
| `results/first_pass_corner.csv`, `.log` | 560 recorded trial rows and run progress |
| `results/eval50k_refactored.log` | saved python run reporting 61.82%; printed 62.14% float baseline is a hardcoded string in the script |
| `fault_model_week5.md` | design plus week 6 amendments; some original random-sampling and due requirements remain unmet |
| `architecture_week4.md` | manual architecture/stages and first-pass layer selection; early block-count table conflicts with its later operator table |
| `baseline_experiment_week3.md`, `quantization_notes.md`, `week*_notes.md` | historical methodology/results; several descriptions have become stale or are incorrect |
| `README.md`, `requirements.txt`, `.python-version`, `.gitignore` | onboarding, incomplete environment pins, python 3.11.9 selection, partial artifact exclusions |

the local `subset/`, `venv311/`, model exports, executables and debug bundles are not supplied by a clone. one compiled python cache file is nevertheless tracked.

## what ran

exported the committed source with `git archive HEAD` into `/private/tmp/mcunet-audit-20261021`, then built both executables from source with `make -B ... all`. compilation succeeded with the configured warnings and the existing external c library. copied the existing subset into that scratch tree and linked its python environment to the existing venv. `make test` regenerated reference outputs there and passed **99 checks, zero failures, zero skips**. this is a clean source build using local prerequisites, not proof of a fresh machine install.

the original suite checks all 2,048 int8 value/bit pairs, one-byte changes and exact restoration in 1,320 cases, all 55 inventory entries, and 110 persistent-versus-fresh interpreter cases. persistent and fresh outputs agreed in all 110 cases; 105 changed logits on the four images used per fault. clean outputs were repeatable.

additional checks, kept separate from the injector:

- `check_outputs.py`: checked all 560 csv rows against the actual model using the python flatbuffer schema. independently recomputed corner rules, original values, flipped values and splitmix seeds, and checked outcome counts, rates and accuracy bounds.
- verified every image hash, every image length, labels hash and model hash against the subset manifest. 200 distinct labels, last selected stream index 225. these checks establish internal consistency, not independent authenticity of imagenet labels or the downloaded model.
- ran random mode twice for 18 trials each, covering ops 1, 62, 120 and bits 0, 7 on 20 images. csvs were byte-identical. `check_random.py` independently recomputed their seeds and selected elements.
- `score_fixture.c`: compiled the unchanged scorer against controlled c-api responses. known examples produced exactly one masked, one subtle, one catastrophic and one due image, with correct accuracy accounting. separately exercised returned errors from input copy, invoke and output copy, and first-maximum tie breaking. this simulates returned errors; it does not simulate a real process crash or hang.
- `independent_python.py`: locates the buffer through the schema, modifies a separate bytearray, builds fresh python reference interpreters and classifies their raw outputs with numpy. this bypasses the c locator, mutation code and scoring implementation.

full campaign rerun and independent classifier check: see final verification below.

model sha256: `780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1`.

original campaign csv sha256: `69b643888dd2732860b1f05cf4132f71b7d1afab6dff7dab03c1e3300a2e9582`.

## meeting claims

| claim | verdict and evidence |
|---|---|
| faults went into weights | **confirmed.** locator follows convolution input 1 to its constant buffer; `fi_apply` xors `m->buf[layer->offset + elem_idx]`. images are separate allocations. |
| worst subset accuracy was about 0.135 from 0.65 | **contradicted.** trial 79: op 3, last element 143, bit 7, -2 to 126; accuracy 0.005 (1/200), clean 0.65, 199 top-1 changes. fresh python reference execution independently reproduced these counts and accuracy. no csv row has accuracy exactly 0.135. |
| bit 7 is most damaging | **confirmed for pooled means in this campaign.** mean top-1 mismatch 12.50%, mean accuracy drop 4.407 percentage points; bit 6 has 10.114% mismatch and 3.086-point drop. this does not establish an ordering for every individual weight. |
| the stem is most sensitive | **contradicted under both reported mean measures.** op 3: 26.675% mismatch and 10.275-point accuracy drop. stem op 1: 17.3625% mismatch and 4.675-point drop. |
| low bits have minimal impact | **contradicted as a blanket statement.** bits 0, 1, 2 cause 2.15%, 2.657%, 2.907% pooled top-1 changes. trial 74, op 3 element 143 bit 2, changes 29/200 predictions (14.5%). pooled accuracy drops for these bits are slightly negative, so top-1 changes should not be confused with accuracy loss. high bits are worse on average. |
| roughly 70% of trials changed output | **contradicted if output means any int8 output-vector change.** 504/560 trials (90%) change at least one vector. 411/560 (73.393%) change at least one top-1 prediction. 79,710/112,000 trial-image evaluations (71.170%) change vectors. these are three different denominators. |
| op 120 has zero catastrophic outcomes | **confirmed for the 40 tested faults on these 200 images**, with the independent python reference verification recorded below. this is not a claim about all 160,000 classifier weights or other images. |
| c/python baseline gap is unexplained | **contradicted at the measured configuration level; exact numerical implementation cause remains unverified.** same 200 images: c 130 correct, python default 131, no delegates 133, reference 130. c versus reference has identical top-1 on all 200, 70 bit-exact vectors, max difference 1. c versus default: 10 top-1 differences, max difference 8. selecting kernel configurations changes predictions even within python. no kernel-by-kernel rounding diagnosis or c-source provenance verification was done. |

the 61.82% figure is a different run: python default on 50,000 images. comparing it directly with c's 65% on 200 selected images confounds dataset and runtime. the saved log supports that the python run reported 61.82%; this audit did not rerun the full baseline or independently verify the float32 number.

## exactly what touches images

`eval_int8_tflite.py:preprocess_to_int8` converts to rgb, resizes the short edge to `int(176*256/224)`, center crops to 176, converts pixels to float32, computes `pixel/255*255 - 128`, casts to int8 and reshapes to nhwc. the arithmetic matched `pixel - 128` for all 256 pixel values in the installed numpy. `make_subset.py` saves those bytes unchanged. `fi_subset_load` reads them into `s->images`; `fi_run_image` copies each image into the interpreter's input tensor. none of those actions injects an image fault. the fault loop only mutates the model copy.

the preprocessing is hardcoded rather than derived from input quantization metadata. the local model's input scale is approximately 1/127, zero point 0. exact equivalence to quantizing the float baseline's normalized input is not established by matching tensor type alone. changing this would alter the experiment and needs a separate validated experiment, not an audit edit.

## methodology and code issues

1. **due does not implement crash/hang detection.** a returned c-api error increments `n_due`, including copy failures. a crash terminates the process and a hang stalls it. there is no worker isolation, timeout or durable failed-trial row. therefore zero recorded due means zero returned errors, not proof of zero possible crashes/hangs. add process isolation and timeouts in a separate change if the original definition is required.
2. **quantization documentation is wrong.** all 55 weights have multiple scale entries in their metadata; convolution uses dimension 0 and depthwise dimension 3, with zero points 0. examples: op 1 has 16 scales, op 3 has 16, op 120 has 1,000. classifier scales happen to be equal. `layer_inventory.py:describe_tensor` only keeps entry 0 and its per-tensor comment is false. the xor is correct, but a single recorded scale cannot describe every fault's real-valued change. preserve full scale arrays and quantized dimension in a future inventory format.
3. **corner rows are not random independent samples.** four duplicate selections occur: op 1 min-nonzero/first, op 9 min-nonzero/first, op 36 zero/first, op 42 min-nonzero/first. 560 rows represent 528 distinct `(op,element,bit)` faults. the handoff's claim that first-element never coincides with another rule is false. rule-weighted means are valid descriptions of this design, not unbiased fault probabilities. random-mode k=30 and confidence-interval discussion does not describe this campaign.
4. **scoring matches the top-1 definitions.** masked means identical quantized output bytes, subtle means changed bytes with the same first-maximum argmax, catastrophic means a different argmax, including wrong-to-right and wrong-to-wrong changes. an api error contributes to mismatch and counts as incorrect. catastrophic rate is not accuracy drop. output vectors are final quantized model outputs; no pre-quantization comparison is available.
5. **kernel selection is library-dependent.** `fi_runner_init` sets one thread and uses the default c resolver. it does not explicitly request reference kernels, despite the header comment. the measured library supports in-place changes in all tested layers; swapping in a library with delegates or weight packing requires repeating that check.
6. **input validation is weak.** unknown layers are skipped and the program can report success after zero trials. tested `--layers 999`: exit 0, header-only csv. numeric parsing uses `atoi`, list lengths can be silently truncated, negative k can produce no trials, allocation and csv write failures are not consistently checked. the subset reader silently caps an oversized image request, ignores trailing image bytes and partial label bytes, and does not verify manifest hashes or label ranges.
7. **the flatbuffer reader is for a trusted model.** the capacity check happens after a pointer into `layers[n_layers]` has been used to write shapes, so more than 64 eligible layers can write out of bounds. offset arithmetic can overflow and shape multiplication is unchecked. this does not invalidate the tested 55-layer model, but contradicts the blanket bounds-safety comment. check capacity before any array access and validate lengths/arithmetic in a separate core fix.
8. **restore is stronger than the spec's hash-only check.** successful trials verify the entire flatbuffer against a clean copy with hash plus `memcmp`, not just weights. the hash uses a nonstandard fnv-1a initial constant, but byte comparison still establishes exact equality when the check passes. the xor restore relies on the byte not being modified by anything else between apply and restore.
9. **provenance is incomplete.** csv/log omit model, subset and library hashes, dependency versions and full command. the installed library reports `2.19.0`; a directory named `tflite-2.21.0` and the build note do not prove its source tag or build flags. retain source revision, build configuration and binary hash with future runs.

## what stops a fresh clone

- `requirements.txt` omits tensorflow, the `tflite` schema package and mcunet. the existing venv has tensorflow 2.21.0, tflite 2.18.0, numpy 2.4.6, torch 2.14.0 and torchvision 0.29.0. several disagree with the pinned file. `pip check` passes, which does not mean the file reproduces this environment. importing mcunet in this venv currently raises `ModuleNotFoundError`, so the float/export/quantization scripts cannot run here as supplied.
- cloning sibling repositories does not put mcunet on python's import path. document its installation and tested revision. tinyengine is not used by the current c runner.
- README says python 3.6+, creates `venv/`, and omits c setup; `.python-version` and the working environment use 3.11.9 and the makefile expects `venv311/`.
- `eval_int8_tflite.py` hardcodes `/Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite`. the subset builder and reference generator import it. inventory/inspection scripts assume the same file under `~/.torch`. the int8 evaluator does not download a model, contrary to README. add a model argument shared by these entry points and a documented acquisition/checksum step.
- the c makefile expects an externally installed library and headers under `~/.local/tflite-2.21.0`; the build note omits the actual header/library installation commands. cmake, ninja and a compiler are separate prerequisites. rpath/install-name instructions are mac-specific. paths containing spaces are not quoted in make recipes.
- model and image subset are absent from git. rebuilding the subset needs licensed imagenet access, network availability and a pinned dataset revision/order; the streaming loader currently specifies no revision. this audit verified the existing files but did not rebuild the subset from the remote source. locally cached hub snapshot paths found during inspection contain training shards, not validation parquet files.
- README references missing `baseline_report.md`; the actual file is `baseline_experiment_week3.md`. README describes 1,000-image evaluation and progress every 50 for int8, while code evaluates 50,000 and prints every 100. its week 4-6 status and project tree are stale.
- `.gitignore` already covers `venv311/`, onnx/pth exports, subset, c binaries and debug bundles, and `__pycache__/`. however `__pycache__/inspect.cpython-311.pyc` is still tracked. remove it from the index in a separate housekeeping commit. README's `venv/`, generic venv names, `*.pyc`, `.tflite`, `.npy`, and potential local dataset/cache directories are not generally excluded. add targeted patterns while retaining intentional experimental csv/log artifacts. no tracked `.npy` or `.tflite` files were found.

## what the readme should say

describe a host-side, weight-memory fault campaign on one particular official tflite artifact. document the exact model and subset hashes, input preprocessing, python and c kernel configurations, source/library versions, dependencies, model acquisition, subset construction, build, test and run commands. make paths configurable and describe every prerequisite not shipped in git.

state that a trial flips one bit in one persistent int8 weight and runs all 200 fixed images, then restores and checks the entire model buffer. the subset is the first example of each of the first 200 newly encountered classes, not a random or representative sample of all imagenet classes. explain per-channel quantization and that only weights are injected, excluding biases, metadata, activations and compute faults.

report the 560 rule-weighted trials, 528 unique faults, 14 selected layers, eight bits and five deterministic rules. define output change, top-1 mismatch, accuracy drop and all denominators explicitly. compare faulted outputs to clean c outputs on the same images; keep the 50,000-image python accuracy separate. report early-layer sensitivity and high-bit effects as observations of this design, without claiming a causal depthwise/shortcut explanation or population confidence intervals.

document due's current returned-error limitation, the library-dependent in-place mutation assumption, pending corner-rule approval, manual stages, and the limited image/fault coverage of the classifier result. remove unsupported claims that matching a paper accuracy proves calibration or isolates quantization as the only difference.

no hardware is needed to rerun this host prototype. claims about microcontroller behavior require an identified board, deployment runtime and a separate on-device validation. stronger population estimates require an approved sampling design and more image/fault coverage. c library provenance requires the original checkout/build record or a fresh recorded rebuild.

## final verification

full clean-source c rerun completed all 560 trials; the csv is byte-identical to the original, with sha256 `69b643888dd2732860b1f05cf4132f71b7d1afab6dff7dab03c1e3300a2e9582`. see `rerun_corner.log` for timing and the final clean-buffer check. `make_test.log` records regenerated python references and 99 passing checks. regenerating the inventory also matched all 122 original rows and every stored field.

independent fresh python reference interpreters verified all 40 op-120 faults, totaling 8,000 image evaluations, with zero top-1 changes. the separate worst-case check reproduced trial 79 exactly: zero masked, one subtle, 199 catastrophic, accuracy 0.005. fresh outputs for all three clean python configurations matched their saved files exactly.

all historical tracked files still match the checkout. the only additions are this audit directory; the pre-existing handoff remains untracked and unchanged. no fix commits were made.
