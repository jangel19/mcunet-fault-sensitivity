# activation tracing prerequisites

this frontend uses the historical c runtime, not the python interpreter.
`c/fault_inject.c` and `c/run_trials.c` are unchanged. no HISTORICAL campaign
was rerun and no existing `results/` files were written. teh only edits to
existing code are documentation COMMENTS in `c/fault_inject.h` and
`layer_inventory.py`.

## saved artifacts

everything new is under `artifacts/propagation/`:

* `frozen_c/quantization.json`: deterministic, complete metadata for all 288
  tensors: 55 weights, 55 biases, 123 activations (including input and final
  output), and 55 constants. includes all scales, zero points, quantized axes,
  float32 scale bit patterns, shapes, and producer/consumer operators. empty
  quantizaton fields are retained for unquantized constants. the old csv is
  intentionally unchanged; its scale columns remain entry 0 summaries.
* `scale_investigation.json`: raw byte, upstream download, and checkpoint
  evidence described below.
* `frozen_c/runtime/`: a copy of teh exact campaign dylib and the separately
  compiled capture executable, using `@loader_path/lib`. the launcher removes
  dyld overrides and verifies the library actually loaded with `dladdr`.
* `frozen_c/source/`: exact source snapshots used for the new executable and
  generator, with hashes in `provenance.json`.
* `frozen_c/baseline/output_000.bin`: raw signed int8 `[200,1000]` final outputs.
  `predictions.csv` has Image index, ground truth and first maximum prediction.
  `restored_logits.bin` is the second clean pass and is byte identical.
* `frozen_c/provenance.json`: model/subset/library/executable/source/data hashes,
  runtime version, thread count, resolver, compiler version and exact frontend
  compile/link flags. **historical library source revision and build flags
  remain unknown**, explicitly recorded as null. new executable flags are not
  presented as the library's flags. binary identity freezes the runtime.
* `frozen_c/block_points.json`: output slot/tensor/operator/shape/quantization
  mapping for the stem and all 18 inverted residual blocks.
* `clean_200/` and `fault_op3_elem17_bit7_200/`: clean and single fault captures
  on all 200 images. the fault is a persistent bit 7 flip at op 3, element 17;
  this is a capture validation, not a new campaign. each manifest records the
  weight before/after, runtime results, file hashes, and equivalence checks.
* `existing_suite.log`, `propagation_tests.log`, `verification.json`: validation
  record. generated capture arrays total approximately 150 MB for the two runs.

the c clean baseline reproduces **130/200 = 65%**. the library is version
`2.19.0`, sha256
`2eb9b9b2b108a4af52e91deafc05590b3e9672e7b3775c894d761d2b615e0a16`.
it uses the default c resolver, one thread, no explicit delegates and no
explicit reference kernel selection. this pins this macOS/Apple silicon
binary; it does not establish portability to another runtime or architecture.

## signed scale investigation

a fresh download from the [official release](https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-320kb-1mb_imagenet.tflite)
is byte identical to the local model, sha256
`780563f544bccba413b1ee4e051159996ac8b5be39eba8bb38761a51b58a07d1`.
the official fp32 checkpoint also matches the cached checkpoint. raw
little endian ieee754 decoding agrees with the schema: stem output channel 7
really has scale `-0.020387113094329834`, zero point 0. thus this is a property
of the released export, not a local corruption, entry 0 reporting issue, or
schema sign decoding error.

there Are **79 negative weight scale entries across 14 tensors**, with 79
corresponding negative bias scale entries. no activation scale is negative.
the stem's negative channel matches the checkpoint's sole negative stem bn
gamma (`-0.18509936332702637`). all convolution bias scale signs match
`input_scale * weight_scale`. these observations support signed bn folding
as an explanation. however, folding the supplied fp32 checkpoint with its
configured epsilon `1e-5` does not exactly reconstruct exported weights
(relative l2 difference about 4.8%). the original exporter/calibration
procedure is unavailable here, so the precise cause is **not established**.
the model is preserved; no scales are repaired or replaced by absolute values.

for conversion, apply the stored [affine quantization formula](https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/g3doc/performance/quantization_spec.md)
with the channel selected by `quantized_dimension`:

```
real[channel] = (q[channel] - zero_point[channel]) * scale[channel]
delta_real[channel] = (q_faulted[channel] - q_clean[channel]) * scale[channel]
```

at stem channel 7, q=10 represents approximately -0.20387113, not +0.20387113.
promote integers BEFORE subtraction to avoid int8 overflow. for unsigned error
magnitude use `abs(delta_real)`; do not discard the scale sign when reconstructing
values. `scripts/propagation.py:dequantize` implements broadcasting. conv weight
axis is 0; depthwise weight axis is 3. activation metadata is per tensor.
the [tflite v2.19 convolution multiplier code](https://github.com/tensorflow/tensorflow/blob/v2.19.0/tensorflow/lite/kernels/kernel_util.cc)
uses channel scales in its effective output multipliers, but that source is
not proof of the historical dylib's exact source/build. other backends may
handle these unusual signed scales differently.

## retaining residual stream outputs

the tracing flatbuffer is a separate derived file. its only changes are a
patched `subgraph.outputs` pointer and an appended output vector; every
operator, tensor table, constant, and original weight byte remains unchanged.
the original final output stays at slot 0. the extra 19 Graph outputs remain
live through invoke and are copied immediately afterward through the c api.
this avoids relying on recycled arena data or a python interpreter's kernels.
it increases memory use; this is host side tracing, not mcu memory profiling.

boundaries are discovered and checked through graph edges: each depthwise
convolution's projection, followed by the residual add when present. both
the main path and bypass must match the preceding block boundary. there are
11 residual additions. captured operator outputs are:

```
stem: 1
blocks 00..17: 5, 11, 18, 25, 32, 38, 45, 52, 58, 65, 72, 78, 85, 92,
              98, 105, 112, 118
```

the implementation intentionally rejects a different graph. capturing a
projection before a residual add would be the wrong block boundary; these
captures use the add output instead. the classifier is available separately
through the FINAL output, not labeled as an inverted residual block.

each `output_NNN.bin` is raw int8 in image major order followed by nhwc tensor
order. its per image shape (including batch dimension 1) is in `capture.json`.
for example:

```python
import json
import numpy as np
from pathlib import Path
capture = Path('artifacts/propagation/clean_200')
info = json.loads((capture / 'capture.json').read_text())
point = info['points'][0]  # stem
array = np.fromfile(capture / f"activations/output_{point['output_slot']:03d}.bin",
                    dtype=np.int8).reshape(info['images'], *point['tensor']['shape'])
# dequantize(array[image_index], point['tensor'])
```

each capture is checked against a separate uninstrumented c execution with
the same fault and images. final vectors and predictions must be byte identical;
after full buffer restoration, outputs must match the frozen clean baseline.
a failed run leaves an incomplete directory without a verified `capture.json`.
this checks output equivalence per capture rather than assuming instrumentation
will preserve every possible future execution.

## reproduce or extend

use the existing venv and local licensed subset. all output destinations must
be new; the tools Refuse to overwrite existing directories/files.

```sh
venv311/bin/python scripts/propagation.py metadata --out /tmp/mcunet-quantization.json
venv311/bin/python scripts/propagation.py freeze --out artifacts/propagation/my_frozen_c
venv311/bin/python scripts/propagation.py capture \
  --frozen artifacts/propagation/my_frozen_c --out artifacts/propagation/my_clean
venv311/bin/python scripts/propagation.py capture \
  --frozen artifacts/propagation/my_frozen_c --out artifacts/propagation/my_fault \
  --fault 3 17 7
venv311/bin/python -B -m unittest discover -s tests -p 'test_propagation.py' -v
```

`freeze` accepts `--model`, `--subset` and `--tflite`; it rejects model,
library, subset or core injector bytes different from the historical campaign.
`capture --images N` uses subset indices 0 through n 1, with n in 1..200.
the integration tests use the delivered `frozen_c`, `clean_200` and
`fault_op3_elem17_bit7_200` directories.

the investigation script is reproducible from explicit local files; it does
not download anything itself:

```sh
venv311/bin/python scripts/investigate_quantization.py \
  --upstream-model /path/to/fresh-official.tflite \
  --upstream-checkpoint /path/to/fresh-official.pth \
  --out /tmp/mcunet-scale-investigation.json
```

verification includes raw quantization vector decoding, deterministic metadata,
signed dequantization, original/tracing equality for all 200 clean and faulted
images, single representative faults in all 55 layers on two images, and
independent truncated prefix c interpreters checking all 19 boundaries on two
clean and faulted images (76 tensor comparisons). the existing injector's
99 check suite is run directly against read only original reference files;
`make test` is avoided because it would REGENERATE those files.

the unknown historical library build flags, original export procedure,
crash/hang isolation and future propagation metrics remain open. this adds
capture and prerequisites, not activation fault injection or masked/attenuated/
amplified classification. the checked graph, binary and input bytes bound the
current verification claims.
