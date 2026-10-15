"""Dump python tf.lite.Interpreter logits for the first K subset images.

test_fi.c reads these files and checks the C runtime gives the same logits.
Three python configurations are written, because tflite has several int8 kernel
sets whose rounding differs slightly and the C library is a fourth one:
    ref_logits_default.bin   tf.lite.Interpreter(path)            (what eval_int8_tflite.py uses)
    ref_logits_nodeleg.bin   default delegates (XNNPACK) switched off
    ref_logits_ref.bin       reference kernels only (BUILTIN_REF)
Each file is K * 1000 int8 values.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_int8_tflite as ev  # noqa: E402  (only for the model path)


def logits_for(interpreter, images):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    res = []
    for img in images:
        interpreter.set_tensor(inp["index"], img.reshape(inp["shape"]))
        interpreter.invoke()
        res.append(interpreter.get_tensor(out["index"]).reshape(-1).astype(np.int8))
    return np.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="subset")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--model", default=ev.tflite_path)
    args = ap.parse_args()

    images = []
    for i in range(args.k):
        raw = np.fromfile(os.path.join(args.subset, f"img_{i:03d}.bin"), dtype=np.int8)
        images.append(raw)

    R = tf.lite.experimental.OpResolverType
    configs = {
        "default": tf.lite.Interpreter(args.model),
        "nodeleg": tf.lite.Interpreter(
            args.model, experimental_op_resolver_type=R.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
        "ref": tf.lite.Interpreter(
            args.model, experimental_op_resolver_type=R.BUILTIN_REF),
    }
    for name, interp in configs.items():
        interp.allocate_tensors()
        lg = logits_for(interp, images)
        path = os.path.join(args.subset, f"ref_logits_{name}.bin")
        lg.tofile(path)
        print(f"wrote {path}: {lg.shape}")


if __name__ == "__main__":
    main()
