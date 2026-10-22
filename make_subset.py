"""Build the fixed image subset used by every fault-injection trial.

Rule (fault_model_week5.md, step 7): walk the imagenet validation stream in its
fixed order and keep the first image of every class not seen yet, until N
images are collected. No randomness anywhere, so running this twice gives a
byte-identical subset.

Output directory (default subset/):
    img_000.bin ...   raw int8 NHWC [1,H,W,3], exactly what goes into the model
    labels.bin        int32 little-endian, one label per image
    manifest.json     stream index, label and sha256 per image, plus the model sha256

Preprocessing comes from eval_int8_tflite.py so these images are exactly the
ones the 50k baseline evaluation fed to the model.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import tensorflow as tf

from model_config import add_model_argument
import eval_int8_tflite as ev


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def build_subset(n_images, scan_limit, model_path):
    interpreter = tf.lite.Interpreter(model_path)
    input_shape = interpreter.get_input_details()[0]["shape"]
    transform = ev.make_transform(int(input_shape[1]))

    seen = set()
    picked = []  # (stream_index, label, int8 array)
    for i, img, label in ev.iter_val_stream(scan_limit):
        if label in seen:
            continue
        seen.add(label)
        picked.append((i, label, ev.preprocess_to_int8(img, transform, input_shape)))
        if len(picked) == n_images:
            break
    if len(picked) < n_images:
        raise SystemExit(f"only found {len(picked)} classes in the first {scan_limit} samples")
    return picked, [int(d) for d in input_shape]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of images (one per class)")
    ap.add_argument("--out", default="subset", help="output directory")
    ap.add_argument("--scan-limit", type=int, default=50000, help="max stream samples to scan")
    add_model_argument(ap)
    args = ap.parse_args()

    picked, input_shape = build_subset(args.n, args.scan_limit, args.model)

    os.makedirs(args.out, exist_ok=True)
    entries = []
    for k, (stream_idx, label, arr) in enumerate(picked):
        raw = np.ascontiguousarray(arr, dtype=np.int8).tobytes()
        name = f"img_{k:03d}.bin"
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(raw)
        entries.append({"file": name, "stream_index": stream_idx,
                        "label": int(label), "sha256": sha256_bytes(raw)})

    labels = np.array([e["label"] for e in entries], dtype="<i4")
    with open(os.path.join(args.out, "labels.bin"), "wb") as f:
        f.write(labels.tobytes())

    manifest = {
        "n_images": len(entries),
        "input_shape": input_shape,
        "dtype": "int8",
        "labels_dtype": "int32 little-endian",
        "rule": "first image of each new class in stream order",
        "model_sha256": sha256_file(args.model),
        "labels_sha256": sha256_bytes(labels.tobytes()),
        "images": entries,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    print(f"wrote {len(entries)} images to {args.out}/ "
          f"(last stream index {entries[-1]['stream_index']})", flush=True)


if __name__ == "__main__":
    main()
    # the streaming dataset leaves background threads that can keep the process
    # alive after the work is done, so exit explicitly (files are already closed)
    os._exit(0)
