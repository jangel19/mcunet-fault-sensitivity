"""Record raw stem-scale evidence and a bounded batch-normalization comparison.

Only reads supplied files. Downloads are explicit separate steps, documented
in PROPAGATION.md. torch.load uses weights_only=True.
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from propagation import graph, metadata, sha, write_json


def investigate(model_path, checkpoint_path, upstream_model=None, upstream_checkpoint=None):
    raw = Path(model_path).read_bytes()
    model, sg, _ = graph(raw)
    stem = sg.Tensors(sg.Operators(1).Inputs(1))
    q = stem.Quantization()
    scales = np.array([q.Scale(i) for i in range(q.ScaleLength())])
    start = q._tab.Vector(q._tab.Offset(8))  # QuantizationParameters field 2: scale
    decoded = list(struct.unpack_from('<' + 'f'*q.ScaleLength(), raw, start))
    if decoded != scales.tolist():
        raise ValueError('raw IEEE754 scale decoding disagrees with schema')
    sd = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
    fp = sd['first_conv.conv.weight'].numpy()
    gamma = sd['first_conv.bn.weight'].numpy()
    var = sd['first_conv.bn.running_var'].numpy()
    weights = np.frombuffer(model.Buffers(stem.Buffer()).DataAsNumpy().tobytes(), dtype=np.int8)
    weights = weights.reshape(16, 3, 3, 3).transpose(0, 3, 1, 2).astype(np.float64)
    stored_real = weights * scales[:, None, None, None]
    folded = fp * (gamma / np.sqrt(var + 1e-5))[:, None, None, None]
    negative = []
    for tensor in metadata(raw)['tensors']:
        quant = tensor['quantization']
        if quant:
            for ch, scale in enumerate(quant['scales']):
                if scale < 0:
                    negative.append({'tensor': tensor['index'], 'role': tensor['role'],
                                     'channel': ch, 'scale': scale})
    bias_scale_checks = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        if max(code.BuiltinCode(), code.DeprecatedBuiltinCode()) not in (3, 4):
            continue
        inp_q = sg.Tensors(op.Inputs(0)).Quantization()
        w_q = sg.Tensors(op.Inputs(1)).Quantization()
        b_q = sg.Tensors(op.Inputs(2)).Quantization()
        expected = np.array([inp_q.Scale(0)*w_q.Scale(j) for j in range(w_q.ScaleLength())])
        actual = np.array([b_q.Scale(j) for j in range(b_q.ScaleLength())])
        bias_scale_checks.append({'op_index': i,
                                  'signs_match': bool(np.all(np.sign(expected) == np.sign(actual))),
                                  'max_relative_difference': float(np.max(abs(expected-actual)/np.maximum(abs(expected), 1e-300)))})
    report = {'model_sha256': sha(model_path), 'checkpoint_sha256': sha(checkpoint_path),
              'official_model_url': 'https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-320kb-1mb_imagenet.tflite',
              'official_checkpoint_url': 'https://hanlab18.mit.edu/projects/tinyml/mcunet/release/mcunet-320kb-1mb_imagenet.pth',
              'negative_scale_entries': negative,
              'bias_scale_vs_input_times_weight_scale': bias_scale_checks,
              'stem_scale_vector_file_offset': start,
              'stem_scale_raw_bytes_hex': raw[start:start+4*len(scales)].hex(),
              'stem_scales': scales.tolist(), 'stem_bn_gamma': gamma.tolist(),
              'negative_stem_scale_channels': np.flatnonzero(scales < 0).tolist(),
              'negative_stem_bn_gamma_channels': np.flatnonzero(gamma < 0).tolist(),
              'bn_fold_comparison_eps': 1e-5,
              'bn_fold_relative_l2_error': float(np.linalg.norm(folded-stored_real)/np.linalg.norm(folded)),
              'bn_fold_max_absolute_error': float(np.max(abs(folded-stored_real))),
              'interpretation': 'Signed scales are consistent with absorbing the negative batch-normalization gamma into the scale. The supplied checkpoint does not reconstruct the exported weights exactly, so the original conversion procedure is not established.',
              'formula': 'real = (q - zero_point[channel]) * signed_scale[channel]',
              'example': {'channel': 7, 'q': 10, 'zero_point': int(q.ZeroPoint(7)),
                          'real': float((10-q.ZeroPoint(7))*scales[7])}}
    for name, path, original in [('model', upstream_model, model_path),
                                 ('checkpoint', upstream_checkpoint, checkpoint_path)]:
        if path:
            report[f'fresh_official_{name}_sha256'] = sha(path)
            report[f'fresh_official_{name}_byte_identical'] = Path(path).read_bytes() == Path(original).read_bytes()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default = Path.home() / '.torch/mcunet/mcunet-320kb-1mb_imagenet'
    ap.add_argument('--model', default=str(default.with_suffix('.tflite')))
    ap.add_argument('--checkpoint', default=str(default.with_suffix('.pth')))
    ap.add_argument('--upstream-model')
    ap.add_argument('--upstream-checkpoint')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    write_json(args.out, investigate(args.model, args.checkpoint, args.upstream_model, args.upstream_checkpoint))


if __name__ == '__main__':
    main()
