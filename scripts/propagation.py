"""Frozen C baseline and retained block outputs for the official mcunet-in3.

No campaign runner or historical files are written. Every destination must be new.
Run with venv311/bin/python scripts/propagation.py --help.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import tflite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_config import add_model_argument


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def graph(raw):
    model = tflite.Model.GetRootAsModel(raw, 0)
    if model.SubgraphsLength() != 1:
        raise ValueError('only the single-subgraph mcunet-in3 artifact is supported')
    sg = model.Subgraphs(0)
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        ops.append({'index': i,
                    'type': tflite.opcode2name(max(code.BuiltinCode(), code.DeprecatedBuiltinCode())),
                    'inputs': [int(x) for x in op.InputsAsNumpy()],
                    'outputs': [int(x) for x in op.OutputsAsNumpy()]})
    return model, sg, ops


def metadata(raw):
    model, sg, ops = graph(raw)
    weights = {o['inputs'][1] for o in ops if o['type'] in ('CONV_2D', 'DEPTHWISE_CONV_2D')}
    biases = {o['inputs'][2] for o in ops if o['type'] in ('CONV_2D', 'DEPTHWISE_CONV_2D')}
    activations = set(int(x) for x in sg.InputsAsNumpy())
    activations.update(t for o in ops for t in o['outputs'])
    tensors = []
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        q = t.Quantization()
        quant = None
        if q is not None:
            quant = {'scales': [float(q.Scale(j)) for j in range(q.ScaleLength())],
                     'zero_points': [int(q.ZeroPoint(j)) for j in range(q.ZeroPointLength())],
                     'quantized_dimension': int(q.QuantizedDimension()),
                     'min': [float(q.Min(j)) for j in range(q.MinLength())],
                     'max': [float(q.Max(j)) for j in range(q.MaxLength())]}
            # Preserve float32 bit patterns independently of JSON float formatting.
            quant['scale_float32_hex'] = [struct.pack('<f', x).hex() for x in quant['scales']]
        tensors.append({'index': i, 'name': t.Name().decode() if t.Name() else '',
                        'shape': [int(t.Shape(j)) for j in range(t.ShapeLength())],
                        'type': int(t.Type()), 'buffer': int(t.Buffer()),
                        'role': 'weight' if i in weights else 'bias' if i in biases else
                                'activation' if i in activations else 'constant',
                        'quantization': quant,
                        'producer_ops': [o['index'] for o in ops if i in o['outputs']],
                        'consumer_ops': [o['index'] for o in ops if i in o['inputs']]})
    return {'schema_version': 1, 'model_sha256': hashlib.sha256(raw).hexdigest(),
            'formula': 'real = (int8 - zero_point[channel]) * scale[channel]; channel follows quantized_dimension',
            'operators': ops, 'tensors': tensors}


def dequantize(values, tensor):
    """Apply the stored affine map without discarding signed scales.

    Accepts one tensor with its declared shape; batch capture files must first
    be reshaped to [images, *shape] and passed one image at a time.
    """
    values = np.asarray(values)
    if list(values.shape) != tensor['shape']:
        raise ValueError('tensor shape mismatch')
    q = tensor['quantization']
    if not q or not q['scales']:
        raise ValueError('tensor has no affine quantization')
    scale = np.asarray(q['scales'], dtype=np.float64)
    zp = np.asarray(q['zero_points'], dtype=np.float64)
    if len(zp) not in (1, len(scale)):
        raise ValueError('incompatible scale/zero-point lengths')
    shape = [1] * values.ndim
    if len(scale) > 1:
        axis = q['quantized_dimension']
        if len(scale) != values.shape[axis]:
            raise ValueError('per-channel scale count mismatch')
        shape[axis] = len(scale)
    return (values.astype(np.float64) - zp.reshape(shape if len(zp) > 1 else [1]*values.ndim)) * scale.reshape(shape)


def block_points(raw):
    """Follow the actual depthwise -> projection -> optional residual ADD graph.

    This is deliberately artifact-specific and fails closed on other graphs.
    Residual inputs must be the preceding block boundary, not an arbitrary ADD.
    """
    _, sg, ops = graph(raw)
    if len(ops) != 122 or ops[1]['type'] != 'CONV_2D':
        raise ValueError('unexpected mcunet-in3 graph')
    points = [{'name': 'stem', 'op_index': 1, 'tensor_index': ops[1]['outputs'][0],
               'residual_add': False}]
    producer = {t: o for o in ops for t in o['outputs']}

    def unpad(t):
        while t in producer and producer[t]['type'] == 'PAD':
            t = producer[t]['inputs'][0]
        return t

    for dw in (o for o in ops if o['type'] == 'DEPTHWISE_CONV_2D'):
        j = dw['index'] + 1
        while ops[j]['type'] == 'PAD':
            j += 1
        project = ops[j]
        if project['type'] != 'CONV_2D' or unpad(project['inputs'][0]) != dw['outputs'][0]:
            raise ValueError('depthwise projection edge mismatch')
        previous = points[-1]['tensor_index']
        entry = unpad(dw['inputs'][0])
        if entry != previous:
            expand = producer[entry]
            if expand['type'] != 'CONV_2D' or unpad(expand['inputs'][0]) != previous:
                raise ValueError('block entry edge mismatch')
        end = project
        residual = ops[j + 1]['type'] == 'ADD'
        if residual:
            end = ops[j + 1]
            if sorted(end['inputs']) != sorted([project['outputs'][0], previous]):
                raise ValueError('residual bypass edge mismatch')
        points.append({'name': f'block_{len(points)-1:02d}', 'depthwise_op': dw['index'],
                       'project_op': j, 'op_index': end['index'],
                       'tensor_index': end['outputs'][0], 'residual_add': residual})
    if len(points) != 19 or sum(p['residual_add'] for p in points) != 11:
        raise ValueError('expected stem plus 18 blocks and 11 residual additions')
    meta = metadata(raw)
    for i, p in enumerate(points, 1):
        p.update({'output_slot': i, 'tensor': meta['tensors'][p['tensor_index']]})
    return points


def tracing_model(raw):
    """Append a new output vector; preserve all existing tables/weights verbatim.

    Only subgraph.outputs' relative offset is patched. Graph outputs are live
    until the caller reads them, so the arena cannot recycle these tensors.
    """
    _, sg, _ = graph(raw)
    if sg.OutputsLength() != 1:
        raise ValueError('expected one original final output')
    points = block_points(raw)
    output_ids = [int(sg.Outputs(0))] + [p['tensor_index'] for p in points]
    raw = bytearray(raw)
    raw.extend(b'\0' * (-len(raw) % 4))
    vector = len(raw)
    raw.extend(struct.pack('<I' + 'i' * len(output_ids), len(output_ids), *output_ids))
    field = sg._tab.Pos + sg._tab.Offset(8)  # SubGraph field 2: outputs
    struct.pack_into('<I', raw, field, vector - field)
    _, result, _ = graph(raw)
    if [int(x) for x in result.OutputsAsNumpy()] != output_ids:
        raise ValueError('tracing output vector verification failed')
    return bytes(raw), points


def checked_subset(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text())
    entries = manifest['images']
    if manifest['n_images'] != 200 or len(entries) != 200:
        raise ValueError('expected the 200-image campaign subset')
    if sha(path / 'labels.bin') != manifest['labels_sha256']:
        raise ValueError('labels hash mismatch')
    labels = np.fromfile(path / 'labels.bin', dtype='<i4')
    for i, entry in enumerate(entries):
        image = path / f'img_{i:03d}.bin'
        if entry['file'] != image.name or image.stat().st_size != 176*176*3 or sha(image) != entry['sha256']:
            raise ValueError(f'image {i} hash or size mismatch')
    if labels.tolist() != [e['label'] for e in entries]:
        raise ValueError('manifest label mismatch')
    return path, manifest


def execute(runtime, model, subset, images, out, fault=None):
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith('DYLD_')}
    args = [str(runtime / 'capture_activations'), str(model), str(subset), str(images), str(out)]
    args += [str(x) for x in (fault if fault else (-1, 0, 0))]
    proc = subprocess.run(args, env=env, check=True, capture_output=True, text=True)
    loaded = [x.removeprefix('loaded_library=') for x in proc.stderr.splitlines()
              if x.startswith('loaded_library=')]
    if len(loaded) != 1 or Path(loaded[0]).resolve() != (runtime / 'lib/libtensorflowlite_c.dylib').resolve():
        raise ValueError(f'unexpected loaded runtime: {loaded}')
    result = json.loads(proc.stdout)
    write_json(out / 'runtime.json', result)
    return result


def freeze(args):
    campaign = json.loads((ROOT / 'results/random_k30_provenance.json').read_text())
    model = Path(args.model).resolve()
    subset, manifest = checked_subset(args.subset)
    lib = Path(args.tflite).resolve() / 'lib/libtensorflowlite_c.dylib'
    if sha(model) != campaign['model_sha256'] or sha(lib) != campaign['c_library_sha256']:
        raise ValueError('must use the exact historical model and C library')
    if sha(subset / 'manifest.json') != campaign['subset_sha256']:
        raise ValueError('must use the exact historical subset')
    for name in ('c/fault_inject.c', 'c/run_trials.c'):
        if sha(ROOT / name) != campaign['source_sha256'][name]:
            raise ValueError('historical injection implementation changed')
    dest = Path(args.out).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    runtime = dest / 'runtime'
    (runtime / 'lib').mkdir(parents=True)
    shutil.copy2(lib, runtime / 'lib' / lib.name)
    source_names = ['c/capture_activations.c', 'c/fault_inject.c', 'c/fault_inject.h',
                    'scripts/propagation.py', 'model_config.py']
    for name in source_names:
        snapshot = dest / 'source' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, snapshot)
    flags = ['-std=c11', '-O2', '-g', '-Wall', '-Wextra']
    command = ['cc', *flags, '-I' + str(Path(args.tflite).resolve() / 'include'),
               str(dest / 'source/c/capture_activations.c'), str(dest / 'source/c/fault_inject.c'),
               '-L' + str(runtime / 'lib'), '-ltensorflowlite_c',
               '-Wl,-rpath,@loader_path/lib', '-o', str(runtime / 'capture_activations')]
    subprocess.run(command, check=True)
    result = execute(runtime, model, subset, 200, dest / 'baseline')
    if result['correct'] != 130 or result['library_version'] != campaign['c_library_reported_version']:
        raise ValueError('historical C clean baseline not reproduced')
    original = model.read_bytes()
    traced, points = tracing_model(original)
    (dest / 'tracing.tflite').write_bytes(traced)
    write_json(dest / 'quantization.json', metadata(original))
    write_json(dest / 'block_points.json', points)
    shutil.copy2(subset / 'manifest.json', dest / 'subset_manifest.json')
    prov = {'schema_version': 1, 'model_sha256': sha(model), 'model_path': str(model),
            'subset_path': str(subset), 'subset_manifest_sha256': sha(subset / 'manifest.json'),
            'library_sha256': sha(lib), 'library_version': result['library_version'],
            'library_original_source_revision': campaign['c_library_source_revision'],
            'library_original_build_flags': campaign['c_library_build_flags'],
            'library_build_limitation': 'Historical source revision and build flags were not recorded; binary identity is frozen instead.',
            'resolver': 'default TFLite C resolver; no explicit delegates or reference-kernel selection',
            'threads': 1, 'capture_build_command': command, 'capture_compiler_flags': flags,
            'compiler': subprocess.check_output(['cc', '--version'], text=True),
            'source_sha256': {n: sha(dest / 'source' / n) for n in source_names},
            'baseline': result, 'format': 'output_000.bin: raw int8 [200,1000]; predictions.csv: image_index,label,prediction',
            'tool_versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                              'tflite_schema': __import__('importlib.metadata', fromlist=['version']).version('tflite')}}
    prov['files_sha256'] = {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    write_json(dest / 'provenance.json', prov)
    print(f'frozen: {dest}; clean C accuracy 130/200 = 65%; {len(points)} capture points')


def verify_freeze(dest):
    prov = json.loads((dest / 'provenance.json').read_text())
    for name, expected in prov['files_sha256'].items():
        if sha(dest / name) != expected:
            raise ValueError(f'frozen artifact changed: {name}')
    if sha(prov['model_path']) != prov['model_sha256']:
        raise ValueError('original model changed')
    subset, _ = checked_subset(prov['subset_path'])
    if sha(subset / 'manifest.json') != prov['subset_manifest_sha256']:
        raise ValueError('subset manifest changed')
    return prov


def capture(args):
    if not 1 <= args.images <= 200:
        raise ValueError('images must be in 1..200')
    if args.fault:
        op, elem, bit = args.fault
        if not (0 <= op <= 121 and elem >= 0 and 0 <= bit <= 7):
            raise ValueError('fault requires OP in 0..121, nonnegative ELEMENT, BIT in 0..7')
    dest = Path(args.frozen).resolve()
    prov = verify_freeze(dest)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='mcunet-capture-') as scratch:
        reference = Path(scratch) / 'original'
        ref = execute(dest / 'runtime', Path(prov['model_path']), Path(prov['subset_path']),
                      args.images, reference, args.fault)
        result = execute(dest / 'runtime', dest / 'tracing.tflite', Path(prov['subset_path']),
                         args.images, out / 'activations', args.fault)
        for name in ('output_000.bin', 'restored_logits.bin', 'predictions.csv'):
            if (reference / name).read_bytes() != (out / 'activations' / name).read_bytes():
                raise ValueError(f'capturing changes {name}')
    clean = (dest / 'baseline/output_000.bin').read_bytes()[:args.images*1000]
    if (out / 'activations/restored_logits.bin').read_bytes() != clean:
        raise ValueError('restored outputs differ from frozen clean baseline')
    if not args.fault and (out / 'activations/output_000.bin').read_bytes() != clean:
        raise ValueError('clean capture differs from frozen clean baseline')
    points = json.loads((dest / 'block_points.json').read_text())
    for p in points:
        shape = p['tensor']['shape']
        path = out / f"activations/output_{p['output_slot']:03d}.bin"
        if path.stat().st_size != args.images * int(np.prod(shape)):
            raise ValueError('activation file size mismatch')
    write_json(out / 'capture.json', {'images': args.images, 'image_indices': list(range(args.images)),
               'fault': args.fault, 'original_model_run': ref, 'tracing_model_run': result,
               'final_outputs_bit_identical': True, 'restored_outputs_match_frozen_baseline': True,
               'frozen_provenance_sha256': sha(dest / 'provenance.json'), 'points': points,
               'format': 'raw int8, image-major then NHWC; shape per image is tensor.shape',
               'files_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(f'captured {args.images} images at {len(points)} boundaries; final outputs byte-identical')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    meta = commands.add_parser('metadata', help='deterministic complete tensor quantization JSON')
    add_model_argument(meta)
    meta.add_argument('--out', required=True)
    frozen = commands.add_parser('freeze', help='snapshot the exact historical C binary and save its clean baseline')
    add_model_argument(frozen)
    frozen.add_argument('--subset', default='subset')
    frozen.add_argument('--tflite', default=str(Path.home() / '.local/tflite-2.21.0'))
    frozen.add_argument('--out', required=True)
    cap = commands.add_parser('capture', help='capture clean or one persistent weight fault, checking original outputs')
    cap.add_argument('--frozen', required=True)
    cap.add_argument('--out', required=True)
    cap.add_argument('--images', type=int, default=200)
    cap.add_argument('--fault', type=int, nargs=3, metavar=('OP', 'ELEMENT', 'BIT'))
    args = parser.parse_args()
    if args.command == 'metadata':
        write_json(args.out, metadata(Path(args.model).read_bytes()))
    elif args.command == 'freeze':
        freeze(args)
    else:
        capture(args)


if __name__ == '__main__':
    main()
