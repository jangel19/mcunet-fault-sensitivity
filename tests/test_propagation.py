"""Integration checks against the frozen C library; no historical files written.

Requires the documented freeze and clean_200 / fault_op3_elem17_bit7_200 runs.
Prefix interpreters independently execute only through each boundary, making
buffer reuse or a wrong full-graph tensor selection observable.
"""
import ctypes as ct
import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import propagation as p

ART = ROOT / 'artifacts/propagation'
FROZEN = ART / 'frozen_c'


class PrefixInterpreter:
    """Independent C API calls, without fi_runner_init or its output assumptions."""
    def __init__(self, lib, raw):
        self.lib = lib
        self.raw = ct.create_string_buffer(raw)
        self.model = lib.TfLiteModelCreate(self.raw, len(raw))
        self.options = lib.TfLiteInterpreterOptionsCreate()
        lib.TfLiteInterpreterOptionsSetNumThreads(self.options, 1)
        self.interpreter = lib.TfLiteInterpreterCreate(self.model, self.options)
        if not self.interpreter or lib.TfLiteInterpreterAllocateTensors(self.interpreter):
            raise RuntimeError('prefix interpreter setup failed')

    def run(self, image):
        lib = self.lib
        inp = lib.TfLiteInterpreterGetInputTensor(self.interpreter, 0)
        data = ct.create_string_buffer(image)
        if lib.TfLiteTensorCopyFromBuffer(inp, data, len(image)) or lib.TfLiteInterpreterInvoke(self.interpreter):
            raise RuntimeError('prefix invoke failed')
        out = lib.TfLiteInterpreterGetOutputTensor(self.interpreter, 0)
        size = lib.TfLiteTensorByteSize(out)
        result = ct.create_string_buffer(size)
        if lib.TfLiteTensorCopyToBuffer(out, result, size):
            raise RuntimeError('prefix output copy failed')
        return result.raw

    def close(self):
        self.lib.TfLiteInterpreterDelete(self.interpreter)
        self.lib.TfLiteInterpreterOptionsDelete(self.options)
        self.lib.TfLiteModelDelete(self.model)


class PropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prov = p.verify_freeze(FROZEN)
        cls.raw = Path(cls.prov['model_path']).read_bytes()
        cls.meta = p.metadata(cls.raw)
        cls.points = p.block_points(cls.raw)
        cls.lib = ct.CDLL(str(FROZEN / 'runtime/lib/libtensorflowlite_c.dylib'))
        v, i, z = ct.c_void_p, ct.c_int, ct.c_size_t
        signatures = {
            'TfLiteModelCreate': ([v,z],v), 'TfLiteModelDelete': ([v],None),
            'TfLiteInterpreterOptionsCreate': ([],v),
            'TfLiteInterpreterOptionsSetNumThreads': ([v,i],None),
            'TfLiteInterpreterOptionsDelete': ([v],None),
            'TfLiteInterpreterCreate': ([v,v],v), 'TfLiteInterpreterDelete': ([v],None),
            'TfLiteInterpreterAllocateTensors': ([v],i), 'TfLiteInterpreterInvoke': ([v],i),
            'TfLiteInterpreterGetInputTensor': ([v,i],v),
            'TfLiteInterpreterGetOutputTensor': ([v,i],v),
            'TfLiteTensorByteSize': ([v],z),
            'TfLiteTensorCopyFromBuffer': ([v,v,z],i),
            'TfLiteTensorCopyToBuffer': ([v,v,z],i)}
        for name, (args, result) in signatures.items():
            func = getattr(cls.lib, name)
            func.argtypes, func.restype = args, result

    def test_complete_metadata_and_raw_arrays(self):
        self.assertEqual(self.meta, json.loads((FROZEN / 'quantization.json').read_text()))
        self.assertEqual(self.meta, p.metadata(self.raw))
        _, sg, _ = p.graph(self.raw)
        self.assertEqual(len(self.meta['tensors']), sg.TensorsLength())
        self.assertEqual(sum(t['role'] == 'weight' for t in self.meta['tensors']), 55)
        for tensor in self.meta['tensors']:
            q = sg.Tensors(tensor['index']).Quantization()
            if q is None:
                self.assertIsNone(tensor['quantization'])
                continue
            for field, fmt, getter, length in [(8, 'f', 'scales', q.ScaleLength()),
                                               (10, 'q', 'zero_points', q.ZeroPointLength())]:
                if length:
                    offset = q._tab.Vector(q._tab.Offset(field))
                    decoded = list(struct.unpack_from('<'+fmt*length, self.raw, offset))
                    self.assertEqual(decoded, tensor['quantization'][getter])

    def test_signed_per_channel_dequantization(self):
        tensor = next(t for t in self.meta['tensors'] if t['producer_ops'] == [] and
                      t['role'] == 'weight' and t['consumer_ops'] == [1])
        values = np.full(tensor['shape'], 10, dtype=np.int8)
        real = p.dequantize(values, tensor)
        self.assertAlmostEqual(float(real[7,0,0,0]), -0.20387113094329834)
        self.assertGreater(float(real[0,0,0,0]), 0)
        output = self.meta['tensors'][self.points[0]['tensor_index']]
        zero = np.full(output['shape'], output['quantization']['zero_points'][0], np.int8)
        self.assertTrue(np.all(p.dequantize(zero, output) == 0))

    def test_only_outputs_pointer_and_appended_vector_change(self):
        traced, points = p.tracing_model(self.raw)
        _, sg, _ = p.graph(self.raw)
        field = sg._tab.Pos + sg._tab.Offset(8)
        self.assertEqual(traced[:field], self.raw[:field])
        self.assertEqual(traced[field+4:len(self.raw)], self.raw[field+4:])
        self.assertEqual(len(points), 19)
        self.assertEqual(sum(x['residual_add'] for x in points), 11)
        self.assertEqual(traced, (FROZEN / 'tracing.tflite').read_bytes())

    def test_full_subset_final_outputs_and_restoration(self):
        clean = (FROZEN / 'baseline/output_000.bin').read_bytes()
        self.assertEqual(len(clean), 200000)
        labels = np.fromfile(Path(self.prov['subset_path']) / 'labels.bin', '<i4')
        pred = np.frombuffer(clean, np.int8).reshape(200,1000).argmax(axis=1)
        self.assertEqual(int(np.sum(pred == labels)), 130)
        for name in ['clean_200', 'fault_op3_elem17_bit7_200']:
            manifest = json.loads((ART / name / 'capture.json').read_text())
            self.assertTrue(manifest['final_outputs_bit_identical'])
            for path, expected in manifest['files_sha256'].items():
                self.assertEqual(p.sha(ART / name / path), expected)
            self.assertEqual((ART / name / 'activations/restored_logits.bin').read_bytes(), clean)
        self.assertEqual((ART / 'clean_200/activations/output_000.bin').read_bytes(), clean)
        faulty = (ART / 'fault_op3_elem17_bit7_200/activations/output_000.bin').read_bytes()
        self.assertNotEqual(faulty, clean)

    def test_every_boundary_against_prefix_execution_clean_and_faulted(self):
        for faulted in [False, True]:
            raw = bytearray(self.raw)
            if faulted:
                model, sg, _ = p.graph(raw)
                weight = sg.Tensors(sg.Operators(3).Inputs(1))
                buf = model.Buffers(weight.Buffer())
                raw[buf._tab.Vector(buf._tab.Offset(4)) + 17] ^= 128
            for point in self.points:
                _, sg, _ = p.graph(raw)
                prefix = bytearray(raw)
                op_vector = sg._tab.Vector(sg._tab.Offset(10)) - 4
                struct.pack_into('<I', prefix, op_vector, point['op_index'] + 1)
                field = sg._tab.Pos + sg._tab.Offset(8)
                prefix.extend(b'\0' * (-len(prefix) % 4))
                struct.pack_into('<I', prefix, field, len(prefix) - field)
                prefix.extend(struct.pack('<Ii', 1, point['tensor_index']))
                interp = PrefixInterpreter(self.lib, bytes(prefix))
                folder = 'fault_op3_elem17_bit7_200' if faulted else 'clean_200'
                retained = (ART / folder / f"activations/output_{point['output_slot']:03d}.bin").read_bytes()
                size = int(np.prod(point['tensor']['shape']))
                try:
                    for image_index in range(2):
                        image = (Path(self.prov['subset_path']) / f'img_{image_index:03d}.bin').read_bytes()
                        self.assertEqual(interp.run(image), retained[image_index*size:(image_index+1)*size],
                                         f'{folder} {point["name"]} image {image_index}')
                finally:
                    interp.close()

    def test_weight_fault_equivalence_all_55_layers(self):
        layers = [o['index'] for o in self.meta['operators']
                  if o['type'] in ('CONV_2D', 'DEPTHWISE_CONV_2D')]
        self.assertEqual(len(layers), 55)
        with tempfile.TemporaryDirectory(prefix='mcunet-all-layers-') as directory:
            for op in layers:
                with self.subTest(op=op), contextlib.redirect_stdout(io.StringIO()):
                    out = Path(directory) / f'op_{op}'
                    p.capture(argparse.Namespace(frozen=str(FROZEN), out=str(out),
                                                images=2, fault=[op, 0, 7]))
                    result = json.loads((out / 'capture.json').read_text())
                    self.assertTrue(result['final_outputs_bit_identical'])
                    self.assertTrue(result['restored_outputs_match_frozen_baseline'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
