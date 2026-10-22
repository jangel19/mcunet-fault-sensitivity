import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

import tflite


LAYERS = [1, 3, 9, 22, 36, 42, 56, 89, 60, 62, 64, 102, 116, 120]
MASK = (1 << 64) - 1


def next64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) & MASK


def cell_seed(base, op, bit, trial):
    return next64((next64((next64((base + op) & MASK) + bit) & MASK) + trial) & MASK)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def interval(cells, metric, bounds):
    # each cell has its own draws, keep them seperate for the error bars
    samples = [[metric(row) for row in cell] for cell in cells]
    mean = statistics.mean(statistics.mean(values) for values in samples)
    count = sum(len(values) for values in samples)
    if all(value == 0 for values in samples for value in values):
        q = 1 - .05 ** (1 / count)
        return f"{100*mean:.3f} [{100*bounds[0]*q:.3f}, {100*bounds[1]*q:.3f}]†"
    variance = sum(statistics.variance(values) / len(values) for values in samples) / len(cells)**2
    margin = 1.96 * math.sqrt(variance)
    if variance == 0:
        margin = (bounds[1] - bounds[0]) * math.sqrt(math.log(40) / (2 * count))
    return f"{100*mean:.3f} [{100*(mean-margin):.3f}, {100*(mean+margin):.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--subset', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    subset = Path(args.subset)
    manifest_raw = (subset / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    raw = Path(args.model).read_bytes()
    assert sha(raw) == manifest['model_sha256']
    labels = (subset / 'labels.bin').read_bytes()
    assert sha(labels) == manifest['labels_sha256']
    assert len(labels) == 200 * 4
    assert len(manifest['images']) == 200
    for index, image in enumerate(manifest['images']):
        assert struct.unpack_from('<i', labels, index * 4)[0] == image['label']
        payload = (subset / image['file']).read_bytes()
        assert len(payload) == 176 * 176 * 3
        assert sha(payload) == image['sha256']
    model = tflite.Model.GetRootAsModel(raw, 0)
    graph = model.Subgraphs(0)
    rows = read_rows(args.csv)
    cells = collections.defaultdict(list)
    assert len(rows) == len(LAYERS) * 8 * args.k
    for index, row in enumerate(rows):
        op, bit, element = (int(row[key]) for key in ('op_index', 'bit', 'elem_idx'))
        assert op in LAYERS and 0 <= bit < 8
        cell = cells[op, bit]
        expected = cell_seed(args.seed, op, bit, len(cell))
        assert int(row['trial_id']) == index
        assert int(row['seed']) == expected
        assert row['elem_rule'] == 'random'
        operator = graph.Operators(op)
        opcode = model.OperatorCodes(operator.OpcodeIndex()).BuiltinCode()
        assert row['op_type'] == {3: 'CONV_2D', 4: 'DEPTHWISE_CONV_2D'}[opcode]
        assert row['tensor'] == 'weight'
        tensor = graph.Tensors(operator.Inputs(1))
        weights = model.Buffers(tensor.Buffer()).DataAsNumpy()
        assert element == next64(expected) % len(weights)
        original = int(weights[element])
        assert int(row['orig_val']) == (original + 128) % 256 - 128
        assert int(row['new_val']) == ((original ^ (1 << bit)) + 128) % 256 - 128
        assert sum(int(row[key]) for key in ('n_masked', 'n_subtle', 'n_catastrophic', 'n_due')) == 200
        assert int(row['n_images']) == 200
        assert float(row['acc_clean']) == .65
        assert abs(float(row['acc_faulted'])*200-round(float(row['acc_faulted'])*200)) < 1e-8
        assert abs(float(row['mismatch_rate']) - (int(row['n_catastrophic'])+int(row['n_due']))/200) < 1e-8
        assert abs(.65 - float(row['acc_faulted'])) <= float(row['mismatch_rate']) + 1e-8
        cell.append(row)
    assert len(cells) == 112 and {len(cell) for cell in cells.values()} == {args.k}
    pilot = read_rows(args.pilot)
    assert len(pilot) == 112
    for row in pilot:
        repeat = cells[int(row['op_index']), int(row['bit'])][0]
        assert {key: value for key, value in row.items() if key != 'trial_id'} == {
            key: value for key, value in repeat.items() if key != 'trial_id'}
    mismatch = lambda row: (int(row['n_catastrophic']) + int(row['n_due'])) / 200
    drop = lambda row: .65 - float(row['acc_faulted'])
    totals = {key: sum(int(row[key]) for row in rows)
              for key in ('n_masked', 'n_subtle', 'n_catastrophic', 'n_due')}
    unique = len({(row['op_index'], row['elem_idx'], row['bit']) for row in rows})
    lines = ['# seeded random campaign', '',
             f'seed `{args.seed}`, k={args.k}, 14 default layers, all eight bits, 200 fixed images.',
             f'{len(rows):,} trials, {len(rows)*200:,} trial-image evaluations, {unique:,} distinct physical faults.', '',
             '## checks and repeatability', '',
             'independently recomputed every cell seed and element through the python schema,',
             'checked original/xor bytes, trial counts, outcome totals and accuracy/rate consistency.',
             'verified every subset payload against the manifest and the model against its recorded hash.',
             'the separate k=1 pilot repeats the first draw in every layer-bit cell: all 112',
             'rows agree in every field except the sequential trial id. this includes seeds,',
             'selected weights and all outcomes. the runner also repeats its clean baseline',
             'and checks full-buffer restoration after every fault. the full 3,360 rows',
             'were not executed twice.', '',
             f'model sha256: `{sha(raw)}`.', '',
             f'subset hash (sha256 of manifest.json): `{sha(manifest_raw)}`.', '',
             f'campaign csv sha256: `{sha(Path(args.csv).read_bytes())}`.', '',
             '## measures and denominators', '',
             'one trial flips one stored weight bit, keeps it flipped for all 200 images,',
             'then restores it. masked means an identical int8 output vector; subtle means',
             'changed vector with unchanged top-1; catastrophic means changed top-1, including',
             'wrong-to-right and wrong-to-wrong. due records returned c-api errors only.',
             'top-1 mismatch is (catastrophic + due) / trial-image evaluations. accuracy',
             'drop is clean accuracy minus faulted accuracy, in percentage points, with',
             'returned errors counted incorrect. clean accuracy is 130/200 = 65%.', '',
             f'outcome counts over {len(rows)*200:,} trial-image evaluations: `{json.dumps(totals, sort_keys=True)}`.', '',
             f'changed output vectors: {totals["n_subtle"] + totals["n_catastrophic"]}/{len(rows)*200} trial-image evaluations.',
             f'trials with any changed output vector: {sum(int(r["n_subtle"]) + int(r["n_catastrophic"]) > 0 for r in rows)}/{len(rows)}.',
             f'trials with any changed top-1 or returned error: {sum(mismatch(r) > 0 for r in rows)}/{len(rows)}.', '',
             '## conditional summaries', '',
             'entries are mean [approximate 95% monte carlo interval]. mismatch is percent;',
             'accuracy drop is percentage points. intervals use 1.96 times the stratified',
             'standard error of trial-level rates, with sample variance estimated separately',
             'in each layer-bit cell. images are fixed and reused, never independent samples.',
             'each interval is marginal; comparisons across many groups need separate',
             'multiplicity handling. these intervals describe uncertainty from weight',
             'draws under this pseudorandom',
             'design, conditional on these images, model and library. k=30 is small for rare',
             'large effects; normal intervals may under-cover and can cross physical bounds.',
             'zero observed effects cannot rule out rare effects.', '',
             '† when every trial in a group is zero for a measure, the interval instead',
             'allows unseen rare effects. q = 1 - 0.05^(1/n) is a conservative 95% upper',
             'bound on the equal-weight probability of a nonzero trial; multiply q by',
             'the measure bounds [0, 1] for mismatch or [-0.35, 0.65] for accuracy drop.',
             'independent bounded draws in equal-sized strata support this bound even',
             'when stratum event probabilities differ: the probability of no events is',
             'at most (1 - mean event probability)^n. this does not assume image independence.',
             'other zero-variance cases use a conservative bounded-variable hoeffding',
             'interval instead of a zero-width normal interval.', '',
             'the sampler draws with replacement via splitmix64 modulo the layer size,',
             'with negligible modulo imbalance. repeats are retained. seeds are deterministic,',
             'so probabilistic interpretation assumes the seeded generator approximates',
             'independent uniform draws. summaries weight selected layers and bits equally,',
             'not in proportion to weight counts. they do not estimate a uniform fault',
             'probability over the whole model, other layers, other images or hardware.', '',
             '| group | trials | trial-image evaluations | top-1 mismatch % | accuracy drop pp |',
             '|---|---:|---:|---:|---:|']
    def add_group(name, selected):
        count = sum(len(cell) for cell in selected)
        lines.append(f'| {name} | {count} | {count*200} | {interval(selected, mismatch, (0, 1))} | {interval(selected, drop, (-.35, .65))} |')
    add_group('all selected cells', list(cells.values()))
    for bit in range(8):
        add_group(f'bit {bit}', [cells[op, bit] for op in LAYERS])
    for op in LAYERS:
        add_group(f'op {op}', [cells[op, bit] for bit in range(8)])
    worst = min(rows, key=lambda row: float(row['acc_faulted']))
    lines += ['', f'lowest observed faulted accuracy: {worst["acc_faulted"]} on trial {worst["trial_id"]},',
              f'op {worst["op_index"]}, element {worst["elem_idx"]}, bit {worst["bit"]}.', '',
              '## different from the corner campaign', '',
              'the historical campaign used five deterministic rules per layer-bit cell:',
              '560 rule-weighted trials, 528 distinct faults. this campaign uses 30 seeded',
              'element draws per cell. its means and uncertainty belong to that different',
              'design. neither design makes the 200 selected images representative of imagenet.',
              'zero returned errors is not evidence of crash/hang detection. all 55 weight',
              'tensors have per-channel scale metadata; the inventory retains only scale 0.',
              'no injector logic, input preprocessing or historical results were changed.', '']
    with Path(args.out).open('x') as handle:
        handle.write('\n'.join(lines))
    print(f'verified {len(rows)} rows, all 112 repeated draws and every subset payload')
    print(json.dumps(totals, sort_keys=True))


if __name__ == '__main__':
    main()
