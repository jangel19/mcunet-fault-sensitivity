from check_outputs import pathlib, csv, seed, next64, sg, model
out=pathlib.Path(__file__).parent
a=(out/'random_a.csv').read_bytes(); b=(out/'random_b.csv').read_bytes()
assert a==b
random_rows=list(csv.DictReader(a.decode().splitlines()))
assert len(random_rows)==18
for r in random_rows:
    op,bit=int(r['op_index']),int(r['bit'])
    cell=[x for x in random_rows if x['op_index']==r['op_index'] and x['bit']==r['bit']]
    k=cell.index(r)
    expected=seed(op,bit,k)
    t=sg.Tensors(sg.Operators(op).Inputs(1))
    n=model.Buffers(t.Buffer()).DataLength()
    assert int(r['seed'])==expected
    assert int(r['elem_idx'])==next64(expected)%n
print('pass: 18 random rows, byte-identical repeats; seeds and elements independently recomputed')
