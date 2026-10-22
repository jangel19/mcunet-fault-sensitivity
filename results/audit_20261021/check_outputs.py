import csv, hashlib, json, pathlib, collections
import numpy as np
import tflite

repo = pathlib.Path('/Users/jordiangel/Desktop/research/ee research/mcunet-fault-sensitivity')
model_path = pathlib.Path.home()/'.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite'
raw = model_path.read_bytes()
model = tflite.Model.GetRootAsModel(raw, 0)
sg = model.Subgraphs(0)
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((repo/'subset/manifest.json').read_text())
assert sha(raw) == manifest['model_sha256']
labels_raw = (repo/'subset/labels.bin').read_bytes()
assert sha(labels_raw) == manifest['labels_sha256']
labels = np.frombuffer(labels_raw, dtype='<i4')
for i, e in enumerate(manifest['images']):
    b = (repo/'subset'/e['file']).read_bytes()
    assert sha(b) == e['sha256'] and len(b) == 176*176*3
    assert labels[i] == e['label']
print('manifest verified:', len(labels), 'images, unique labels',len(set(labels)), 'last stream index',manifest['images'][-1]['stream_index'])
print('model sha256', sha(raw))
print('manifest sha256', sha((repo/'subset/manifest.json').read_bytes()))
rows = list(csv.DictReader((repo/'results/first_pass_corner.csv').open()))
assert len(rows)==560
cells=collections.Counter((r['op_index'],r['bit']) for r in rows)
assert len(cells)==112 and set(cells.values())=={5}
assert {int(r['op_index']) for r in rows}=={1,3,9,22,36,42,56,89,60,62,64,102,116,120}
mask = (1 << 64)-1
def next64(x):
    x = (x + 0x9E3779B97F4A7C15)&mask
    z = ((x^(x>>30))*0xBF58476D1CE4E5B9)&mask
    z = ((z^(z>>27))*0x94D049BB133111EB)&mask
    return (z^(z>>31))&mask
def seed(op, bit, k, base=2026):
    return next64((next64((next64((base+op)&mask)+bit)&mask)+k)&mask)
rules = ['max_abs','min_nonzero_abs','first_zero','first_elem','last_elem']
duplicates=[]
for j,r in enumerate(rows):
    op, bit, e = (int(r[x]) for x in ['op_index','bit','elem_idx'])
    code=model.OperatorCodes(sg.Operators(op).OpcodeIndex()).BuiltinCode()
    assert r['op_type']=={3:'CONV_2D',4:'DEPTHWISE_CONV_2D'}[code]
    assert r['tensor']=='weight' and 0<=bit<=7
    stage=next(name for cutoff,name in [(2,'stem'),(7,'1'),(34,'2'),(54,'3'),(94,'4'),(120,'5'),(123,'tail')] if op<cutoff)
    assert r['stage']==stage
    t = sg.Tensors(sg.Operators(op).Inputs(1))
    b = model.Buffers(t.Buffer()).DataAsNumpy()
    w = b.view(np.int8).astype(int)
    picks = [int(abs(w).argmax()),int(np.flatnonzero(abs(w)==min(abs(w[w!=0])))[0]),int(np.flatnonzero(w==0)[0]),0,len(w)-1]
    assert e == picks[rules.index(r['elem_rule'])]
    assert int(r['orig_val']) == w[e]
    assert int(r['new_val']) == ((int(b[e])^(1<<bit))+128)%256-128
    assert int(r['seed']) == seed(op,bit,1000+rules.index(r['elem_rule']))
    assert int(r['trial_id']) == j
    counts=[int(r[x]) for x in ['n_masked','n_subtle','n_catastrophic','n_due']]
    assert sum(counts)==int(r['n_images'])==200
    assert abs(float(r['mismatch_rate'])-(counts[2]+counts[3])/200)<1e-9
    assert float(r['acc_clean'])==.65
    assert abs(float(r['acc_faulted'])*200-round(float(r['acc_faulted'])*200))<1e-8
    assert abs(float(r['acc_faulted'])-.65) <= float(r['mismatch_rate'])+1e-9
print('all 560 rows independently verified: rule, original byte, XOR, seed, counts, rates')
print('worst', min(rows,key=lambda r:float(r['acc_faulted'])))
print('trials with changed logits',sum(int(r['n_masked'])<200 for r in rows),'of',len(rows))
print('trials with changed top1',sum(int(r['n_catastrophic'])>0 for r in rows))
print('outcome totals', {k:sum(int(r[k]) for r in rows) for k in ['n_masked','n_subtle','n_catastrophic','n_due']})
for key in ['op_index','bit']:
    for value in dict.fromkeys(r[key] for r in rows):
        group=[r for r in rows if r[key]==value]
        print(key,value,'catastrophic',sum(int(r['n_catastrophic']) for r in group)/(200*len(group)),'mean accuracy drop',sum(.65-float(r['acc_faulted']) for r in group)/len(group))
for op in dict.fromkeys(int(r['op_index']) for r in rows):
    chosen={r['elem_rule']:int(r['elem_idx']) for r in rows if int(r['op_index'])==op}
    byelem=collections.defaultdict(list)
    for rule,e in chosen.items(): byelem[e].append(rule)
    duplicates.extend((op,e,names) for e,names in byelem.items() if len(names)>1)
print('duplicate rule picks',duplicates)
print('unique physical faults',len({(r['op_index'],r['elem_idx'],r['bit']) for r in rows}))
