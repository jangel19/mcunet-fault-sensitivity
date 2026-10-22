import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv,json,pathlib,hashlib
import numpy as np
import tensorflow as tf
import tflite

root=pathlib.Path('/Users/jordiangel/Desktop/research/ee research/mcunet-fault-sensitivity')
out=pathlib.Path(__file__).parent
raw=(pathlib.Path.home()/'.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite').read_bytes()
sg=tflite.Model.GetRootAsModel(raw,0).Subgraphs(0)
images=[np.fromfile(root/'subset'/f'img_{i:03d}.bin',dtype=np.int8).reshape(1,176,176,3) for i in range(200)]
labels=np.fromfile(root/'subset/labels.bin',dtype='<i4')
def run(buf,config):
    interp=tf.lite.Interpreter(model_content=bytes(buf),num_threads=1,experimental_op_resolver_type=config)
    interp.allocate_tensors()
    inp=interp.get_input_details()[0]['index']; output=interp.get_output_details()[0]['index']
    outputs=[]
    for image in images:
        interp.set_tensor(inp,image); interp.invoke()
        outputs.append(interp.get_tensor(output).reshape(-1))
    return np.array(outputs)
R=tf.lite.experimental.OpResolverType
clean=run(raw,R.BUILTIN_REF)
print('independent python reference clean accuracy',np.mean(clean.argmax(1)==labels),flush=True)
rows=list(csv.DictReader((root/'results/first_pass_corner.csv').open()))
for r in rows:
    if r['op_index']!='120' and r['trial_id']!='79': continue
    op=sg.Operators(int(r['op_index'])); tensor=sg.Tensors(op.Inputs(1))
    model=tflite.Model.GetRootAsModel(raw,0)
    data=model.Buffers(tensor.Buffer()).DataAsNumpy()
    offset=data.ctypes.data-np.frombuffer(raw,dtype=np.uint8).ctypes.data
    buf=bytearray(raw); pos=offset+int(r['elem_idx']); buf[pos]^=1<<int(r['bit'])
    assert sum(a!=b for a,b in zip(buf,raw))==1
    fault=run(buf,R.BUILTIN_REF)
    same=np.all(fault==clean,axis=1); top=fault.argmax(1); clean_top=clean.argmax(1)
    result=dict(trial_id=int(r['trial_id']),op=int(r['op_index']),masked=int(same.sum()),subtle=int((~same & (top==clean_top)).sum()),catastrophic=int((top!=clean_top).sum()),accuracy=float(np.mean(top==labels)))
    print(json.dumps(result),flush=True)
    if r['op_index']=='120': assert result['catastrophic']==0
for name,config in [('default',R.AUTO),('nodeleg',R.BUILTIN_WITHOUT_DEFAULT_DELEGATES),('ref',R.BUILTIN_REF)]:
    lg=run(raw,config)
    lg.tofile(out/f'ref_logits_{name}.bin')
    old=np.fromfile(root/'subset'/f'ref_logits_{name}.bin',dtype=np.int8).reshape(200,1000)
    print('fresh reference',name,'matches saved',bool(np.array_equal(lg,old)),'accuracy',float(np.mean(lg.argmax(1)==labels)),flush=True)
