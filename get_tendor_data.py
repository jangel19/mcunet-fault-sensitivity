# need to get the input tensor shape, weight tensor shape, output tensor shape, and quantization params

# the operator holds the tensor indices so we need to look up in the subgraphs for tensor list

import os, tflite
import numpy as np


model_path = os.path.expanduser("~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite")

with open(model_path, "rb") as f:
    buf = f.read()

model = tflite.Model.GetRootAsModel(buf, 0)
subgraph = model.Subgraphs(0)

def describe_tensor(tensor_index):
    tensor = subgraph.Tensors(tensor_index)

    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    quant = tensor.Quantization()
    scale = None
    zero_point = None
    if quant is not None and quant.ScaleLength() > 0:
        scale = quant.Scale(0)
        zero_point = quant.ZeroPoint(0)

    return {
        "name": tensor.Name().decode("utf-8") if tensor.Name() else f"tensor_{tensor_index}",
        "shape": shape,
        "scale": scale,
        "zero_point": zero_point,
    }

for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    opcode_index = op.OpcodeIndex()
    opcode = model.OperatorCodes(opcode_index)
    op_type = tflite.opcode2name(opcode.BuiltinCode())

    if op_type in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        input_tensor = describe_tensor(op.Inputs(0))
        weight_tensor = describe_tensor(op.Inputs(1))
        output_tensor = describe_tensor(op.Outputs(0))

        print(f"op {i:3d}: {op_type}")
        print(f"    input:  shape={input_tensor['shape']}  scale={input_tensor['scale']}  zp={input_tensor['zero_point']}")
        print(f"    weight: shape={weight_tensor['shape']}  scale={weight_tensor['scale']}  zp={weight_tensor['zero_point']}")
        print(f"    output: shape={output_tensor['shape']}  scale={output_tensor['scale']}  zp={output_tensor['zero_point']}")
        print()
