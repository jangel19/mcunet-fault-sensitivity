import tflite, os

model_path = os.path.expanduser("~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite")


#load the raw bytes of the model
with open(model_path, "rb") as f:
    buf = f.read()


# parse the flatbuffer into the model object
model = tflite.Model.GetRootAsModel(buf, 0)

#subgraph zero is the actual network
subgraph = model.Subgraphs(0)

print(f"total operators in the graph: {subgraph.OperatorsLength()}")
print(f"total tensors in graph: {subgraph.TensorsLength()}")
print()

for i in range(subgraph.OperatorsLength()): 
    op = subgraph.Operators(i)

    #lookup into models global opcode table where the actual type name lives
    opcode_index = op.OpcodeIndex()
    opcode = model.OperatorCodes(opcode_index)
    op_type = tflite.opcode2name(opcode.BuiltinCode())

    print(f"op {i:3d}: {op_type}")



