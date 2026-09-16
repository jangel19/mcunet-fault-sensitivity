import os, csv, tflite


model_path = os.path.expanduser("~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite")


def describe_tensor(subgraph, tensor_index):
    #pull out shape and quantization params

    tensor = subgraph.Tensors(tensor_index)

    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]

    quant = tensor.Quantization()
    scale = None
    zero_point = None
    if quant is not None and quant.ScaleLength() > 0:
        # TFLite supports per-channel quantization (multiple scale/zero_point
        # pairs per tensor) but MCUNet's export uses per-tensor quantization,
        # so index 0 is the only entry that exists here.
        scale = quant.Scale(0)
        zero_point = quant.ZeroPoint(0)
 
    return {"shape": shape, "scale": scale, "zero_point": zero_point}


def param_count_andMACs(op_type, weight_shape, output_shape):
    # do diff formulas depending on convolution type

    if op_type == "CONV_2D":
        c_out, k_h, k_w, c_in = weight_shape
        _, h_out, w_out, _  = output_shape
        params = c_out * k_h * k_w * c_in
        macs = k_h * k_w * c_in * c_out * h_out * w_out
        return params, macs

    if op_type == "DEPTHWISE_CONV_2D":
        _, k_h, k_w, c_in = weight_shape
        _, h_out, w_out, _ = output_shape
        params = k_h * k_w * c_in
        macs =k_h * k_w * c_in * h_out * w_out
        return params, macs
    #pad add and reshape and others have no learned weights 
    return None, None

def build_inventory(model_path):
    with open(model_path, "rb") as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)

    #0 is the whole network
    subgraph = model.Subgraphs(0)



    rows = []

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode_index = op.OpcodeIndex()
        opcode = model.OperatorCodes(opcode_index)
        op_type = tflite.opcode2name(opcode.BuiltinCode())
 
        row = {
            "op_index": i,
            "op_type": op_type,
            "input_shape": None,
            "weight_shape": None,
            "output_shape": None,
            "input_scale": None,
            "input_zp": None,
            "weight_scale": None,
            "weight_zp": None,
            "output_scale": None,
            "output_zp": None,
            "param_count": None,
            "macs": None,
            "injection_candidate": False,
        }



        #every op we need has one primary output tensor

        output_info = describe_tensor(subgraph, op.Outputs(0))
        row["output_shape"] = output_info["shape"]
        row["output_scale"] = output_info["scale"]
        row["output_zp"] = output_info["zero_point"]



        if op.InputsLength() > 0:
            input_info = describe_tensor(subgraph, op.Inputs(0))
            row["input_shape"] = input_info["shape"]
            row["input_scale"] = input_info["scale"]
            row["input_zp"] = input_info["zero_point"]



    #only the 2 convolution we have carry weight tensor
        if op_type in ("CONV_2D", "DEPTHWISE_CONV_2D") and op.InputsLength() > 1:
            weight_info = describe_tensor(subgraph, op.Inputs(1))
            row["weight_shape"] = weight_info["shape"]
            row["weight_scale"] = weight_info["scale"]
            row["weight_zp"] = weight_info["zero_point"]
            params, macs = param_count_andMACs(op_type, weight_info["shape"],
                                               output_info["shape"])


            row["param_count"] = params
            row["macs"] = macs
            row["injection_candidate"] = True

        rows.append(row)

    return rows


def write_csv(rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = build_inventory(model_path)
 
    output_path = "layer_inventory.csv"
    write_csv(rows, output_path)
 
    print(f"Wrote {len(rows)} rows to {output_path}")

    #make sure only include what we need


    total_macs = sum(r["macs"] for r in rows if r["macs"] is not None)
    total_params = sum(r["param_count"] for r in rows if r["param_count"] is not None)
    print(f"Total MACs across conv/depthwise ops: {total_macs:,}")
    print(f"Total params across conv/depthwise ops: {total_params:,}")



