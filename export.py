import torch
from mcunet.model_zoo import build_model

# load the model
model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)
model.eval()

#dummy input 
dummy_input = torch.randn(1,3, image_size, image_size)

#export to onnx
torch.onnx.export(
        model,
        dummy_input,
        "mcunet-in3.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
)

print("exported to mcunet-in3.onnx")


