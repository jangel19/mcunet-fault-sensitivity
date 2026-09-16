import os
import torch
from mcunet.model_zoo import build_model

model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)

# count parameters
total_params = sum(p.numel() for p in model.parameters())
float32_size = total_params * 4  # 4 bytes per float32
int8_size = total_params * 1     # 1 byte per int8

print(f"Parameters: {total_params:,}")
print(f"Float32 size: {float32_size/1024:.1f} KB")
print(f"INT8 size: {int8_size/1024:.1f} KB")
