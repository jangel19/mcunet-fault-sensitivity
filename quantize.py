import torch
from mcunet.model_zoo import build_model
from datasets import load_dataset
from torchvision import transforms

#need this for quantization
torch.backends.quantized.engine = 'qnnpack'
#laod the mdoel
model, image_size, description = build_model(net_id="mcunet-in3",
                                             pretrained=True,)
model.eval()

#prepare for quantize using qnnpack instead of fbgemm since im on apple silicon
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

print("Model ready for cal")

# calibration dataset
dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print("running cal on 1000 samples (same as baseline)")

with torch.no_grad():
    for i, sample in enumerate(dataset):
        if i >= 1000:
            break
        img = sample["image"]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = transform(img).unsqueeze(0) #converts to tensor and adds batch dimension
        model(x)
        print(f"Calibration: {i}/1000")


#convert to int8
torch.quantization.convert(model, inplace=True)
torch.save(model.state_dict(), "mcunet_int8.pth")
print("int8 model saved to mcunet_int8.pth")


