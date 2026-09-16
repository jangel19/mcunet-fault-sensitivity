import torch
from mcunet.model_zoo import build_model
from datasets import load_dataset
from torchvision import transforms

torch.backends.quantized.engine = 'qnnpack'

model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)
model.eval()

# calibration
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

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
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = transform(img).unsqueeze(0)
        model(x)
        if i % 100 == 0:
            print(f"calibration: {i}/1000")

# convert to int8
torch.quantization.convert(model, inplace=True)
print("converted to int8, running eval...")

# eval on fresh stream
dataset2 = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
correct = 0
total = 1000

with torch.no_grad():
    for i, sample in enumerate(dataset2):
        if i >= total:
            break
        img = sample['image']
        label = sample['label']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        x = transform(img).unsqueeze(0)
        out = model(x)
        pred = out.argmax(dim=1).item()
        if pred == label:
            correct += 1
        if i % 100 == 0:
            print(f"progress: {i}/{total} | running accuracy: {correct/(i+1)*100:.2f}%")

print(f"\nmodel: {description}")
print(f"final int8 accuracy on {total} samples: {correct/total*100:.2f}%")
