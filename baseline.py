import torch
from mcunet.model_zoo import build_model
from datasets import load_dataset
from torchvision import transforms

model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)
model.eval()

dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

transform = transforms.Compose([
    transforms.Resize(int(image_size * 256 / 224)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

correct = 0
total = 1000

for i, sample in enumerate(dataset):
    if i >= total:
        break
    img = sample['image']
    label = sample['label']
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    pred = out.argmax(dim=1).item()
    if pred == label:
        correct += 1
    
    if i % 50 == 0:
        print(f"Progress: {i}/{total} | Running accuracy: {correct/(i+1)*100:.2f}%")
labels_seen = set()
for i, sample in enumerate(dataset):
    if i >= 1000:
        break
    labels_seen.add(sample['label'])
print(f"unique classes seen: {len(labels_seen)}")

print(f"\nModel: {description}")
print(f"Final accuracy on {total} samples: {correct/total*100:.2f}%")
