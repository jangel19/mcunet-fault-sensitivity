import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
# load the pre-built int8 tflite model from mit han lab
tflite_path = "/Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite"

# set up the tflite interpreter (this is what runs the model)
interpreter = tf.lite.Interpreter(tflite_path)
interpreter.allocate_tensors()

# get input and output tensor info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
resolution = input_shape[1]

transform = transforms.Compose([
    transforms.Resize(int(resolution * 256 / 224)),
    transforms.CenterCrop(resolution),
])


print(f"model input shape: {input_shape}")
print(f"resolution: {resolution}")

# stream imagenet validation set (same 1000 samples as baseline)
dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

correct = 0
total = 1000

for i, sample in enumerate(dataset):
    if i >= total:
        break

    img = sample['image']
    label = sample['label']

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # resize to model input size
    img = transform(img)

    # convert to int8 — tflite int8 models expect values in [-128, 127]
    # pixel values are 0-255, so we normalize to 0-1 then shift by -128
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np * 255 - 128).astype(np.int8)
    img_np = img_np.reshape(input_shape)

    # run inference
    interpreter.set_tensor(input_details[0]['index'], img_np)
    interpreter.invoke()

    # get prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output)

    if pred == label:
        correct += 1

    if i % 50 == 0:
        print(f"progress: {i}/{total} | running accuracy: {correct/(i+1)*100:.2f}%")

print(f"\nmodel: mcunet-in3 (320kb sram / 1mb flash)")
print(f"quantization: int8 tflite (mit han lab official)")
print(f"final int8 accuracy on {total} samples: {correct/total*100:.2f}%")
print(f"baseline float32 accuracy: 64.00%")
