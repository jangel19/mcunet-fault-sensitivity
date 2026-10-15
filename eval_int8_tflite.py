import numpy as np
import tensorflow as tf
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
# load the pre-built int8 tflite model from mit han lab
tflite_path = "/Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite"


def make_transform(resolution):
    # resize then center crop, same recipe as the float baseline
    return transforms.Compose([
        transforms.Resize(int(resolution * 256 / 224)),
        transforms.CenterCrop(resolution),
    ])


def iter_val_stream(total=50000):
    # stream imagenet validation set, yields (index, PIL image, label).
    # order is fixed by the dataset, so the same call always gives the same stream
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    for i, sample in enumerate(dataset):
        if i >= total:
            break
        yield i, sample['image'], sample['label']


def preprocess_to_int8(img, transform, input_shape):
    # PIL image -> int8 array shaped like the model input
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # resize to model input size
    img = transform(img)

    # convert to int8 — tflite int8 models expect values in [-128, 127]
    # pixel values are 0-255, so we normalize to 0-1 then shift by -128
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np * 255 - 128).astype(np.int8)
    return img_np.reshape(input_shape)


def main():
    # set up the tflite interpreter (this is what runs the model)
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    # get input and output tensor info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    resolution = input_shape[1]

    transform = make_transform(resolution)

    print(f"model input shape: {input_shape}")
    print(f"resolution: {resolution}")

    correct = 0
    total = 50000

    for i, img, label in iter_val_stream(total):
        img_np = preprocess_to_int8(img, transform, input_shape)

        # run inference
        interpreter.set_tensor(input_details[0]['index'], img_np)
        interpreter.invoke()

        # get prediction
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output)

        if pred == label:
            correct += 1

        if i % 100 == 0:
            print(f"progress: {i}/{total} | running accuracy: {correct/(i+1)*100:.2f}%")

    print(f"\nmodel: mcunet-in3 (320kb sram / 1mb flash)")
    print(f"quantization: int8 tflite (mit han lab official)")
    print(f"final int8 accuracy on {total} samples: {correct/total*100:.2f}%")
    print(f"baseline float32 accuracy: 62.14%")


if __name__ == "__main__":
    main()
