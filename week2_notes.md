# week 2 notes

## what i did

cloned both mcunet and tinyengine repos from mit han lab. set up a python venv
and installed all dependencies. main goal was getting a reproducible baseline
run working w/ the pretrained mcunet-in3 model.

ran into a few issues:
- typo in net_id (mcunet-int3 instead of mcunet-in3), model zoo just throws
  an assertion error if the id is wrong
- imagenet-1k downloads the entire 150gb dataset by default if u dont use
  streaming mode, killed that fast
- tried tiny-imagenet first but it has diff class labels (200 classes, 0-199)
  vs imagenet (1000 classes) so accuracy was 0%, labels just never matched
- fixed by using ILSVRC/imagenet-1k on huggingface w/ streaming=True so it
  pulls images one at a time instead of downloading everything

## what baseline.py is doing

loads the pretrained mcunet-in3 model, a mobilenet-based architecture designed
by tinynas to fit within 320kb sram and 1mb flash. weights come from mit han
lab and are in float32 for the pytorch version (the tflite version is int8,
thats what we'll be working w/ for fault injection).

preprocessing pipeline:
- resize each image to the models expected input resolution
- convert from PIL image to pytorch tensor
- normalize pixel values using imagenet mean/std, model was trained w/ this
  normalization so it expects inputs in this range

inference loop:
- streams 1000 samples from imagenet validation set one at a time
- runs each image thru the model, takes argmax of the 1000-class output vector
  as the predicted class
- compares prediction to ground truth label, tracks running top-1 accuracy

top-1 accuracy means the models single best guess has to match the true label.
top-5 would be more lenient (true label just has to be in top 5 guesses) and
would give a higher number.

## result

model: mcunet-in3 (320kb sram / 1mb flash target)
dataset: imagenet-1k validation, 1000 samples streamed from huggingface
top-1 accuracy: 40.20%

paper reports ~62% top-1 on the full 50k validation set. the gap is bc:
1. only using 1000 samples so theres more variance
2. using simple resize instead of the standard center crop preprocessing
3. evaluating the pytorch float32 model not the int8 tflite version

this 40.20% is the clean baseline, all fault injection results will be
compared against this number to measure accuracy degradation.

## next steps (week 3)

build a layer inventory of mcunet-in3, list every layer w/ its type,
parameter count, tensor shape, and mac count. this is the foundation for
deciding where to inject faults.
