# week 2 notes

## what i did

cloned both mcunet and tinyengine repos from mit han lab. set up a venv and got
all deps installed. ran into a few issues getting the baseline working:

- typo in net_id (mcunet-int3 instead of mcunet-in3)
- imagenet-1k tries to download the whole 150gb dataset if u dont use streaming
- tiny-imagenet has diff labels than imagenet so accuracy was 0% lol
- fixed by using ILSVRC/imagenet-1k w/ streaming=True on huggingface

ended up w/ a working baseline running 1000 imagenet validation samples thru
mcunet-in3. accuracy came out to 40.20% top-1 which is lower than the papers
~62%m diff is bc were using simple resize instead of center crop and only
1000 samples instead of the full 50k val set (which would be crazy especially on my machine).

## what baseline.py is doing

1. loads pretrained mcunet-in3 weights from mit han lab (320kb sram / 1mb flash)
2. streams 1000 samples from imagenet validation set (streaming so it doesnt
   download 150gb)
3. preprocesses each imag resize to model input size, convert to tensor,
   normalize w/ imagenet mean/std (model was trained w/ this so it expects it)
4. runs forward pass, takes argmax of output as predicted class
5. compares pred to true label, tracks running accuracy
6. prints final top-1 accuracy

## result

model: mcunet-in3 (320kb sram / 1mb flash)
dataset: imagenet-1k validation, 1000 samples
top-1 accuracy: 40.20%

this is the clean baseline everything from here gets compared against this
number when we start injecting faults
