# mcunet fault sensitivity - fault model (week five)

model: mcunet-in3, 320kb sram variant, mit han lab
file: mcunet-320kb-1mb_imagenet.tflite
goal: define exactly what a fault is, where it goes, and how i score it so the injection framework has a spec to build against


## summary

- fault = single bit flip in an int8 tensor
- weights first, activations second
- bits 0-7 swept uniformly, lit used as a hypothesis not an assumption
- random elements, seeded, fixed image subset
- outcomes: masked / subtle sdc / catastrophic sdc / due, scored against the clean run


## step 1 - fault type

single bit flip, one flip per trial. no multi-bit, no stuck-at, no random value replacement in v1

why single bit: its the common fault model in this lit (li et al. sc17, binfi, the bit flip attack paper) and it keeps each trial attributable to exactly one (layer, element, bit)

the flip is an xor on the stored int8 byte (2's complement):
- bits 0-6 change the stored value by +/- 2^bit
- bit 7 is the sign bit, flips value by -128 or +128 depending on the sign it started with
- int8 is bounded so no nan / exponent blowup like fp32, worst case is a 128 jump in stored units


## step 2 - target tensors

in scope (v1):
- int8 weight tensors of conv_2d and depthwise_conv_2d (55 ops, from layer_inventory.csv)

in scope (phase 2):
- int8 activation tensors (output of each conv/depthwise op)

out of scope, and why:
- biases - int32, different bit width, not in my param count yet either
- scale / zero_point - quant metadata, not data. corrupting these is a different fault class
- pad / add / pooling / reshape - no weights, nothing storable to corrupt
- anything below the tensor level (registers, alu, instruction faults like sassifi does) - not what this project is about

weight zero point is 0 on every conv in the csv so a flip on the stored byte maps cleanly onto the real weight. activation zero points are -128 or 0 depending on the layer so activation flips happen on the stored int8 value, not the dequantized one


## step 3 - weight faults vs activation faults

these are two different things and i'm keeping them separate:

| | weight fault | activation fault |
|---|---|---|
| models | corruption of stored weights (flash / sram) | transient upset while a layer is computing |
| lifetime | persistent, stays until restored | one inference only |
| affects | every image run while its in place | one image |
| spatial reach | every output position that weight touches | one element of one feature map |
| status | v1 (week 6) | phase 2 (week 11) |

ares (reagen et al. dac 2018) found activations were as tolerant as or more tolerant than weights at the same bit error rate, up to ~50x for resnet50 / vgg16. they put it down to weight reuse, one bad weight gets multiplied across the whole feature map. so expect a weight flip to hurt more than an activation flip. that was fixed point models with faults in memory, so still something to check on mcunet

note on activations: the tflite python interpreter has no per-op hook so activation injection needs a different mechanism than weights (split model, or the c route). deciding that when i get to week 11


## step 4 - injection timing

weights:
1. pick (layer, element, bit)
2. flip the bit in the model buffer
3. run the whole fixed image subset with the corrupted model
4. flip the same bit again to restore
5. check the model buffer hash matches the clean one before the next trial (cheaper than rerunning inference)

activations (phase 2):
1. run forward pass up to layer L
2. flip one bit in one element of layer L's int8 output
3. continue the forward pass
4. next image gets a clean run again


## step 5 - bit position

uniform sweep over bits 0-7. no weighting toward the msb.

hypothesis: the high bits matter most and bits 0-2 are mostly noise. support for it is indirect:
- bit k changes the stored int8 value by 2^k (sign bit by 128), thats just arithmetic
- li et al. sc17 found only high order bits cause sdcs (float and fixed point, gpu accelerators)
- binfi argues the effect of a flip grows with bit position (32 bit fixed point, approximate for some nets)
- rakin et al. (bit flip attack) found 100 random flips on an 8 bit resnet-18 cost under ~1% top-1, so random int8 flips are mostly benign and the critical ones are rare
- i didnt find anything that tests int8 bit by bit on a tinyml model, so thats the gap this project fills

sweeping all 8 evenly is what lets me test this on mcunet, and it's cheap enough that there's no reason to prune. since critical flips are rare, subtle sdc (logit change) is worth tracking and not just top-1


## step 6 - layer selection

full space: 55 injectable layers x 8 bits = 440 cells

first pass is 14 layers picked off the stage table (see architecture_week4.md):

| op | stage | layer |
|---|---|---|
| 1 | stem | conv 3x3 s2 |
| 3 | 1 | depthwise 3x3 |
| 9 | 2 | depthwise 7x7 s2, entry |
| 22 | 2 | depthwise 7x7, shortcut |
| 36 | 3 | depthwise 5x5 s2, entry |
| 42 | 3 | depthwise 5x5, shortcut |
| 56 | 4 | depthwise 3x3 s2, entry |
| 89 | 4 | depthwise 3x3, shortcut |
| 60 | 4 | expand 1x1 |
| 62 | 4 | depthwise 7x7, shortcut |
| 64 | 4 | project 1x1 |
| 102 | 5 | depthwise 7x7, shortcut |
| 116 | 5 | depthwise 7x7, no shortcut |
| 120 | tail | classifier 1x1 |

the pairs control for kernel size (9/22, 36/42, 56/89) and 102 vs 116 have identical shape + macs so thats the clean shortcut test. 60/62/64 is depthwise vs pointwise inside one block. whether depthwise is actually more fragile is an open question, i couldnt find a paper that shows it (see the end)

once the pipeline is validated this widens to all 55


## step 7 - sampling

- element: uniform random within the layer's weight tensor, seeded
- one flip per trial
- k trials per (layer, bit) cell
- images: a fixed subset, same images every trial. not all 50k per trial

on how many trials: cochran's formula gives ~385 trials for +/-5% at 95% confidence (n = z^2 p(1-p) / e^2, p = 0.5). thats per estimate, and 385 x 440 cells is ~169k trials, too many. so k per cell is small (starting point ~30) and i report confidence intervals per cell, then pool across layers per bit to get tight per-bit estimates

sample size math reference: leveugle et al., statistical fault injection: quantified error and confidence. for the layer x bit stratified setup see ruospo et al. date 2023, they stratify by layer and bit and compare against exhaustive runs, but on fp32 stuck-at faults (resnet-20 / mobilenetv2 on cifar-10), not int8 flips

first pass budget check: 14 layers x 8 bits x 30 trials = 3,360 trials. at 200 images per trial thats 672,000 inferences. need to time one inference before locking k and the subset size

image subset: ~200 images, seeded, spread across classes (the first 1000 streamed samples only covered 638 classes so not just taking the first n). cached once as a .npy so nothing re-streams per trial. the 50k run stays the baseline accuracy number, separate from the injection runs


## step 8 - outcome classes

per image, compared against the clean run on that same image:

| class | definition |
|---|---|
| masked | faulted logits identical to clean logits |
| subtle sdc | logits changed, top-1 same |
| catastrophic sdc | top-1 changed |
| due | crash or hang (no output) |

due should be ~0. int8 data flips dont crash anything, crashes come from corrupted metadata like indices / shapes / pointers and thats out of scope. worth reporting as a result anyway (breier et al. 2026 report int8 weights stay bounded so no nan or extreme values like the float formats)

per trial metrics (over the subset):
- mismatch rate = fraction of images where faulted top-1 != clean top-1
- accuracy under fault vs labels, and drop vs clean accuracy on the same subset
- counts of masked / subtle / catastrophic / due

heatmap cell (layer x bit) = mean over its trials. sdc is defined against the clean prediction, accuracy drop is what goes in the degradation heatmap

open: does catastrophic mean any top-1 change, or only the clean class dropping out of top-5. leaning any top-1 change for v1, can add top-5 later


## step 9 - reproducibility

- seed per trial, logged. numpy default_rng, seed -> (element, bit)
- image subset saved as .npy, sha256 recorded in the log
- model file sha256 recorded
- requirements.txt pinned
- before trusting anything: clean run twice gives identical logits, and flip + restore gives identical logits to the clean run

log row per trial:

```
trial_id, seed, op_index, op_type, stage, tensor (weight/act),
elem_idx, bit, orig_val, new_val, n_images,
mismatch_rate, acc_faulted, acc_clean,
n_masked, n_subtle, n_catastrophic, n_due
```


## references

- lin et al., mcunet: tiny deep learning on iot devices, neurips 2020
  https://arxiv.org/abs/2007.10319

- hari et al., sassifi: an architecture-level fault injection tool for gpu application resilience evaluation, ispass 2017
  https://research.nvidia.com/publication/2017-04_sassifi-architecture-level-fault-injection-tool-gpu-application-resilience

- chen et al., tensorfi: a flexible fault injection framework for tensorflow applications, issre 2020
  https://arxiv.org/abs/2004.01743

- li et al., understanding error propagation in deep learning neural network (dnn) accelerators and applications, sc17
  https://research.nvidia.com/publication/2017-11_understanding-error-propagation-deep-learning-neural-network-dnn-accelerators

- reagen et al., ares: a framework for quantifying the resilience of deep neural networks, dac 2018
  https://vlsiarch.eecs.harvard.edu/publications/ares-framework-quantifying-resilience-deep-neural-networks

- chen et al., binfi: an efficient fault injector for safety-critical machine learning systems, sc19
  https://people.ece.ubc.ca/zitaoc/files/sc19-binfi.pdf

- rakin et al., bit-flip attack: crushing neural network with progressive bit search, iccv 2019
  https://arxiv.org/abs/1903.12269

- breier et al., the weight of a bit: emfi sensitivity analysis of embedded deep learning models, 2026
  https://arxiv.org/abs/2602.16309

- ruospo et al., assessing convolutional neural networks reliability through statistical fault injections, date 2023
  https://past.date-conference.com/proceedings-archive/2023/DATA/807.pdf

- statistical fault injection: quantified error and confidence
  https://hal.archives-ouvertes.fr/hal-00386107

- sharif et al., efficient software-implemented hw fault tolerance for tinyml inference in safety-critical applications, date 2023
  https://past.date-conference.com/proceedings-archive/2023/DATA/6005.pdf

