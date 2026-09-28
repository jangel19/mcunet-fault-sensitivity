# mcunet fault sensitivity — architecture / pipeline
 
model: mcunet-in3, 320kb sram variant, mit han lab
file: mcunet-320kb-1mb_imagenet.tflite
goal: characterize accuracy degradation from int8 bit-flip faults, layer by layer + bit position 0-7


## Pipeline

```
.tflite file
    -> raw flatbuffers parse (tflite pack age not the tf intepreter)
    -> layer_inventory.py -> layer_inventory.csv (122 ops)
    -> stage/block grouping from stride data
    -> layer inventory csv
    -> injection candidate selection
```

## Step 1 - layer inventory

layer_inventory.py parses the tflite file straight from raw flatbuffers. walks all 122 ops, records per op: type, stride, input/weight/output shapes, quant params (scale/zero_point), param count, macs, injection_candidate flag.

injection candidate = true only for conv_2d / depthwise_conv_2d. pad/add/reshape/pooling are excluded, nothing storable to corrupt.

verified against han lab's published model card:
- macs: 81,785,520 computed vs 81.8m published - exact match
- params: 721,032 vs ~0.74m - 2.7% low, bias tensors not counted yet (noted, not ignored)


## step 2 - architecture structure
stride pulled directly from op options (conv2doptions / depthwiseconv2doptions), verified against real shape transitions in teh csv.

network = repeating inverted residual block + stem no other structure really

block pattern :
1. expand - 1x1 conv, bumps channel count
2. depthwise - kxk conv (k = 3/5/7, varies per block), filters each channel independaently. only step where stride can be 2
3. project - 1x1 conv, shrinks channels back down
4. shortcut (some blocks only) - input added straight to output. only works if block doesnt change channels or spatial size, so first block of every stage never has one

kernel size + expansial ratio vary block to block, not just stage to stage - its tinynas output, not hand designed searched under a fixed sram budget

### five stages
| stage | res | entry | blocks | notes |
|---|---|---|---|---|
| stem | 176x176 -> 88x88 | 3x3 conv, stride 2 | - | one time entry, not a repeating block |
| 1 | 88x88 | stride 1 (already downsampled) | 1 | channels 8 -> 24 -> 16 |
| 2 | 44x44 | stride 2 | 3 | |
| 3 | 22x22 | stride 2 | 2 | |
| 4 | 11x11 | stride 2 | 5 | most blocks of any stage |
| 5 | 6x6 | stride 2 | 3 | ends at 160 channels |


4 stride-2 depthwise convs (stage boundaries 2-5) + stem's own stride-2 conv = 5 downsampling points total. 176x176 -> 6x6


### tail
6x6x160 -> global avg pool -> 160 values (1 per channel) -> 1x1 conv classifier -> 1000 imagenet class scores


## step 3 - injection candidate selection (in progress)
 
candidates being weighed:
- depth - early vs mid vs late layers (tests error propagation distance)
- mac/param weight - high-mac vs low-mac layers
- shortcut presence - residual blocks vs not

