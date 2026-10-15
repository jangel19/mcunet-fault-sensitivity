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


## step 3 - injection candidate selection

### the pool
55 injectable ops out of 122 (counted from the csv, conv_2d / depthwise_conv_2d only):
- 1 stem conv (3x3, stride 2)
- 18 depthwise
- 35 pointwise 1x1 (expand + project)
- 1 classifier (1x1, 160 -> 1000)

other 67 ops (pad, add, pooling, reshape) have no weights so nothing to flip. so the full heatmap is 55 layers x 8 bits = 440 cells

### what i'm choosing between
- depth - stem, stage 1 through 5, then the classifier (tests error propagation distance)
- mac/param weight - high-mac vs low-mac layers
- shortcut presence - residual blocks vs not
- layer type - depthwise vs pointwise. lit says depthwise should be more fragile (no cross channel averaging to dilute the error). treating that as a hypothesis to test on mcunet, not something i assume

### stages -> csv ops
the five stages from step 2 mapped onto the op indexes in layer_inventory.csv. depthwise ops listed entry block first (the stride 2 one)

| stage | res | depthwise ops | shortcut adds |
|---|---|---|---|
| stem | 176x176 -> 88x88 | op 1 (3x3 conv, s2) | - |
| 1 | 88x88 | 3 | none |
| 2 | 44x44 | 9, 15, 22, 29 | 18, 25, 32 |
| 3 | 22x22 | 36, 42, 49 | 45, 52 |
| 4 | 11x11 | 56, 62, 69, 76, 82, 89 | 65, 72, 85, 92 |
| 5 | 6x6 | 96, 102, 109, 116 | 105, 112 |
| tail | 1x1 | op 120 (classifier) | - |

11 shortcut adds total. blocks without a shortcut are the stride 2 entry block of each stage, plus op 76 (40 -> 48 channels) and op 116 (96 -> 160 channels) since those change channel count

### approach
representative / corner cases first, not an exhaustive sweep. so start with a small hand picked set off the csv, get the injection pipeline working end to end, then widen to all 55 layers once its validated

### first pass (draft, built off the stage table)
walk the stages in order. in stages 2-4 take the entry depthwise (stride 2, no shortcut) and a shortcut block depthwise with the same kernel size, so depth moves and kernel size doesnt

| op | stage | layer | why |
|---|---|---|---|
| 1 | stem | conv 3x3 s2 | first layer, everything downstream sees the corruption |
| 3 | 1 | depthwise 3x3 | only block in stage 1, earliest depthwise |
| 9 | 2 | depthwise 7x7 s2, entry | no shortcut |
| 22 | 2 | depthwise 7x7, shortcut | same kernel as op 9, highest macs in the net (7.59m) |
| 36 | 3 | depthwise 5x5 s2, entry | no shortcut |
| 42 | 3 | depthwise 5x5, shortcut | same kernel as op 36 |
| 56 | 4 | depthwise 3x3 s2, entry | no shortcut |
| 89 | 4 | depthwise 3x3, shortcut | same kernel as op 56 |
| 60, 62, 64 | 4 | expand 1x1, depthwise 7x7, project 1x1 | one shortcut block, depthwise vs pointwise at the same depth |
| 102 | 5 | depthwise 7x7, shortcut | late stage |
| 116 | 5 | depthwise 7x7, no shortcut | identical shape + macs to op 102, only difference is the shortcut |
| 120 | tail | classifier 1x1 | most params (160k), feeds the class scores directly |

14 of the 55. entry vs shortcut pairs are a bit confounded since the entry block also downsamples with stride 2, so 102 vs 116 is the clean shortcut test

### per layer
- weights first, activations later
- pick random elements in the layer, flip each bit 0-7, one flip per trial
- run on a fixed subset of images (not all 50k per trial), compare against the clean run
- log layer, element, bit, seed, mismatch vs clean, accuracy

## still open
- programmatic stage/block grouping (manual right now)
- bias tensor counting (int32, not in the 721k yet)
- confirm the first pass layer list
- how big the image subset is, need to time one trial first
