Week Four
--

Goal: Map layer index to operation, tensor shape, quantization parameters, parameter count, and MACs. Identify potential injection points.

firs have to build out the map. 

in a neural network we are storing numberss in memory which are the weights...then some input data is introduced then a sequence of operation combine the output weihts until we get an output (ie 87% flower or like cat depending on the image since we are using an image model)

each operation in this sequence is what the layer iis 


convolution: slide a small window of numbers across a grid of nums and then multiply and add at each position then record res

lets say we have an input, for th efirst layer, we have an image thats 224x224 pixels and has 3 channels so its really 224x224x3 block numbers.
we have a kernel(filter), which is a small grid of weights that is for ex 3x3x3 

we slide this over the image. at each pos, we do an elementsie multiply between the filter and patch the image underneath then sum everything into one number. that number is the output produces one pixel of the output

we repeat this until we get all of the grid

the thing is we dont only do this once. if we only do this once we would only detect one pattern. to have a good convolution layer we need many, each one looks for something different. 

this is where MACs come in which are multiply accumulate operations. every single multiply the filter against the patch and sum are one MAC per multiplication. thats why this would be a good idea if i wasnt working on an mcu. macs are in essence counting how many multiply adds the hardware has to do to execute this layer.



macs are good...but for this they are not as it is too expensive for the limitations that i have

formula: MACs = K_h × K_w × C_in × C_out × H_out × W_out

every term multiplied tg...if you have for ex 64 input channels and want 128 output channels with a 3x3 filter on a 56x56 map, thats basically
3x64x128x56x56 which is roughly 231 MILLION macs for one layer

an mcu with kb of ram and no gpu cant handle this, especially repeated a tens of times through a net

MobileNet does this in a quite unique way: splits this one expensive operation into two cheap ones

Depthwise separable convolution:

instead of one filter that only looks at one cahnnel, it is split into two jobs:

Depthwise convolution: each input channel gets its own small filter, and that filter only looks at that one channel, no mixing. therefore if you 64 input channel, you have 64 tiny filters and each are only producing one output channel. 

with this we have no C_out

MACs = K_h × K_w × C_in × H_out × W_out

Pointwise convolution:
    one step we need is to mix info across channels. for this we do a plain 1x1 convolution, a per pixel weighted combination of channels, but it all mixes C_in into C_out channels.

MACs = C_in × C_out × H_out × W_out

depthwise costs 3×3×64×56×56 ≈ 5.6 million MACs, then pointwise (64→128) costs 64×128×56×56 ≈ 25.7 million MACs. Total ≈ 31 million versus 231 million for plain conv. Roughly 7-8x cheaper, for a very similar result.

we have to do this int8 as we are dealing with constraints so we have to compress things to one bytw 

the trained weights are not optimized for this as some might be 0.00054ot like -1.87, not integers.

to fix this we need a scale and zero point mapping 
we pick a linear mapping between the int8 range and the real floating range that the weights would occupy. the mapping is defined by two numbers stored per tensor
- scale how much real number distance one integet step represents (scale = 0.03 means each increment of the int8 val -> a real value change of .03)

- zero point- which int8 value represents the real number 0.0

real_value = (int8_value - zero_point) × scale
int8_value = round(real_value / scale) + zero_point

so if a tensors weights range from about -2 and 2, TFLite's converter picks a scale that maps that whole range onto the int8 range and stores the scale and zero point alongside the tensor so any code reading the tensor knows how to reconstruct the approx real value.

every tensor needs its own scale and zero_point because different layer's weights and activations have diff natural ranger. once layer might cluster around [-0.1,.1] and another might be 100x that. so if we have a global scale this would plummet the precision on whichever tensors are far from that scale. the convertor calibrates each tensor independently and thats exactly why we need a scale and zero_point column per op

this is specially important for fault injection. if we flip a bit of a stored int8 value, the real world magnitude of that corruption depends entirely on the tensor's scale. a bit flip on a tensor with a very small scale like .0001 would barely move the value while a scale of 5 would swing this a lot. 






