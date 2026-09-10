neural networks by default start weight as float32 which is fine on a computer but when we are dealing with mcu's we have real constraints 

in this case we would use int8 for example which gives us 4x memory reduction 

float32 weight = 4 bytes while int8 weight = 1 byte

----- 

math behind this
for every float32 weight x there is a mapping 

x_quantized = round(x/ scale) + zero_point)
when we look at this see we see that scale and zero_point are computed from the 
range of values in that layer. 

to get the float back 
x_approx = (x_approx - zero_point )*scale 

when doing this precision because we are cramming things in a continuos range 
into 256 possible values

ptq vs qat

        ptq                     qat
when    after training is done  during training 
effort  low - js convert        high - retrain 
accuracy small but real         minimal
tflite uses     ptq             optional

this task uses ptq 

---

    
