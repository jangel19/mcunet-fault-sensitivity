## quantization notes

**model used:** mcunet-in3 (proxylessnas based, tinynas search, mit han lab)


**design constraints:** 320KB sram / 1MB flash
**measured footprint (from PAPER):** 293KB sram / 897KB flash
**footprint from ths experiment:** `.tflite` file size
**parameters:** 737,408
**input res:** 176x176

**quantization scheme:** int8 post training STATIC quantization (ptq), per tensor. used the official pre built `.tflite` model shipped by mit han lab via `download_tflite(net_id="mcunet-in3")`, rather than a custom quantization pipeline.

**why not a custom pipeline:** i tried to build an independent int8 quantization pipeline using pytorch's eager mode quantization api (`torch.quantization.prepare` / `convert`, `qnnpack` backend). this failed consistently across different environments that i tried such as apple silicon (python 3.14), a clean pyenv 3.11 environment, and google colab (python 3.13 all producing the same error:
    `NotImplementedError: Could not run 'quantized::conv2d.new' with arguments from the 'CPU' backend`


**root cause**: pytorch's eager mode static quantization for conv2d is not really that supported on anything older than python 3.12. after figuring this out i pivoted to what i shouldve done in the beginning, use mit han lab's official pre quantized tflite Artifact, which is the actual deployment target for this model family (tinyengine on mcus) and avoids reproducing a broken toolchain.


**preprocessing:** mcunet models are not trained with the standard imagenet normalization (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`). they use `mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]`, scaling inputs to `[-1,1]`. during my first baseline using standard normalization the float32 was a surprising ~40 to 45%. but when i fixed the normalization i was able to bump up the accuracy to ~64%.both float32 and int8 evaluation pipelines also require `Resize(int(resolution*256/224)) → CenterCrop(resolution)`, not a direct resize to the target resolution. omitting the center crop step measurably reduced accuracy on the int8 pipeline (59.00% to 63.50% after adding it back).



