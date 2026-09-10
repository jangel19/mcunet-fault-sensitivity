## Quantization notes

**Model Used:** mcunet-in3 (ProxylessNAS-based, TinyNAS search, MIT HAN Lab)


**design constraints:** 320KB SRAM / 1MB Flash
**measured footprint (from paper):** 293KB SRAM / 897KB Flash
**footprint from this experiment:** `.tflite` file size
**parameters:** 737,408
**Input Res:** 176x176

**Quantization scheme:** INT8 post-training static quantization (PTQ), per-tensor. Used the official pre-built `.tflite` model shipped by MIT HAN Lab via `download_tflite(net_id="mcunet-in3")`, rather than a custom quantization pipeline.

**Why not a custom pipeline:** i tried to build an independent INT8 quantization pipeline using PyTorch's eager mode quantization API (`torch.quantization.prepare` / `convert`, `qnnpack` backend). This failed consistently across different environments that i tried such as Apple Silicon (Python 3.14), a clean pyenv 3.11 environment, and Google Colab (Python 3.13 all producing the same error:
    `NotImplementedError: Could not run 'quantized::conv2d.new' with arguments from the 'CPU' backend`


**Root cause**: pytorch's eager mode static quantization for conv2d is not really that supported on anything older than Python 3.12. After figuring this out i pivoted to what i shouldve done in the beginning, use MIT HAN Lab's official pre-quantized TFLite artifact, which is the actual deployment target for this model family (TinyEngine on MCUs) and avoids reproducing a broken toolchain.


**preprocessing:** mcunet models are not trained with the standard ImageNet normalization (`mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`). They use `mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]`, scaling inputs to `[-1,1]`. during my first baseline using standard normalization the float32 was a surprising ~40-45%. But when i fixed the normalization i was able to bump up the accuracy to ~64%.Both float32 and INT8 evaluation pipelines also require `Resize(int(resolution*256/224)) → CenterCrop(resolution)`, not a direct resize to the target resolution. Omitting the center crop step measurably reduced accuracy on the INT8 pipeline (59.00% → 63.50% after adding it back).



