## Baseline Experiment Report

**Objective:** Establish float32 and INT8 top-1 accuracy baselines for mcunet-in3 on ImageNet, to serve as the reference point for subsequent fault injection experiments.


**Dataset:** ILSVRC/imagenet-1k, validation split, streamed via HuggingFace Datasets. 1000 samples evaluated for both float32 and INT8 runs (same preprocessing pipeline structure, different normalization per model requirements). Also ran a full 50,000-sample evaluation matching the paper's validation set size.


**Results:**

| Model | Format | n=1000 | n=50,000 | Paper reported (n=50,000) |
|---|---|---|---|---|
| mcunet-in3 | Float32 (PyTorch) | 64.00% | 62.14% | 62.2% |
| mcunet-in3 | INT8 (TFLite) | 63.50% | 61.82% | 61.8% |
| — | Accuracy drop from quantization | 0.50 pp | 0.32 pp | 0.4 pp |


**Interpretation:** the measured quantized induced accuracy dropped half a percent which closely matched the paper's reported drop of around .4%, this confirms that the INT8 model is well calibrated and the experiment pipeline is correctly producing the expected behavior. Both the float32 and int8 numbers run a bit higher than what the paper shows, which is expected given the smaller sample size of 1,000 being used by me compared to the full 50,000-image validation set being used in the paper.

ran the full 50,000-sample evaluation to confirm this, and the numbers came in almost exactly matching the paper, 62.14% vs 62.2% for float32 and 61.82% vs 61.8% for int8, both within a couple hundredths of a point. quantization drop at full scale was 0.32 pp, close to the paper's 0.4 pp. this confirms the 1000-sample gap was just sampling noise and the pipeline itself is solid.


**Verified consistency:** Both float32 and INT8 evaluations use the same underlying architecture and trained weights (`net_id="mcunet-in3"`), differing only in export format (PyTorch checkpoint vs. TFLite INT8 export), ensuring the accuracy comparison isolates the effect of quantization rather than confounding it with architectural differences.


**Known limitations**:
- Using only 1000 of 50000 validation samples, absolute accuracy valyes carry sampling variance and should not be over interpreted even though the delta for float32/int8 is consistent with the published results. (the full 50,000-sample run confirms this, results landed almost exactly on the paper's numbers once sample size matched.)
- HuggingFace streaming order might be playing a factor due streaming order possibly not being class-balanced, not being spread out evenly, across the 1000 sample subset


**Class coverage**: 
got curious and ran to see how many classes are being shown in the 1000 subset. 638 of 1000 ImageNet classes were represented in the 1000-sample evaluation subset (streamed via HuggingFace, non-shuffled order). This confirms the sample is not perfectly class-balanced, some classes appear multiple times while ~36% of classes are entirely absent, but the coverage is broad enough that results are unlikely to be dominated by a small number of classes. (at 50,000 samples, all 1000 classes are covered, so this isn't an issue for the full run.)
