# MCUNet Fault Sensitivity

Research project studying layer-wise fault sensitivity of INT8 quantized MCUNet models on embedded hardware. Built on the MCUNet / TinyEngine platform from MIT HAN Lab.

**Baseline results (mcunet-in3, 320KB SRAM / 1MB Flash, 1000 ImageNet validation samples):**
- Float32 (PyTorch): 64.00% top-1 accuracy
- INT8 (TFLite, official MIT HAN Lab export): 63.50% top-1 accuracy
- Quantization-induced accuracy drop: 0.50 pp (paper reports 0.4 pp for this model)

See [`quantization_notes.md`](./quantization_notes.md) and [`baseline_report.md`](./baseline_report.md) for full details.

## Prerequisites

- Python 3.6+
- Git
- A HuggingFace account with access to the ILSVRC/imagenet-1k dataset

## HuggingFace Setup

1. Create an account at huggingface.co
2. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k and click Access repository to accept the license
3. Go to https://huggingface.co/settings/tokens and create a new token (read permissions)
4. In your terminal, log in:

```bash
pip install huggingface_hub
huggingface-cli login
# paste your token when prompted and follow instructions for 8 digit/letter code
```

## Setup

### 1. Clone the required repositories

```bash
git clone https://github.com/jangel19/mcunet-fault-sensitivity.git
git clone --recursive https://github.com/mit-han-lab/mcunet.git
git clone --recursive https://github.com/mit-han-lab/tinyengine.git
```

### 2. Set up virtual environment

```bash
cd mcunet-fault-sensitivity
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Float32 Baseline

```bash
python baseline.py
```

This will:
- Download the pretrained mcunet-in3 weights (~320KB SRAM / 1MB Flash) from MIT HAN Lab
- Stream 1000 samples from the ImageNet validation set
- Apply the correct MCUNet preprocessing (resize + center crop, `mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]` normalization)
- Report top-1 accuracy with progress updates every 100 samples

Expected output:

```
Progress: 0/1000 | Running accuracy: 100.00%
Progress: 100/1000 | Running accuracy: 65.35%
...
Model: MCUNet model that fits 320KB SRAM and 1MB Flash (ImageNet)
Final accuracy on 1000 samples: 64.00%
```

## Running the INT8 TFLite Baseline

```bash
python eval_int8_tflite.py
```

This will:
- Download the official pre-quantized INT8 `.tflite` model for mcunet-in3 from MIT HAN Lab
- Stream 1000 samples from the ImageNet validation set
- Apply matching preprocessing (resize + center crop, int8 pixel shift)
- Report top-1 accuracy with progress updates every 50 samples

Expected output:

```
progress: 0/1000 | running accuracy: 100.00%
...
final int8 accuracy on 1000 samples: 63.50%
```

## Research Plan

| Week | Task | Status |
|---|---|---|
| 1 | TinyML and MCUNet foundation | [x] Done |
| 2 | MCUNet environment | [x] Done |
| 3 | INT8 quantization and baseline | [x] Done |
| 4 | Model and layer instrumentation | [ ] Next up |
| 5 | Fault-model design | [ ] Upcoming |
| 6 | Fault-injection framework | [ ] Upcoming |
| 7 | Fault-injection validation | [ ] Upcoming |
| 8 | Pilot experiment | [ ] Upcoming |
| 9 | Large-scale weight fault campaign | [ ] Upcoming |
| 10 | Bit-position sensitivity | [ ] Upcoming |
| 11 | Activation fault experiments | [ ] Upcoming |
| 12 | Architectural pattern analysis | [ ] Upcoming |
| 13 | Robustness and reproducibility | [ ] Upcoming |
| 14 | Synthesis and implications | [ ] Upcoming |
| 15 | Final report and research artifact | [ ] Upcoming |

### Week 4: Model and Layer Instrumentation (current)

**Goal:** Map layer index to operation, tensor shape, quantization parameters, parameter count, and MACs. Identify potential fault injection points and define fault assumptions.

**Approach:**
- Revisit MCUNet paper sections on TinyEngine and memory scheduling
- Inspect source/model definitions in the mcunet and tinyengine repos to understand how layers are represented internally

**Deliverable:** Layer inventory table/CSV + architecture map

## Project Structure

```
mcunet-fault-sensitivity/
├── baseline.py              # float32 PyTorch inference baseline
├── eval_int8_tflite.py      # INT8 TFLite inference baseline
├── quantization_notes.md    # model architecture, footprint, quantization details
├── baseline_report.md       # baseline experiment results and analysis
├── requirements.txt         # dependencies
└── README.md
```

## References

- [MCUNet: Tiny Deep Learning on IoT Devices, NeurIPS 2020](https://arxiv.org/abs/2007.10319)
- [MIT HAN Lab MCUNet GitHub](https://github.com/mit-han-lab/mcunet)
- [TinyEngine GitHub](https://github.com/mit-han-lab/tinyengine)
- [PyTorchFI](https://github.com/DependableSystemsLab/pytorchfi)
