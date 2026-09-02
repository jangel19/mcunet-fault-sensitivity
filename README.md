# MCUNet Fault Sensitivity

research project studying layer-wise fault sensitivity of INT8 quantized MCUNet models on embedded hardware. Built on the MCUNet / TinyEngine platform from MIT HAN Lab.

**Baseline result:** 40.20% top-1 accuracy (mcunet-in3, 320KB SRAM / 1MB Flash, 1000 ImageNet validation samples)

---

## Prerequisites

- Python 3.6+
- Git
- A [HuggingFace account](https://huggingface.co) with access to the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset

### HuggingFace Setup
1. Create an account at huggingface.co
2. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k and click **Access repository** to accept the license
3. Go to https://huggingface.co/settings/tokens and create a new token (read permissions)
4. In your terminal, log in:
```bash
pip install huggingface_hub
huggingface-cli login
# paste your token when prompted and follow instructions for 8 digit/letter code
```

---

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

---

## Running the Baseline

```bash
python baseline.py
```

This will:
- Download the pretrained mcunet-in3 weights (~320KB SRAM / 1MB Flash) from MIT HAN Lab
- Stream 1000 samples from the ImageNet validation set
- Report top-1 accuracy with progress updates every 100 samples

Expected output:

Progress: 0/1000 | Running accuracy: 100.00%
Progress: 100/1000 | Running accuracy: 37.62%
...
Model: MCUNet model that fits 320KB SRAM and 1MB Flash (ImageNet)
Final accuracy on 1000 samples: 40.20%


---

## Project Structure

mcunet-fault-sensitivity/
├── baseline.py # clean inference baseline
├── requirements.txt # dependencies
└── README.md


---

## Research Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Baseline + layer inventory | In progress |
| 2 | Weight fault injection (PyTorchFI) | Upcoming |
| 3 | Layer-wise sensitivity heatmap | Upcoming |
| 4 | Activation fault injection | Stretch goal |

---

## References

- [MCUNet: Tiny Deep Learning on IoT Devices, NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html)
- [MIT HAN Lab MCUNet GitHub](https://github.com/mit-han-lab/mcunet)
- [TinyEngine GitHub](https://github.com/mit-han-lab/tinyengine)
- [PyTorchFI](https://github.com/pytorchfi/pytorchfi)
