import os
from pathlib import Path


DEFAULT_MODEL = Path.home() / ".torch/mcunet/mcunet-320kb-1mb_imagenet.tflite"


def add_model_argument(parser):
    # one path for teh python tools and make
    parser.add_argument("--model", default=os.environ.get("MCUNET_MODEL", str(DEFAULT_MODEL)),
                        help="tflite file (default: MCUNET_MODEL or ~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite)")
