# building the tflite c library on mac

this is an external prerequisite, not supplied by pip tensorflow or this repo.
the local campaign used apple silicon, apple clang and the installed binary
whose sha256 is `2eb9b9b2b108a4af52e91deafc05590b3e9672e7b3775c894d761d2b615e0a16`.
it reports `2.19.0`. its original source revision and flags are unverified.
the steps below describe a fresh v2.21.0 build to record, not proven provenance
for that old binary. rerun all tests when changing the library.

## prerequisites

install git, xcode command line tools (`xcode-select --install`), python 3.11,
cmake and ninja. the previous build notes used cmake 4.4 and ninja. keep these
build tools separate from the experiment environment:

```sh
python3.11 -m venv /tmp/tflite-build-tools
source /tmp/tflite-build-tools/bin/activate
python -m pip install cmake==4.4.0 ninja==1.13.2
cmake --version
ninja --version
cc --version
```

## build and install

run from a scratch directory with room for tensorflow and its downloaded
build dependencies. network access is needed. use absolute paths:

```sh
git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git tf-src
export TF_SOURCE="$(cd tf-src && pwd)"
export TF_BUILD="$PWD/tflite-build"
export TFLITE="$HOME/.local/tflite-2.21.0"
cmake -S "$TF_SOURCE/tensorflow/lite/c" -B "$TF_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORFLOW_SOURCE_DIR="$TF_SOURCE" \
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF -DTFLITE_ENABLE_RUY=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$TF_BUILD" --parallel 10
mkdir -p "$TFLITE/lib" "$TFLITE/include"
cp "$TF_BUILD/libtensorflowlite_c.dylib" "$TFLITE/lib/"
```

install the tensorflow headers with their relative directories intact. copying
all public `.h` files avoids missing transitive includes in the c api:

```sh
python3.11 - <<'PY'
import os
import shutil
from pathlib import Path
source = Path(os.environ['TF_SOURCE'])
include = Path(os.environ['TFLITE']) / 'include'
for header in (source / 'tensorflow').rglob('*.h'):
    destination = include / header.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(header, destination)
PY
install_name_tool -id @rpath/libtensorflowlite_c.dylib "$TFLITE/lib/libtensorflowlite_c.dylib"
codesign --force --sign - "$TFLITE/lib/libtensorflowlite_c.dylib"
otool -L "$TFLITE/lib/libtensorflowlite_c.dylib"
git -C "$TF_SOURCE" rev-parse HEAD > "$TFLITE/source-revision.txt"
cp "$TF_BUILD/CMakeCache.txt" "$TFLITE/CMakeCache.txt"
shasum -a 256 "$TFLITE/lib/libtensorflowlite_c.dylib" > "$TFLITE/library-sha256.txt"
```

`TENSORFLOW_SOURCE_DIR` matters: without it the c wrapper can fetch a different
source checkout. disabling xnnpack and ruy follows the previous local build
recipe. the runner itself selects the default c resolver with one thread;
it does not explicitly select reference kernels. tests check whether this
particular library observes in-place weight mutations.

on mac, the dylib install name should be `@rpath/libtensorflowlite_c.dylib`.
after changing the install name, ad-hoc signing refreshes the mac arm64
signature. this is local signing, not distribution notarization. the makefile
embeds an rpath to `$(TFLITE)/lib`. `otool -L c/run_trials` should
show that install name after linking. if it instead shows a build-directory
path, fix the installed dylib id above and rebuild the executables. inspect
other dependencies with `otool -L` if dyld still fails. linux would need a
`.so` build and separate loader instructions; it is not tested here.

from the project root, after creating `subset/` and `venv311/`:

```sh
make -C c all TFLITE="$TFLITE"
make -C c test TFLITE="$TFLITE" MODEL="$MCUNET_MODEL"
make -C c smoke TFLITE="$TFLITE" MODEL="$MCUNET_MODEL"
```

`TFLITE`, `MODEL`, `SUBSET`, `CSV` and `PYTHON` can be overridden. recipe paths
are quoted for spaces. paths passed through `make -C c` are relative to `c/`,
so use absolute paths for custom locations. `make test` uses 200 images and
writes reference files into the supplied subset. `make smoke` uses 20 images,
ops 1, 62 and 120, bits 0 and 7, with a 30-trial limit.

## verified fresh setup

these steps were exercised in separate temporary directories on 2026-10-21:
source tag `v2.21.0`, revision `a481b10260dfdf833a1b16007eead49c1d7febf3`,
cmake 4.4.0, ninja 1.13.2, release build with the flags above and two build
jobs. the installed headers and dylib built both c executables from source.
with an empty venv installed from the pinned requirements, `make test`
passed 99 checks, zero failures and zero skips. `make smoke` completed all
30 trials with a clean model buffer. repository, model, subset, python and
library paths included spaces.

see `results/setup_fresh_tflite_provenance.json` for the source and dependency
revisions, exact configuration and installed binary hash, plus
`results/setup_tflite_CMakeCache.txt` and the associated build/test logs.
this new library was used for setup validation only. it did not replace the
original library or produce the random campaign results.
