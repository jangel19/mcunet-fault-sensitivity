# Building libtensorflowlite_c (v2.21.0)

Python tensorflow in venv311 is 2.21.0, and the pip wheel does not export the
TFLite C api, so the C library is built from the matching source tag.

Installed result (what the Makefile expects, `TFLITE=`):

    ~/.local/tflite-2.21.0/lib/libtensorflowlite_c.dylib
    ~/.local/tflite-2.21.0/include/tensorflow/...   (only the c_api headers)

Build steps used (macOS arm64, Apple clang, cmake 4.4 + ninja from a throwaway pip venv):

    git clone --depth 1 --branch v2.21.0 https://github.com/tensorflow/tensorflow.git tf-src
    pip install cmake ninja            # in a scratch venv, not the project venv
    mkdir build && cd build
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DTENSORFLOW_SOURCE_DIR=$PWD/../tf-src \
          -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF -DTFLITE_ENABLE_RUY=OFF \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ../tf-src/tensorflow/lite/c
    cmake --build . -j 10
    install_name_tool -id @rpath/libtensorflowlite_c.dylib libtensorflowlite_c.dylib

Things that matter:

- `-DTENSORFLOW_SOURCE_DIR` is required. Without it cmake silently fetches an
  older TensorFlow checkout and the build fails on a flatbuffers version assert.
- XNNPACK and RUY are off, so this is plain builtin kernels. Python's default
  interpreter uses XNNPACK; see the parity numbers printed by `make test`.
- `TfLiteVersion()` prints "2.19.0" even though the source is v2.21.0: the
  version number is hard coded in tensorflow/lite/CMakeLists.txt upstream.
