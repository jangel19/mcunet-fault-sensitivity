/* fault_inject.h - single-bit weight fault injection for the int8 mcunet tflite model.
 *
 * The idea in four steps (see fault_model_week5.md):
 *   1. keep our own mutable copy of the .tflite file in memory
 *   2. find where every conv / depthwise weight tensor lives inside that copy
 *   3. flip one bit of one int8 weight byte, run the image subset, flip it back
 *   4. check the buffer is byte-identical to the clean file again
 *
 * The tflite interpreter reads constant weights straight out of the buffer we
 * hand it, so changing a byte in the buffer changes the model.
 */
#ifndef FAULT_INJECT_H
#define FAULT_INJECT_H

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/c/c_api.h"

#define FI_MAX_LAYERS   64      /* 55 injectable layers in this model */
#define FI_NUM_CLASSES  1000

/* tflite schema enum values (checked against schema.fbs in TF v2.21.0) */
#define FI_OP_CONV_2D            3
#define FI_OP_DEPTHWISE_CONV_2D  4
#define FI_TYPE_INT8             9

/* ---------- model buffer + weight locator ---------- */

typedef struct {
    int    op_index;      /* position in the subgraph's operator list (same as layer_inventory.csv) */
    int    op_type;       /* FI_OP_CONV_2D or FI_OP_DEPTHWISE_CONV_2D */
    int    tensor_index;  /* index of the weight tensor in the subgraph */
    int    shape[4];      /* weight tensor shape, e.g. [16,3,3,3] */
    size_t n_elems;       /* number of int8 weights */
    size_t offset;        /* byte offset of weight 0 inside the model buffer */
} fi_layer_t;

typedef struct {
    uint8_t *buf;         /* working copy of the model, this is what gets faulted */
    uint8_t *clean;       /* pristine copy, used to check restore */
    size_t   len;
    uint64_t clean_hash;  /* FNV-1a of the clean model */
    fi_layer_t layers[FI_MAX_LAYERS];
    int      n_layers;
} fi_model_t;

/* Reads the .tflite file and locates every conv weight tensor.
 * Returns 0 on success, -1 on error (message printed to stderr). */
int  fi_model_load(fi_model_t *m, const char *path);
void fi_model_free(fi_model_t *m);

/* Finds the layer with this op_index, or NULL. */
const fi_layer_t *fi_find_layer(const fi_model_t *m, int op_index);

/* ---------- applying and undoing a fault ---------- */

typedef struct {
    size_t  byte_pos;     /* absolute position in the model buffer */
    int     bit;          /* 0..7, bit 7 is the sign bit */
    int8_t  orig_val;     /* weight value before the flip */
    int8_t  new_val;      /* weight value after the flip */
    int     active;       /* 1 while the flip is applied */
} fi_fault_t;

/* int8 value you get by flipping `bit` of `v` (pure function, used by the tests too) */
int8_t fi_flipped_value(int8_t v, int bit);

/* Flip bit `bit` of weight `elem_idx` of `layer` in m->buf. Returns 0 or -1. */
int fi_apply(fi_model_t *m, const fi_layer_t *layer, size_t elem_idx, int bit, fi_fault_t *f);

/* Undo the flip (same xor again). Returns 0 or -1. */
int fi_restore(fi_model_t *m, fi_fault_t *f);

/* 1 if the working buffer is byte-identical to the clean model (hash and memcmp). */
int fi_is_clean(const fi_model_t *m);

/* How many bytes differ between the working buffer and the clean model. */
size_t fi_diff_bytes(const fi_model_t *m);

/* ---------- running the network ---------- */

typedef struct {
    TfLiteModel              *model;
    TfLiteInterpreterOptions *opts;
    TfLiteInterpreter        *interp;
    size_t in_bytes;               /* bytes per input image (176*176*3) */
} fi_runner_t;

/* Creates an interpreter on top of m->buf (single thread, reference kernels).
 * Must be called after fi_model_load; m must outlive the runner. */
int  fi_runner_init(fi_runner_t *r, const fi_model_t *m);
void fi_runner_free(fi_runner_t *r);

/* Runs one image (int8, in_bytes long). Writes FI_NUM_CLASSES int8 logits.
 * Returns 0 on success, -1 if invoke failed (a DUE). */
int fi_run_image(fi_runner_t *r, const int8_t *image, int8_t *logits);

/* index of the largest logit, first one wins on ties (same as numpy argmax) */
int fi_argmax(const int8_t *logits);

/* ---------- image subset ---------- */

typedef struct {
    int      n;
    size_t   img_bytes;
    int8_t  *images;      /* n * img_bytes */
    int32_t *labels;      /* n */
} fi_subset_t;

/* Loads subset/img_000.bin ... plus labels.bin written by make_subset.py. */
int  fi_subset_load(fi_subset_t *s, const char *dir, int max_images, size_t img_bytes);
void fi_subset_free(fi_subset_t *s);

/* ---------- scoring one trial ---------- */

typedef struct {
    int n_images;
    int n_masked;         /* faulted logits identical to clean logits */
    int n_subtle;         /* logits differ, top-1 same */
    int n_catastrophic;   /* top-1 changed */
    int n_due;            /* invoke failed */
    int n_correct_faulted;
    int n_correct_clean;
} fi_score_t;

/* clean_logits: n * FI_NUM_CLASSES int8 values from a run with no fault. */
int fi_clean_run(fi_runner_t *r, const fi_subset_t *s, int8_t *clean_logits);

/* Runs the subset with whatever is currently in the model buffer and scores it
 * against clean_logits. Always returns 0 unless setup is broken. */
int fi_score_run(fi_runner_t *r, const fi_subset_t *s, const int8_t *clean_logits, fi_score_t *out);

/* ---------- seeding ---------- */

/* splitmix64: one step of the generator. State is advanced, output returned. */
uint64_t fi_splitmix64_next(uint64_t *state);

/* order independent seed for a (layer, bit, k) cell, so the same cell always
 * gets the same seed no matter which other cells ran before it */
uint64_t fi_cell_seed(uint64_t base_seed, int op_index, int bit, int k);

#endif
