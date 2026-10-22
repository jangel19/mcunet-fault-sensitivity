# Week 6 handoff bundle

Generated 2026-10-15 from branch main at `6fd39f0`. No code was changed; only this file was written. Code blocks use `~~~~` fences so embedded backticks survive.

## 1. Repo state

### git log --oneline (all commits)

~~~~
6fd39f0 fault_model_week5: record week 6 changes
906d0a6 results: week 6 first pass (corner mode, 14 layers x 5 rules x 8 bits)
f66796b c: trial loop with corner and random element modes
dead483 c: sanity tests for the fault injector
f828813 c: single-bit weight fault injector on the tflite C api
9f50a72 make_subset: build the fixed fault-injection image subset
cfc98da eval_int8_tflite: extract preprocessing and stream into functions
fb42ad3 python script to make subsets to test
586b38c week 5 fault model spec, week 4 architecture step 3
ef69a51 Update architecture_week4.md
f62f7d1 udpates...
c7dc106 Update dataset details and results in baseline report
2bf7786 Revise baseline results format in README.md
f86999b stride added
26fae34 parses mcunet-320kb-1mb_imagenet.tflite via the raw FlatBuffers schema (tflite package) rather than the TF interpreter, so the graph structure (op list, tensor shapes, quant params) is read directly rather than through an abstraction layer. does not take into account bias tensors or stride
287e81c week3: int8 quantization baseline using mit han lab tflite model and also fixing baseline run
d50cd54 more detailed week2notes
8b66d5e add reproducible setup instructions and baseline result
f625a65 Enhance README with project details and setup instructions
112b626 baseline: 40.20% top-1 accuracy on 1000 ImageNet validation samples
2f9e70f Initial commit
~~~~

### git status

~~~~
On branch main
Your branch is ahead of 'origin/main' by 7 commits.
  (use "git push" to publish your local commits)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	results/week6_handoff.md

nothing added to commit but untracked files present (use "git add" to track)
~~~~

### git diff --stat 586b38c HEAD (week 6 commits: 586b38c is the week 5 spec commit; the 8 commits after it, fb42ad3..6fd39f0)

~~~~
 .gitignore                     |  10 +
 c/BUILD_TFLITE.md              |  30 +++
 c/Makefile                     |  39 +++
 c/fault_inject.c               | 444 ++++++++++++++++++++++++++++++++
 c/fault_inject.h               | 143 +++++++++++
 c/ref_logits.py                |  62 +++++
 c/run_trials.c                 | 258 +++++++++++++++++++
 c/test_fi.c                    | 513 +++++++++++++++++++++++++++++++++++++
 eval_int8_tflite.py            | 100 +++++---
 fault_model_week5.md           |  33 ++-
 make_subset.py                 | 100 ++++++++
 results/eval50k_refactored.log | 514 +++++++++++++++++++++++++++++++++++++
 results/first_pass_corner.csv  | 561 +++++++++++++++++++++++++++++++++++++++++
 results/first_pass_corner.log  |  30 +++
 14 files changed, 2787 insertions(+), 50 deletions(-)
~~~~

### file tree (excluding venv311, subset/, __pycache__, .git)

~~~~
./.gitignore
./.python-version
./architecture_week4.md
./baseline_experiment_week3.md
./baseline.py
./c/BUILD_TFLITE.md
./c/fault_inject.c
./c/fault_inject.h
./c/Makefile
./c/ref_logits.py
./c/run_trials
./c/run_trials.c
./c/run_trials.dSYM/Contents/Info.plist
./c/run_trials.dSYM/Contents/Resources/DWARF/run_trials
./c/run_trials.dSYM/Contents/Resources/Relocations/aarch64/run_trials.yml
./c/test_fi
./c/test_fi.c
./c/test_fi.dSYM/Contents/Info.plist
./c/test_fi.dSYM/Contents/Resources/DWARF/test_fi
./c/test_fi.dSYM/Contents/Resources/Relocations/aarch64/test_fi.yml
./c/trials_smoke.csv
./counting.py
./eval_int8_tflite.py
./export.py
./fault_model_week5.md
./get_tendor_data.py
./int8_eval.py
./layer_inventory.csv
./layer_inventory.py
./make_subset.py
./mcunet_int8.pth
./mcunet-in3.onnx
./mcunet-in3.onnx.data
./modelsize.py
./quantization_notes.md
./quantize.py
./README.md
./requirements.txt
./results/eval50k_refactored.log
./results/first_pass_corner.csv
./results/first_pass_corner.log
./results/week6_handoff.md
./week2_notes.md
./week3_notes.md
./week4_notes.md
~~~~

### versions

~~~~
os: macOS 26.6.2 (arm64), Darwin 25.6.0
python (venv311): Python 3.11.9
tensorflow (venv311): 2.21.0
tflite C library: built from TF source tag v2.21.0 (c/BUILD_TFLITE.md), installed ~/.local/tflite-2.21.0, XNNPACK/RUY/GPU off
  TfLiteVersion() at runtime prints 2.19.0 (see section 8, gap 1)
compiler: Apple clang version 17.0.0 (clang-1700.6.4.2)
~~~~

## 2. Full source (verbatim)

### c/fault_inject.h

~~~~
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
~~~~

### c/fault_inject.c

~~~~
/* fault_inject.c - see fault_inject.h for the overview */
#include "fault_inject.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =====================================================================
 * 1. Minimal flatbuffer reader
 *
 * A .tflite file is a flatbuffer. We only need to walk
 *   Model -> subgraphs[0] -> operators / tensors, Model -> buffers
 * so instead of linking a flatbuffers library we read the few fields we need.
 *
 * Flatbuffer rules we rely on (all little endian):
 *   - a table starts with an int32 "soffset"; vtable = table - soffset
 *   - the vtable is: u16 vtable_size, u16 table_size, then one u16 per field
 *     giving the field's offset inside the table (0 = field absent)
 *   - offsets to tables / vectors / strings are u32 relative to where the
 *     offset itself is stored
 *   - a vector is: u32 length, then the elements
 * Every read is bounds checked; a bad offset sets `err` instead of crashing.
 * ===================================================================== */

typedef struct {
    const uint8_t *b;
    size_t         n;
    int            err;
} fb_t;

static uint32_t rd_u32(fb_t *f, size_t pos)
{
    uint32_t v = 0;
    if (pos + 4 > f->n) { f->err = 1; return 0; }
    memcpy(&v, f->b + pos, 4);
    return v;
}

static int32_t rd_i32(fb_t *f, size_t pos) { return (int32_t)rd_u32(f, pos); }

static uint16_t rd_u16(fb_t *f, size_t pos)
{
    uint16_t v = 0;
    if (pos + 2 > f->n) { f->err = 1; return 0; }
    memcpy(&v, f->b + pos, 2);
    return v;
}

static int8_t rd_i8(fb_t *f, size_t pos)
{
    if (pos + 1 > f->n) { f->err = 1; return 0; }
    return (int8_t)f->b[pos];
}

/* position of field `id` of the table at `table`, or 0 if the field is absent.
 * Field ids are the order fields are declared in schema.fbs (0, 1, 2, ...). */
static size_t fb_field(fb_t *f, size_t table, int id)
{
    int32_t soff    = rd_i32(f, table);
    size_t  vtable  = (size_t)((int64_t)table - soff);
    size_t  vt_size = rd_u16(f, vtable);
    size_t  slot    = 4 + 2 * (size_t)id;
    if (f->err || slot + 2 > vt_size) return 0;
    uint16_t off = rd_u16(f, vtable + slot);
    return off ? table + off : 0;
}

/* follow a u32 relative offset stored at `pos` */
static size_t fb_follow(fb_t *f, size_t pos) { return pos + rd_u32(f, pos); }

/* vector stored in field `id`: returns position of element 0 and the length */
static size_t fb_vector(fb_t *f, size_t table, int id, uint32_t *len)
{
    size_t p = fb_field(f, table, id);
    *len = 0;
    if (!p) return 0;
    size_t v = fb_follow(f, p);
    *len = rd_u32(f, v);
    return v + 4;
}

/* i-th element of a vector of tables */
static size_t fb_table_at(fb_t *f, size_t vec0, uint32_t i)
{
    return fb_follow(f, vec0 + 4 * (size_t)i);
}

/* schema.fbs field ids (verified against tensorflow v2.21.0 schema.fbs) */
enum { MODEL_OPERATOR_CODES = 1, MODEL_SUBGRAPHS = 2, MODEL_BUFFERS = 4 };
enum { SUBGRAPH_TENSORS = 0, SUBGRAPH_OPERATORS = 3 };
enum { OPERATOR_OPCODE_INDEX = 0, OPERATOR_INPUTS = 1 };
enum { OPCODE_DEPRECATED_BUILTIN = 0, OPCODE_BUILTIN = 3 };
enum { TENSOR_SHAPE = 0, TENSOR_TYPE = 1, TENSOR_BUFFER = 2 };
enum { BUFFER_DATA = 0 };

/* =====================================================================
 * 2. Locating the conv weights
 * ===================================================================== */

static uint64_t fnv1a(const uint8_t *p, size_t n)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static int locate_layers(fi_model_t *m)
{
    fb_t f = { m->buf, m->len, 0 };
    uint32_t n_codes, n_subgraphs, n_buffers, n_tensors, n_ops;

    size_t root      = fb_follow(&f, 0);
    size_t codes0    = fb_vector(&f, root, MODEL_OPERATOR_CODES, &n_codes);
    size_t sgs0      = fb_vector(&f, root, MODEL_SUBGRAPHS, &n_subgraphs);
    size_t buffers0  = fb_vector(&f, root, MODEL_BUFFERS, &n_buffers);
    if (f.err || n_subgraphs < 1) { fprintf(stderr, "bad model header\n"); return -1; }

    size_t sg       = fb_table_at(&f, sgs0, 0);
    size_t tensors0 = fb_vector(&f, sg, SUBGRAPH_TENSORS, &n_tensors);
    size_t ops0     = fb_vector(&f, sg, SUBGRAPH_OPERATORS, &n_ops);
    if (f.err) { fprintf(stderr, "bad subgraph\n"); return -1; }

    m->n_layers = 0;
    for (uint32_t i = 0; i < n_ops; i++) {
        size_t op = fb_table_at(&f, ops0, i);

        /* which kind of op is this: look it up in the opcode table */
        size_t   p        = fb_field(&f, op, OPERATOR_OPCODE_INDEX);
        uint32_t code_idx = p ? rd_u32(&f, p) : 0;
        if (code_idx >= n_codes) { fprintf(stderr, "op %u: bad opcode index\n", i); return -1; }
        size_t  code_tbl = fb_table_at(&f, codes0, code_idx);
        size_t  pd = fb_field(&f, code_tbl, OPCODE_DEPRECATED_BUILTIN);
        size_t  pb = fb_field(&f, code_tbl, OPCODE_BUILTIN);
        int     code = pd ? rd_i8(&f, pd) : 0;
        if (pb && rd_i32(&f, pb) > code) code = rd_i32(&f, pb);   /* tflite uses the max of the two */
        if (code != FI_OP_CONV_2D && code != FI_OP_DEPTHWISE_CONV_2D) continue;

        /* conv inputs are [activation, weights, bias] */
        uint32_t n_in;
        size_t   in0 = fb_vector(&f, op, OPERATOR_INPUTS, &n_in);
        if (n_in < 2) { fprintf(stderr, "op %u: conv with %u inputs\n", i, n_in); return -1; }
        int32_t t_idx = rd_i32(&f, in0 + 4);
        if (t_idx < 0 || (uint32_t)t_idx >= n_tensors) { fprintf(stderr, "op %u: bad tensor\n", i); return -1; }

        size_t   t   = fb_table_at(&f, tensors0, (uint32_t)t_idx);
        size_t   pt  = fb_field(&f, t, TENSOR_TYPE);
        int      typ = pt ? rd_i8(&f, pt) : 0;
        if (typ != FI_TYPE_INT8) { fprintf(stderr, "op %u: weight type %d is not int8\n", i, typ); return -1; }

        uint32_t rank;
        size_t   sh0 = fb_vector(&f, t, TENSOR_SHAPE, &rank);
        if (rank != 4) { fprintf(stderr, "op %u: weight rank %u\n", i, rank); return -1; }

        fi_layer_t *L = &m->layers[m->n_layers];
        size_t n = 1;
        for (int d = 0; d < 4; d++) { L->shape[d] = rd_i32(&f, sh0 + 4 * (size_t)d); n *= (size_t)L->shape[d]; }

        /* weights live in buffers[tensor.buffer].data */
        size_t   pbuf = fb_field(&f, t, TENSOR_BUFFER);
        uint32_t bidx = pbuf ? rd_u32(&f, pbuf) : 0;
        if (bidx >= n_buffers) { fprintf(stderr, "op %u: bad buffer index\n", i); return -1; }
        uint32_t data_len;
        size_t   data0 = fb_vector(&f, fb_table_at(&f, buffers0, bidx), BUFFER_DATA, &data_len);
        if (f.err || data0 == 0 || data_len != n || data0 + n > m->len) {
            fprintf(stderr, "op %u: weight buffer has %u bytes, shape needs %zu\n", i, data_len, n);
            return -1;
        }

        if (m->n_layers >= FI_MAX_LAYERS) { fprintf(stderr, "too many conv layers\n"); return -1; }
        L->op_index     = (int)i;
        L->op_type      = code;
        L->tensor_index = t_idx;
        L->n_elems      = n;
        L->offset       = data0;
        m->n_layers++;
    }
    return f.err ? -1 : 0;
}

int fi_model_load(fi_model_t *m, const char *path)
{
    memset(m, 0, sizeof *m);
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (len <= 0) { fclose(fp); return -1; }

    m->len   = (size_t)len;
    m->buf   = malloc(m->len);
    m->clean = malloc(m->len);
    if (!m->buf || !m->clean || fread(m->buf, 1, m->len, fp) != m->len) {
        fprintf(stderr, "could not read %s\n", path);
        fclose(fp);
        fi_model_free(m);
        return -1;
    }
    fclose(fp);
    memcpy(m->clean, m->buf, m->len);
    m->clean_hash = fnv1a(m->clean, m->len);

    if (locate_layers(m) != 0) { fi_model_free(m); return -1; }
    return 0;
}

void fi_model_free(fi_model_t *m)
{
    free(m->buf);
    free(m->clean);
    memset(m, 0, sizeof *m);
}

const fi_layer_t *fi_find_layer(const fi_model_t *m, int op_index)
{
    for (int i = 0; i < m->n_layers; i++)
        if (m->layers[i].op_index == op_index) return &m->layers[i];
    return NULL;
}

/* =====================================================================
 * 3. Apply / restore a single bit flip
 * ===================================================================== */

int8_t fi_flipped_value(int8_t v, int bit)
{
    /* xor on the stored byte. bit 7 is the sign bit of two's complement */
    return (int8_t)((uint8_t)v ^ (uint8_t)(1u << bit));
}

int fi_apply(fi_model_t *m, const fi_layer_t *layer, size_t elem_idx, int bit, fi_fault_t *f)
{
    if (f->active) { fprintf(stderr, "fi_apply: a fault is already active\n"); return -1; }
    if (elem_idx >= layer->n_elems || bit < 0 || bit > 7) return -1;

    f->byte_pos = layer->offset + elem_idx;
    f->bit      = bit;
    f->orig_val = (int8_t)m->buf[f->byte_pos];
    m->buf[f->byte_pos] ^= (uint8_t)(1u << bit);
    f->new_val  = (int8_t)m->buf[f->byte_pos];
    f->active   = 1;
    return 0;
}

int fi_restore(fi_model_t *m, fi_fault_t *f)
{
    if (!f->active) return -1;
    m->buf[f->byte_pos] ^= (uint8_t)(1u << f->bit);
    f->active = 0;
    return 0;
}

int fi_is_clean(const fi_model_t *m)
{
    return fnv1a(m->buf, m->len) == m->clean_hash && memcmp(m->buf, m->clean, m->len) == 0;
}

size_t fi_diff_bytes(const fi_model_t *m)
{
    size_t d = 0;
    for (size_t i = 0; i < m->len; i++) d += (m->buf[i] != m->clean[i]);
    return d;
}

/* =====================================================================
 * 4. Running the network through the tflite C api
 * ===================================================================== */

int fi_runner_init(fi_runner_t *r, const fi_model_t *m)
{
    memset(r, 0, sizeof *r);
    r->model = TfLiteModelCreate(m->buf, m->len);
    if (!r->model) { fprintf(stderr, "TfLiteModelCreate failed\n"); return -1; }

    r->opts = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(r->opts, 1);   /* no threads: deterministic and simple */

    r->interp = TfLiteInterpreterCreate(r->model, r->opts);
    if (!r->interp || TfLiteInterpreterAllocateTensors(r->interp) != kTfLiteOk) {
        fprintf(stderr, "interpreter setup failed\n");
        fi_runner_free(r);
        return -1;
    }

    const TfLiteTensor *in  = TfLiteInterpreterGetInputTensor(r->interp, 0);
    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(r->interp, 0);
    r->in_bytes = TfLiteTensorByteSize(in);
    if (TfLiteTensorType(in) != kTfLiteInt8 || TfLiteTensorType(out) != kTfLiteInt8 ||
        TfLiteTensorByteSize(out) != FI_NUM_CLASSES) {
        fprintf(stderr, "unexpected input/output tensor type or size\n");
        fi_runner_free(r);
        return -1;
    }
    return 0;
}

void fi_runner_free(fi_runner_t *r)
{
    if (r->interp) TfLiteInterpreterDelete(r->interp);
    if (r->opts)   TfLiteInterpreterOptionsDelete(r->opts);
    if (r->model)  TfLiteModelDelete(r->model);
    memset(r, 0, sizeof *r);
}

int fi_run_image(fi_runner_t *r, const int8_t *image, int8_t *logits)
{
    TfLiteTensor *in = TfLiteInterpreterGetInputTensor(r->interp, 0);
    if (TfLiteTensorCopyFromBuffer(in, image, r->in_bytes) != kTfLiteOk) return -1;
    if (TfLiteInterpreterInvoke(r->interp) != kTfLiteOk) return -1;
    const TfLiteTensor *out = TfLiteInterpreterGetOutputTensor(r->interp, 0);
    if (TfLiteTensorCopyToBuffer(out, logits, FI_NUM_CLASSES) != kTfLiteOk) return -1;
    return 0;
}

int fi_argmax(const int8_t *logits)
{
    int best = 0;
    for (int i = 1; i < FI_NUM_CLASSES; i++)
        if (logits[i] > logits[best]) best = i;   /* strict > keeps the first max */
    return best;
}

/* =====================================================================
 * 5. Image subset
 * ===================================================================== */

int fi_subset_load(fi_subset_t *s, const char *dir, int max_images, size_t img_bytes)
{
    memset(s, 0, sizeof *s);
    char path[1024];

    snprintf(path, sizeof path, "%s/labels.bin", dir);
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int n = (int)(bytes / 4);
    if (max_images > 0 && max_images < n) n = max_images;
    if (n <= 0) { fclose(fp); return -1; }

    s->n         = n;
    s->img_bytes = img_bytes;
    s->labels    = malloc((size_t)n * sizeof *s->labels);
    s->images    = malloc((size_t)n * img_bytes);
    if (!s->labels || !s->images || fread(s->labels, 4, (size_t)n, fp) != (size_t)n) {
        fclose(fp);
        fi_subset_free(s);
        return -1;
    }
    fclose(fp);

    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/img_%03d.bin", dir, i);
        fp = fopen(path, "rb");
        if (!fp || fread(s->images + (size_t)i * img_bytes, 1, img_bytes, fp) != img_bytes) {
            fprintf(stderr, "could not read %s (expected %zu bytes)\n", path, img_bytes);
            if (fp) fclose(fp);
            fi_subset_free(s);
            return -1;
        }
        fclose(fp);
    }
    return 0;
}

void fi_subset_free(fi_subset_t *s)
{
    free(s->images);
    free(s->labels);
    memset(s, 0, sizeof *s);
}

/* =====================================================================
 * 6. Scoring
 *
 * Per image, compared with the clean run on the same image:
 *   due          invoke failed, no output
 *   masked       int8 logits identical to clean
 *   subtle sdc   logits differ, top-1 same
 *   catastrophic top-1 changed
 * ===================================================================== */

int fi_clean_run(fi_runner_t *r, const fi_subset_t *s, int8_t *clean_logits)
{
    for (int i = 0; i < s->n; i++) {
        if (fi_run_image(r, s->images + (size_t)i * s->img_bytes,
                         clean_logits + (size_t)i * FI_NUM_CLASSES) != 0) {
            fprintf(stderr, "clean run failed on image %d\n", i);
            return -1;
        }
    }
    return 0;
}

int fi_score_run(fi_runner_t *r, const fi_subset_t *s, const int8_t *clean_logits, fi_score_t *out)
{
    int8_t logits[FI_NUM_CLASSES];
    memset(out, 0, sizeof *out);
    out->n_images = s->n;

    for (int i = 0; i < s->n; i++) {
        const int8_t *clean = clean_logits + (size_t)i * FI_NUM_CLASSES;
        int clean_top = fi_argmax(clean);
        if (clean_top == s->labels[i]) out->n_correct_clean++;

        if (fi_run_image(r, s->images + (size_t)i * s->img_bytes, logits) != 0) {
            out->n_due++;                 /* no output: counts as wrong for accuracy */
            continue;
        }
        int top = fi_argmax(logits);
        if (top == s->labels[i]) out->n_correct_faulted++;

        if (memcmp(logits, clean, FI_NUM_CLASSES) == 0) out->n_masked++;
        else if (top == clean_top)                       out->n_subtle++;
        else                                             out->n_catastrophic++;
    }
    return 0;
}

/* =====================================================================
 * 7. Seeding (splitmix64, Vigna's reference constants)
 * ===================================================================== */

uint64_t fi_splitmix64_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static uint64_t mix64(uint64_t x)
{
    return fi_splitmix64_next(&x);
}

uint64_t fi_cell_seed(uint64_t base_seed, int op_index, int bit, int k)
{
    uint64_t h = mix64(base_seed + (uint64_t)op_index);
    h = mix64(h + (uint64_t)bit);
    h = mix64(h + (uint64_t)k);
    return h;
}
~~~~

### c/test_fi.c

~~~~
/* test_fi.c - sanity tests for the fault injector.
 *
 *   ./test_fi --model <.tflite> --subset <dir> --csv <layer_inventory.csv>
 *
 * Each test prints PASS/FAIL lines. Exit code is non-zero if anything failed.
 * A test that could not run (e.g. python reference files missing) prints
 * SKIP and is counted separately, it is never reported as a pass.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fault_inject.h"

static int n_pass, n_fail, n_skip;

#define CHECK(cond, ...) do {                                   \
        if (cond) { n_pass++; }                                 \
        else { n_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
               printf(__VA_ARGS__); printf("\n"); }             \
    } while (0)

static void header(const char *name) { printf("\n[%s]\n", name); }

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---------------------------------------------------------------- */
/* 1. locator finds exactly the 55 convs listed in layer_inventory.csv */

/* split one csv line into fields, honouring "quoted, fields". Returns count. */
static int csv_split(char *line, char **f, int max)
{
    int n = 0;
    char *p = line;
    while (*p && n < max) {
        if (*p == '"') {
            f[n++] = ++p;
            while (*p && *p != '"') p++;
            if (*p) *p++ = 0;
            if (*p == ',') p++;
        } else {
            f[n++] = p;
            while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
            if (*p == ',') *p++ = 0;
            else { *p = 0; break; }
        }
    }
    return n;
}

static int parse_shape(const char *s, int shape[4])
{
    return sscanf(s, "[%d, %d, %d, %d]", &shape[0], &shape[1], &shape[2], &shape[3]) == 4;
}

static void test_locator(const fi_model_t *m, const char *csv_path)
{
    header("locator vs layer_inventory.csv");
    FILE *fp = fopen(csv_path, "r");
    if (!fp) { CHECK(0, "cannot open %s", csv_path); return; }

    char line[4096], *f[32];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); CHECK(0, "empty csv"); return; }
    int nf = csv_split(line, f, 32);
    int c_op = -1, c_type = -1, c_wshape = -1, c_params = -1, c_inj = -1;
    for (int i = 0; i < nf; i++) {
        if (!strcmp(f[i], "op_index"))             c_op = i;
        if (!strcmp(f[i], "op_type"))              c_type = i;
        if (!strcmp(f[i], "weight_shape"))         c_wshape = i;
        if (!strcmp(f[i], "param_count"))          c_params = i;
        if (!strcmp(f[i], "injection_candidate"))  c_inj = i;
    }
    CHECK(c_op >= 0 && c_type >= 0 && c_wshape >= 0 && c_params >= 0 && c_inj >= 0, "csv header columns");

    int expected = 0, matched = 0;
    while (fgets(line, sizeof line, fp)) {
        nf = csv_split(line, f, 32);
        if (nf <= c_inj || strcmp(f[c_inj], "True") != 0) continue;
        expected++;

        int op = atoi(f[c_op]);
        const fi_layer_t *L = fi_find_layer(m, op);
        int ok = L != NULL;
        if (ok) {
            int shape[4] = {0};
            int want_type = !strcmp(f[c_type], "CONV_2D") ? FI_OP_CONV_2D : FI_OP_DEPTHWISE_CONV_2D;
            ok = parse_shape(f[c_wshape], shape) && memcmp(shape, L->shape, sizeof shape) == 0
              && L->op_type == want_type && (long)L->n_elems == atol(f[c_params]);
        }
        CHECK(ok, "csv op %d does not match the locator", op);
        matched += ok;
    }
    fclose(fp);

    CHECK(expected == 55, "csv lists %d injectable ops, expected 55", expected);
    CHECK(m->n_layers == 55, "locator found %d conv layers, expected 55", m->n_layers);
    CHECK(matched == expected, "%d of %d csv rows matched", matched, expected);
    printf("  locator: %d layers, csv: %d injectable, %d matched (op index, type, shape, count)\n",
           m->n_layers, expected, matched);
}

/* ---------------------------------------------------------------- */
/* 2. the value change is exactly the intended int8 change.
 *    Expected values are computed with plain arithmetic, not with xor,
 *    so this is independent of the implementation. */

static int expected_after_flip(int v, int bit)
{
    int u = v & 0xFF;                                  /* stored byte 0..255 */
    u = (u & (1 << bit)) ? u - (1 << bit) : u + (1 << bit);
    return u > 127 ? u - 256 : u;                      /* back to two's complement */
}

static void test_value_change(fi_model_t *m)
{
    header("intended int8 value change: all 256 values x 8 bits");

    /* (a) the pure function */
    int bad = 0, bad_sign = 0, bad_mag = 0;
    for (int v = -128; v <= 127; v++) {
        for (int bit = 0; bit < 8; bit++) {
            int got  = fi_flipped_value((int8_t)v, bit);
            int want = expected_after_flip(v, bit);
            bad += (got != want);

            if (bit == 7) {
                /* sign bit: always crosses zero, and moves the value by exactly 128 */
                int want_sign = (v >= 0) ? v - 128 : v + 128;
                bad_sign += (got != want_sign) || ((v < 0) == (got < 0)) || (abs(got - v) != 128);
            } else {
                /* magnitude bits: move by exactly 2^bit, sign of the value never changes */
                bad_mag += (abs(got - v) != (1 << bit)) || ((v < 0) != (got < 0));
            }
        }
    }
    CHECK(bad == 0, "%d of 2048 (value,bit) pairs differ from the arithmetic expectation", bad);
    CHECK(bad_sign == 0, "sign bit (bit 7): %d of 256 values wrong (must be -128 for v>=0, +128 for v<0)", bad_sign);
    CHECK(bad_mag == 0, "bits 0-6: %d of 1792 pairs wrong (must change value by 2^bit, keep sign)", bad_mag);

    /* explicit sign bit examples so they are easy to read in the output */
    CHECK(fi_flipped_value(0, 7) == -128,   "0 with bit7 -> %d, want -128", fi_flipped_value(0, 7));
    CHECK(fi_flipped_value(127, 7) == -1,   "127 with bit7 -> %d, want -1", fi_flipped_value(127, 7));
    CHECK(fi_flipped_value(-1, 7) == 127,   "-1 with bit7 -> %d, want 127", fi_flipped_value(-1, 7));
    CHECK(fi_flipped_value(-128, 7) == 0,   "-128 with bit7 -> %d, want 0", fi_flipped_value(-128, 7));
    CHECK(fi_flipped_value(5, 7) == -123,   "5 with bit7 -> %d, want -123", fi_flipped_value(5, 7));
    CHECK(fi_flipped_value(-123, 7) == 5,   "-123 with bit7 -> %d, want 5", fi_flipped_value(-123, 7));

    /* (b) the same thing through the real buffer path: put every value into a
     * real weight byte, apply, and read back what the model buffer holds */
    const fi_layer_t *L = fi_find_layer(m, 120);
    size_t pos = L->offset + 7;
    uint8_t keep = m->buf[pos];
    int bad_buf = 0;
    for (int v = -128; v <= 127; v++) {
        for (int bit = 0; bit < 8; bit++) {
            m->buf[pos] = (uint8_t)v;
            fi_fault_t f = {0};
            if (fi_apply(m, L, 7, bit, &f) != 0) { bad_buf++; continue; }
            int want = expected_after_flip(v, bit);
            bad_buf += ((int8_t)m->buf[pos] != want) || f.orig_val != v || f.new_val != want;
            if (fi_restore(m, &f) != 0 || (int8_t)m->buf[pos] != v) bad_buf++;
        }
    }
    m->buf[pos] = keep;
    CHECK(bad_buf == 0, "%d of 2048 buffer-level flips differ from expectation", bad_buf);
    CHECK(fi_is_clean(m), "buffer not clean after the 2048-flip sweep");
    printf("  2048 (value, bit) pairs checked arithmetically and through the buffer\n");
}

/* ---------------------------------------------------------------- */
/* 3. exactly one byte differs while a fault is active; restore -> clean */

static void test_one_byte_and_restore(fi_model_t *m)
{
    header("exactly one byte differs while faulted; restore returns clean");
    int n_cases = 0, bad_one = 0, bad_where = 0, bad_xor = 0, bad_restore = 0;

    for (int li = 0; li < m->n_layers; li++) {
        const fi_layer_t *L = &m->layers[li];
        size_t elems[3] = { 0, L->n_elems / 2, L->n_elems - 1 };
        for (int e = 0; e < 3; e++) {
            for (int bit = 0; bit < 8; bit++) {
                fi_fault_t f = {0};
                if (fi_apply(m, L, elems[e], bit, &f) != 0) { bad_one++; continue; }
                n_cases++;

                if (fi_diff_bytes(m) != 1) bad_one++;
                size_t pos = L->offset + elems[e];
                if (m->buf[pos] == m->clean[pos]) bad_where++;
                if ((m->buf[pos] ^ m->clean[pos]) != (1u << bit)) bad_xor++;
                if (fi_is_clean(m)) bad_one++;      /* must NOT look clean while faulted */

                fi_restore(m, &f);
                if (fi_diff_bytes(m) != 0 || !fi_is_clean(m)) bad_restore++;
            }
        }
    }
    CHECK(bad_one == 0,     "%d cases did not have exactly 1 differing byte", bad_one);
    CHECK(bad_where == 0,   "%d cases changed a different byte than intended", bad_where);
    CHECK(bad_xor == 0,     "%d cases changed more or other than the one intended bit", bad_xor);
    CHECK(bad_restore == 0, "%d cases not clean after restore", bad_restore);
    printf("  %d cases (55 layers x first/middle/last element x 8 bits)\n", n_cases);

    /* misuse is rejected, not silently accepted */
    fi_fault_t f = {0};
    const fi_layer_t *L = fi_find_layer(m, 1);
    CHECK(fi_apply(m, L, L->n_elems, 0, &f) != 0, "out of range element accepted");
    CHECK(fi_apply(m, L, 0, 8, &f) != 0,          "bit 8 accepted");
    CHECK(fi_is_clean(m), "rejected apply still changed the buffer");
}

/* ---------------------------------------------------------------- */
/* 4. runtime behaviour: clean determinism, zero faults, fault visible,
 *    restore gives clean logits, same seed gives same result            */

static int8_t *alloc_logits(int n) { return malloc((size_t)n * FI_NUM_CLASSES); }

static size_t pick_max_abs(const fi_model_t *m, const fi_layer_t *L)
{
    size_t best = 0; int bv = -1;
    for (size_t i = 0; i < L->n_elems; i++) {
        int v = abs((int8_t)m->buf[L->offset + i]);
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}

static void test_runtime(fi_model_t *m, fi_runner_t *r, const fi_subset_t *s,
                         const int8_t *clean_a)
{
    header("zero faults / determinism / fault visible / restore / seed");
    int n = s->n;
    int8_t *again = alloc_logits(n);
    int8_t *faulted = alloc_logits(n);

    /* zero faults: a second clean run must give identical logits */
    CHECK(fi_clean_run(r, s, again) == 0, "second clean run failed");
    CHECK(memcmp(clean_a, again, (size_t)n * FI_NUM_CLASSES) == 0, "two clean runs differ");

    /* zero faults through the scoring path: everything masked */
    fi_score_t sc;
    fi_score_run(r, s, clean_a, &sc);
    CHECK(sc.n_masked == n && sc.n_subtle == 0 && sc.n_catastrophic == 0 && sc.n_due == 0,
          "zero-fault score: masked=%d subtle=%d cat=%d due=%d (want %d,0,0,0)",
          sc.n_masked, sc.n_subtle, sc.n_catastrophic, sc.n_due, n);
    CHECK(sc.n_correct_faulted == sc.n_correct_clean, "zero-fault accuracy differs from clean");
    printf("  zero faults: %d/%d masked, accuracy %d/%d\n", sc.n_masked, n, sc.n_correct_clean, n);

    /* a sign bit flip on the biggest classifier weight must be visible: this
     * is also the proof that the interpreter reads weights from our buffer */
    const fi_layer_t *L = fi_find_layer(m, 120);
    size_t e = pick_max_abs(m, L);
    fi_fault_t f = {0};
    CHECK(fi_apply(m, L, e, 7, &f) == 0, "apply failed");
    fi_score_run(r, s, clean_a, &sc);
    CHECK(sc.n_masked < n, "fault in op 120 elem %zu bit 7 changed nothing: weights not read from buffer?", e);
    printf("  op 120 elem %zu bit 7 (%d -> %d): masked=%d subtle=%d catastrophic=%d due=%d of %d\n",
           e, f.orig_val, f.new_val, sc.n_masked, sc.n_subtle, sc.n_catastrophic, sc.n_due, n);
    CHECK(sc.n_masked + sc.n_subtle + sc.n_catastrophic + sc.n_due == n, "outcome counts do not add up");

    /* restore: model buffer clean AND logits back to the clean logits */
    CHECK(fi_restore(m, &f) == 0, "restore failed");
    CHECK(fi_is_clean(m), "buffer not clean after restore");
    fi_clean_run(r, s, again);
    CHECK(memcmp(clean_a, again, (size_t)n * FI_NUM_CLASSES) == 0, "logits after restore differ from clean");

    /* same seed -> same element -> same result, different seed -> (very likely) different element */
    uint64_t seed = fi_cell_seed(2026, 62, 6, 3);
    size_t elem_of[2];
    fi_score_t res[2];
    const fi_layer_t *L2 = fi_find_layer(m, 62);
    for (int rep = 0; rep < 2; rep++) {
        uint64_t st = seed;
        elem_of[rep] = (size_t)(fi_splitmix64_next(&st) % L2->n_elems);
        fi_fault_t g = {0};
        fi_apply(m, L2, elem_of[rep], 6, &g);
        fi_score_run(r, s, clean_a, &res[rep]);
        fi_restore(m, &g);
    }
    CHECK(elem_of[0] == elem_of[1], "same seed picked different elements");
    CHECK(memcmp(&res[0], &res[1], sizeof res[0]) == 0, "same seed gave different scores");
    CHECK(fi_cell_seed(2026, 62, 6, 3) == seed, "cell seed not repeatable");
    CHECK(fi_cell_seed(2026, 62, 6, 4) != seed && fi_cell_seed(2026, 62, 5, 3) != seed &&
          fi_cell_seed(2026, 60, 6, 3) != seed && fi_cell_seed(2027, 62, 6, 3) != seed,
          "different cells share a seed");
    printf("  seed %llu -> op 62 elem %zu twice: scores identical (masked=%d subtle=%d cat=%d)\n",
           (unsigned long long)seed, elem_of[0], res[0].n_masked, res[0].n_subtle, res[0].n_catastrophic);

    free(again);
    free(faulted);
}

/* The persistent interpreter is created once and reused for every trial. That
 * is only valid if every kernel reads its weights from the model buffer at
 * invoke time. A kernel that packed or copied its weights during
 * AllocateTensors would silently ignore later flips. So for every layer: fault
 * the buffer, run on the persistent interpreter, then build a brand new
 * interpreter on the same faulted buffer and demand identical logits. */
static void test_inplace_vs_fresh(fi_model_t *m, fi_runner_t *r, const fi_subset_t *s,
                                  const int8_t *clean)
{
    header("persistent interpreter sees flips in all 55 layers (vs fresh interpreter)");
    int n_img = s->n < 4 ? s->n : 4;
    int8_t a[FI_NUM_CLASSES], b[FI_NUM_CLASSES];
    int cases = 0, mismatch = 0, visible = 0;

    for (int li = 0; li < m->n_layers; li++) {
        const fi_layer_t *L = &m->layers[li];
        size_t elems[2]; int bits[2]; int n_f = 0;
        elems[n_f] = pick_max_abs(m, L); bits[n_f++] = 7;
        for (size_t i = 0; i < L->n_elems; i++)
            if (m->buf[L->offset + i] == 0) { elems[n_f] = i; bits[n_f++] = 6; break; }

        for (int fidx = 0; fidx < n_f; fidx++) {
            fi_fault_t f = {0};
            fi_apply(m, L, elems[fidx], bits[fidx], &f);
            fi_runner_t fresh;
            if (fi_runner_init(&fresh, m) != 0) { CHECK(0, "fresh runner failed"); fi_restore(m, &f); continue; }
            int changed = 0, differ = 0;
            for (int i = 0; i < n_img; i++) {
                const int8_t *img = s->images + (size_t)i * s->img_bytes;
                fi_run_image(r, img, a);
                fi_run_image(&fresh, img, b);
                differ  += memcmp(a, b, FI_NUM_CLASSES) != 0;
                changed += memcmp(a, clean + (size_t)i * FI_NUM_CLASSES, FI_NUM_CLASSES) != 0;
            }
            fi_runner_free(&fresh);
            fi_restore(m, &f);
            cases++;
            mismatch += differ != 0;
            visible  += changed != 0;
            if (differ) printf("  op %d elem %zu bit %d: persistent and fresh interpreters disagree\n",
                               L->op_index, elems[fidx], bits[fidx]);
        }
    }
    CHECK(mismatch == 0, "%d of %d faulted cases: persistent interpreter ignored the flip", mismatch, cases);
    CHECK(fi_is_clean(m), "buffer not clean after the sweep");
    printf("  %d faulted cases over 55 layers: persistent == fresh in %d, flip changed the logits in %d (%d images each)\n",
           cases, cases - mismatch, visible, n_img);
}

/* splitmix64 against reference vectors produced by an independent python
 * implementation (and equal to the published values for seed 1234567) */
static void test_splitmix(void)
{
    header("splitmix64 reference vectors");
    static const uint64_t want[5] = {
        6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL,
        4593380528125082431ULL, 16408922859458223821ULL };
    uint64_t st = 1234567;
    int bad = 0;
    for (int i = 0; i < 5; i++) bad += (fi_splitmix64_next(&st) != want[i]);
    CHECK(bad == 0, "%d of 5 outputs differ from the reference", bad);
    CHECK(fi_cell_seed(12345, 120, 7, 3) == 17806289111753477034ULL, "cell seed vector 1");
    CHECK(fi_cell_seed(0, 1, 0, 0)       == 12793040940332582595ULL, "cell seed vector 2");
    CHECK(fi_cell_seed(2026, 89, 4, 29)  == 15804760654010310752ULL, "cell seed vector 3");
}

/* ---------------------------------------------------------------- */
/* 5. C runtime vs python tf.lite.Interpreter */

static int8_t *read_ref(const char *path, int *n_images)  /* returns NULL if missing */
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    long bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    int8_t *p = malloc((size_t)bytes);
    if (!p || fread(p, 1, (size_t)bytes, fp) != (size_t)bytes) { fclose(fp); free(p); return NULL; }
    fclose(fp);
    *n_images = (int)(bytes / FI_NUM_CLASSES);
    return p;
}

/* max |difference| between two logit sets, number of images whose top-1 differs */
static void compare_logits(const int8_t *a, const int8_t *b, int k, int *exact, int *top1_diff, int *max_diff)
{
    *exact = *top1_diff = *max_diff = 0;
    for (int i = 0; i < k; i++) {
        const int8_t *x = a + (size_t)i * FI_NUM_CLASSES;
        const int8_t *y = b + (size_t)i * FI_NUM_CLASSES;
        int same = 1;
        for (int j = 0; j < FI_NUM_CLASSES; j++) {
            int d = abs((int)x[j] - (int)y[j]);
            if (d) { same = 0; if (d > *max_diff) *max_diff = d; }
        }
        *exact += same;
        *top1_diff += (fi_argmax(x) != fi_argmax(y));
    }
}

/* tflite has several int8 kernel sets (python default with XNNPACK, plain
 * builtin, reference) and their rounding differs slightly, so two correct
 * runtimes are not bit-identical. The honest question is whether the C runtime
 * is any further from python than python's own configurations are from each
 * other. That spread is measured here, not assumed. */
static void test_python_parity(const fi_subset_t *s, const int8_t *clean, const char *subset_dir)
{
    header("C runtime vs python tf.lite.Interpreter");
    enum { NCFG = 3 };
    const char *names[NCFG] = { "default", "nodeleg", "ref" };
    const char *desc[NCFG]  = { "python default (XNNPACK)", "python no delegates", "python reference kernels" };
    int8_t *ref[NCFG] = { NULL };
    int k = s->n;

    for (int c = 0; c < NCFG; c++) {
        char path[1024];
        int kc = 0;
        snprintf(path, sizeof path, "%s/ref_logits_%s.bin", subset_dir, names[c]);
        ref[c] = read_ref(path, &kc);
        if (!ref[c]) { printf("  SKIP: %s not found (run ref_logits.py)\n", path); n_skip++; goto done; }
        if (kc < k) k = kc;
    }

    /* spread of python against itself */
    int py_max_diff = 0, py_top1_diff = 0;
    for (int a = 0; a < NCFG; a++) {
        for (int b = a + 1; b < NCFG; b++) {
            int ex, t1, md;
            compare_logits(ref[a], ref[b], k, &ex, &t1, &md);
            printf("  %-26s vs %-26s: bit-exact %3d/%d, top-1 differs %d, max |diff| %d\n",
                   desc[a], desc[b], ex, k, t1, md);
            if (md > py_max_diff) py_max_diff = md;
            if (t1 > py_top1_diff) py_top1_diff = t1;
        }
    }

    /* C against each python configuration */
    int c_max_diff = 0, c_top1_diff = 0;
    for (int c = 0; c < NCFG; c++) {
        int ex, t1, md;
        compare_logits(clean, ref[c], k, &ex, &t1, &md);
        printf("  %-26s vs %-26s: bit-exact %3d/%d, top-1 differs %d, max |diff| %d\n",
               "C runtime", desc[c], ex, k, t1, md);
        if (md > c_max_diff) c_max_diff = md;
        if (t1 > c_top1_diff) c_top1_diff = t1;
    }

    /* accuracy on this subset, for information */
    int acc_c = 0, acc_py[NCFG] = {0};
    for (int i = 0; i < k; i++) {
        acc_c += fi_argmax(clean + (size_t)i * FI_NUM_CLASSES) == s->labels[i];
        for (int c = 0; c < NCFG; c++)
            acc_py[c] += fi_argmax(ref[c] + (size_t)i * FI_NUM_CLASSES) == s->labels[i];
    }
    printf("  subset accuracy (%d images): C %d, python default %d, no delegates %d, reference %d\n",
           k, acc_c, acc_py[0], acc_py[1], acc_py[2]);

    CHECK(c_max_diff <= py_max_diff,
          "C is further from python (max |diff| %d) than python configs are from each other (%d)",
          c_max_diff, py_max_diff);
    CHECK(c_top1_diff <= py_top1_diff,
          "C top-1 disagrees on %d images, python configs disagree with each other on at most %d",
          c_top1_diff, py_top1_diff);
done:
    for (int c = 0; c < NCFG; c++) free(ref[c]);
}

/* ---------------------------------------------------------------- */

int main(int argc, char **argv)
{
    const char *model_path = NULL, *subset_dir = "../subset", *csv = "../layer_inventory.csv";
    int n_images = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--model"))  model_path = argv[i + 1];
        else if (!strcmp(argv[i], "--subset")) subset_dir = argv[i + 1];
        else if (!strcmp(argv[i], "--csv"))    csv = argv[i + 1];
        else if (!strcmp(argv[i], "--images")) n_images = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
    }
    if (!model_path) { fprintf(stderr, "usage: test_fi --model <tflite> [--subset dir] [--csv file] [--images N]\n"); return 2; }

    printf("TfLiteVersion: %s\n", TfLiteVersion());

    fi_model_t m;
    if (fi_model_load(&m, model_path) != 0) return 2;
    printf("model: %zu bytes, %d conv layers located\n", m.len, m.n_layers);

    test_splitmix();
    test_locator(&m, csv);
    test_value_change(&m);
    test_one_byte_and_restore(&m);

    fi_runner_t r;
    fi_subset_t s;
    if (fi_runner_init(&r, &m) != 0) return 2;
    if (fi_subset_load(&s, subset_dir, n_images, r.in_bytes) != 0) return 2;

    int8_t *clean = alloc_logits(s.n);
    double t0 = now_sec();
    if (fi_clean_run(&r, &s, clean) != 0) return 2;
    printf("clean run: %d images, %.1f ms/image\n", s.n, 1000.0 * (now_sec() - t0) / s.n);

    test_runtime(&m, &r, &s, clean);
    test_inplace_vs_fresh(&m, &r, &s, clean);
    test_python_parity(&s, clean, subset_dir);

    printf("\n==== %d checks passed, %d failed, %d skipped ====\n", n_pass, n_fail, n_skip);
    free(clean);
    fi_subset_free(&s);
    fi_runner_free(&r);
    fi_model_free(&m);
    return n_fail ? 1 : 0;
}
~~~~

### c/run_trials.c

~~~~
/* run_trials.c - the fault injection trial loop.
 *
 * For every (layer, element, bit): flip, run the image subset, restore, check
 * the model buffer is clean again, write one csv row.
 *
 *   ./run_trials --model <.tflite> --subset <dir> --mode corner|random
 *                [--layers 1,3,9,...] [--bits 0,1,...,7] [--k 30]
 *                [--seed 2026] [--images 200] [--limit N] [--out trials.csv]
 *
 * mode corner: per layer, 5 corner elements x the chosen bits
 *     max_abs         element with the largest |w|
 *     min_nonzero_abs element with the smallest nonzero |w|
 *     first_zero      first element equal to 0
 *     first_elem      element 0
 *     last_elem       last element
 *   (ties go to the lowest index; the same element can win several rules,
 *    each rule still gets its own rows)
 * mode random: per layer and bit, k trials with a seeded random element.
 *
 * --limit stops after N trials in total, for smoke tests.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fault_inject.h"

/* first pass layers from fault_model_week5.md, step 6 */
static const int DEFAULT_LAYERS[] = { 1, 3, 9, 22, 36, 42, 56, 89, 60, 62, 64, 102, 116, 120 };
#define N_DEFAULT_LAYERS ((int)(sizeof DEFAULT_LAYERS / sizeof DEFAULT_LAYERS[0]))

enum { RULE_MAX_ABS, RULE_MIN_NONZERO_ABS, RULE_FIRST_ZERO, RULE_FIRST_ELEM, RULE_LAST_ELEM, N_RULES };
static const char *RULE_NAMES[N_RULES] = { "max_abs", "min_nonzero_abs", "first_zero", "first_elem", "last_elem" };

/* stage of a conv op, from the stage table in architecture_week4.md.
 * A block is [expand 1x1, pad, depthwise, project 1x1], and the stage entry
 * depthwise ops are 9, 36, 56, 96, so a stage starts at its entry block's
 * expand conv: ops 7, 34, 54, 94. */
static const char *stage_of(int op)
{
    if (op <= 1)   return "stem";
    if (op < 7)    return "1";
    if (op < 34)   return "2";
    if (op < 54)   return "3";
    if (op < 94)   return "4";
    if (op < 120)  return "5";
    return "tail";
}

/* ---------- element selection ---------- */

/* returns the element index for a corner rule, or -1 if the rule has no candidate */
static long corner_element(const fi_model_t *m, const fi_layer_t *L, int rule)
{
    const int8_t *w = (const int8_t *)(m->buf + L->offset);
    long best = -1;
    int  best_abs = 0;

    switch (rule) {
    case RULE_MAX_ABS:
        for (size_t i = 0; i < L->n_elems; i++) {
            int a = w[i] < 0 ? -w[i] : w[i];             /* |-128| = 128 */
            if (best < 0 || a > best_abs) { best = (long)i; best_abs = a; }
        }
        return best;
    case RULE_MIN_NONZERO_ABS:
        for (size_t i = 0; i < L->n_elems; i++) {
            int a = w[i] < 0 ? -w[i] : w[i];
            if (a != 0 && (best < 0 || a < best_abs)) { best = (long)i; best_abs = a; }
        }
        return best;
    case RULE_FIRST_ZERO:
        for (size_t i = 0; i < L->n_elems; i++)
            if (w[i] == 0) return (long)i;
        return -1;
    case RULE_FIRST_ELEM:
        return 0;
    case RULE_LAST_ELEM:
        return (long)L->n_elems - 1;
    }
    return -1;
}

/* ---------- csv ---------- */

static const char *CSV_HEADER =
    "trial_id,seed,op_index,op_type,stage,tensor,elem_idx,bit,orig_val,new_val,"
    "n_images,mismatch_rate,acc_faulted,acc_clean,n_masked,n_subtle,n_catastrophic,"
    "n_due,elem_rule\n";

typedef struct {
    FILE        *csv;
    fi_model_t  *model;
    fi_runner_t *runner;
    fi_subset_t *subset;
    int8_t      *clean_logits;
    int          trial_id;
    int          limit;       /* <= 0: no limit */
    double       t_start;
} ctx_t;

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* one trial. Returns 0 ok, -1 fatal (buffer could not be restored). */
static int run_trial(ctx_t *c, const fi_layer_t *L, size_t elem, int bit,
                     uint64_t seed, const char *rule)
{
    fi_fault_t f = {0};
    fi_score_t sc;

    if (fi_apply(c->model, L, elem, bit, &f) != 0) return -1;
    fi_score_run(c->runner, c->subset, c->clean_logits, &sc);
    fi_restore(c->model, &f);

    if (!fi_is_clean(c->model)) {
        fprintf(stderr, "FATAL: model buffer not clean after restore (op %d elem %zu bit %d)\n",
                L->op_index, elem, bit);
        return -1;
    }

    /* mismatch rate: images whose top-1 differs from the clean top-1.
     * A DUE has no top-1 at all, so it is counted as a mismatch too. */
    double n = (double)sc.n_images;
    fprintf(c->csv, "%d,%llu,%d,%s,%s,weight,%zu,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%d,%s\n",
            c->trial_id, (unsigned long long)seed, L->op_index,
            L->op_type == FI_OP_CONV_2D ? "CONV_2D" : "DEPTHWISE_CONV_2D",
            stage_of(L->op_index), elem, bit, f.orig_val, f.new_val, sc.n_images,
            (sc.n_catastrophic + sc.n_due) / n,
            sc.n_correct_faulted / n, sc.n_correct_clean / n,
            sc.n_masked, sc.n_subtle, sc.n_catastrophic, sc.n_due, rule);
    fflush(c->csv);

    c->trial_id++;
    if (c->trial_id % 20 == 0)
        fprintf(stderr, "  %d trials, %.1f s\n", c->trial_id, now_sec() - c->t_start);
    return 0;
}

static int limit_reached(const ctx_t *c) { return c->limit > 0 && c->trial_id >= c->limit; }

/* ---------- argument helpers ---------- */

static int parse_int_list(const char *s, int *out, int max)
{
    int n = 0;
    char *copy = strdup(s), *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok && n < max; tok = strtok_r(NULL, ",", &save))
        out[n++] = atoi(tok);
    free(copy);
    return n;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL, *subset_dir = "../subset", *out_path = "trials.csv";
    const char *mode = "corner";
    int layers[FI_MAX_LAYERS], n_layers = 0, bits[8] = {0, 1, 2, 3, 4, 5, 6, 7}, n_bits = 8;
    int k = 30, n_images = 0, limit = 0;
    uint64_t seed = 2026;

    for (int i = 0; i < N_DEFAULT_LAYERS; i++) layers[n_layers++] = DEFAULT_LAYERS[i];

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "flag %s needs a value\n", argv[i]); return 2; }
        const char *v = argv[i + 1];
        if      (!strcmp(argv[i], "--model"))  model_path = v;
        else if (!strcmp(argv[i], "--subset")) subset_dir = v;
        else if (!strcmp(argv[i], "--out"))    out_path = v;
        else if (!strcmp(argv[i], "--mode"))   mode = v;
        else if (!strcmp(argv[i], "--layers")) n_layers = parse_int_list(v, layers, FI_MAX_LAYERS);
        else if (!strcmp(argv[i], "--bits"))   n_bits = parse_int_list(v, bits, 8);
        else if (!strcmp(argv[i], "--k"))      k = atoi(v);
        else if (!strcmp(argv[i], "--seed"))   seed = strtoull(v, NULL, 10);
        else if (!strcmp(argv[i], "--images")) n_images = atoi(v);
        else if (!strcmp(argv[i], "--limit"))  limit = atoi(v);
        else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
    }
    int corner = !strcmp(mode, "corner");
    if (!model_path || (!corner && strcmp(mode, "random"))) {
        fprintf(stderr, "usage: run_trials --model <tflite> [--mode corner|random] [--subset dir] "
                        "[--layers a,b] [--bits a,b] [--k N] [--seed S] [--images N] [--limit N] [--out file]\n");
        return 2;
    }
    for (int i = 0; i < n_bits; i++)
        if (bits[i] < 0 || bits[i] > 7) { fprintf(stderr, "bit %d out of range\n", bits[i]); return 2; }

    fi_model_t m;
    fi_runner_t r;
    fi_subset_t s;
    if (fi_model_load(&m, model_path) != 0) return 2;
    if (fi_runner_init(&r, &m) != 0) return 2;
    if (fi_subset_load(&s, subset_dir, n_images, r.in_bytes) != 0) return 2;

    /* clean reference run, done twice: if it is not repeatable nothing else means anything */
    int8_t *clean = malloc((size_t)s.n * FI_NUM_CLASSES);
    int8_t *check = malloc((size_t)s.n * FI_NUM_CLASSES);
    if (fi_clean_run(&r, &s, clean) != 0 || fi_clean_run(&r, &s, check) != 0) return 2;
    if (memcmp(clean, check, (size_t)s.n * FI_NUM_CLASSES) != 0) {
        fprintf(stderr, "FATAL: two clean runs differ\n");
        return 2;
    }
    free(check);

    FILE *csv = fopen(out_path, "w");
    if (!csv) { perror(out_path); return 2; }
    fputs(CSV_HEADER, csv);

    ctx_t c = { csv, &m, &r, &s, clean, 0, limit, now_sec() };
    fprintf(stderr, "mode=%s layers=%d bits=%d images=%d seed=%llu -> %s\n",
            mode, n_layers, n_bits, s.n, (unsigned long long)seed, out_path);

    int rc = 0;
    for (int li = 0; li < n_layers && !rc && !limit_reached(&c); li++) {
        const fi_layer_t *L = fi_find_layer(&m, layers[li]);
        if (!L) { fprintf(stderr, "op %d is not an injectable layer, skipped\n", layers[li]); continue; }

        if (corner) {
            for (int rule = 0; rule < N_RULES && !rc && !limit_reached(&c); rule++) {
                long e = corner_element(&m, L, rule);
                if (e < 0) {
                    fprintf(stderr, "op %d: rule %s has no candidate, skipped\n", L->op_index, RULE_NAMES[rule]);
                    continue;
                }
                for (int b = 0; b < n_bits && !rc && !limit_reached(&c); b++) {
                    /* corner choice does not use the seed; it is logged for the record */
                    uint64_t tseed = fi_cell_seed(seed, L->op_index, bits[b], 1000 + rule);
                    rc = run_trial(&c, L, (size_t)e, bits[b], tseed, RULE_NAMES[rule]);
                }
            }
        } else {
            for (int b = 0; b < n_bits && !rc; b++) {
                for (int t = 0; t < k && !rc && !limit_reached(&c); t++) {
                    /* the cell seed picks the element: same (seed, layer, bit, t) -> same element */
                    uint64_t tseed = fi_cell_seed(seed, L->op_index, bits[b], t);
                    uint64_t st = tseed;
                    size_t e = (size_t)(fi_splitmix64_next(&st) % L->n_elems);
                    rc = run_trial(&c, L, e, bits[b], tseed, "random");
                }
            }
        }
    }

    fclose(csv);
    fprintf(stderr, "done: %d trials in %.1f s, model buffer clean: %s\n", c.trial_id,
            now_sec() - c.t_start, fi_is_clean(&m) ? "yes" : "NO");

    free(clean);
    fi_subset_free(&s);
    fi_runner_free(&r);
    fi_model_free(&m);
    return rc ? 3 : 0;
}
~~~~

### c/Makefile

~~~~
# Build the fault injection tests and the trial runner.
#
#   make            build test_fi and run_trials
#   make test       write python reference logits, then run test_fi
#   make smoke      short run_trials run (see SMOKE_LIMIT)
#
# TFLITE points at the libtensorflowlite_c build (v2.21.0, same as python
# tensorflow in venv311). Override on the command line if you moved it.

TFLITE ?= $(HOME)/.local/tflite-2.21.0
MODEL  ?= $(HOME)/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite
SUBSET ?= ../subset
CSV    ?= ../layer_inventory.csv
PYTHON ?= ../venv311/bin/python
SMOKE_LIMIT ?= 30

CC      = cc
CFLAGS  = -std=c11 -O2 -g -Wall -Wextra -I$(TFLITE)/include
LDFLAGS = -L$(TFLITE)/lib -ltensorflowlite_c -Wl,-rpath,$(TFLITE)/lib

all: test_fi run_trials

test_fi: test_fi.c fault_inject.c fault_inject.h
	$(CC) $(CFLAGS) -o $@ test_fi.c fault_inject.c $(LDFLAGS)

run_trials: run_trials.c fault_inject.c fault_inject.h
	$(CC) $(CFLAGS) -o $@ run_trials.c fault_inject.c $(LDFLAGS)

test: test_fi
	$(PYTHON) ref_logits.py --subset $(SUBSET) --k 200 --model $(MODEL)
	./test_fi --model $(MODEL) --subset $(SUBSET) --csv $(CSV) --images 200

smoke: run_trials
	./run_trials --model $(MODEL) --subset $(SUBSET) --mode corner --layers 1,62,120 --bits 0,7 --limit $(SMOKE_LIMIT) --images 20 --out trials_smoke.csv

clean:
	rm -rf test_fi run_trials *.dSYM trials*.csv

.PHONY: all test smoke clean
~~~~

### c/ref_logits.py

~~~~
"""Dump python tf.lite.Interpreter logits for the first K subset images.

test_fi.c reads these files and checks the C runtime gives the same logits.
Three python configurations are written, because tflite has several int8 kernel
sets whose rounding differs slightly and the C library is a fourth one:
    ref_logits_default.bin   tf.lite.Interpreter(path)            (what eval_int8_tflite.py uses)
    ref_logits_nodeleg.bin   default delegates (XNNPACK) switched off
    ref_logits_ref.bin       reference kernels only (BUILTIN_REF)
Each file is K * 1000 int8 values.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_int8_tflite as ev  # noqa: E402  (only for the model path)


def logits_for(interpreter, images):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    res = []
    for img in images:
        interpreter.set_tensor(inp["index"], img.reshape(inp["shape"]))
        interpreter.invoke()
        res.append(interpreter.get_tensor(out["index"]).reshape(-1).astype(np.int8))
    return np.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="subset")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--model", default=ev.tflite_path)
    args = ap.parse_args()

    images = []
    for i in range(args.k):
        raw = np.fromfile(os.path.join(args.subset, f"img_{i:03d}.bin"), dtype=np.int8)
        images.append(raw)

    R = tf.lite.experimental.OpResolverType
    configs = {
        "default": tf.lite.Interpreter(args.model),
        "nodeleg": tf.lite.Interpreter(
            args.model, experimental_op_resolver_type=R.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
        "ref": tf.lite.Interpreter(
            args.model, experimental_op_resolver_type=R.BUILTIN_REF),
    }
    for name, interp in configs.items():
        interp.allocate_tensors()
        lg = logits_for(interp, images)
        path = os.path.join(args.subset, f"ref_logits_{name}.bin")
        lg.tofile(path)
        print(f"wrote {path}: {lg.shape}")


if __name__ == "__main__":
    main()
~~~~

### make_subset.py

~~~~
"""Build the fixed image subset used by every fault-injection trial.

Rule (fault_model_week5.md, step 7): walk the imagenet validation stream in its
fixed order and keep the first image of every class not seen yet, until N
images are collected. No randomness anywhere, so running this twice gives a
byte-identical subset.

Output directory (default subset/):
    img_000.bin ...   raw int8 NHWC [1,H,W,3], exactly what goes into the model
    labels.bin        int32 little-endian, one label per image
    manifest.json     stream index, label and sha256 per image, plus the model sha256

Preprocessing comes from eval_int8_tflite.py so these images are exactly the
ones the 50k baseline evaluation fed to the model.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import tensorflow as tf

import eval_int8_tflite as ev


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def build_subset(n_images, scan_limit):
    interpreter = tf.lite.Interpreter(ev.tflite_path)
    input_shape = interpreter.get_input_details()[0]["shape"]
    transform = ev.make_transform(int(input_shape[1]))

    seen = set()
    picked = []  # (stream_index, label, int8 array)
    for i, img, label in ev.iter_val_stream(scan_limit):
        if label in seen:
            continue
        seen.add(label)
        picked.append((i, label, ev.preprocess_to_int8(img, transform, input_shape)))
        if len(picked) == n_images:
            break
    if len(picked) < n_images:
        raise SystemExit(f"only found {len(picked)} classes in the first {scan_limit} samples")
    return picked, [int(d) for d in input_shape]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of images (one per class)")
    ap.add_argument("--out", default="subset", help="output directory")
    ap.add_argument("--scan-limit", type=int, default=50000, help="max stream samples to scan")
    args = ap.parse_args()

    picked, input_shape = build_subset(args.n, args.scan_limit)

    os.makedirs(args.out, exist_ok=True)
    entries = []
    for k, (stream_idx, label, arr) in enumerate(picked):
        raw = np.ascontiguousarray(arr, dtype=np.int8).tobytes()
        name = f"img_{k:03d}.bin"
        with open(os.path.join(args.out, name), "wb") as f:
            f.write(raw)
        entries.append({"file": name, "stream_index": stream_idx,
                        "label": int(label), "sha256": sha256_bytes(raw)})

    labels = np.array([e["label"] for e in entries], dtype="<i4")
    with open(os.path.join(args.out, "labels.bin"), "wb") as f:
        f.write(labels.tobytes())

    manifest = {
        "n_images": len(entries),
        "input_shape": input_shape,
        "dtype": "int8",
        "labels_dtype": "int32 little-endian",
        "rule": "first image of each new class in stream order",
        "model_sha256": sha256_file(ev.tflite_path),
        "labels_sha256": sha256_bytes(labels.tobytes()),
        "images": entries,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")

    print(f"wrote {len(entries)} images to {args.out}/ "
          f"(last stream index {entries[-1]['stream_index']})", flush=True)


if __name__ == "__main__":
    main()
    # the streaming dataset leaves background threads that can keep the process
    # alive after the work is done, so exit explicitly (files are already closed)
    os._exit(0)
~~~~

### fault_model_week5.md

~~~~
# mcunet fault sensitivity - fault model (week five)

model: mcunet-in3, 320kb sram variant, mit han lab
file: mcunet-320kb-1mb_imagenet.tflite
goal: define exactly what a fault is, where it goes, and how i score it so the injection framework has a spec to build against


## summary

- fault = single bit flip in an int8 tensor
- weights first, activations second
- bits 0-7 swept uniformly, lit used as a hypothesis not an assumption
- corner case elements first (random still available as a mode, seeded), fixed image subset. updated week 6, see the week 6 changes section at the bottom
- outcomes: masked / subtle sdc / catastrophic sdc / due, scored against the clean run


## step 1 - fault type

single bit flip, one flip per trial. no multi-bit, no stuck-at, no random value replacement in v1

why single bit: its the common fault model in this lit (li et al. sc17, binfi, the bit flip attack paper) and it keeps each trial attributable to exactly one (layer, element, bit)

the flip is an xor on the stored int8 byte (2's complement):
- bits 0-6 change the stored value by +/- 2^bit
- bit 7 is the sign bit, flips value by -128 or +128 depending on the sign it started with
- int8 is bounded so no nan / exponent blowup like fp32, worst case is a 128 jump in stored units


## step 2 - target tensors

in scope (v1):
- int8 weight tensors of conv_2d and depthwise_conv_2d (55 ops, from layer_inventory.csv)

in scope (phase 2):
- int8 activation tensors (output of each conv/depthwise op)

out of scope, and why:
- biases - int32, different bit width, not in my param count yet either
- scale / zero_point - quant metadata, not data. corrupting these is a different fault class
- pad / add / pooling / reshape - no weights, nothing storable to corrupt
- anything below the tensor level (registers, alu, instruction faults like sassifi does) - not what this project is about

weight zero point is 0 on every conv in the csv so a flip on the stored byte maps cleanly onto the real weight. activation zero points are -128 or 0 depending on the layer so activation flips happen on the stored int8 value, not the dequantized one


## step 3 - weight faults vs activation faults

these are two different things and i'm keeping them separate:

| | weight fault | activation fault |
|---|---|---|
| models | corruption of stored weights (flash / sram) | transient upset while a layer is computing |
| lifetime | persistent, stays until restored | one inference only |
| affects | every image run while its in place | one image |
| spatial reach | every output position that weight touches | one element of one feature map |
| status | v1 (week 6) | phase 2 (week 11) |

ares (reagen et al. dac 2018) found activations were as tolerant as or more tolerant than weights at the same bit error rate, up to ~50x for resnet50 / vgg16. they put it down to weight reuse, one bad weight gets multiplied across the whole feature map. so expect a weight flip to hurt more than an activation flip. that was fixed point models with faults in memory, so still something to check on mcunet

note on activations: the tflite python interpreter has no per-op hook so activation injection needs a different mechanism than weights (split model, or the c route). deciding that when i get to week 11


## step 4 - injection timing

weights:
1. pick (layer, element, bit)
2. flip the bit in the model buffer
3. run the whole fixed image subset with the corrupted model
4. flip the same bit again to restore
5. check the model buffer hash matches the clean one before the next trial (cheaper than rerunning inference)

activations (phase 2):
1. run forward pass up to layer L
2. flip one bit in one element of layer L's int8 output
3. continue the forward pass
4. next image gets a clean run again


## step 5 - bit position

uniform sweep over bits 0-7. no weighting toward the msb.

hypothesis: the high bits matter most and bits 0-2 are mostly noise. support for it is indirect:
- bit k changes the stored int8 value by 2^k (sign bit by 128), thats just arithmetic
- li et al. sc17 found only high order bits cause sdcs (float and fixed point, gpu accelerators)
- binfi argues the effect of a flip grows with bit position (32 bit fixed point, approximate for some nets)
- rakin et al. (bit flip attack) found 100 random flips on an 8 bit resnet-18 cost under ~1% top-1, so random int8 flips are mostly benign and the critical ones are rare
- i didnt find anything that tests int8 bit by bit on a tinyml model, so thats the gap this project fills

sweeping all 8 evenly is what lets me test this on mcunet, and it's cheap enough that there's no reason to prune. since critical flips are rare, subtle sdc (logit change) is worth tracking and not just top-1


## step 6 - layer selection

full space: 55 injectable layers x 8 bits = 440 cells

first pass is 14 layers picked off the stage table (see architecture_week4.md):

| op | stage | layer |
|---|---|---|
| 1 | stem | conv 3x3 s2 |
| 3 | 1 | depthwise 3x3 |
| 9 | 2 | depthwise 7x7 s2, entry |
| 22 | 2 | depthwise 7x7, shortcut |
| 36 | 3 | depthwise 5x5 s2, entry |
| 42 | 3 | depthwise 5x5, shortcut |
| 56 | 4 | depthwise 3x3 s2, entry |
| 89 | 4 | depthwise 3x3, shortcut |
| 60 | 4 | expand 1x1 |
| 62 | 4 | depthwise 7x7, shortcut |
| 64 | 4 | project 1x1 |
| 102 | 5 | depthwise 7x7, shortcut |
| 116 | 5 | depthwise 7x7, no shortcut |
| 120 | tail | classifier 1x1 |

the pairs control for kernel size (9/22, 36/42, 56/89) and 102 vs 116 have identical shape + macs so thats the clean shortcut test. 60/62/64 is depthwise vs pointwise inside one block. whether depthwise is actually more fragile is an open question, i couldnt find a paper that shows it (see the end)

once the pipeline is validated this widens to all 55


## step 7 - sampling

- element (week 6 update, from the advisor meeting): corner cases first, not uniform random. per layer: max |w|, smallest nonzero |w|, first zero weight, first element, last element, each with all 8 bits. random within the layer's weight tensor stays as `--mode random`. my reading of "corner cases" is the five rules above and the advisor still has to confirm it, so treat the rules as pending
- one flip per trial
- k trials per (layer, bit) cell
- images: a fixed subset, same images every trial. not all 50k per trial

on how many trials: cochran's formula gives ~385 trials for +/-5% at 95% confidence (n = z^2 p(1-p) / e^2, p = 0.5). thats per estimate, and 385 x 440 cells is ~169k trials, too many. so k per cell is small (starting point ~30) and i report confidence intervals per cell, then pool across layers per bit to get tight per-bit estimates

sample size math reference: leveugle et al., statistical fault injection: quantified error and confidence. for the layer x bit stratified setup see ruospo et al. date 2023, they stratify by layer and bit and compare against exhaustive runs, but on fp32 stuck-at faults (resnet-20 / mobilenetv2 on cifar-10), not int8 flips

first pass budget check: 14 layers x 8 bits x 30 trials = 3,360 trials. at 200 images per trial thats 672,000 inferences. need to time one inference before locking k and the subset size

image subset: ~200 images, spread across classes (the first 1000 streamed samples only covered 638 classes so not just taking the first n). built once by make_subset.py as the first image of each new class in stream order, so there is no randomness in it at all. saved as raw int8 .bin files plus labels.bin and a manifest with sha256s (not .npy anymore, the c code reads the raw bytes) so nothing re-streams per trial. the 200 images come from the first 226 streamed samples, and two runs of the script give byte identical files. the 50k run stays the baseline accuracy number, separate from the injection runs


## step 8 - outcome classes

per image, compared against the clean run on that same image:

| class | definition |
|---|---|
| masked | faulted logits identical to clean logits |
| subtle sdc | logits changed, top-1 same |
| catastrophic sdc | top-1 changed |
| due | crash or hang (no output) |

due should be ~0. int8 data flips dont crash anything, crashes come from corrupted metadata like indices / shapes / pointers and thats out of scope. worth reporting as a result anyway (breier et al. 2026 report int8 weights stay bounded so no nan or extreme values like the float formats)

per trial metrics (over the subset):
- mismatch rate = (catastrophic + due) / n. a due has no top-1 at all so it counts as a mismatch. this is slightly different from "faulted top-1 != clean top-1" above, but due has been 0 so far so the numbers are the same
- accuracy under fault vs labels, and drop vs clean accuracy on the same subset
- counts of masked / subtle / catastrophic / due

heatmap cell (layer x bit) = mean over its trials. sdc is defined against the clean prediction, accuracy drop is what goes in the degradation heatmap

open: does catastrophic mean any top-1 change, or only the clean class dropping out of top-5. leaning any top-1 change for v1, can add top-5 later


## step 9 - reproducibility

- seed per trial, logged. week 6 update: splitmix64 instead of numpy default_rng since the injector is in c. the trial seed comes from (base seed, op index, bit, k) so a cell gets the same seed no matter what else ran, and in random mode the seed picks the element. same seed gives the same element and the same result, checked (two full runs gave byte identical csvs). in corner mode the element does not depend on the seed, it is still logged
- image subset saved as raw int8 .bin files, sha256 of every image in subset/manifest.json (not written into the log yet)
- model file sha256 recorded
- requirements.txt pinned
- before trusting anything: clean run twice gives identical logits, and flip + restore gives identical logits to the clean run

log row per trial:

```
trial_id, seed, op_index, op_type, stage, tensor (weight/act),
elem_idx, bit, orig_val, new_val, n_images,
mismatch_rate, acc_faulted, acc_clean,
n_masked, n_subtle, n_catastrophic, n_due, elem_rule
```

elem_rule is the corner rule name (max_abs, min_nonzero_abs, first_zero, first_elem, last_elem) or "random". added week 6 so a row says why that element was picked


## week 6 changes (what the c prototype actually does)

these are the places the built injector differs from or adds to the spec above. everything in c/ and results/ backs these up

- element selection is corner cases first (max |w|, smallest nonzero |w|, first zero weight, first, last, all 8 bits per rule), random kept as `--mode random`. corner rules are pending advisor confirmation. ties go to the lowest index and one element can win more than one rule, each rule still gets its own rows
- new elem_rule column at the end of the log row
- seeding is splitmix64, same seed gives the same result (verified)
- mismatch_rate = (catastrophic + due) / n
- stage column is derived from op index ranges (stem is op 1, stage 1 ops 2-6, stage 2 ops 7-33, stage 3 ops 34-53, stage 4 ops 54-93, stage 5 ops 94-119, tail op 120). a stage starts at the expand conv of its entry block, so it matches the 14 layer table in step 6 but it is my own derivation, not something read from the model
- runtime is the tflite c api built from the v2.21.0 source (same version as the python tensorflow) with xnnpack and ruy off. so it is not the same kernel path as the python eval, which uses the default interpreter with xnnpack. measured on the 200 image subset: the c logits are within 1 unit of python's reference kernels (top-1 identical on all 200) but up to 8 units away from python's default and no-delegate kernels, with 9-10 of 200 top-1 flips. python's own three configurations differ from each other by about the same amount
- because of that the c clean run is the reference for every trial (outcomes compare faulted c logits to clean c logits), and c subset accuracy will not equal 61.82%. on the 200 image subset c gets 130/200 = 65.0%, python default gets 131/200. the 61.82% is the 50k python number, re-checked after the refactor
- fault model framing: this is a weight memory fault, a bit of the stored int8 weight flips and every inference after that uses the corrupted value. it is not emulated cpu compute corruption (no alu or register faults, nothing about how the kernel computes). activation faults come in week 11
- the interpreter is created once and reused across trials. checked that it sees flips in all 55 layers by comparing it to a freshly built interpreter on the same faulted buffer (identical logits in all 110 cases tried)

open: the advisor needs to confirm the corner rules


- lin et al., mcunet: tiny deep learning on iot devices, neurips 2020
  https://arxiv.org/abs/2007.10319

- hari et al., sassifi: an architecture-level fault injection tool for gpu application resilience evaluation, ispass 2017
  https://research.nvidia.com/publication/2017-04_sassifi-architecture-level-fault-injection-tool-gpu-application-resilience

- chen et al., tensorfi: a flexible fault injection framework for tensorflow applications, issre 2020
  https://arxiv.org/abs/2004.01743

- li et al., understanding error propagation in deep learning neural network (dnn) accelerators and applications, sc17
  https://research.nvidia.com/publication/2017-11_understanding-error-propagation-deep-learning-neural-network-dnn-accelerators

- reagen et al., ares: a framework for quantifying the resilience of deep neural networks, dac 2018
  https://vlsiarch.eecs.harvard.edu/publications/ares-framework-quantifying-resilience-deep-neural-networks

- chen et al., binfi: an efficient fault injector for safety-critical machine learning systems, sc19
  https://people.ece.ubc.ca/zitaoc/files/sc19-binfi.pdf

- rakin et al., bit-flip attack: crushing neural network with progressive bit search, iccv 2019
  https://arxiv.org/abs/1903.12269

- breier et al., the weight of a bit: emfi sensitivity analysis of embedded deep learning models, 2026
  https://arxiv.org/abs/2602.16309

- ruospo et al., assessing convolutional neural networks reliability through statistical fault injections, date 2023
  https://past.date-conference.com/proceedings-archive/2023/DATA/807.pdf

- statistical fault injection: quantified error and confidence
  https://hal.archives-ouvertes.fr/hal-00386107

- sharif et al., efficient software-implemented hw fault tolerance for tinyml inference in safety-critical applications, date 2023
  https://past.date-conference.com/proceedings-archive/2023/DATA/6005.pdf

~~~~

### eval_int8_tflite.py: git diff for the refactor commit cfc98da

~~~~
diff --git a/eval_int8_tflite.py b/eval_int8_tflite.py
index 0a61068..ee3d05b 100644
--- a/eval_int8_tflite.py
+++ b/eval_int8_tflite.py
@@ -6,38 +6,27 @@ from torchvision import transforms
 # load the pre-built int8 tflite model from mit han lab
 tflite_path = "/Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite"
 
-# set up the tflite interpreter (this is what runs the model)
-interpreter = tf.lite.Interpreter(tflite_path)
-interpreter.allocate_tensors()
 
-# get input and output tensor info
-input_details = interpreter.get_input_details()
-output_details = interpreter.get_output_details()
-input_shape = input_details[0]['shape']
-resolution = input_shape[1]
+def make_transform(resolution):
+    # resize then center crop, same recipe as the float baseline
+    return transforms.Compose([
+        transforms.Resize(int(resolution * 256 / 224)),
+        transforms.CenterCrop(resolution),
+    ])
 
-transform = transforms.Compose([
-    transforms.Resize(int(resolution * 256 / 224)),
-    transforms.CenterCrop(resolution),
-])
 
+def iter_val_stream(total=50000):
+    # stream imagenet validation set, yields (index, PIL image, label).
+    # order is fixed by the dataset, so the same call always gives the same stream
+    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
+    for i, sample in enumerate(dataset):
+        if i >= total:
+            break
+        yield i, sample['image'], sample['label']
 
-print(f"model input shape: {input_shape}")
-print(f"resolution: {resolution}")
-
-# stream imagenet validation set (same 1000 samples as baseline)
-dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
-
-correct = 0
-total = 50000
-
-for i, sample in enumerate(dataset):
-    if i >= total:
-        break
-
-    img = sample['image']
-    label = sample['label']
 
+def preprocess_to_int8(img, transform, input_shape):
+    # PIL image -> int8 array shaped like the model input
     if img.mode != 'RGB':
         img = img.convert('RGB')
 
@@ -48,23 +37,50 @@ for i, sample in enumerate(dataset):
     # pixel values are 0-255, so we normalize to 0-1 then shift by -128
     img_np = np.array(img).astype(np.float32) / 255.0
     img_np = (img_np * 255 - 128).astype(np.int8)
-    img_np = img_np.reshape(input_shape)
+    return img_np.reshape(input_shape)
+
+
+def main():
+    # set up the tflite interpreter (this is what runs the model)
+    interpreter = tf.lite.Interpreter(tflite_path)
+    interpreter.allocate_tensors()
+
+    # get input and output tensor info
+    input_details = interpreter.get_input_details()
+    output_details = interpreter.get_output_details()
+    input_shape = input_details[0]['shape']
+    resolution = input_shape[1]
+
+    transform = make_transform(resolution)
+
+    print(f"model input shape: {input_shape}")
+    print(f"resolution: {resolution}")
+
+    correct = 0
+    total = 50000
+
+    for i, img, label in iter_val_stream(total):
+        img_np = preprocess_to_int8(img, transform, input_shape)
+
+        # run inference
+        interpreter.set_tensor(input_details[0]['index'], img_np)
+        interpreter.invoke()
+
+        # get prediction
+        output = interpreter.get_tensor(output_details[0]['index'])
+        pred = np.argmax(output)
 
-    # run inference
-    interpreter.set_tensor(input_details[0]['index'], img_np)
-    interpreter.invoke()
+        if pred == label:
+            correct += 1
 
-    # get prediction
-    output = interpreter.get_tensor(output_details[0]['index'])
-    pred = np.argmax(output)
+        if i % 100 == 0:
+            print(f"progress: {i}/{total} | running accuracy: {correct/(i+1)*100:.2f}%")
 
-    if pred == label:
-        correct += 1
+    print(f"\nmodel: mcunet-in3 (320kb sram / 1mb flash)")
+    print(f"quantization: int8 tflite (mit han lab official)")
+    print(f"final int8 accuracy on {total} samples: {correct/total*100:.2f}%")
+    print(f"baseline float32 accuracy: 62.14%")
 
-    if i % 100 == 0:
-        print(f"progress: {i}/{total} | running accuracy: {correct/(i+1)*100:.2f}%")
 
-print(f"\nmodel: mcunet-in3 (320kb sram / 1mb flash)")
-print(f"quantization: int8 tflite (mit han lab official)")
-print(f"final int8 accuracy on {total} samples: {correct/total*100:.2f}%")
-print(f"baseline float32 accuracy: 62.14%")
+if __name__ == "__main__":
+    main()
~~~~

## 3. The sheet's week 6 text

**Not in the repo.** Searched, case-insensitive: every tracked and untracked text file in the working tree (md, txt, py, c, h, csv; venv311 and subset/ excluded), and every blob in every commit on every ref (`git grep` over `git rev-list --all`, plus `git log --all -S"Verify intended"`). There are no PDFs in the repo.

- The exact string "Verify intended INT8": **zero matches** anywhere, in the working tree or in history.
- The only near match is a comment/header in `c/test_fi.c` ("intended int8 value change: all 256 values x 8 bits"), which is my own test title, not sheet text.
- "week 6" appears only in `fault_model_week5.md` (the "week 6 update" notes) and in two commit messages. None of those is a copy of the sheet row.

I do not have the sheet's wording, so I can't check the work against it.

## 4. Test inventory

Source: `c/test_fi.c`. "Last clean run" = the `make test` I ran in this session on 2026-10-15 (output below); it exited 0 with 99 passed, 0 failed, 0 skipped. Count column = number of `CHECK` assertions executed, summing to 99.

| # | check (test function) | asserts | cases | CHECKs | last run |
|---|---|---|---|---|---|
| 1 | splitmix64 vectors (`test_splitmix`) | 5 outputs for seed 1234567 equal reference values | 5 | 1 | pass |
| 2 | cell seed vectors (`test_splitmix`) | `fi_cell_seed` equals 3 fixed expected 64-bit values | 3 | 3 | pass |
| 3 | locator csv header (`test_locator`) | all 5 needed columns present in layer_inventory.csv | 5 columns | 1 | pass |
| 4 | locator per op (`test_locator`) | for every `injection_candidate==True` row: op index found, shape, op type and param count match the flatbuffer locator | 55 rows | 55 | pass |
| 5 | locator counts (`test_locator`) | csv lists 55 injectable ops; locator finds 55 layers; matched == expected | 3 | 3 | pass |
| 6 | value change, full table (`test_value_change`) | `fi_flipped_value` equals plain-arithmetic expectation for every int8 value x bit | 256 x 8 = 2048 | 1 | pass |
| 7 | **sign bit (bit 7), own check** (`test_value_change`) | for every value, bit 7 flip crosses zero and moves value by exactly 128 (-128 if v>=0, +128 if v<0) | 256 values | 1 | pass |
| 8 | magnitude bits (`test_value_change`) | bits 0-6 change value by exactly 2^bit and keep its sign | 7 x 256 = 1792 | 1 | pass |
| 9 | sign bit explicit examples (`test_value_change`) | 0->-128, 127->-1, -1->127, -128->0, 5->-123, -123->5 | 6 | 6 | pass |
| 10 | buffer-level flips (`test_value_change`) | writing each value into a real weight byte of op 120, `fi_apply` gives the expected stored byte, recorded orig/new, and `fi_restore` returns the original | 2048 | 1 | pass |
| 11 | buffer clean after sweep (`test_value_change`) | `fi_is_clean` after the 2048-flip sweep | 1 | 1 | pass |
| 12 | exactly one byte differs (`test_one_byte_and_restore`) | while faulted: 1 differing byte, not "clean" | 1320 | 1 | pass |
| 13 | right byte (`test_one_byte_and_restore`) | the byte at layer offset + element changed | 1320 | 1 | pass |
| 14 | right bit (`test_one_byte_and_restore`) | xor with clean byte equals exactly 1<<bit | 1320 | 1 | pass |
| 15 | restore (`test_one_byte_and_restore`) | after `fi_restore`: 0 differing bytes and clean | 1320 | 1 | pass |
| 16 | misuse rejected (`test_one_byte_and_restore`) | element out of range and bit 8 refused; buffer untouched | 3 | 3 | pass |
| 17 | second clean run (`test_runtime`) | clean run succeeds and gives identical logits to the first | 200 images | 2 | pass |
| 18 | zero-fault score (`test_runtime`) | all images masked, 0 subtle/catastrophic/due; faulted accuracy == clean accuracy | 200 images | 2 | pass |
| 19 | fault visible (`test_runtime`) | op 120 max-abs element bit 7 applies, and n_masked < n (weights are read from our buffer); counts add to n | 1 fault, 200 images | 3 | pass |
| 20 | restore gives clean logits (`test_runtime`) | restore ok, buffer clean, logits identical to clean | 200 images | 3 | pass |
| 21 | seed determinism (`test_runtime`) | same seed -> same element and byte-identical score struct; cell seed repeatable; 4 neighbouring cells get different seeds | 1 cell run twice | 4 | pass |
| 22 | persistent vs fresh interpreter (`test_inplace_vs_fresh`) | per layer: persistent and freshly built interpreter give identical logits on the faulted buffer; buffer clean after sweep | 110 faulted cases (55 layers x 2), 4 images each | 2 | pass |
| 23 | C vs python (`test_python_parity`) | C max abs logit diff and top-1 disagreement are no larger than the spread among python's own 3 kernel configs | 200 images x 3 configs | 2 | pass |

Total: 4 + 59 + 11 + 7 + 14 + 2 + 2 = 99 (splitmix 4, locator 59, value change 11, one-byte 7, runtime 14, in-place 2, parity 2). I counted these from the source; the run printed only the total.

Not covered by any test: the random element mode end to end, the csv writer / column order, the `due` path (cannot be triggered with int8 flips), corner-rule selection itself (max_abs etc.), and an independent check of masked/subtle/catastrophic classification against known expected counts (only the zero-fault all-masked case and a sum-to-n check).

### make test output (verbatim, run 2026-10-15)

~~~~
../venv311/bin/python ref_logits.py --subset ../subset --k 200 --model /Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite
/Users/jordiangel/Desktop/research/ee research/mcunet-fault-sensitivity/venv311/lib/python3.11/site-packages/tensorflow/lite/python/interpreter.py:457: UserWarning:     Warning: tf.lite.Interpreter is deprecated and is scheduled for deletion in
    TF 2.20. Please use the LiteRT interpreter from the ai_edge_litert package.
    See the [migration guide](https://ai.google.dev/edge/litert/migration)
    for details.
    
  warnings.warn(_INTERPRETER_DELETION_WARNING)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
wrote ../subset/ref_logits_default.bin: (200, 1000)
wrote ../subset/ref_logits_nodeleg.bin: (200, 1000)
wrote ../subset/ref_logits_ref.bin: (200, 1000)
./test_fi --model /Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite --subset ../subset --csv ../layer_inventory.csv --images 200
TfLiteVersion: 2.19.0
model: 1018208 bytes, 55 conv layers located

[splitmix64 reference vectors]

[locator vs layer_inventory.csv]
  locator: 55 layers, csv: 55 injectable, 55 matched (op index, type, shape, count)

[intended int8 value change: all 256 values x 8 bits]
  2048 (value, bit) pairs checked arithmetically and through the buffer

[exactly one byte differs while faulted; restore returns clean]
  1320 cases (55 layers x first/middle/last element x 8 bits)
clean run: 200 images, 4.3 ms/image

[zero faults / determinism / fault visible / restore / seed]
  zero faults: 200/200 masked, accuracy 130/200
  op 120 elem 22334 bit 7 (127 -> -1): masked=16 subtle=184 catastrophic=0 due=0 of 200
  seed 17404210285331851499 -> op 62 elem 10354 twice: scores identical (masked=0 subtle=195 cat=5)

[persistent interpreter sees flips in all 55 layers (vs fresh interpreter)]
  110 faulted cases over 55 layers: persistent == fresh in 110, flip changed the logits in 105 (4 images each)

[C runtime vs python tf.lite.Interpreter]
  python default (XNNPACK)   vs python no delegates       : bit-exact   0/200, top-1 differs 10, max |diff| 7
  python default (XNNPACK)   vs python reference kernels  : bit-exact   0/200, top-1 differs 10, max |diff| 8
  python no delegates        vs python reference kernels  : bit-exact   0/200, top-1 differs 9, max |diff| 7
  C runtime                  vs python default (XNNPACK)  : bit-exact   0/200, top-1 differs 10, max |diff| 8
  C runtime                  vs python no delegates       : bit-exact   0/200, top-1 differs 9, max |diff| 7
  C runtime                  vs python reference kernels  : bit-exact  70/200, top-1 differs 0, max |diff| 1
  subset accuracy (200 images): C 130, python default 131, no delegates 133, reference 130

==== 99 checks passed, 0 failed, 0 skipped ====

exit status: 0
~~~~

### make smoke output (verbatim, run 2026-10-15)

~~~~
./run_trials --model /Users/jordiangel/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite --subset ../subset --mode corner --layers 1,62,120 --bits 0,7 --limit 30 --images 20 --out trials_smoke.csv
mode=corner layers=3 bits=2 images=20 seed=2026 -> trials_smoke.csv
  20 trials, 1.8 s
done: 30 trials in 2.6 s, model buffer clean: yes
exit status: 0
~~~~

(`make test` also regenerates `subset/ref_logits_*.bin` via ref_logits.py; subset/ is gitignored. The smoke run overwrote the gitignored `c/trials_smoke.csv`. The tracked results/ files were not touched.)

## 5. Results

All tables below were recomputed in this session from `results/first_pass_corner.csv` with a throwaway script (not committed). **Comparison with earlier reports: I cannot say whether anything differs, because the numbers I reported earlier are not available in this session's context and this repo contains no earlier copy of these aggregate tables.** What I can say: the csv has 560 rows = 14 layers x 5 rules x 8 bits; row counts, `n_masked+n_subtle+n_catastrophic+n_due == n_images` (0 violations), `mismatch_rate == (cat+due)/n` (0 violations), and `new_val == orig_val with bit flipped` (0 violations) all hold; the figures match the log (560 trials, 525.4 s) and the spec's 130/200 = 0.65 clean accuracy.

### csv header

~~~~
trial_id,seed,op_index,op_type,stage,tensor,elem_idx,bit,orig_val,new_val,n_images,mismatch_rate,acc_faulted,acc_clean,n_masked,n_subtle,n_catastrophic,n_due,elem_rule
~~~~

### first 15 rows

~~~~
0,1072633581730028176,1,CONV_2D,stem,weight,391,0,126,127,200,0.040000,0.645000,0.650000,0,192,8,0,max_abs
1,1635523166987473384,1,CONV_2D,stem,weight,391,1,126,124,200,0.075000,0.650000,0.650000,0,185,15,0,max_abs
2,6828833762828906199,1,CONV_2D,stem,weight,391,2,126,122,200,0.065000,0.640000,0.650000,0,187,13,0,max_abs
3,3342511902841367514,1,CONV_2D,stem,weight,391,3,126,118,200,0.100000,0.650000,0.650000,0,180,20,0,max_abs
4,5206593807927925153,1,CONV_2D,stem,weight,391,4,126,110,200,0.180000,0.620000,0.650000,0,164,36,0,max_abs
5,3884384688863450241,1,CONV_2D,stem,weight,391,5,126,94,200,0.295000,0.575000,0.650000,0,141,59,0,max_abs
6,4086863498266995436,1,CONV_2D,stem,weight,391,6,126,62,200,0.520000,0.400000,0.650000,0,96,104,0,max_abs
7,17766090701463367350,1,CONV_2D,stem,weight,391,7,126,-2,200,0.640000,0.315000,0.650000,0,72,128,0,max_abs
8,14496589544018203799,1,CONV_2D,stem,weight,0,0,-1,-2,200,0.065000,0.650000,0.650000,0,187,13,0,min_nonzero_abs
9,10652776191592778466,1,CONV_2D,stem,weight,0,1,-1,-3,200,0.055000,0.645000,0.650000,0,189,11,0,min_nonzero_abs
10,15388055438921416265,1,CONV_2D,stem,weight,0,2,-1,-5,200,0.080000,0.675000,0.650000,0,184,16,0,min_nonzero_abs
11,15060502914887403428,1,CONV_2D,stem,weight,0,3,-1,-9,200,0.050000,0.645000,0.650000,0,190,10,0,min_nonzero_abs
12,48108898571796392,1,CONV_2D,stem,weight,0,4,-1,-17,200,0.070000,0.650000,0.650000,0,186,14,0,min_nonzero_abs
13,4687270348959600097,1,CONV_2D,stem,weight,0,5,-1,-33,200,0.110000,0.645000,0.650000,0,178,22,0,min_nonzero_abs
14,12012983118536039135,1,CONV_2D,stem,weight,0,6,-1,-65,200,0.140000,0.660000,0.650000,0,172,28,0,min_nonzero_abs
~~~~

### Aggregates

Percentages are of images (each trial has 200). "mean acc drop" = mean(acc_clean - acc_faulted), negative means the faulted run scored higher than clean. Every cell of layer x bit holds 5 trials (one per corner rule), not 1.

### outcome by bit

| bit | trials | images | masked % | subtle % | catastrophic % | due % | mean mismatch | mean acc drop | trials with >=1 catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 70 | 14000 | 38.74 | 59.11 | 2.15 | 0.00 | 0.0215 | -0.0014 | 49 |
| 1 | 70 | 14000 | 35.80 | 61.54 | 2.66 | 0.00 | 0.0266 | -0.0024 | 51 |
| 2 | 70 | 14000 | 33.03 | 64.06 | 2.91 | 0.00 | 0.0291 | -0.0029 | 49 |
| 3 | 70 | 14000 | 30.14 | 66.44 | 3.41 | 0.00 | 0.0341 | 0.0004 | 51 |
| 4 | 70 | 14000 | 26.92 | 68.49 | 4.59 | 0.00 | 0.0459 | 0.0028 | 51 |
| 5 | 70 | 14000 | 23.61 | 69.64 | 6.75 | 0.00 | 0.0675 | 0.0149 | 53 |
| 6 | 70 | 14000 | 20.70 | 69.19 | 10.11 | 0.00 | 0.1011 | 0.0309 | 53 |
| 7 | 70 | 14000 | 21.69 | 65.81 | 12.50 | 0.00 | 0.1250 | 0.0441 | 54 |
| ALL | 560 | 112000 | 28.83 | 65.53 | 5.64 | 0.00 | | | |

### outcome by layer (op_index, in spec step 6 order)

| op_index | trials | images | masked % | subtle % | catastrophic % | due % | mean mismatch | mean acc drop | trials with >=1 catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 8000 | 0.00 | 82.64 | 17.36 | 0.00 | 0.1736 | 0.0468 | 40 |
| 3 | 40 | 8000 | 0.00 | 73.33 | 26.68 | 0.00 | 0.2667 | 0.1028 | 40 |
| 9 | 40 | 8000 | 0.09 | 91.90 | 8.01 | 0.00 | 0.0801 | 0.0002 | 40 |
| 22 | 40 | 8000 | 0.00 | 94.61 | 5.39 | 0.00 | 0.0539 | -0.0033 | 40 |
| 36 | 40 | 8000 | 0.00 | 95.76 | 4.24 | 0.00 | 0.0424 | -0.0038 | 40 |
| 42 | 40 | 8000 | 20.00 | 71.60 | 8.40 | 0.00 | 0.0840 | 0.0095 | 32 |
| 56 | 40 | 8000 | 45.09 | 53.23 | 1.69 | 0.00 | 0.0169 | 0.0004 | 24 |
| 89 | 40 | 8000 | 58.45 | 40.84 | 0.71 | 0.00 | 0.0071 | -0.0008 | 25 |
| 60 | 40 | 8000 | 49.89 | 48.95 | 1.16 | 0.00 | 0.0116 | 0.0006 | 30 |
| 62 | 40 | 8000 | 38.16 | 59.26 | 2.58 | 0.00 | 0.0258 | 0.0000 | 31 |
| 64 | 40 | 8000 | 40.92 | 57.40 | 1.68 | 0.00 | 0.0168 | -0.0008 | 30 |
| 102 | 40 | 8000 | 32.33 | 66.75 | 0.93 | 0.00 | 0.0093 | 0.0001 | 32 |
| 116 | 40 | 8000 | 64.21 | 35.70 | 0.09 | 0.00 | 0.0009 | -0.0006 | 7 |
| 120 | 40 | 8000 | 54.49 | 45.51 | 0.00 | 0.00 | 0.0000 | 0.0000 | 0 |
| ALL | 560 | 112000 | 28.83 | 65.53 | 5.64 | 0.00 | | | |

### outcome by rule

| elem_rule | trials | images | masked % | subtle % | catastrophic % | due % | mean mismatch | mean acc drop | trials with >=1 catastrophic |
|---|---|---|---|---|---|---|---|---|---|
| max_abs | 112 | 22400 | 9.40 | 83.76 | 6.84 | 0.00 | 0.0684 | 0.0085 | 100 |
| min_nonzero_abs | 112 | 22400 | 21.55 | 74.17 | 4.28 | 0.00 | 0.0428 | 0.0018 | 91 |
| first_zero | 112 | 22400 | 32.74 | 60.94 | 6.33 | 0.00 | 0.0633 | 0.0171 | 82 |
| first_elem | 112 | 22400 | 26.71 | 68.59 | 4.70 | 0.00 | 0.0470 | 0.0052 | 90 |
| last_elem | 112 | 22400 | 53.76 | 40.21 | 6.03 | 0.00 | 0.0603 | 0.0215 | 48 |
| ALL | 560 | 112000 | 28.83 | 65.53 | 5.64 | 0.00 | | | |

### layer x bit catastrophic rate (% of images, pooled over the 5 corner rules = 5 trials x 200 images per cell; due counted with catastrophic, due is 0 everywhere)

| op | bit 0 | bit 1 | bit 2 | bit 3 | bit 4 | bit 5 | bit 6 | bit 7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.2 | 6.4 | 7.6 | 7.9 | 12.6 | 19.5 | 33.7 | 45.0 |
| 3 | 5.3 | 7.5 | 10.1 | 14.6 | 21.4 | 38.8 | 57.1 | 58.6 |
| 9 | 4.5 | 4.9 | 5.7 | 5.9 | 6.6 | 8.0 | 10.9 | 17.6 |
| 22 | 3.4 | 4.4 | 4.1 | 4.4 | 4.4 | 6.5 | 7.6 | 8.3 |
| 36 | 2.5 | 3.3 | 3.6 | 5.0 | 4.2 | 4.1 | 4.8 | 6.4 |
| 42 | 2.8 | 3.8 | 3.2 | 3.0 | 6.3 | 9.4 | 16.5 | 22.2 |
| 56 | 1.2 | 1.7 | 1.1 | 1.5 | 2.3 | 1.6 | 1.5 | 2.6 |
| 89 | 0.4 | 0.6 | 0.5 | 0.4 | 0.7 | 0.9 | 1.2 | 1.0 |
| 60 | 0.9 | 0.8 | 0.8 | 0.9 | 1.2 | 1.1 | 1.5 | 2.1 |
| 62 | 1.6 | 1.4 | 1.7 | 1.8 | 2.2 | 1.8 | 3.3 | 6.8 |
| 64 | 1.0 | 1.6 | 1.7 | 1.6 | 1.5 | 1.6 | 2.3 | 2.1 |
| 102 | 0.3 | 0.8 | 0.6 | 0.7 | 0.8 | 1.1 | 1.1 | 2.0 |
| 116 | 0.0 | 0.0 | 0.0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 |
| 120 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |


Other numbers from the same script: 112 (layer, bit) cells, 5 trials per cell; n_images = 200 on every row; acc_clean = 0.65 on every row; total due = 0.

Elements chosen per layer (rule: index), from the csv:

~~~~
1 [('first_elem', 0), ('first_zero', 29), ('last_elem', 431), ('max_abs', 391), ('min_nonzero_abs', 0)]
3 [('first_elem', 0), ('first_zero', 1), ('last_elem', 143), ('max_abs', 16), ('min_nonzero_abs', 3)]
9 [('first_elem', 0), ('first_zero', 8), ('last_elem', 1175), ('max_abs', 783), ('min_nonzero_abs', 0)]
22 [('first_elem', 0), ('first_zero', 6), ('last_elem', 3919), ('max_abs', 1963), ('min_nonzero_abs', 1)]
36 [('first_elem', 0), ('first_zero', 0), ('last_elem', 1999), ('max_abs', 1446), ('min_nonzero_abs', 2)]
42 [('first_elem', 0), ('first_zero', 12), ('last_elem', 2999), ('max_abs', 1487), ('min_nonzero_abs', 0)]
56 [('first_elem', 0), ('first_zero', 72), ('last_elem', 1079), ('max_abs', 531), ('min_nonzero_abs', 80)]
89 [('first_elem', 0), ('first_zero', 1), ('last_elem', 2159), ('max_abs', 1045), ('min_nonzero_abs', 3)]
60 [('first_elem', 0), ('first_zero', 8), ('last_elem', 9599), ('max_abs', 4452), ('min_nonzero_abs', 9)]
62 [('first_elem', 0), ('first_zero', 7), ('last_elem', 11759), ('max_abs', 5761), ('min_nonzero_abs', 2)]
64 [('first_elem', 0), ('first_zero', 19), ('last_elem', 9599), ('max_abs', 4008), ('min_nonzero_abs', 36)]
102 [('first_elem', 0), ('first_zero', 1), ('last_elem', 23519), ('max_abs', 11689), ('min_nonzero_abs', 6)]
116 [('first_elem', 0), ('first_zero', 33), ('last_elem', 23519), ('max_abs', 11521), ('min_nonzero_abs', 29)]
120 [('first_elem', 0), ('first_zero', 156), ('last_elem', 159999), ('max_abs', 22334), ('min_nonzero_abs', 9)]
~~~~

Note first_elem is index 0 in all 14 layers and never coincides with another rule's pick in this csv.

## 6. Params breakdown

Computed this session with `tflite` flatbuffer parsing on `~/.torch/mcunet/mcunet-320kb-1mb_imagenet.tflite` (every tensor whose buffer is non-empty = "constant tensor") and `torch.load` on `~/.torch/mcunet/mcunet-320kb-1mb_imagenet.pth` (state_dict).

### tflite constant tensors (165 total)

| role | dtype | tensors | elements |
|---|---|---|---|
| CONV_2D weight | INT8 | 37 | 616,624 |
| DEPTHWISE_CONV_2D weight | INT8 | 18 | 104,408 |
| CONV_2D bias | INT32 | 37 | 5,272 |
| DEPTHWISE_CONV_2D bias | INT32 | 18 | 3,416 |
| PAD paddings | INT32 | 54 | 432 |
| RESHAPE shape | INT32 | 1 | 2 |
| total | | 165 | 730,154 |

### pytorch checkpoint (326 state_dict keys)

| kind | tensors | elements |
|---|---|---|
| conv / fc weight | 55 | 721,032 |
| bias (54 BN beta + 1 classifier bias) | 55 | 8,688 |
| BN gamma | 54 | 7,688 |
| BN running_mean | 54 | 7,688 |
| BN running_var | 54 | 7,688 |
| BN num_batches_tracked | 54 | 54 |
| total | 326 | 752,838 |

### reconciliation

- weights: tflite 616,624 + 104,408 = 721,032 = checkpoint conv/fc weight 721,032. Also equals `param_count` summed over the 55 injection candidates in layer_inventory.csv and the 721,032 in architecture_week4.md.
- biases: tflite 5,272 + 3,416 = 8,688 = checkpoint bias/beta 8,688 (BN folded into conv bias at export; 37 + 18 = 55 tensors on both sides).
- tflite total: 721,032 + 8,688 + 432 (PAD) + 2 (RESHAPE) = 730,154.
- checkpoint total: 721,032 + 8,688 + 3 x 7,688 (gamma, mean, var = 23,064) + 54 = 752,838.
- trainable parameters (weights + bias/beta + gamma; excludes BN running stats and counters): 721,032 + 8,688 + 7,688 = 737,408, which is the ~0.74M that architecture_week4.md compares its 721,032 against.
- checkpoint minus tflite: 752,838 - 730,154 = 22,684. Checkpoint-only: gamma + running_mean + running_var + counters = 3 x 7,688 + 54 = 23,118. tflite-only: PAD 432 + RESHAPE 2 = 434. 23,118 - 434 = 22,684, so the two counts reconcile exactly; BN statistics have no separate tflite tensor (folded into weights and bias at export).

## 7. PyTorchFI comparison (best effort)

Reached github (raw.githubusercontent.com/pytorchfi/pytorchfi/master). Caveat: WebFetch returns a summary produced by a small model, not the raw file, so what follows is what that reading reported, and the quoted fragments are its quotes. I did not clone the repo or read the files line by line. Files read: `README.md`, `pytorchfi/core.py`, `pytorchfi/neuron_error_models.py`, `pytorchfi/weight_error_models.py`. (`pytorchfi/util.py`, where `random_value` lives, was not read.)

| aspect | PyTorchFI | this project (c/fault_inject.c, run_trials.c) |
|---|---|---|
| purpose (README) | "a runtime perturbation tool for deep neural networks (DNNs)", weights or neurons, PyTorch | single int8 weight bit flip on a tflite model, C api |
| layer choice | user gives an index, or `random_weight_location` / `random_neuron_location` with layer == -1 draws `random.randint(0, pfi.get_total_layers() - 1)` (weight_error_models.py, neuron_error_models.py); layer types default to `[nn.Conv2d]` (core.py) | explicit op index list (14 first-pass layers from the spec) over the 55 conv/depthwise ops found by a flatbuffer locator |
| element choice | uniform random per dimension: `k = random.randint(0, dim0_shape - 1)` and the same for c/h/w | corner rules (max abs, min nonzero abs, first zero, first, last), or seeded random in `--mode random` |
| bit choice | weight_error_models.py has **no bit flips**: random value in a range, or multiply by zero. `neuron_error_models.py` has `single_bit_flip_func` with `rand_bit = random.randint(0, self.bits - 1)` on activations (quantised to `self.bits`, default 8, flipped in two's complement) | all 8 bits swept deterministically, weights only |
| how it injects into weights | `copy.deepcopy` of the model, then `layer.weight[corrupt_idx] = corrupt_value` under `torch.no_grad()` on the copy (core.py `declare_weight_fault_injection`); neuron faults go through forward hooks | writes one flipped byte directly into the loaded tflite flatbuffer buffer the interpreter reads from |
| restore | no weight restore: the fault lives only in the deep copy; `reset_fault_injection` clears state and sets `corrupted_model = None` | xor the same bit back, then check buffer against a clean copy by fnv1a hash and memcmp (`fi_is_clean`) |
| seeding | none in the files read; no `random.seed`/`torch.manual_seed`, uses the global `random` module so reproducibility is up to the caller | splitmix64, per cell seed from (base seed, op, bit, k), logged per trial |

Source urls: https://raw.githubusercontent.com/pytorchfi/pytorchfi/master/README.md, .../pytorchfi/core.py, .../pytorchfi/neuron_error_models.py, .../pytorchfi/weight_error_models.py. (A guess at `github.com/mahmoodlab/pytorchfi` returned 404; it was the wrong org.)

## 8. Known gaps

Could not verify:
1. **Sheet wording.** The week 6 row ("Verify intended INT8 ...") is not in the repo or its history, so conformance to the sheet is unchecked.
2. **TFLite C library version.** `TfLiteVersion()` prints 2.19.0 at runtime. `c/BUILD_TFLITE.md` says that string is hard coded upstream and the source tag was v2.21.0. I confirmed the string in the dylib and read the note, but did not rebuild or otherwise verify the library really is 2.21.0 source.
3. **Reproducibility claims.** The spec says two full runs gave byte identical csvs. I did not rerun the 525 s first pass, so that is unverified by me. Same for the splitmix reference vectors being "from an independent python implementation": I did not regenerate them.
4. **Model identity.** The model file sha256 is "recorded" per the spec, but it is not in the csv or the log, and the subset manifest hashes are not written into the log either (the spec says so). The model lives outside the repo (`~/.torch/mcunet/`).
5. **Comparison with earlier reported aggregates** (section 5): not possible in this session.
6. **PyTorchFI section** is based on summarised fetches, not full file reads.

Differs from / unresolved against the week 5 spec:
7. **Corner rules are pending the advisor** (spec says so too). "Smallest nonzero |w|" and "first zero weight" are my reading; ties go to the lowest index.
8. **Trial count.** Spec step 7 says k ~ 30 trials per (layer, bit) cell and 14 x 8 x 30 = 3,360 trials. Corner mode is deterministic, so the first pass has 5 trials per layer x bit (one per rule), 560 total, no repeats, and no confidence intervals. Random mode was not run for the first pass.
9. **Per-trial seed in corner mode is irrelevant**: the element does not depend on it (spec says logged only).
10. **Hash check.** Spec step 4 says compare a hash of the model buffer; the code compares both fnv1a and a full memcmp against a clean copy, which is stronger but not the same.
11. **Scoring definitions** (c/fault_inject.c, `fi_score_run`): due = `invoke` failed; masked = all 1000 int8 logits identical to the C clean run; subtle = logits differ, argmax same; catastrophic = argmax differs from the clean C run (not "clean class out of top-5", which the spec leaves open). Things a reviewer should check: (a) logits are int8, so "masked" means identical after output quantisation, not identical pre-quantisation; (b) `fi_argmax` tie-breaking on int8 logits is not covered by a test; (c) a "catastrophic" image can be a flip from a wrong clean class to another wrong class, so catastrophic rate is not accuracy drop (the accuracy-drop column is also reported); (d) due images are not counted as correct, so `acc_faulted` can drop even if no logits were scored; due never occurred (0 of 112,000).
12. **Runtime differs from the python baseline.** The C runtime (XNNPACK and RUY off) is within 1 logit unit of python reference kernels but up to 8 from python default and no-delegate, with 9-10 of 200 top-1 flips. The 61.82% baseline is from python; the C subset accuracy is 130/200 = 65.0% (python default 131/200). Fault outcomes are relative to the C clean run, so this is self-consistent but not directly comparable to the python numbers.
13. **Stage labels** are my own op-index-range derivation, not read from the model.
14. **Subset of 200 images** is the first image of each new class in stream order (first 226 samples): not random, and classes with a late first appearance are absent, so it is not a uniform sample of ImageNet.

What I'd flag as a reviewer:
15. Op 120 (classifier) shows 0 catastrophic outcomes in all 40 trials, including bit 7 on its max-abs weight (127 -> -1 gave masked=16, subtle=184, catastrophic=0 in the test). That fits one weight touching one class logit, but it is also a case where the C runtime result is the only evidence; no independent check.
16. The "accuracy drop" is slightly negative on several rows/layers (e.g. ops 22, 36, 89), i.e. a flip improved accuracy on 200 images: noise at this sample size.
17. Early layers (ops 1, 3) are far more fragile than late ones (ops 1 and 3: 17.4% and 26.7% catastrophic; ops 89, 102, 116, 120: under 1%), which is confounded with depthwise vs pointwise and with kernel size in the 14 layer design, only partly controlled by the pairs.
18. `first_zero` and `min_nonzero_abs` rules fall on very different elements per layer (e.g. op 120 first_zero is element 156), so the rule names do not mean the same position class across layers; the five rules are not independent samples of a layer.
19. Weight-only, memory-fault framing: nothing here models compute faults; activation faults are not implemented.
20. The test suite exercises one model and one subset; the persistent-interpreter check used 4 images per fault and 2 faults per layer.
