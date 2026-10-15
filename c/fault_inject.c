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
