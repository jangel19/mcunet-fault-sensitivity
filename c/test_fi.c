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
