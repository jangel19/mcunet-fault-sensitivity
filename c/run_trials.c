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
