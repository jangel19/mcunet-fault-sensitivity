/* Separate tracing frontend. Uses the unchanged injector and C runner.
 * Usage: capture_activations MODEL SUBSET IMAGES OUTDIR OP ELEM BIT
 * OP=-1 selects a clean run. OUTDIR must already exist and be empty.
 * Output 0 is the original final vector. Additional outputs in a tracing
 * FlatBuffer are retained by TFLite's memory planner, then copied after Invoke.
 */
#include "fault_inject.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

static long number(const char *s)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !*s || *end) { fprintf(stderr, "invalid integer: %s\n", s); exit(2); }
    return v;
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr, "usage: capture_activations MODEL SUBSET IMAGES OUTDIR OP ELEM BIT\n");
        return 2;
    }
    Dl_info library;
    if (!dladdr((void *)TfLiteVersion, &library)) return 2;
    fprintf(stderr, "loaded_library=%s\n", library.dli_fname);
    long images = number(argv[3]), op = number(argv[5]);
    long elem = number(argv[6]), bit = number(argv[7]);
    if (images < 1 || images > 200 || op < -1 || op > 121 || elem < 0 || bit < 0 || bit > 7)
        return 2;
    fi_model_t m;
    fi_runner_t r;
    fi_subset_t s;
    if (fi_model_load(&m, argv[1]) || fi_runner_init(&r, &m) ||
        fi_subset_load(&s, argv[2], (int)images, r.in_bytes)) return 2;
    if (s.n != images) return 2;
    fi_fault_t fault = {0};
    if (op >= 0) {
        const fi_layer_t *layer = fi_find_layer(&m, (int)op);
        if (!layer || fi_apply(&m, layer, (size_t)elem, (int)bit, &fault)) return 2;
    }
    int count = TfLiteInterpreterGetOutputTensorCount(r.interp);
    FILE **files = calloc((size_t)count, sizeof *files);
    if (!files) return 2;
    char path[4096];
    for (int j = 0; j < count; j++) {
        const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(r.interp, j);
        if (TfLiteTensorType(t) != kTfLiteInt8) return 2;
        snprintf(path, sizeof path, "%s/output_%03d.bin", argv[4], j);
        files[j] = fopen(path, "wbx");
        if (!files[j]) { perror(path); return 2; }
    }
    snprintf(path, sizeof path, "%s/predictions.csv", argv[4]);
    FILE *pred = fopen(path, "wx");
    if (!pred) return 2;
    fprintf(pred, "image_index,label,prediction\n");
    int correct = 0;
    int8_t *first = malloc((size_t)s.n * FI_NUM_CLASSES);
    if (!first) return 2;
    for (int i = 0; i < s.n; i++) {
        int8_t *logits = first + (size_t)i * FI_NUM_CLASSES;
        if (fi_run_image(&r, s.images + (size_t)i * s.img_bytes, logits)) return 2;
        int top = fi_argmax(logits);
        correct += top == s.labels[i];
        fprintf(pred, "%d,%d,%d\n", i, s.labels[i], top);
        for (int j = 0; j < count; j++) {
            const TfLiteTensor *t = TfLiteInterpreterGetOutputTensor(r.interp, j);
            size_t bytes = TfLiteTensorByteSize(t);
            void *copy = malloc(bytes);
            if (!copy || TfLiteTensorCopyToBuffer(t, copy, bytes) != kTfLiteOk ||
                fwrite(copy, 1, bytes, files[j]) != bytes) return 2;
            free(copy);
        }
    }
    if (fclose(pred)) return 2;
    for (int j = 0; j < count; j++) if (fclose(files[j])) return 2;
    free(files);
    if (fault.active && fi_restore(&m, &fault)) return 2;
    if (!fi_is_clean(&m)) return 2;
    int8_t *restored = malloc((size_t)s.n * FI_NUM_CLASSES);
    if (!restored || fi_clean_run(&r, &s, restored)) return 2;
    if (op < 0 && memcmp(first, restored, (size_t)s.n * FI_NUM_CLASSES)) return 2;
    snprintf(path, sizeof path, "%s/restored_logits.bin", argv[4]);
    FILE *restore_file = fopen(path, "wbx");
    if (!restore_file || fwrite(restored, FI_NUM_CLASSES, (size_t)s.n, restore_file) != (size_t)s.n ||
        fclose(restore_file)) return 2;
    printf("{\"library_version\":\"%s\",\"images\":%d,\"correct\":%d,"
           "\"outputs\":%d,\"model_restored\":true,\"original_weight\":%d,\"faulted_weight\":%d}\n",
           TfLiteVersion(), s.n, correct, count, fault.orig_val, fault.new_val);
    free(first);
    free(restored);
    fi_subset_free(&s);
    fi_runner_free(&r);
    fi_model_free(&m);
    return 0;
}
