#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#include "neural_network_cnn.h"
#include "bmp_loader.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s <bmp_28x28_path> [--no-invert]\n"
        "  %s --watch <latest.bmp> [--fps N] [--no-invert]\n\n"
        "Examples:\n"
        "  %s data/custom_digits/2-0.bmp\n"
        "  %s --watch $HOME/mnist_live/latest.bmp --fps 10\n",
        prog, prog, prog, prog
    );
}

static int get_mtime(const char *path, time_t *out_mtime) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    *out_mtime = st.st_mtime;
    return 1;
}

static int is_stable_file(const char *path, int wait_ms) {
    struct stat a, b;
    if (stat(path, &a) != 0) return 0;
    usleep((useconds_t)(wait_ms * 1000));
    if (stat(path, &b) != 0) return 0;
    return (a.st_size == b.st_size) && (a.st_mtime == b.st_mtime);
}

static void print_logits(const float logits[10]) {
    printf("Logits:\n");
    for (int i = 0; i < 10; ++i) printf("  %d: %.5f\n", i, logits[i]);
}

int main(int argc, char **argv) {
    const char *weights_path = "models/cnn_weights.txt";

    if (argc < 2) { usage(argv[0]); return 1; }

    // flags
    int watch_mode = 0;
    const char *bmp_path = NULL;
    int fps = 10;
    int invert = 1;

    // parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--watch") == 0) {
            watch_mode = 1;
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            bmp_path = argv[++i];
        } else if (strcmp(argv[i], "--fps") == 0) {
            if (i + 1 >= argc) { usage(argv[0]); return 1; }
            fps = atoi(argv[++i]);
            if (fps <= 0) fps = 10;
        } else if (strcmp(argv[i], "--no-invert") == 0) {
            invert = 0;
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 1;
        } else {
            bmp_path = argv[i];
        }
    }

    if (!bmp_path) { usage(argv[0]); return 1; }

    // load weights once
    CNN cnn = (CNN){0};
    if (!cnn_load_weights_from_txt(&cnn, weights_path)) {
        fprintf(stderr, "Failed to load CNN weights from %s\n", weights_path);
        return 1;
    }

    // single-shot mode
    if (!watch_mode) {
        float input[28][28];
        if (!load_bmp_28x28_to_float(bmp_path, input, invert)) {
            fprintf(stderr, "Failed to load BMP: %s\n", bmp_path);
            cnn_free(&cnn);
            return 1;
        }

        float logits[10] = {0};
        cnn_forward(&cnn, input, logits);

        print_logits(logits);
        printf("Prediction = %d\n", argmax10(logits));

        cnn_free(&cnn);
        return 0;
    }

    // watch mode
    printf("[INFO] Watch mode on: %s (fps=%d, invert=%d)\n", bmp_path, fps, invert);
    time_t last_mtime = 0;
    int sleep_us = (int)(1000000 / fps);
    if (sleep_us < 1000) sleep_us = 1000;

    while (1) {
        time_t mt = 0;
        if (!get_mtime(bmp_path, &mt)) {
            // file not created yet
            usleep((useconds_t)sleep_us);
            continue;
        }

        if (mt != last_mtime) {
            // wait a tiny bit to avoid reading during write
            if (!is_stable_file(bmp_path, 20)) {
                usleep((useconds_t)sleep_us);
                continue;
            }

            float input[28][28];
            if (load_bmp_28x28_to_float(bmp_path, input, invert)) {
                float logits[10] = {0};
                cnn_forward(&cnn, input, logits);
                int pred = argmax10(logits);

                // compact live output
                printf("pred=%d | logits[0..9]= ", pred);
                for (int i = 0; i < 10; ++i) printf("%.3f ", logits[i]);
                printf("\n");
                fflush(stdout);

                last_mtime = mt;
            } else {
                // could be transient if file is being rewritten
            }
        }

        usleep((useconds_t)sleep_us);
    }

}