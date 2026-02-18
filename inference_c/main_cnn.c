#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#include "neural_network_cnn.h"
#include "neural_network_mlp.h"
#include "bmp_loader.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s (-c|-m) <bmp_28x28_path> [--no-invert]\n"
        "  %s (-c|-m) --watch <latest.bmp> [--fps N] [--no-invert]\n\n"
        "Examples:\n"
        "  %s -c data/custom_digits/2-0.bmp\n"
        "  %s -m --watch data/custom_digits/2-0.bmp --fps 10\n",
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

static int argmax_local_10(const float v[10]) {
    int best = 0;
    for (int i = 1; i < 10; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(int argc, char **argv) {
    const char *cnn_weights_path = "models/cnn_weights.txt";
    const char *mlp_weights_path = "models/mlp_weights.txt";

    if (argc < 2) { usage(argv[0]); return 1; }

    // flags
    int model_cnn = 0;
    int model_mlp = 0;
    int watch_mode = 0;
    const char *bmp_path = NULL;
    int fps = 10;
    int invert = 1;

    // parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) {
            model_cnn = 1;
        } else if (strcmp(argv[i], "-m") == 0) {
            model_mlp = 1;
        } else if (strcmp(argv[i], "--watch") == 0) {
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

    if ((model_cnn + model_mlp) != 1) {
        fprintf(stderr, "You must choose exactly one model: -c (CNN) or -m (MLP).\n");
        usage(argv[0]);
        return 1;
    }

    if (!bmp_path) { usage(argv[0]); return 1; }

    // load chosen model once
    CNN cnn = (CNN){0};
    MLP mlp = (MLP){0};
    if (model_cnn) {
        if (!cnn_load_weights_from_txt(&cnn, cnn_weights_path)) {
            fprintf(stderr, "Failed to load CNN weights from %s\n", cnn_weights_path);
            return 1;
        }
    } else {
        if (!mlp_load_weights(&mlp, mlp_weights_path)) {
            fprintf(stderr, "Failed to load MLP weights from %s\n", mlp_weights_path);
            return 1;
        }
    }

    // single-shot mode
    if (!watch_mode) {
        float input[28][28];
        if (!load_bmp_28x28_to_float(bmp_path, input, invert)) {
            fprintf(stderr, "Failed to load BMP: %s\n", bmp_path);
            if (model_cnn) cnn_free(&cnn);
            return 1;
        }

        float scores[10] = {0};
        if (model_cnn) {
            cnn_forward(&cnn, input, scores);
        } else {
            float mlp_input[INPUT_SIZE];
            for (int y = 0; y < 28; ++y)
                for (int x = 0; x < 28; ++x)
                    mlp_input[y * 28 + x] = input[y][x];
            mlp_forward(&mlp, mlp_input, NULL);
            for (int i = 0; i < 10; ++i) scores[i] = mlp.output[i];
        }

        print_logits(scores);
        printf("Prediction = %d\n", model_cnn ? argmax10(scores) : argmax_local_10(scores));

        if (model_cnn) cnn_free(&cnn);
        return 0;
    }

    // watch mode
    printf("[INFO] Watch mode on (%s): %s (fps=%d, invert=%d)\n",
           model_cnn ? "CNN" : "MLP", bmp_path, fps, invert);
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
                float scores[10] = {0};
                int pred = 0;

                if (model_cnn) {
                    cnn_forward(&cnn, input, scores);
                    pred = argmax10(scores);
                } else {
                    float mlp_input[INPUT_SIZE];
                    for (int y = 0; y < 28; ++y)
                        for (int x = 0; x < 28; ++x)
                            mlp_input[y * 28 + x] = input[y][x];
                    mlp_forward(&mlp, mlp_input, NULL);
                    for (int i = 0; i < 10; ++i) scores[i] = mlp.output[i];
                    pred = argmax_local_10(scores);
                }

                // compact live output
                printf("pred=%d | logits[0..9]= ", pred);
                for (int i = 0; i < 10; ++i) printf("%.3f ", scores[i]);
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