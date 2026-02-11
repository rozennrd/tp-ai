#include "neural_network_cnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// ===== Required symbols for linking (must NOT be static) =====

int argmax10(const float v[10]) {
    int best = 0;
    for (int i = 1; i < 10; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

static void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; ++i) x[i] = relu(x[i]);
}

static void conv3x3_pad1(
    const float *in, int Cin, int H, int W,
    float *out, int Cout,
    const float *w,  // [Cout][Cin][3][3]
    const float *b   // [Cout]
) {
    for (int oc = 0; oc < Cout; ++oc) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float acc = b[oc];
                for (int ic = 0; ic < Cin; ++ic) {
                    const float *w_ic = w + (oc * Cin + ic) * 9;
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = y + ky - 1;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = x + kx - 1;
                            if (ix < 0 || ix >= W) continue;
                            float v = in[ic * H * W + iy * W + ix];
                            acc += v * w_ic[ky * 3 + kx];
                        }
                    }
                }
                out[oc * H * W + y * W + x] = acc;
            }
        }
    }
}

static void maxpool2x2(
    const float *in, int C, int H, int W,
    float *out
) {
    int H2 = H / 2, W2 = W / 2;
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H2; ++y) {
            for (int x = 0; x < W2; ++x) {
                int iy = 2*y, ix = 2*x;
                float m  = in[c*H*W + (iy)*W + (ix)];
                float v1 = in[c*H*W + (iy)*W + (ix+1)];
                float v2 = in[c*H*W + (iy+1)*W + (ix)];
                float v3 = in[c*H*W + (iy+1)*W + (ix+1)];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                out[c*H2*W2 + y*W2 + x] = m;
            }
        }
    }
}

static void linear(
    const float *x, int in_dim,
    float *y, int out_dim,
    const float *W,  // [out_dim][in_dim]
    const float *b
) {
    for (int o = 0; o < out_dim; ++o) {
        const float *wrow = W + o * in_dim;
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) acc += wrow[i] * x[i];
        y[o] = acc;
    }
}

// IMPORTANT: must NOT be static (main calls it)
void cnn_forward(const CNN *cnn, const float input[28][28], float logits[10]) {
    static float x0[1 * 28 * 28];
    static float c1[32 * 28 * 28];
    static float p1[32 * 14 * 14];
    static float c2[64 * 14 * 14];
    static float p2[64 * 7 * 7];
    static float flat[64 * 7 * 7];
    static float h1[256];

    for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
            x0[y*28 + x] = input[y][x];

    conv3x3_pad1(x0, 1, 28, 28, c1, 32, cnn->conv1_w, cnn->conv1_b);
    relu_inplace(c1, 32*28*28);
    maxpool2x2(c1, 32, 28, 28, p1);

    conv3x3_pad1(p1, 32, 14, 14, c2, 64, cnn->conv2_w, cnn->conv2_b);
    relu_inplace(c2, 64*14*14);
    maxpool2x2(c2, 64, 14, 14, p2);

    for (int i = 0; i < 64*7*7; ++i) flat[i] = p2[i];

    linear(flat, 3136, h1, 256, cnn->fc1_w, cnn->fc1_b);
    relu_inplace(h1, 256);

    linear(h1, 256, logits, 10, cnn->fc2_w, cnn->fc2_b);
}

static void free_all(CNN *cnn) {
    free(cnn->conv1_w); free(cnn->conv1_b);
    free(cnn->conv2_w); free(cnn->conv2_b);
    free(cnn->fc1_w);   free(cnn->fc1_b);
    free(cnn->fc2_w);   free(cnn->fc2_b);
    memset(cnn, 0, sizeof(*cnn));
}

void cnn_free(CNN *cnn) { free_all(cnn); }

static size_t prod_shape(const int *shape, int nd) {
    size_t n = 1;
    for (int i = 0; i < nd; ++i) n *= (size_t)shape[i];
    return n;
}

static bool parse_shape_line(const char *line, int *shape, int *ndims_out) {
    // line: "# shape: 32 1 3 3"
    const char *p = strstr(line, ":");
    if (!p) return false;
    p++; // after ':'
    int nd = 0;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (nd >= 8) return false;
        shape[nd] = (int)strtol(p, (char**)&p, 10);
        nd++;
    }
    *ndims_out = nd;
    return nd > 0;
}

static bool want_tensor(const char *name,
                        const char **out_key,
                        int *out_shape, int *out_ndims) {
    // Map PyTorch state_dict names -> our buffers + expected shapes
    if (strcmp(name, "features.0.weight") == 0) {
        *out_key = "conv1_w";
        int s[] = {32,1,3,3}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 4; return true;
    }
    if (strcmp(name, "features.0.bias") == 0) {
        *out_key = "conv1_b";
        int s[] = {32}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 1; return true;
    }
    if (strcmp(name, "features.3.weight") == 0) {
        *out_key = "conv2_w";
        int s[] = {64,32,3,3}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 4; return true;
    }
    if (strcmp(name, "features.3.bias") == 0) {
        *out_key = "conv2_b";
        int s[] = {64}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 1; return true;
    }
    if (strcmp(name, "classifier.1.weight") == 0) {
        *out_key = "fc1_w";
        int s[] = {256,3136}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 2; return true;
    }
    if (strcmp(name, "classifier.1.bias") == 0) {
        *out_key = "fc1_b";
        int s[] = {256}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 1; return true;
    }
    if (strcmp(name, "classifier.4.weight") == 0) {
        *out_key = "fc2_w";
        int s[] = {10,256}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 2; return true;
    }
    if (strcmp(name, "classifier.4.bias") == 0) {
        *out_key = "fc2_b";
        int s[] = {10}; memcpy(out_shape, s, sizeof(s)); *out_ndims = 1; return true;
    }
    return false;
}

static float **buffer_ptr(CNN *cnn, const char *key) {
    if (strcmp(key,"conv1_w")==0) return &cnn->conv1_w;
    if (strcmp(key,"conv1_b")==0) return &cnn->conv1_b;
    if (strcmp(key,"conv2_w")==0) return &cnn->conv2_w;
    if (strcmp(key,"conv2_b")==0) return &cnn->conv2_b;
    if (strcmp(key,"fc1_w")==0)   return &cnn->fc1_w;
    if (strcmp(key,"fc1_b")==0)   return &cnn->fc1_b;
    if (strcmp(key,"fc2_w")==0)   return &cnn->fc2_w;
    if (strcmp(key,"fc2_b")==0)   return &cnn->fc2_b;
    return NULL;
}

bool cnn_load_weights_from_txt(CNN *cnn, const char *weights_txt_path) {
    memset(cnn, 0, sizeof(*cnn));

    FILE *f = fopen(weights_txt_path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s: %s\n", weights_txt_path, strerror(errno));
        return false;
    }

    char line[4096];
    char cur_name[256] = {0};
    int cur_shape[8] = {0};
    int cur_ndims = 0;

    const char *key = NULL;
    int exp_shape[8] = {0};
    int exp_ndims = 0;
    size_t exp_n = 0;
    float *buf = NULL;
    size_t idx = 0;
    bool reading_values = false;

    while (fgets(line, sizeof(line), f)) {
        // trim newline
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;

        if (strncmp(line, "# name:", 7) == 0) {
            // if we were reading a tensor, ensure completed
            if (reading_values) {
                if (idx != exp_n) {
                    fprintf(stderr, "Tensor %s incomplete: got %zu/%zu values\n", cur_name, idx, exp_n);
                    fclose(f); free_all(cnn); return false;
                }
                reading_values = false;
                buf = NULL;
                idx = 0;
            }

            // parse name
            const char *p = line + 7;
            while (*p == ' ' || *p == '\t') p++;
            strncpy(cur_name, p, sizeof(cur_name)-1);
            cur_name[sizeof(cur_name)-1] = 0;

            // decide if we care about it
            key = NULL;
            if (want_tensor(cur_name, &key, exp_shape, &exp_ndims)) {
                // we will allocate after we read shape line
            } else {
                key = NULL; // ignore this tensor
            }
            continue;
        }

        if (strncmp(line, "# shape:", 8) == 0) {
            if (!parse_shape_line(line, cur_shape, &cur_ndims)) {
                fprintf(stderr, "Bad shape line for tensor %s\n", cur_name);
                fclose(f); free_all(cnn); return false;
            }

            if (!key) {
                // ignore tensor -> skip its values
                reading_values = true;
                buf = NULL;
                idx = 0;
                exp_n = 0;
                continue;
            }

            // verify shape matches expected
            if (cur_ndims != exp_ndims) {
                fprintf(stderr, "Dims mismatch for %s: got %d dims, expected %d\n",
                        cur_name, cur_ndims, exp_ndims);
                fclose(f); free_all(cnn); return false;
            }
            for (int i = 0; i < exp_ndims; ++i) {
                if (cur_shape[i] != exp_shape[i]) {
                    fprintf(stderr, "Shape mismatch for %s at dim %d: got %d expected %d\n",
                            cur_name, i, cur_shape[i], exp_shape[i]);
                    fclose(f); free_all(cnn); return false;
                }
            }

            exp_n = prod_shape(exp_shape, exp_ndims);

            float **dst = buffer_ptr(cnn, key);
            if (!dst) {
                fprintf(stderr, "Internal error: unknown key %s\n", key);
                fclose(f); free_all(cnn); return false;
            }

            *dst = (float*)malloc(exp_n * sizeof(float));
            if (!*dst) {
                fprintf(stderr, "OOM allocating %s (%zu floats)\n", key, exp_n);
                fclose(f); free_all(cnn); return false;
            }
            buf = *dst;
            idx = 0;
            reading_values = true;
            continue;
        }

        // values / blank lines
        if (reading_values) {
            if (line[0] == 0) {
                // end of tensor block
                if (key && idx != exp_n) {
                    fprintf(stderr, "Tensor %s incomplete: got %zu/%zu values\n", cur_name, idx, exp_n);
                    fclose(f); free_all(cnn); return false;
                }
                reading_values = false;
                buf = NULL;
                idx = 0;
                continue;
            }

            if (key && buf) {
                // parse floats from this line
                char *p = line;
                while (*p) {
                    while (*p == ' ' || *p == '\t') p++;
                    if (!*p) break;
                    if (idx >= exp_n) {
                        fprintf(stderr, "Tensor %s: too many values\n", cur_name);
                        fclose(f); free_all(cnn); return false;
                    }
                    buf[idx++] = strtof(p, &p);
                }
            } else {
                // ignored tensor: do nothing
            }
        }
    }

    // finalize last tensor if file didn't end with blank line
    if (reading_values && key && idx != exp_n) {
        fprintf(stderr, "Tensor %s incomplete at EOF: got %zu/%zu values\n", cur_name, idx, exp_n);
        fclose(f); free_all(cnn); return false;
    }

    fclose(f);

    // sanity: ensure all required pointers exist
    if (!cnn->conv1_w || !cnn->conv1_b || !cnn->conv2_w || !cnn->conv2_b ||
        !cnn->fc1_w || !cnn->fc1_b || !cnn->fc2_w || !cnn->fc2_b) {
        fprintf(stderr, "Missing one or more required tensors in %s\n", weights_txt_path);
        free_all(cnn);
        return false;
    }

    return true;
}
