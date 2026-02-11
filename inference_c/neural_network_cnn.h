#ifndef NEURAL_NETWORK_CNN_H
#define NEURAL_NETWORK_CNN_H

#include <stdbool.h>

typedef struct {
    float *conv1_w; // 32*1*3*3
    float *conv1_b; // 32
    float *conv2_w; // 64*32*3*3
    float *conv2_b; // 64
    float *fc1_w;   // 256*3136
    float *fc1_b;   // 256
    float *fc2_w;   // 10*256
    float *fc2_b;   // 10
} CNN;

bool cnn_load_weights_from_txt(CNN *cnn, const char *weights_txt_path);
void cnn_free(CNN *cnn);
void cnn_forward(const CNN *cnn, const float input[28][28], float logits[10]);
int argmax10(const float v[10]);

#endif
