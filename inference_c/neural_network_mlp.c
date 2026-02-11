#include "neural_network_mlp.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

void softmax(float *x, int size) {
    // Softmax stable (soustraction du max pour éviter l'overflow)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Forward pass complet
void mlp_forward(MLP *mlp, const float *input, float *output_probs) {
    // Couche 1: RELU
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += mlp->W1[i][j] * input[j];
        }
        mlp->hidden1[i] = relu(sum);
    }
    
    // Couche 2 : RELU
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = mlp->b2[i];
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += mlp->W2[i][j] * mlp->hidden1[j];
        }
        mlp->hidden2[i] = relu(sum);
    }
    
    // Couche 3 : Linear puis softmax
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = mlp->b3[i];
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            sum += mlp->W3[i][j] * mlp->hidden2[j];
        }
        mlp->output[i] = sum;
    }
    
    // Softmax sur la sortie
    softmax(mlp->output, OUTPUT_SIZE);
    
}

// Prédiction : retourne l'indice de la classe avec la plus haute probabilité
int mlp_predict(MLP *mlp, const float *input) {
    mlp_forward(mlp, input, NULL);
    
    int best_class = 0;
    float best_prob = mlp->output[0];
    
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (mlp->output[i] > best_prob) {
            best_prob = mlp->output[i];
            best_class = i;
        }
    }
    
    return best_class;
}


bool mlp_load_weights(MLP *mlp, const char *prefix) {
    char filename[256];
    FILE *f;
    uint32_t rows, cols;
    
    // W1: 256 x 784
    snprintf(filename, sizeof(filename), "%s_W1.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(&cols, sizeof(uint32_t), 1, f);
    if (rows != HIDDEN1_SIZE || cols != INPUT_SIZE) {
        fprintf(stderr, "W1 dimensions mismatch: %ux%u vs %dx%d\n", 
                rows, cols, HIDDEN1_SIZE, INPUT_SIZE);
        fclose(f); return false;
    }
    fread(mlp->W1, sizeof(float), rows * cols, f);
    fclose(f);
    
    // b1: 256
    snprintf(filename, sizeof(filename), "%s_b1.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(mlp->b1, sizeof(float), rows, f);
    fclose(f);
    
    // W2: 128 x 256
    snprintf(filename, sizeof(filename), "%s_W2.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(&cols, sizeof(uint32_t), 1, f);
    if (rows != HIDDEN2_SIZE || cols != HIDDEN1_SIZE) {
        fprintf(stderr, "W2 dimensions mismatch\n");
        fclose(f); return false;
    }
    fread(mlp->W2, sizeof(float), rows * cols, f);
    fclose(f);
    
    // b2: 128
    snprintf(filename, sizeof(filename), "%s_b2.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(mlp->b2, sizeof(float), rows, f);
    fclose(f);
    
    // W3: 10 x 128
    snprintf(filename, sizeof(filename), "%s_W3.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(&cols, sizeof(uint32_t), 1, f);
    if (rows != OUTPUT_SIZE || cols != HIDDEN2_SIZE) {
        fprintf(stderr, "W3 dimensions mismatch\n");
        fclose(f); return false;
    }
    fread(mlp->W3, sizeof(float), rows * cols, f);
    fclose(f);
    
    // b3: 10
    snprintf(filename, sizeof(filename), "%s_b3.bin", prefix);
    f = fopen(filename, "rb");
    if (!f) { perror(filename); return false; }
    fread(&rows, sizeof(uint32_t), 1, f);
    fread(mlp->b3, sizeof(float), rows, f);
    fclose(f);
    
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <preprocessed_image.bin>\n", argv[0]);
        printf("Image format: 784 floats (28x28), déjà normalisées\n");
        return 1;
    }
    
    MLP mlp = {0};
    
    if (!mlp_load_weights(&mlp, "weights")) {
        fprintf(stderr, "Erreur chargement poids\n");
        return 1;
    }
    
    // Charger l'image prétraitée (784 floats)
    float input[INPUT_SIZE];
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("Erreur ouverture image");
        return 1;
    }
    
    size_t n = fread(input, sizeof(float), INPUT_SIZE, f);
    fclose(f);
    
    if (n != INPUT_SIZE) {
        fprintf(stderr, "Erreur: fichier trop court (%zu floats, attendu %d)\n", n, INPUT_SIZE);
        return 1;
    }
    
    // Inférence
    float probs[OUTPUT_SIZE];
    mlp_forward(&mlp, input, probs);
    int prediction = mlp_predict(&mlp, input);
    
    printf("Prédiction: %d\n", prediction);
    printf("Confiance: %.2f%%\n", probs[prediction] * 100);
    printf("\nDistribution:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  %d: %.4f %s\n", i, probs[i], (i == prediction) ? "<--" : "");
    }
    
    return 0;
}