#ifndef MLP_H
#define MLP_H


#include <stdint.h>
#include <stdbool.h>

// Dimensions
#define INPUT_SIZE   784   // 28*28
#define HIDDEN1_SIZE 256
#define HIDDEN2_SIZE 128
#define OUTPUT_SIZE  10

// Structure du MLP
typedef struct {
    // Poids et biais
    float W1[HIDDEN1_SIZE][INPUT_SIZE];   // 256 x 784
    float b1[HIDDEN1_SIZE];               // 256
    
    float W2[HIDDEN2_SIZE][HIDDEN1_SIZE]; // 128 x 256
    float b2[HIDDEN2_SIZE];               // 128
    
    float W3[OUTPUT_SIZE][HIDDEN2_SIZE];  // 10 x 128
    float b3[OUTPUT_SIZE];                // 10
    
    // Buffers pour les activations
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output[OUTPUT_SIZE];
} MLP;

// Fonctions
bool mlp_load_weights(MLP *mlp, const char *weights_path);
void mlp_forward(MLP *mlp, const float *input, float *output_probs);
int mlp_predict(MLP *mlp, const float *input); // retourne la classe prédite (0-9)

// Fonctions d'activation
float relu(float x);
void softmax(float *x, int size);

#endif