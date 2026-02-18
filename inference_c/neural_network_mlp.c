#include "neural_network_mlp.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

// Structure d'en-tête BMP
#pragma pack(push, 1)
typedef struct {
    uint16_t type;              // "BM" = 0x4D42
    uint32_t file_size;         // Taille fichier
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;            // Offset des données pixels
    uint32_t header_size;       // Taille de cet en-tête (40)
    int32_t width;              // Largeur
    int32_t height;             // Hauteur
    uint16_t planes;            // 1
    uint16_t bpp;               // Bits par pixel (8 ou 24)
    uint32_t compression;       // 0 = non compressé
    uint32_t image_size;        // Taille des données
    int32_t x_pixels_per_meter;
    int32_t y_pixels_per_meter;
    uint32_t colors_used;
    uint32_t important_colors;
} BMPHeader;
#pragma pack(pop)

// Charger une image BMP 28x28 et convertir en floats normalisés [0,1]
bool load_bmp_image(const char *filepath, float *output) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("Erreur ouverture BMP");
        return false;
    }
    
    BMPHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fprintf(stderr, "Erreur: impossible de lire l'en-tête BMP\n");
        fclose(f);
        return false;
    }
    
    // Vérifier signature BMP
    if (header.type != 0x4D42) {
        fprintf(stderr, "Erreur: fichier n'est pas un BMP valide (signature: 0x%04X)\n", header.type);
        fclose(f);
        return false;
    }
    
    // Vérifier dimensions (doivent être 28x28)
    if (header.width != 28 || abs(header.height) != 28) {
        fprintf(stderr, "Erreur: dimensions BMP %dx%d, attendu 28x28\n", header.width, abs(header.height));
        fclose(f);
        return false;
    }
    
    // Vérifier format (8-bit grayscale ou 24-bit RGB)
    if (header.bpp != 8 && header.bpp != 24) {
        fprintf(stderr, "Erreur: BMP doit être 8-bit ou 24-bit, trouvé %d-bit\n", header.bpp);
        fclose(f);
        return false;
    }
    
    // Positionner au début des données pixels
    // L'offset dans le header pointe directement sur les pixels
    fseek(f, header.offset, SEEK_SET);
    
    int width = header.width;
    int height = abs(header.height);
    int row_padding = (4 - (width * (header.bpp / 8)) % 4) % 4;
    int is_bottom_up = header.height > 0;  // BMP standard = bottom-up
    
    uint8_t *row = malloc(width * (header.bpp / 8));
    if (!row) {
        fprintf(stderr, "Erreur: allocation mémoire\n");
        fclose(f);
        return false;
    }
    
    for (int y = 0; y < height; y++) {
        int read_y = is_bottom_up ? (height - 1 - y) : y;
        
        if (fread(row, 1, width * (header.bpp / 8), f) != width * (header.bpp / 8)) {
            fprintf(stderr, "Erreur: lecture des pixels ligne %d\n", y);
            free(row);
            fclose(f);
            return false;
        }
        
        // Sauter le padding
        if (row_padding > 0) {
            fseek(f, row_padding, SEEK_CUR);
        }
        
        for (int x = 0; x < width; x++) {
            float pixel_value;
            if (header.bpp == 8) {
                // 8-bit grayscale
                // Inverser pour MNIST: fond blanc (255) -> noir (0), chiffre noir -> blanc
                pixel_value = (255.0f - row[x]) / 255.0f;
            } else {
                // 24-bit RGB: convertir en grayscale
                int idx = x * 3;
                // Formule standard: 0.299*R + 0.587*G + 0.114*B
                float gray = 0.299f * row[idx + 2] + 0.587f * row[idx + 1] + 0.114f * row[idx];
                // Inverser pour MNIST
                pixel_value = (255.0f - gray) / 255.0f;
            }
            output[read_y * width + x] = pixel_value;
        }
    }
    
    free(row);
    fclose(f);
    printf("Image BMP chargée: %dx%d, %d-bit (offset=%d)\n", width, height, header.bpp, header.offset);

    
    return true;
}

// Vérifier si le fichier est un BMP
bool is_bmp_file(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    
    uint16_t signature;
    bool result = (fread(&signature, 2, 1, f) == 1 && signature == 0x4D42);
    fclose(f);
    return result;
}

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
    float sum0 = mlp->b1[0];
    for (int j = 0; j < INPUT_SIZE; j++) {
        sum0 += mlp->W1[0][j] * input[j];
    }
    int count_nonzero = 0;
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        float sum = mlp->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += mlp->W1[i][j] * input[j];
        }
        mlp->hidden1[i] = relu(sum);
        if (mlp->hidden1[i] > 0) count_nonzero++;
    }

    // Couche 2 : RELU
    int count_h2 = 0;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        float sum = mlp->b2[i];
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            sum += mlp->W2[i][j] * mlp->hidden1[j];
        }
        mlp->hidden2[i] = relu(sum);
        if (mlp->hidden2[i] > 0) count_h2++;
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


// Fonction utilitaire : lire une ligne et ignorer les commentaires
// Cette version gère plusieurs valeurs par ligne (format: 16 floats par ligne)
static int read_next_value(FILE *f, float *value) {
    static char line[4096];  // Buffer plus grand pour les lignes longues
    static char *ptr = NULL;
    static int has_data = 0;
    
    // Si on a encore des données dans la ligne courante, les parser
    if (has_data && ptr != NULL && *ptr != '\0' && *ptr != '\n') {
        // Chercher le prochain nombre
        while (*ptr && (*ptr == ' ' || *ptr == '\t')) ptr++;
        
        if (*ptr && *ptr != '\n' && *ptr != '#') {
            char *end;
            float val = strtof(ptr, &end);
            if (ptr != end) {  // Conversion réussie
                *value = val;
                ptr = end;
                return 1;
            }
        }
    }
    
    // Lire une nouvelle ligne
    while (fgets(line, sizeof(line), f)) {
        ptr = line;
        
        // Ignorer les espaces au début
        while (*ptr == ' ' || *ptr == '\t') ptr++;
        
        // Ignorer les lignes vides et commentaires
        if (*ptr == '\0' || *ptr == '\n' || *ptr == '#') {
            continue;
        }
        
        // Parser le premier nombre de la ligne
        char *end;
        float val = strtof(ptr, &end);
        if (ptr != end) {  // Conversion réussie
            *value = val;
            ptr = end;
            has_data = 1;
            return 1;
        }
    }
    
    return 0; // Plus de données
}

bool mlp_load_weights(MLP *mlp, const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        perror("Erreur ouverture fichier");
        return false;
    }
    
    float val;
    int count;
    
    // IMPORTANT: Les poids Keras sont stockés transposés
    // Keras: W[input][output] -> C: W[output][input]
    // Donc on doit transposer pendant le chargement
    
    // W1 : Keras [784, 256] -> C [256][784]
    printf("Chargement W1 (256x784) avec transposition...\n");
    count = 0;
    // D'abord lire tous les poids dans un buffer temporaire
    float W1_temp[INPUT_SIZE][HIDDEN1_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            if (!read_next_value(f, &val)) {
                fprintf(stderr, "Erreur: pas assez de valeurs pour W1 (%d lues)\n", count);
                fclose(f);
                return false;
            }
            W1_temp[i][j] = val;
            count++;
        }
    }
    // Transposer: W1[j][i] = W1_temp[i][j]
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            mlp->W1[j][i] = W1_temp[i][j];
        }
    }
    printf("  %d valeurs lues et transposées\n", count);
    
    // b1 : 256 (pas de transposition pour les biais)
    printf("Chargement b1 (256)...\n");
    count = 0;
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        if (!read_next_value(f, &val)) {
            fprintf(stderr, "Erreur: pas assez de valeurs pour b1\n");
            fclose(f);
            return false;
        }
        mlp->b1[i] = val;
        count++;
    }
    printf("  %d valeurs lues\n", count);
    
    // W2 : Keras [256, 128] -> C [128][256]
    printf("Chargement W2 (128x256) avec transposition...\n");
    count = 0;
    float W2_temp[HIDDEN1_SIZE][HIDDEN2_SIZE];
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            if (!read_next_value(f, &val)) {
                fprintf(stderr, "Erreur: pas assez de valeurs pour W2\n");
                fclose(f);
                return false;
            }
            W2_temp[i][j] = val;
            count++;
        }
    }
    // Transposer
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            mlp->W2[j][i] = W2_temp[i][j];
        }
    }
    printf("  %d valeurs lues et transposées\n", count);
    
    // b2 : 128
    printf("Chargement b2 (128)...\n");
    count = 0;
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        if (!read_next_value(f, &val)) {
            fprintf(stderr, "Erreur: pas assez de valeurs pour b2\n");
            fclose(f);
            return false;
        }
        mlp->b2[i] = val;
        count++;
    }
    printf("  %d valeurs lues\n", count);
    
    // W3 : Keras [128, 10] -> C [10][128]
    printf("Chargement W3 (10x128) avec transposition...\n");
    count = 0;
    float W3_temp[HIDDEN2_SIZE][OUTPUT_SIZE];
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (!read_next_value(f, &val)) {
                fprintf(stderr, "Erreur: pas assez de valeurs pour W3\n");
                fclose(f);
                return false;
            }
            W3_temp[i][j] = val;
            count++;
        }
    }
    // Transposer
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            mlp->W3[j][i] = W3_temp[i][j];
        }
    }
    printf("  %d valeurs lues et transposées\n", count);
    
    // b3 : 10
    printf("Chargement b3 (10)...\n");
    count = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (!read_next_value(f, &val)) {
            fprintf(stderr, "Erreur: pas assez de valeurs pour b3\n");
            fclose(f);
            return false;
        }
        mlp->b3[i] = val;
        count++;
    }
    printf("  %d valeurs lues\n", count);
    
    fclose(f);
    printf("Chargement terminé avec succès !\n");
    return true;
}

#ifdef MLP_STANDALONE_MAIN
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <image_file>\n", argv[0]);
        printf("  Supports: .bmp (28x28) or .bin (784 floats)\n");
        return 1;
    }
    
    MLP mlp = {0};
    
    char* filepath = "../models/mlp_weights.txt";

    if (!mlp_load_weights(&mlp, filepath)) {
        fprintf(stderr, "Erreur chargement poids\n");
        return 1;
    };
    
    // Charger l'image
    float input[INPUT_SIZE];
    
    // Détecter si c'est un fichier BMP
    if (is_bmp_file(argv[1])) {
        printf("Fichier BMP détecté, chargement...\n");
        if (!load_bmp_image(argv[1], input)) {
            fprintf(stderr, "Erreur: impossible de charger l'image BMP\n");
            return 1;
        }
    } else {
        // Charger comme fichier binaire de floats (format legacy)
        printf("Chargement fichier binaire...\n");
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
    }
    

    // Inférence
    mlp_forward(&mlp, input, NULL);
    int prediction = mlp_predict(&mlp, input);
    
    printf("\nPrédiction: %d\n", prediction);
    printf("Confiance: %.2f%%\n", mlp.output[prediction] * 100);
    printf("\nDistribution:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("  %d: %.4f %s\n", i, mlp.output[i], (i == prediction) ? "<--" : "");
    }
    
    return 0;
}
#endif
