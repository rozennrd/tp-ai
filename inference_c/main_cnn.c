#include <stdio.h>
#include <stdlib.h>
#include "neural_network_cnn.h"

int main(void) {
    CNN cnn = {0};

    // 1) Charger les poids
    if (!cnn_load_weights_from_txt(&cnn, "./models/cnn_weights.txt")) {
        fprintf(stderr, "Failed to load CNN weights\n");
        return 1;
    }


    // 2) Image de test (dummy ici)
    float input[28][28] = {0};
    input[14][14] = 1.0f;  // pixel blanc au centre (test)

    // 3) Inférence
    float logits[10];
    cnn_forward(&cnn, input, logits);

    // 4) Résultat
    printf("Logits:\n");
    for (int i = 0; i < 10; ++i)
        printf("  %d: %.5f\n", i, logits[i]);

    printf("Prediction = %d\n", argmax10(logits));

    // 5) Nettoyage
    cnn_free(&cnn);

    return 0;
}
