#include <stdio.h>
#include <string.h>
#include "neural_network_cnn.h"
#include "bmp_loader.h"

int main(int argc, char **argv) {
    const char *weights_path = "models/cnn_weights.txt";

    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  %s <bmp_28x28_path> [--no-invert]\n\n"
            "Example:\n"
            "  %s data/custom_digits/2-0.bmp\n",
            argv[0], argv[0]
        );
        return 1;
    }

    const char *bmp_path = argv[1];
    bool invert = true;
    if (argc >= 3 && strcmp(argv[2], "--no-invert") == 0) invert = false;

    CNN cnn = {0};
    if (!cnn_load_weights_from_txt(&cnn, weights_path)) {
        fprintf(stderr, "Failed to load CNN weights\n");
        return 1;
    }

    float input[28][28];
    if (!load_bmp_28x28_to_float(bmp_path, input, invert)) {
        fprintf(stderr, "Failed to load BMP input\n");
        cnn_free(&cnn);
        return 1;
    }

    float logits[10] = {0};
    cnn_forward(&cnn, input, logits);

    printf("Logits:\n");
    for (int i = 0; i < 10; ++i) printf("  %d: %.5f\n", i, logits[i]);
    printf("Prediction = %d\n", argmax10(logits));

    cnn_free(&cnn);
    return 0;
}
