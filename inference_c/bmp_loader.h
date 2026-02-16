#ifndef BMP_LOADER_H
#define BMP_LOADER_H

#include <stdbool.h>

bool load_bmp_28x28_to_float(const char *path, float out[28][28], bool invert);

#endif
