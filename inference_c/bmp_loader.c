#include "bmp_loader.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint16_t rd_u16(FILE *f) {
    uint8_t b[2];
    if (fread(b, 1, 2, f) != 2) return 0;
    return (uint16_t)(b[0] | (b[1] << 8));
}
static uint32_t rd_u32(FILE *f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
}
static int32_t rd_i32(FILE *f) { return (int32_t)rd_u32(f); }

static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

bool load_bmp_28x28_to_float(const char *path, float out[28][28], bool invert) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return false; }

    // --- BITMAPFILEHEADER ---
    uint16_t bfType = rd_u16(f);
    if (bfType != 0x4D42) { // 'BM'
        fprintf(stderr, "Not a BMP: %s\n", path);
        fclose(f);
        return false;
    }
    (void)rd_u32(f);              // bfSize
    (void)rd_u16(f); (void)rd_u16(f); // bfReserved1/2
    uint32_t bfOffBits = rd_u32(f);

    // --- BITMAPINFOHEADER (assume BITMAPINFOHEADER size=40) ---
    uint32_t biSize = rd_u32(f);
    if (biSize < 40) {
        fprintf(stderr, "Unsupported BMP header size (%u): %s\n", biSize, path);
        fclose(f);
        return false;
    }

    int32_t width  = rd_i32(f);
    int32_t height = rd_i32(f);
    uint16_t planes = rd_u16(f);
    uint16_t bpp    = rd_u16(f);
    uint32_t comp   = rd_u32(f);
    (void)rd_u32(f); // biSizeImage
    (void)rd_i32(f); // biXPelsPerMeter
    (void)rd_i32(f); // biYPelsPerMeter
    uint32_t clrUsed = rd_u32(f);
    (void)rd_u32(f); // biClrImportant

    if (planes != 1) {
        fprintf(stderr, "Unsupported planes=%u: %s\n", planes, path);
        fclose(f);
        return false;
    }
    if (comp != 0) {
        fprintf(stderr, "Compressed BMP not supported (comp=%u): %s\n", comp, path);
        fclose(f);
        return false;
    }
    if (width != 28 || (height != 28 && height != -28)) {
        fprintf(stderr, "Expected 28x28 BMP, got %dx%d: %s\n", width, height, path);
        fclose(f);
        return false;
    }
    if (!(bpp == 24 || bpp == 8)) {
        fprintf(stderr, "Unsupported bpp=%u (need 8 or 24): %s\n", bpp, path);
        fclose(f);
        return false;
    }

    // If height > 0 => bottom-up storage
    bool bottom_up = (height > 0);
    int H = (height > 0) ? height : -height;

    // Palette for 8-bit
    uint8_t *palette = NULL;
    uint32_t palette_entries = 0;

    if (bpp == 8) {
        // palette is between current file pos (54 for standard) and bfOffBits
        long pos = ftell(f);
        if (pos < 0) pos = 54;

        uint32_t palette_bytes = (bfOffBits > (uint32_t)pos) ? (bfOffBits - (uint32_t)pos) : 0;
        if (palette_bytes == 0) palette_bytes = 256 * 4; // fallback

        palette_entries = palette_bytes / 4;
        if (clrUsed != 0 && clrUsed < palette_entries) palette_entries = clrUsed;

        palette = (uint8_t*)malloc(palette_entries * 4);
        if (!palette) { fclose(f); return false; }

        // seek to palette start
        fseek(f, pos, SEEK_SET);
        if (fread(palette, 4, palette_entries, f) != palette_entries) {
            fprintf(stderr, "Failed to read palette: %s\n", path);
            free(palette);
            fclose(f);
            return false;
        }
    }

    // Go to pixel data
    fseek(f, (long)bfOffBits, SEEK_SET);

    // row size in bytes incl padding to 4 bytes
    int bytes_per_pixel = (bpp == 24) ? 3 : 1;
    int row_bytes_raw = width * bytes_per_pixel;
    int row_stride = (row_bytes_raw + 3) & ~3;

    uint8_t *row = (uint8_t*)malloc((size_t)row_stride);
    if (!row) {
        free(palette);
        fclose(f);
        return false;
    }

    for (int y = 0; y < H; ++y) {
        int yy = bottom_up ? (H - 1 - y) : y;

        if (fread(row, 1, (size_t)row_stride, f) != (size_t)row_stride) {
            fprintf(stderr, "Failed to read pixel row: %s\n", path);
            free(row);
            free(palette);
            fclose(f);
            return false;
        }

        for (int x = 0; x < 28; ++x) {
            float gray01 = 0.0f;

            if (bpp == 24) {
                uint8_t B = row[x * 3 + 0];
                uint8_t G = row[x * 3 + 1];
                uint8_t R = row[x * 3 + 2];
                // luminance approx
                float gray = 0.2126f * R + 0.7152f * G + 0.0722f * B;
                gray01 = gray / 255.0f;
            } else { // 8-bit indexed
                uint8_t idx = row[x];
                if (palette && idx < palette_entries) {
                    uint8_t B = palette[idx * 4 + 0];
                    uint8_t G = palette[idx * 4 + 1];
                    uint8_t R = palette[idx * 4 + 2];
                    float gray = 0.2126f * R + 0.7152f * G + 0.0722f * B;
                    gray01 = gray / 255.0f;
                } else {
                    gray01 = (float)idx / 255.0f;
                }
            }

            if (invert) gray01 = 1.0f - gray01;
            out[yy][x] = clamp01(gray01);
        }
    }

    free(row);
    free(palette);
    fclose(f);
    return true;
}
