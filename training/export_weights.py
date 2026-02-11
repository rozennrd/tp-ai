#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np


# ============================================================
# Utils
# ============================================================
def write_txt_dump(out_txt, tensors, per_line=16):
    with open(out_txt, "w") as f:
        for name, arr in tensors:
            arr = np.asarray(arr)
            f.write(f"# name: {name}\n")
            f.write("# shape: " + " ".join(map(str, arr.shape)) + "\n")

            flat = arr.reshape(-1)
            for i in range(0, len(flat), per_line):
                chunk = flat[i:i + per_line]
                f.write(" ".join(f"{x:.8f}" for x in chunk) + "\n")
            f.write("\n")


def write_meta(out_json, tensors, extra):
    meta = {
        "tensors": [
            {"name": name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            for name, arr in tensors
        ],
        **extra
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# PyTorch export (CNN)
# ============================================================
def export_pytorch(pth_path):
    import torch

    state = torch.load(pth_path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("cnn.pth does not contain a valid state_dict")

    tensors = []
    for name, tensor in state.items():
        tensors.append((name, tensor.detach().cpu().numpy()))

    return tensors


# ============================================================
# Keras export (MLP)
# ============================================================
def export_keras(keras_path):
    from tensorflow import keras

    model = keras.models.load_model(keras_path)
    weights = model.get_weights()

    tensors = []
    for i, w in enumerate(weights):
        tensors.append((f"weight_{i}", np.asarray(w)))

    return tensors


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--cnn", action="store_true", help="Export CNN weights")
    parser.add_argument("-m", "--mlp", action="store_true", help="Export MLP weights")
    args = parser.parse_args()

    # No option → explain usage
    if not args.cnn and not args.mlp:
        print(
            "\nUsage:\n"
            "  -c   Export CNN weights (from models/cnn.pth)\n"
            "  -m   Export MLP weights (from models/mlp.keras)\n\n"
            "Examples:\n"
            "  python3 training/export_weights.py -c\n"
            "  python3 training/export_weights.py -m\n"
            "  python3 training/export_weights.py -c -m\n"
        )
        return

    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- CNN ----------
    if args.cnn:
        cnn_ckpt = os.path.join(out_dir, "cnn.pth")
        if not os.path.isfile(cnn_ckpt):
            raise FileNotFoundError("models/cnn.pth not found")

        print("[INFO] Exporting CNN weights")
        tensors = export_pytorch(cnn_ckpt)

        write_txt_dump(os.path.join(out_dir, "cnn_weights.txt"), tensors)
        write_meta(
            os.path.join(out_dir, "cnn_meta.json"),
            tensors,
            {"model": "cnn", "dataset": "MNIST", "framework": "pytorch"}
        )
        print("[OK] CNN exported")

    # ---------- MLP ----------
    if args.mlp:
        mlp_ckpt = os.path.join(out_dir, "mlp.keras")
        if not os.path.isfile(mlp_ckpt):
            raise FileNotFoundError("models/mlp.keras not found")

        print("[INFO] Exporting MLP weights")
        tensors = export_keras(mlp_ckpt)

        write_txt_dump(os.path.join(out_dir, "mlp_weights.txt"), tensors)
        write_meta(
            os.path.join(out_dir, "mlp_meta.json"),
            tensors,
            {"model": "mlp", "dataset": "MNIST", "framework": "keras"}
        )
        print("[OK] MLP exported")


if __name__ == "__main__":
    main()
