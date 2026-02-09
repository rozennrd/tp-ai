"""
Il charge un modèle sauvegardé (checkpoint) puis exporte ses tenseurs en .txt (et un .json meta).

Supporté :
Sorties (par défaut dans ./models/) :
- <name>_weights.txt
- <name>_meta.json

Format TXT (parseable facilement en C) :
# name: <tensor_name>
# shape: d0 d1 d2 ...
v0 v1 v2 v3 ...
...

Exemples :
  # CNN PyTorch (ton BasicConvNet)
  python3 training/export_weights.py --pytorch models/cnn.pth --name cnn

  # MLP Keras
  python3 training/export_weights.py --keras models/mlp.keras --name mlp

Options utiles :
  --out-dir models
  --per-line 16
"""

import os
import json
import argparse
from typing import Any, Dict, List, Tuple

import numpy as np


# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chunks_flat(arr: np.ndarray, per_line: int) -> List[np.ndarray]:
    flat = arr.reshape(-1)
    return [flat[i:i + per_line] for i in range(0, len(flat), per_line)]


def write_txt_dump(out_txt: str, tensors: List[Tuple[str, np.ndarray]], per_line: int = 16) -> None:
    with open(out_txt, "w", encoding="utf-8") as f:
        for name, arr in tensors:
            arr = np.asarray(arr)

            # Header
            f.write(f"# name: {name}\n")
            f.write("# shape: " + " ".join(map(str, arr.shape)) + "\n")
            f.write(f"# dtype: {arr.dtype}\n")

            # Values
            for chunk in chunks_flat(arr, per_line):
                f.write(" ".join(f"{float(x):.8f}" for x in chunk) + "\n")
            f.write("\n")


def write_meta(out_json: str, tensors: List[Tuple[str, np.ndarray]], extra: Dict[str, Any]) -> None:
    meta = {
        "tensors": [
            {"name": name, "shape": list(np.asarray(arr).shape), "dtype": str(np.asarray(arr).dtype)}
            for name, arr in tensors
        ],
        **extra
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -------------------------
# Exporters
# -------------------------
def export_pytorch(state_path: str) -> List[Tuple[str, np.ndarray]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch n'est pas disponible dans cet environnement.") from e

    obj = torch.load(state_path, map_location="cpu")

    # obj peut être:
    # - un state_dict directement
    # - un dict contenant 'state_dict' (checkpoint custom)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        # On suppose que c'est déjà un state_dict
        state = obj
    else:
        raise ValueError("Format PyTorch non supporté: attendu dict/state_dict/checkpoint.")

    tensors: List[Tuple[str, np.ndarray]] = []
    for name, t in state.items():
        # t est un torch.Tensor
        arr = t.detach().cpu().numpy()
        tensors.append((name, arr))

    return tensors


def export_keras(model_path: str) -> List[Tuple[str, np.ndarray]]:
    try:
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras n'est pas disponible dans cet environnement.") from e

    model = keras.models.load_model(model_path)

    weights = model.get_weights()  # [W0, b0, W1, b1, ...]
    tensors: List[Tuple[str, np.ndarray]] = []

    # On nomme par index + type (dense/kernel, dense/bias...) quand possible
    # Keras ne donne pas toujours de noms explicites via get_weights().
    for i, w in enumerate(weights):
        tensors.append((f"weight_{i}", np.asarray(w)))

    return tensors


# -------------------------
# Main
# -------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Export des poids (PyTorch/Keras) vers TXT + JSON meta.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pytorch", type=str, help="Chemin vers checkpoint PyTorch (.pth/.pt) contenant un state_dict.")
    g.add_argument("--keras", type=str, help="Chemin vers modèle Keras (.keras/.h5).")

    p.add_argument("--name", type=str, required=True, help="Nom logique du modèle (ex: cnn, mlp).")
    p.add_argument("--out-dir", type=str, default="models", help="Dossier de sortie (défaut: models).")
    p.add_argument("--per-line", type=int, default=16, help="Nombre de valeurs par ligne dans le TXT.")
    p.add_argument("--float32", action="store_true", help="Caster tous les tenseurs en float32 avant export.")
    args = p.parse_args()

    ensure_dir(args.out_dir)

    if args.pytorch:
        in_path = args.pytorch
        tensors = export_pytorch(in_path)
        kind = "pytorch"
    else:
        in_path = args.keras
        tensors = export_keras(in_path)
        kind = "keras"

    if args.float32:
        tensors = [(n, np.asarray(a, dtype=np.float32)) for n, a in tensors]

    out_txt = os.path.join(args.out_dir, f"{args.name}_weights.txt")
    out_json = os.path.join(args.out_dir, f"{args.name}_meta.json")

    write_txt_dump(out_txt, tensors, per_line=args.per_line)
    write_meta(
        out_json,
        tensors,
        extra={
            "name": args.name,
            "source_kind": kind,
            "source_path": in_path,
            "format": "txt+json",
            "per_line": args.per_line,
        }
    )

    print(f"[OK] Export terminé")
    print(f"  - TXT : {out_txt}")
    print(f"  - META: {out_json}")


if __name__ == "__main__":
    main()
