import h5py
from pathlib import Path


SEQUENCE_EMBEDDINGS = Path("data/embeddings/9606.protein.sequence.embeddings.v12.0.h5")
NETWORK_EMBEDDINGS = Path("data/embeddings/9606.protein.network.embeddings.v12.0.h5")


def inspect_h5(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find: {path}")

    print(f"\nInspecting: {path}")

    with h5py.File(path, "r") as f:
        print("Top-level keys:")
        for key in f.keys():
            print(" -", key)

        first_key = list(f.keys())[0]
        obj = f[first_key]

        print(f"\nFirst key: {first_key}")
        print(f"Object type: {type(obj)}")

        if hasattr(obj, "shape"):
            print(f"Shape: {obj.shape}")
            print(f"Dtype: {obj.dtype}")
        else:
            print("This key is a group, not a dataset.")
            print("Subkeys:")
            for subkey in obj.keys():
                print(" -", subkey)


def main():
    inspect_h5(SEQUENCE_EMBEDDINGS)
    inspect_h5(NETWORK_EMBEDDINGS)


if __name__ == "__main__":
    main()