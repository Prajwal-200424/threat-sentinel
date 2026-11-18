import pickle
from pathlib import Path

from data_prep import prepare_data


# Same dataset path used during training
DATA_DIR = r"C:\Users\Tejar\Downloads\archive"

# Models are stored here
MODELS_DIR = Path("models")


def main():
    print("=" * 60)
    print("CREATING TRAINING SCALER")
    print("=" * 60)

    print("\nUsing dataset:")
    print(DATA_DIR)

    print("\nRecreating the same preprocessing used during training...")

    data = prepare_data(
        data_dir=DATA_DIR,
        subsample_frac=0.1,
        test_size=0.2,
        random_state=42,
    )

    scaler = data["scaler"]

    MODELS_DIR.mkdir(exist_ok=True)

    scaler_path = MODELS_DIR / "scaler.pkl"

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 60)
    print("SCALER CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nSaved to:")
    print(scaler_path.resolve())

    print(f"\nNumber of features: {scaler.n_features_in_}")


if __name__ == "__main__":
    main()