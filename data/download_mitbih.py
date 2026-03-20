import os
import wfdb


def main():
    save_dir = "raw"
    os.makedirs(save_dir, exist_ok=True)

    # Downloads full MIT-BIH Arrhythmia Database (v1.0.0) into data/raw
    wfdb.dl_database("mitdb", dl_dir=save_dir)

    print(f"Download complete. Files saved in: {save_dir}")


if __name__ == "__main__":
    main()