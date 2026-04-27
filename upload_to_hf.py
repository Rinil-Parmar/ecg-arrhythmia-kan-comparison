from huggingface_hub import create_repo, upload_folder

# Change this to your Hugging Face model repo: <hf_username>/<repo_name>
REPO_ID = "Rinil-Parmar/ecg-arrhythmia-kan-comparison"

# This will create the repo on HF if it doesn't exist, then upload the folder
create_repo(REPO_ID, repo_type="model", exist_ok=True)

# Upload just the checkpoints folder (contains cnn_kan.pth, cnn_mlp.pth)
upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path="checkpoints",
    path_in_repo="checkpoints",
    commit_message="Upload model checkpoints",
)

print(f"Done: https://huggingface.co/{REPO_ID}")