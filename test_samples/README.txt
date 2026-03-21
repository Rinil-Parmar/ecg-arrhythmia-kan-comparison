ECG TEST SAMPLES
============================================================
Generated from MIT-BIH Arrhythmia Database

FOLDER STRUCTURE:
  test_samples/
  |-- npy_files/    <- upload in UI Tab 2
  |-- csv_files/    <- upload in UI Tab 2
  +-- ecg_images/   <- upload in UI Tab 3

HOW TO USE IN STREAMLIT UI:
  Tab 2 → Upload .npy or .csv file → Predict
  Tab 3 → Upload .png image → Digitize → Predict

SAMPLES:
------------------------------------------------------------
Class N (Normal)
  Dataset index : 0
  .npy file     : test_samples\npy_files\classN_sample1_idx0.npy
  .csv file     : test_samples\csv_files\classN_sample1_idx0.csv
  image         : test_samples\ecg_images\classN_sample1_idx0.png

Class N (Normal)
  Dataset index : 16417
  .npy file     : test_samples\npy_files\classN_sample2_idx16417.npy
  .csv file     : test_samples\csv_files\classN_sample2_idx16417.csv
  image         : test_samples\ecg_images\classN_sample2_idx16417.png

Class S (Supraventricular)
  Dataset index : 16
  .npy file     : test_samples\npy_files\classS_sample1_idx16.npy
  .csv file     : test_samples\csv_files\classS_sample1_idx16.csv
  image         : test_samples\ecg_images\classS_sample1_idx16.png

Class S (Supraventricular)
  Dataset index : 16391
  .npy file     : test_samples\npy_files\classS_sample2_idx16391.npy
  .csv file     : test_samples\csv_files\classS_sample2_idx16391.csv
  image         : test_samples\ecg_images\classS_sample2_idx16391.png

Class V (Ventricular)
  Dataset index : 7
  .npy file     : test_samples\npy_files\classV_sample1_idx7.npy
  .csv file     : test_samples\csv_files\classV_sample1_idx7.csv
  image         : test_samples\ecg_images\classV_sample1_idx7.png

Class V (Ventricular)
  Dataset index : 16412
  .npy file     : test_samples\npy_files\classV_sample2_idx16412.npy
  .csv file     : test_samples\csv_files\classV_sample2_idx16412.csv
  image         : test_samples\ecg_images\classV_sample2_idx16412.png

Class F (Fusion)
  Dataset index : 272
  .npy file     : test_samples\npy_files\classF_sample1_idx272.npy
  .csv file     : test_samples\csv_files\classF_sample1_idx272.csv
  image         : test_samples\ecg_images\classF_sample1_idx272.png

Class F (Fusion)
  Dataset index : 16262
  .npy file     : test_samples\npy_files\classF_sample2_idx16262.npy
  .csv file     : test_samples\csv_files\classF_sample2_idx16262.csv
  image         : test_samples\ecg_images\classF_sample2_idx16262.png

Class Q (Unknown)
  Dataset index : 12
  .npy file     : test_samples\npy_files\classQ_sample1_idx12.npy
  .csv file     : test_samples\csv_files\classQ_sample1_idx12.csv
  image         : test_samples\ecg_images\classQ_sample1_idx12.png

Class Q (Unknown)
  Dataset index : 16399
  .npy file     : test_samples\npy_files\classQ_sample2_idx16399.npy
  .csv file     : test_samples\csv_files\classQ_sample2_idx16399.csv
  image         : test_samples\ecg_images\classQ_sample2_idx16399.png

============================================================
CLASSES:
  N — Normal / Non-ectopic beat
  S — Supraventricular ectopic beat
  V — Ventricular ectopic beat
  F — Fusion beat
  Q — Unknown / Unclassifiable beat
