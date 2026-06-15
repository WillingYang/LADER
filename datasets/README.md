# Dataset Layout

This repository does **not** assume a single fixed dataset format across all benchmarks.
Instead, each dataset is described by a YAML file in `configs/datasets/`.

## Expected workflow

1. Place your split files under a dataset root, for example:

   - `datasets/weibo/weibo_train.csv`
   - `datasets/weibo/weibo_test.csv`
   - `datasets/weibo/images/...`

2. Update the corresponding dataset YAML:

   - `configs/datasets/weibo.yaml`

3. Make sure the following fields are correct:

   - `data_root`
   - `splits`
   - `image_root`
   - `id_column`
   - `text_column`
   - `label_column`
   - `output_label_map`

## Internal label convention

The training code uses a unified internal binary convention:

- `0 = real`
- `1 = fake`

Raw dataset labels are mapped into this convention through `output_label_map`.

## Notes

- Images can be resolved either by explicit file paths or by sample/image IDs.
- Supported split file formats include CSV, TSV/TXT, and JSON.
- If your dataset has a different schema, update only the YAML when possible before changing code.

