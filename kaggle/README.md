# Kaggle code dataset (Qubit-Medic)

This folder holds **`dataset-metadata.json`** for publishing the project as a
[Kaggle Dataset](https://www.kaggle.com/docs/datasets) so notebooks (Kaggle or
Colab with the Kaggle API) always pull the latest Python sources.

## One-time setup

1. Install the CLI: `pip install kaggle` (or `pip install -r requirements-train.txt`)
2. Create **`~/.kaggle/kaggle.json`** with your API credentials from
   [Kaggle → Account → API → Create New Token](https://www.kaggle.com/settings).
   **Never commit this file** — keep it only under your home directory.
   A **`kaggle.json` in the project folder is ignored by the CLI** unless you
   set `KAGGLE_CONFIG_DIR`; the official location is **`~/.kaggle/kaggle.json`**.
3. `chmod 600 ~/.kaggle/kaggle.json`
4. Edit **`dataset-metadata.json`** and set `"id"` to
   `"<your_kaggle_username>/<dataset_slug>"` if it differs from the default.

### `401 Unauthorized` on `datasets create` / `datasets version`

That response means **Kaggle did not accept your API key** (or it never saw one).

Checklist:

- File exists: **`~/.kaggle/kaggle.json`** (not only in the repo root).
- JSON is exactly `{"username":"...","key":"..."}` — no trailing commas, no
  smart quotes, username matches the account that created the key.
- If you **rotated or revoked** the key after a leak, download a **new** token
  and replace the file.
- Optional env override (CI / non-default home): set **`KAGGLE_USERNAME`** and
  **`KAGGLE_KEY`** to the same values as in the JSON.

From the repo root, after `pip install kaggle` in `.venv`:

```bash
make kaggle-auth-test
```

If that prints a 401, fix credentials first; only then retry `make kaggle-push-create`.

## Publish / update

From the **repository root**:

```bash
make kaggle-sync
make kaggle-push-create    # first time only
make kaggle-push MSG='describe your change'   # subsequent updates
```

Or set `KAGGLE_DATASET_ID=username/slug` when syncing to override the id
without editing the JSON.

## Use in a Kaggle notebook

Add this dataset as notebook input, then:

```python
import sys
sys.path.insert(0, "/kaggle/input/qubit-medic-code")  # folder name matches the dataset
```

Install training deps from the bundled `requirements-train.txt` as needed.
