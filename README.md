# NeuroVision AI --- Part 2

## Brain MRI Tumor Classification with MobileNetV2 Transfer Learning and Fine-Tuning

NeuroVision AI is a deep-learning project that classifies brain MRI
images into four categories: **glioma**, **meningioma**, **notumor**,
and **pituitary**. Part 2 builds on the first CNN version by using
**MobileNetV2 transfer learning**, followed by **fine-tuning**, model
evaluation, experiment tracking with **MLflow**, and an image-prediction
interface built with **Streamlit**.

> **Important medical disclaimer:** This project is for educational and
> research demonstration only. It is not a medical device and must not
> be used to diagnose, rule out, or guide treatment for a real patient.
> Predictions can be wrong and require expert clinical interpretation.

------------------------------------------------------------------------

## Contents

-   [Project goals](#project-goals)
-   [Features](#features)
-   [Tech stack](#tech-stack)
-   [Dataset](#dataset)
-   [Project structure](#project-structure)
-   [Model workflow](#model-workflow)
-   [Evaluation results](#evaluation-results)
-   [Environment and reproducibility](#environment-and-reproducibility)
-   [Running on a new Vast.ai
    instance](#running-on-a-new-vastai-instance)
-   [MLflow experiment tracking](#mlflow-experiment-tracking)
-   [Cloudflare Tunnel and browser
    access](#cloudflare-tunnel-and-browser-access)
-   [Streamlit application](#streamlit-application)
-   [SSH connection and terminal
    notes](#ssh-connection-and-terminal-notes)
-   [Problems encountered and fixes](#problems-encountered-and-fixes)
-   [Troubleshooting checklist](#troubleshooting-checklist)
-   [Limitations and next steps](#limitations-and-next-steps)

------------------------------------------------------------------------

## Project goals

1.  Build a four-class brain MRI image classifier using MobileNetV2.
2.  Compare a frozen-backbone transfer-learning phase with a fine-tuning
    phase.
3.  Evaluate the final model on a held-out test set using accuracy, a
    classification report, and a confusion matrix.
4.  Track experiments and evaluation results with MLflow.
5.  Provide a simple Streamlit UI for uploading an MRI image and viewing
    the predicted class and class probabilities.
6.  Document the environment and commands so the project can be run on a
    different Linux/GPU instance.

## Features

-   Four-class image classification.
-   Image resizing to **224 × 224** and RGB input.
-   MobileNetV2-specific `preprocess_input`, mapping pixels to
    approximately **\[-1, 1\]**.
-   Transfer learning followed by fine-tuning.
-   Saved best checkpoints for both training phases.
-   Test-set evaluation, per-class precision/recall/F1, and confusion
    matrix.
-   MLflow tracking server with a local SQLite backend and local
    artifact directory.
-   Streamlit image-upload interface.
-   `uv` for Python version and dependency management.

## Tech stack

-   Python 3.12
-   TensorFlow / Keras
-   MobileNetV2
-   NumPy and scikit-learn (evaluation)
-   MLflow
-   Streamlit
-   `uv`
-   Vast.ai Linux GPU instance
-   Syncthing for synchronizing project files from Windows to the
    instance
-   Cloudflare Quick Tunnel for temporary browser access to MLflow (the
    generated URL can change)

Exact package versions should be taken from `uv.lock` for the project
checkout being used. The working environment during this run included
TensorFlow 2.21.0, MLflow 3.17.2, and Streamlit 1.66.1.

------------------------------------------------------------------------

## Dataset

The dataset contains four class folders:

``` text
dataset/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

The original dataset was kept as a ZIP archive in Google Drive and
transferred/made available to the training environment. The
preprocessing pipeline found **7,023 usable images**. The split used for
the final run was approximately:

  Split                  Images   Approximate share
  ------------ ---------------- -------------------
  Train                   4,494                 64%
  Validation              1,124                 16%
  Test           1,405--1,406\*                 20%

\*The logged final test evaluation reports 1,406 images. Use the exact
split output produced by the current data-preparation run if recreating
the split, since split counts can differ by one depending on the
splitting method and rounding.

Class order used by the model:

``` python
["glioma", "meningioma", "notumor", "pituitary"]
```

**Do not assume the dataset is included in the Git repository.** Keep
the dataset outside the repository unless its licence and size make
redistribution appropriate. On a new instance, download/copy the archive
and extract it, then configure the dataset path expected by the
project's preprocessing code.

------------------------------------------------------------------------

## Project structure

The repository contains the following main components (some generated
files/directories may exist only after training or running MLflow):

``` text
NeuroVision/
├── app/
│   └── app.py                    # Streamlit user interface
├── src/
│   ├── config.py                  # Shared paths, image size, class names
│   ├── preprocessing.py            # Dataset/image preprocessing
│   ├── model_training.py           # Transfer learning and fine-tuning
│   ├── evaluation.py               # Test evaluation and metrics
│   ├── prediction.py               # Load model and predict one uploaded image
│   └── main.py                     # Main project workflow/entry point
├── models/
│   ├── fine_tuning_best.keras      # Best fine-tuned checkpoint
│   └── transfer_learning_best.keras# Best transfer-learning checkpoint
├── notebooks/                      # Exploration/training notebooks, if included
├── mlartifacts/                    # MLflow artifacts; generated locally
├── mlflow.db                       # MLflow SQLite database; generated locally
├── start_mlflow.sh                 # Starts the MLflow server
├── pyproject.toml                  # Project/dependency configuration for uv
├── uv.lock                         # Locked dependency versions
└── README.md
```

Generated model files, the MLflow database, MLflow artifacts, datasets,
`.venv`, and caches can be large. Whether they are tracked by Git should
be decided intentionally. Do not commit a large dataset, `.venv`, cache,
or temporary files by accident. If model weights are too large for the
selected Git host, use Git LFS or a model-artifact hosting service and
document the download step.

------------------------------------------------------------------------

## Model workflow

### 1. Prepare and split data

-   Read the images from the four class directories.
-   Decode each image as RGB.
-   Resize to **224 × 224**.
-   Apply `tensorflow.keras.applications.mobilenet_v2.preprocess_input`.
-   Split into training, validation, and test sets.
-   Preserve the class-to-index order consistently between training and
    inference.

### 2. Transfer learning

-   Use MobileNetV2 pretrained on ImageNet as the feature extractor.
-   Freeze the pretrained base initially.
-   Train the project's classification head.
-   Save the best checkpoint based on validation performance.

Best checkpoint:

``` text
models/transfer_learning_best.keras
```

Recorded best validation accuracy: **about 93.32%** (validation loss
about **0.1838**).

### 3. Fine-tuning

-   Unfreeze the selected part of the pretrained MobileNetV2 backbone
    according to the training code.
-   Continue training with a lower learning rate.
-   Use callbacks to help prevent overfitting and retain the best model:
    -   `EarlyStopping` (patience 5 in the recorded run)
    -   `ModelCheckpoint` (`best_model.keras` during training; final
        project checkpoint saved as `models/fine_tuning_best.keras`)
    -   `ReduceLROnPlateau` (factor 0.2, patience 3, minimum learning
        rate `1e-6`)

Best checkpoint:

``` text
models/fine_tuning_best.keras
```

Recorded best validation accuracy: **about 95.55%** (validation loss
about **0.1280**).

The exact layer-unfreezing strategy, batch size, number of epochs, and
learning rates are defined in the training source/notebook. Check those
files before reproducing training; do not assume a different instance
has the same notebook state.

### 4. Final test evaluation

The final test-set result recorded for the fine-tuned model:

-   **Test accuracy: 96.44%**
-   **Test loss: 0.1106**

Classification report:

  Class          Precision   Recall   F1-score   Support
  ------------ ----------- -------- ---------- ---------
  glioma            0.9686   0.9477     0.9580       325
  meningioma        0.9321   0.9179     0.9250       329
  notumor           0.9900   0.9875     0.9887       400
  pituitary         0.9616   0.9972     0.9791       352

Confusion matrix (rows = true labels; columns = predicted labels, in the
class order above):

``` text
[[308  16   1   0]
 [ 10 302   3  14]
 [  0   5 395   0]
 [  0   1   0 351]]
```

These are the recorded results from the current run, not a promise that
a newly trained model will achieve identical results. Results depend on
the exact dataset, split, seed, package versions, and training
configuration.

------------------------------------------------------------------------

## Environment and reproducibility

The training and serving environment used a Linux Vast.ai instance with
an NVIDIA GPU. The recorded environment included an **NVIDIA RTX 3060 12
GB**. A separate earlier training environment used Tesla T4 GPUs; the
GPU available on a new instance will depend on the selected Vast.ai
offer.

The project uses `uv` and the checked-in `pyproject.toml` / `uv.lock`
files to recreate its Python environment.

### Check the project

``` bash
cd /home/user/NeuroVision
ls
```

The project directory may differ on a new instance. Change into the
directory where the repository was cloned or synced.

### Install `uv` on a fresh Linux instance

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

If the `uv` command is not found in a newly opened shell, run:

``` bash
source "$HOME/.local/bin/env"
```

### Install Python and project dependencies

From the project root:

``` bash
uv python install 3.12
uv sync
```

Verify the Python version and key libraries:

``` bash
uv run python --version
uv run python -c "import tensorflow as tf, mlflow, streamlit; print('TensorFlow:', tf.__version__); print('MLflow:', mlflow.__version__); print('Streamlit:', streamlit.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
nvidia-smi
```

A GPU list containing at least one device indicates that TensorFlow can
see a GPU. If the list is empty, check the Vast.ai template, NVIDIA
driver/CUDA compatibility, and the dependency configuration before
starting training.

> The shell may display
> `VIRTUAL_ENV=/venv/main does not match the project environment path .venv and will be ignored`.
> In this project, that warning was non-blocking when commands ran
> successfully through `uv run`. Prefer `uv run ...` so `uv` uses the
> project environment.

------------------------------------------------------------------------

## Running on a new Vast.ai instance

This section is designed for a fresh instance. **Do not reuse an old
instance's IP address, SSH port, process IDs, or temporary Cloudflare
URL.** Vast.ai may assign different values to every instance.

### Step 1 --- Connect to the instance

Copy the SSH command shown by Vast.ai for the current instance. The
previous instance used a command in this format:

``` bash
ssh -p <SSH_PORT> root@<INSTANCE_IP>
```

An earlier concrete connection looked like this:

``` bash
ssh -p 34470 root@124.60.192.12
```

That IP and port are historical examples only and may no longer work.
Use the exact connection details displayed for your current instance.

To open an SSH local port forward for MLflow instead of using a public
tunnel, the previous command format was:

``` bash
ssh -p <SSH_PORT> root@<INSTANCE_IP> -L 5001:localhost:5000
```

With that forward active, open `http://localhost:5001` in a browser on
the same computer that initiated SSH. This is an alternative access
method; it is not required when using the Cloudflare Quick Tunnel method
below.

### Step 2 --- Disable Vast.ai automatic tmux behavior (optional but useful)

The command used in this workflow was:

``` bash
touch ~/.no_auto_tmux
```

It creates a marker file in the Linux user's home directory to disable
the automatic tmux behavior in the relevant Vast.ai environment. Run it
**inside the instance**, as the same user who will use the terminal.
Reconnect after creating it if needed. This setting is per
instance/user; repeat it on a new instance if you want the same
behavior.

### Step 3 --- Get the project onto the instance

The project files were edited in VS Code on Windows and synchronized to
the Linux instance using Syncthing. On a fresh instance, either set up
Syncthing again or clone the repository.

Example using Git:

``` bash
git clone <https://github.com/ISMAILBENMEZI/NeuroVision-AI-Part-2.git> NeuroVision
cd NeuroVision
```

If you already have the project synced to `/home/user/NeuroVision`, use:

``` bash
cd /home/user/NeuroVision
```

Do not sync or commit `.venv` from Windows to Linux. Python environments
are platform-specific. Exclude `.venv`, Python caches, and temporary
files from Syncthing/Git synchronization.

### Step 4 --- Prepare the environment

``` bash
cd /home/user/NeuroVision
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv sync
```

If `uv` is already installed, skip its installation and just run
`source "$HOME/.local/bin/env"` if needed, then `uv sync`.

### Step 5 --- Verify GPU and files

``` bash
nvidia-smi
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
ls -lh models/
```

The app expects this model by default:

``` text
models/fine_tuning_best.keras
```

Ensure the file exists on the new instance. If model files are not
tracked by Git, copy them from the previous instance or download them
from the configured artifact storage. Do not start Streamlit until the
checkpoint is present.

### Step 6 --- Start MLflow

From the project root:

``` bash
cd /home/user/NeuroVision
chmod +x start_mlflow.sh
./start_mlflow.sh
```

Keep this process running in its own terminal/session. By default the
script binds MLflow to `127.0.0.1:5000`, which means it is accessible
locally on the instance and can be exposed through SSH forwarding or a
tunnel. Do not bind it to all public interfaces unless you understand
and configure the security implications.

### Step 7 --- Start a Cloudflare Quick Tunnel for MLflow

In a **separate terminal on the instance**, run the installed
`cloudflared` command:

``` bash
cloudflared tunnel --url http://127.0.0.1:5000
```

If `cloudflared` is not installed on a fresh instance, install it using
the official Cloudflare instructions for the current Linux distribution
before running the command. The command prints a temporary HTTPS URL
ending in `trycloudflare.com`. Keep the tunnel process running and open
that URL in your browser.

**Important:** Quick Tunnel URLs are temporary and may change each time
the tunnel restarts. The previous URL was:

``` text
https://rid-annotated-configuration-jpg.trycloudflare.com
```

Do not expect that old URL to work on another instance or another run.

When the generated hostname changes, update the `--allowed-hosts` and
`--cors-allowed-origins` values in `start_mlflow.sh` to match the
current tunnel hostname, then restart the MLflow server. See [MLflow
experiment tracking](#mlflow-experiment-tracking).

### Step 8 --- Start Streamlit

In another terminal/session on the instance:

``` bash
cd /home/user/NeuroVision
PYTHONPATH=/home/user/NeuroVision uv run streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

The `PYTHONPATH` is important for this project because `app/app.py`
imports modules using `from src...`. Without it, Streamlit previously
failed with `ModuleNotFoundError: No module named 'src'`.

Vast.ai may provide a direct HTTP service link for port **8501**. Use
the port-forwarding/service exposure controls offered by the current
instance. Keep the Streamlit server process running while using the app.

------------------------------------------------------------------------

## MLflow experiment tracking

### What is tracked

The MLflow experiment used in this project is named:

``` text
NeuroVision-MobileNetV2
```

The experiment ID in the recorded MLflow database was `1`. The recorded
run names include:

-   `Transfer-Learning`
-   `Fine-Tuning`
-   `Final-Evaluation`

Earlier failed attempts may remain in the MLflow history; that is
expected during debugging. Use the successful runs when reviewing the
final results.

The MLflow server used: - SQLite backend store: `sqlite:///mlflow.db` -
Local artifacts directory: `./mlartifacts` - Host: `127.0.0.1` - Port:
`5000`

The `mlflow.db` database and `mlartifacts/` directory are generated
data. Back them up if you need to preserve experiment history; they are
not recreated just by installing the Python dependencies.

### `start_mlflow.sh`

The working script during this run was:

``` bash
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
mkdir -p mlartifacts

uv run mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --allowed-hosts "localhost:*,127.0.0.1:*,rid-annotated-configuration-jpg.trycloudflare.com" \
    --cors-allowed-origins "https://rid-annotated-configuration-jpg.trycloudflare.com" \
    --backend-store-uri sqlite:///mlflow.db \
    --artifacts-destination ./mlartifacts
```

**Before using this script on another instance:** replace
`rid-annotated-configuration-jpg.trycloudflare.com` with the hostname
printed by the currently running `cloudflared tunnel` command. The
`--allowed-hosts` value is the hostname without `https://`; the
`--cors-allowed-origins` value includes `https://`.

For a stable long-lived setup, use a configured named Cloudflare
Tunnel/domain or an SSH port forward instead of relying on a temporary
Quick Tunnel hostname. Do not publish the MLflow server or experiment
data publicly without considering access control and data privacy.

### Starting and stopping MLflow

Start:

``` bash
cd /home/user/NeuroVision
./start_mlflow.sh
```

Stop it by pressing `Ctrl+C` in the terminal running it. This is safer
than copying old process IDs from a previous instance. Process IDs such
as `3562` and `3565` were specific to the earlier session and must not
be reused blindly.

If you must inspect running processes:

``` bash
ps aux | grep '[m]lflow server'
```

Then stop the relevant process deliberately.

### MLflow browser errors encountered

The MLflow UI initially loaded but browser requests to the MLflow API
failed with a cross-origin/host validation error. The logs included a
blocked request from the Cloudflare hostname and a `403` for:

``` text
/ajax-api/2.0/mlflow/runs/search
```

The fix was to configure both: - `--allowed-hosts` with the tunnel
hostname - `--cors-allowed-origins` with the full HTTPS origin

If the tunnel URL changes, update both values and restart MLflow. A
`404` at `/portal-resolver` was also observed; it was not the main cause
of the blocked run-search request. After the CORS/allowed-host
configuration was corrected, the MLflow UI worked.

### Model registry note

The **Models** registry page can be empty even when experiment runs and
local `.keras` files exist. Saving a Keras file in `models/` does not
automatically register it in the MLflow Model Registry. The recorded
experiment tracking was verified separately from the model registry.

------------------------------------------------------------------------

## Streamlit application

The application is in `app/app.py`. It loads the fine-tuned checkpoint
through `src/prediction.py`, accepts an uploaded image, preprocesses it,
and displays a predicted class and class probabilities.

### Prediction preprocessing

Streamlit's uploaded file is an `UploadedFile` object, **not a
filesystem path**. The working inference preprocessing reads its bytes
directly:

``` python
def preprocess_image(image_file):
    image_bytes = image_file.getvalue()

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    return image
```

The preprocessing must remain consistent with training: RGB, the same
`IMAGE_SIZE`, and MobileNetV2 `preprocess_input`.

### Run the app

``` bash
cd /home/user/NeuroVision
PYTHONPATH=/home/user/NeuroVision uv run streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open the instance's exposed Streamlit URL for port `8501`, upload a
supported brain MRI image, and check the prediction and class
probabilities.

------------------------------------------------------------------------

## SSH connection and terminal notes

### SSH command

Use the SSH connection string displayed by Vast.ai for the current
instance. The earlier instance used:

``` bash
ssh -p 34470 root@124.60.192.12
```

For a local MLflow port-forward:

``` bash
ssh -p <SSH_PORT> root@<INSTANCE_IP> -L 5001:localhost:5000
```

Then browse to `http://localhost:5001` on the computer running SSH.

### Useful commands on the instance

``` bash
# Confirm current directory
pwd

# Move to the project
cd /home/user/NeuroVision

# Check GPU and memory
nvidia-smi

# Check Python using the project environment
uv run python --version

# Check model files
ls -lh models/

# Check MLflow script
cat start_mlflow.sh

# Inspect MLflow process
ps aux | grep '[m]lflow server'

# Inspect the no-auto-tmux marker
ls -la ~/.no_auto_tmux
```

Commands such as `touch ~/.no_auto_tmux`, the current SSH endpoint,
process IDs, and the temporary Cloudflare hostname are
environment/session-specific. The README documents them so they can be
recreated, but only the marker command is expected to be repeated
literally; obtain new endpoint/URL values from the new instance.

------------------------------------------------------------------------

## Problems encountered and fixes

### 1. Python environment not available on a new instance

**Problem:** `uv` was missing or Python/dependencies were not ready.

**Fix:**

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv sync
```

### 2. `VIRTUAL_ENV` points to `/venv/main`

**Message:**

``` text
VIRTUAL_ENV=/venv/main does not match the project environment path .venv and will be ignored
```

**Fix/interpretation:** This warning was non-blocking in the recorded
setup. Run commands through `uv run` so `uv` selects the project
environment. Confirm the command's actual output rather than treating
this warning alone as a failure.

### 3. Streamlit could not import `src`

**Error:**

``` text
ModuleNotFoundError: No module named 'src'
```

**Cause:** The app was launched without the project root being on
Python's import path.

**Fix:**

``` bash
cd /home/user/NeuroVision
PYTHONPATH=/home/user/NeuroVision uv run streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

### 4. Uploaded image raised TensorFlow `NotFoundError`

**Error:** TensorFlow attempted to read a string representation of
Streamlit's `UploadedFile` as though it were a filesystem path,
resulting in a `ReadFile` "No such file or directory" error.

**Cause:** The old code called:

``` python
tf.io.read_file(str(image_file))
```

An uploaded Streamlit file is an in-memory file object, not a path on
the server.

**Fix:** Use `image_file.getvalue()` and decode the bytes using
`tf.io.decode_image`, as shown in [Prediction
preprocessing](#prediction-preprocessing). Save the file, ensure
Syncthing has synced it to the instance, and restart/reload Streamlit
before testing again.

### 5. MLflow UI opened but runs/API requests returned `403`

**Cause:** MLflow host validation and CORS did not allow the active
Cloudflare tunnel origin.

**Fix:** Add the current tunnel hostname to `--allowed-hosts` and the
full `https://...` origin to `--cors-allowed-origins` in
`start_mlflow.sh`, then restart the server. Do not copy an old temporary
tunnel URL into a new instance.

### 6. Model file exceeds the Git host's file-size limit

The first NeuroVision version encountered a Git push rejection because a
`.keras` model was approximately **128.05 MB**, exceeding the common 100
MB per-file limit on GitHub.

**Fix:** Do not push oversized model files as ordinary Git blobs. Use
Git LFS or external model storage, or exclude the checkpoint and
document how to obtain it. Check the actual current checkpoint sizes
before pushing:

``` bash
ls -lh models/
```

### 7. GPU differs between instances

A new Vast.ai instance may provide a different GPU. Check with
`nvidia-smi` and TensorFlow's `tf.config.list_physical_devices('GPU')`.
Do not assume the GPU model or available VRAM is the same as the
recorded run.

------------------------------------------------------------------------

## Troubleshooting checklist

Before running the application on a fresh instance:

-   [ ] The repository is present and the terminal is in its root
    directory.
-   [ ] `uv sync` completed successfully.
-   [ ] `uv run python --version` reports Python 3.12.
-   [ ] TensorFlow imports successfully.
-   [ ] The GPU is visible if GPU inference/training is intended.
-   [ ] `models/fine_tuning_best.keras` exists.
-   [ ] `start_mlflow.sh` is executable (`chmod +x start_mlflow.sh`).
-   [ ] MLflow is running on `127.0.0.1:5000`.
-   [ ] If using Cloudflare,
    `cloudflared tunnel --url http://127.0.0.1:5000` is running and the
    current hostname is in the MLflow allowed-host/CORS configuration.
-   [ ] Streamlit is launched with the `PYTHONPATH` command documented
    above.
-   [ ] An uploaded image is decoded from bytes using `getvalue()`, not
    treated as a path.

------------------------------------------------------------------------

## Limitations and next steps

-   Performance metrics reflect one recorded train/validation/test split
    and are not evidence of clinical readiness.
-   The model may perform differently on MRI images from other scanners,
    hospitals, populations, image formats, or acquisition protocols.
-   The dataset source, licence, and redistribution conditions should be
    documented before public release.
-   Record the random seed, split method, exact training parameters, and
    dataset version in the notebook/configuration to improve
    reproducibility.
-   For deployment, consider input validation, clearer error messages,
    model/version metadata, access control, and a stable model-artifact
    download method.
-   MLflow's local SQLite database and `mlartifacts/` folder must be
    persisted/backed up if experiment history needs to survive instance
    deletion.

------------------------------------------------------------------------

## Quick command reference

``` bash
# 1. Connect from your local computer (replace with current Vast.ai details)
ssh -p <SSH_PORT> root@<INSTANCE_IP>

# 2. On the instance, optionally disable automatic tmux behavior
touch ~/.no_auto_tmux

# 3. Enter the project directory
cd /home/user/NeuroVision

# 4. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# 5. Install Python and dependencies
uv python install 3.12
uv sync

# 6. Verify environment/GPU
uv run python --version
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
nvidia-smi

# 7. Start MLflow in its own terminal
cd /home/user/NeuroVision
chmod +x start_mlflow.sh
./start_mlflow.sh

# 8. Start Cloudflare Quick Tunnel in another terminal
cloudflared tunnel --url http://127.0.0.1:5000

# 9. Start Streamlit in another terminal
cd /home/user/NeuroVision
PYTHONPATH=/home/user/NeuroVision uv run streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501

# 10. Optional SSH local port-forward instead of Cloudflare for MLflow
# Run this from your local computer with the current instance's SSH details:
ssh -p <SSH_PORT> root@<INSTANCE_IP> -L 5001:localhost:5000
# Then open http://localhost:5001 locally.
```

**Final reminder:** on a new Vast.ai instance, always use the current
SSH IP/port, current Cloudflare URL, current model file location, and
the project's `pyproject.toml` / `uv.lock`. Do not copy historical IPs,
temporary tunnel URLs, or old process IDs as though they were permanent.
