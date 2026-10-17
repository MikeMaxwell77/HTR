# GPU training with Docker

This image runs the CTC word trainer and its tests. It packages the source so the
same image can later be pushed to a registry and run on an AWS GPU host. The
segmentation/inference tools have additional dependencies and are not included.

Validated on October 17, 2026 with Docker Desktop's Linux engine and an NVIDIA
GeForce RTX 5050 Laptop GPU: image build, GPU matrix multiplication, all three
training tests, and two epochs over 100 IAM images (80 training/20 validation,
batch size 2). The model, metadata, history, and predictions persisted in the
host's `models/` folder. This small run validates execution, not model accuracy.

## Windows prerequisites

Install Docker Desktop with the **WSL 2 backend** and Linux containers enabled.
Keep WSL and the Windows NVIDIA driver current. Docker's
[GPU setup guide](https://docs.docker.com/desktop/features/gpu/) describes GPU
passthrough. No Windows CUDA Toolkit installation is needed for this image.

The base is `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`: Python 3.12,
NVIDIA TensorFlow 2.17, CUDA 12.8, and cuDNN 9.7, with Keras 3.8 added for the
trainer's CTC API. NVIDIA documents Blackwell support and a driver requirement
of release 570 or later in its
[release notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-25-02.html).
Use a driver that also supports your RTX 5050. Actual compatibility must be
confirmed with the smoke test below on your machine.

This is a separate Linux dependency stack from the Windows `.venv` and lockfile.
Do not install `scripts/training/requirements-lock.txt` inside it; doing so would
replace the NVIDIA TensorFlow build. NVIDIA ended this container release line
after 25.02. Reassess the base and run the smoke test before a future production
deployment. The base is pinned by digest and added dependencies are pinned by
version. Deploy the validated final image by digest as well.

## Build and check

From the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Setup
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Smoke
```

`Setup` finds Docker Desktop even when `docker` is missing from PATH, builds the
image, and checks GPU execution and the training tests. `Smoke` builds and trains
on 100 real IAM images for one epoch, saving to a unique `models/docker-smoke-*/`
folder. The execution policy flag applies only to this PowerShell process.

To train using the settings in `train.py`, or supply overrides:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Train
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Train --epochs 50 --batch-size 2
```

The launcher rebuilds before training so source/settings changes are included.
`Check` reruns the GPU tests using the already built image. The precision check
disables TensorFloat-32 approximations to compare saved/reloaded float32 outputs;
normal training retains TensorFlow's default precision settings.

Equivalent direct Docker commands, when `docker` is on PATH:

```powershell
docker compose build
docker compose run --rm --entrypoint nvidia-smi train
docker compose run --rm --entrypoint python train -m docker.gpu_check
```

The last command requires a visible TensorFlow GPU, executes a GPU matrix
multiplication, and runs the existing tests including a CNN/BiLSTM/CTC training
step, decoding, and model save/load. A missing GPU or a failed/skipped test
returns a nonzero exit code. Run this before launching a long training job.
The NVIDIA base can print CUDA plugin registration, NUMA, PTX feature, and GPU
timing warnings. These appeared during validation; use the final test result and
process exit code to determine success.

## Train

Provide these files on the host (they are never copied into the image):

```text
data/
  word_dataset_info.csv
  IAM_Words/
    words/  # Extracted IAM word images, matching the CSV's relative image_path
```

Start with a small run, then a full run:

```powershell
docker compose run --rm train --limit 100 --epochs 1 --batch-size 2
docker compose run --rm train
```

Defaults come from `train.py`; CLI flags override them. Rebuild after editing
source or the defaults in `train.py`. CLI changes need no rebuild:

```powershell
docker compose run --rm train --epochs 100 --batch-size 4 --lstm-units 128
docker compose run --rm train --help
```

The 8 GB RTX 5050 may need a smaller batch size than the default 8, especially
when overriding the default 128 x 32 resize canvas. TensorFlow allocates GPU memory as needed.
The service exposes one GPU; this trainer does not implement distributed training.

`data/` is mounted read-only. Outputs persist in the host's `models/ctc_word/`
even after the container is removed. `outputs/` is also writable; use
`--png-dir /app/outputs/standardized_words` for optional PNG exports.

For datasets stored elsewhere, set `HTR_DATA_DIR` to a folder with the layout
above, or override the in-container paths explicitly:

```powershell
$env:HTR_DATA_DIR = 'D:/datasets/IAM'
docker compose run --rm train --words-dir /app/data/words --csv /app/data/word_dataset_info.csv
```

`HTR_MODELS_DIR`, `HTR_OUTPUTS_DIR`, and `HTR_IMAGE` similarly override the host
output directories and image name. These values can also be set in a local
`.env` file (already ignored by Git). CLI paths refer to paths **inside** the
container, not Windows paths. Keep each run's artifacts separate with
`--output-dir /app/models/run-001`.

## Future AWS integration

The image uses Linux amd64, accepts training arguments, writes logs to stdout,
and needs no source-code mount. Push the built image to ECR later and use its
immutable digest with an x86 NVIDIA GPU host/runtime. AWS uses the GPU attached
to that host, not the RTX 5050 in your laptop. Linux hosts require Docker Engine
and the NVIDIA Container Toolkit configured for GPU access.

Example standalone invocation on a future Linux GPU host, after provisioning
`/srv/htr/data` and `/srv/htr/models`:

```bash
docker run --rm --init --gpus '"device=0"' --shm-size=2g \
  --mount type=bind,src=/srv/htr/data,dst=/app/data,readonly \
  --mount type=bind,src=/srv/htr/models,dst=/app/models \
  htr-training:local --epochs 50 --batch-size 4
```

Use mounted storage for input/output and stage data from S3 outside the
container if needed. No AWS credentials, datasets, model files, or `.venv` are
included in the build context. No AWS resources are created by this setup.
The trainer saves its final model after fitting; interrupted/Spot runs currently
have no periodic checkpoint/resume support.
