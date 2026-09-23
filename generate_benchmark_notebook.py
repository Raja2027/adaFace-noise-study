import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AdaFace 100k - Benchmark and Smoke Tests\n",
                "This notebook mounts Google Drive, copies the 100k dataset to fast local `/content` storage, runs batch size benchmarks, and executes explicit smoke tests for the diagnostic probe."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Mount Google Drive\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Check disk space and copy dataset\n",
                "import shutil\n",
                "import os\n",
                "from pathlib import Path\n",
                "from tqdm import tqdm\n",
                "\n",
                "drive_dataset = Path(\"/content/drive/MyDrive/adaFace-noise-study/data\")\n",
                "local_dataset = Path(\"/content/adaFace-noise-study/data\")\n",
                "\n",
                "print(\"Checking available disk space...\")\n",
                "total, used, free = shutil.disk_usage(\"/content\")\n",
                "free_gb = free / (1024**3)\n",
                "print(f\"Free space in /content: {free_gb:.2f} GB\")\n",
                "\n",
                "# Estimate required: 200k images * ~5KB = ~1GB. Plus overhead, we need at least 5GB free safely.\n",
                "required_gb = 5.0\n",
                "if free_gb < required_gb:\n",
                "    raise RuntimeError(f\"Insufficient disk space. Need {required_gb} GB, have {free_gb:.2f} GB\")\n",
                "\n",
                "print(\"\\nSpace is sufficient. Copying dataset from Google Drive to /content... (This is much faster than Drive -> Drive, but still takes a few minutes)\")\n",
                "local_dataset.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "all_files = []\n",
                "for root, dirs, files in os.walk(drive_dataset):\n",
                "    for f in files:\n",
                "        all_files.append(os.path.join(root, f))\n",
                "        \n",
                "for src_file in tqdm(all_files, desc=\"Copying files to local storage\"):\n",
                "    rel_path = os.path.relpath(src_file, drive_dataset)\n",
                "    dst_file = local_dataset / rel_path\n",
                "    dst_file.parent.mkdir(parents=True, exist_ok=True)\n",
                "    if not dst_file.exists():\n",
                "        shutil.copy2(src_file, dst_file)\n",
                "\n",
                "print(\"\\nLocal copy complete.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Verify copied dataset counts\n",
                "def count_files(directory):\n",
                "    if not directory.exists(): return 0\n",
                "    return sum(1 for _ in directory.glob(\"*\") if _.is_file())\n",
                "\n",
                "train_dir = local_dataset / \"splits\" / \"100k\" / \"images\" / \"train\"\n",
                "heldout_dir = local_dataset / \"splits\" / \"100k\" / \"images\" / \"heldout\"\n",
                "lq_train_dir = local_dataset / \"corrupted\" / \"100k_blur_15x15_s5\" / \"train\"\n",
                "\n",
                "print(f\"HQ Train images: {count_files(train_dir)} (Expected: 100000)\")\n",
                "print(f\"HQ Heldout images: {count_files(heldout_dir)} (Expected: 8000)\")\n",
                "print(f\"LQ Train images: {count_files(lq_train_dir)} (Expected: 100000)\")\n",
                "print(f\"master_100k.csv exists: {(local_dataset / 'splits' / '100k' / 'master_100k.csv').exists()}\")\n",
                "print(f\"validation_report.json exists: {(local_dataset / 'splits' / '100k' / 'validation_report.json').exists()}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# 4. Clone GitHub and checkout specific commits\n",
                "cd /content\n",
                "# Don't overwrite the data folder we just created, so clone to a temp dir and move\n",
                "rm -rf repo_temp\n",
                "git clone https://github.com/Raja2027/adaFace-noise-study.git repo_temp\n",
                "cd repo_temp\n",
                "git checkout __NEW_COMMIT_HASH__\n",
                "git submodule update --init\n",
                "\n",
                "# Move everything EXCEPT data to the working dir\n",
                "cp -r * /content/adaFace-noise-study/\n",
                "cp -r .git /content/adaFace-noise-study/\n",
                "cd /content/adaFace-noise-study\n",
                "\n",
                "echo \"==================================================\"\n",
                "echo \"DATASET BASELINE COMMIT: bcb9ae487269decfc377140dd32e75f1172cd847\"\n",
                "echo -n \"IMPLEMENTATION COMMIT: \"\n",
                "git rev-parse HEAD\n",
                "echo \"==================================================\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# 5. Report Environment\n",
                "echo \"==================================================\"\n",
                "echo \"ENVIRONMENT REPORT\"\n",
                "echo \"==================================================\"\n",
                "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader\n",
                "nvcc --version | grep \"release\"\n",
                "python -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# 6. Run Benchmark\n",
                "cd /content/adaFace-noise-study\n",
                "python scripts/benchmark_100k.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "# 7. Run Smoke Tests & Delta_G Integrity\n",
                "cd /content/adaFace-noise-study\n",
                "python scripts/smoke_test_runner.py"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("colab_benchmark_smoke.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
