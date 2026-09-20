# AdaFace Noise Study Colab Workflow

This document records the strict workflow and execution architecture for training the AdaFace model remotely on Google Colab while maintaining this PC as the absolute source of truth.

## Git / Google Drive Responsibility Matrix
* **LOCAL PC**: Development, debugging, analysis, Git, source code, configs, noise maps, research documentation, and archive of run artifacts.
* **GITHUB**: Version-controlled source, configs, scripts, manifests, lightweight results, experiment metadata. (The reproducibility record).
* **COLAB**: Remote GPU execution, training, evaluation, temporary `/content/` high-speed storage.
* **GOOGLE DRIVE**: Persistent storage for `train.rec`, large `.pt` checkpoints, and large generated artifacts.

---

## The 20-Step Execution Process

Follow this exact sequence to run a training experiment on Colab:

1. **Open project in VS Code** on your local PC.
2. **Ensure latest code is committed** cleanly (e.g. `git add .` and `git commit -m "Configure noise_0 run"`).
3. **Push to GitHub** (`git push origin main`).
4. **Open `run_experiment.ipynb`** locally in VS Code or upload it to Google Colab.
5. **Select Google Colab kernel** in VS Code (or open directly via colab.research.google.com).
6. **Authenticate** your Google credentials when prompted by the Colab runtime.
7. **Connect to Colab runtime** (ensure GPU hardware accelerator is enabled).
8. **Mount Google Drive** by running the first cell in the notebook.
9. **Clone/pull exact Git revision**. In the notebook, set `COMMIT_HASH` to the exact hash you just pushed, and run the Git clone cell.
10. **Copy heavy dataset to `/content/`**. Run `!python colab/setup_colab.py`.
11. **Verify dataset and labels**. (This is done automatically by `setup_colab.py` and the `train_colab.py` fail-safes).
12. **Benchmark/select batch size**. (Optional) Run `!python colab/benchmark_batch_size.py` if your allocated GPU architecture has changed.
13. **Print and save run manifest**. Run `!python colab/train_colab.py ...`. The fail-safe validation runs first, then generates and saves `run_manifest.json` before training begins.
14. **Start training**. (Handled by `train_colab.py`).
15. **Save intermediate checkpoints**. Automatically saved to `/content/` and synced every N epochs.
16. **Sync artifacts to Drive**. (Checkpoints and `train_log.csv` are mirrored to Drive automatically by the script).
17. **Push lightweight experiment metadata/results to Git**. (You can optionally commit `run_manifest.json` and small result CSVs to GitHub).
18. **Download large artifacts back to PC**. After the run finishes, pull the checkpoints from your Google Drive into the local PC `checkpoints/noise_{X}/seed_{Y}/run_{DATE}/` directory.
19. **Pull the Colab Git commit/results**. `git pull` on the PC to synchronize any lightweight logs or manifests committed from Colab.
20. **Archive the complete run locally**. Store the evaluation results and logs in the local `results/` folder to finalize the phase.

*Do NOT diverge your local branch from Colab during an active run. Wait for the run artifacts to be generated, commit them, and sync them back locally before further development.*
