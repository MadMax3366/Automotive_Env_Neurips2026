# Automotive-ENV: Benchmarking Multimodal Models in Automotive Cockpit Environments

**Junfeng Yan, Biao Wu, Meng Fang, Ling Chen**

NeurIPS 2026, Evaluations and Datasets Track

[Paper](https://arxiv.org/abs/2509.21143) | [Code](https://github.com/MadMax3366/Automotive_Env_Neurips2026) | [Project Page](https://automotive-env.github.io/)

Automotive-ENV is an interactive benchmark for evaluating multimodal agents in automotive cockpit environments, built on Android Automotive OS (AAOS) running in the Android Emulator. This repository contains the environment, the task suites, the deterministic validators, the agent implementations, and the evaluation scripts.

## Installation

1. Set up the Android Emulator
   1. Download Android Studio [here](https://developer.android.com/studio)
   2. Create an Android Virtual Device (AVD) by following these instructions. For hardware select **Automotive**, for System Image select **Tiramisu, API Level 33**, and choose AVD name as **AAOS_00**.

2. Launch the Android Emulator from the command line

    Launch the emulator from the command line, not using the Android Studio UI,
    with the `-grpc 8554` flag which is needed communication with accessibility
    forwarding app.

    ```bash
    # Typically it's located in ~/Android/Sdk/emulator/emulator or
    # ~/Library/Android/sdk/emulator/emulator
    EMULATOR_NAME=AAOS_00 # From previous step
    ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554 -writable-system
    ```

3. [Optional] It's recommended to use `conda`, which you can download [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).

    ```
    conda create -n android_world python=3.11.8
    conda activate android_world
    ```

4. Setup. *Note: Python 3.11 or above is required.*

    ```bash
    pip install "setuptools<81"
    pip install -r requirements.txt
    pip install -e . --no-build-isolation
    ```

5. Add model provider APIs as environment variables.

    ```bash
    # Add to .bashrc.
    export OPENAI_API_KEY=your-key
    export GCP_API_KEY=your-key
    ```

6. Install `ffmpeg`, if not already installed.

    ```bash
    # Linux (Ubuntu/Debian)
    # sudo apt update && sudo apt install ffmpeg

    # macOS
    brew install ffmpeg
    ```

## Quickstart

### Running the Automotive task suites with `run_task.py`

`run_task.py` is the main entry point for the Automotive-ENV fork. It loads a
task JSON from `./task/`, initializes the emulator (one env per suite), runs
the chosen agent over every task, and writes results to
`new_test_01/<agent>_<model>/<task_basename>_NN.json`.

The repo ships four task files:

| File | Count | Description |
|---|---|---|
| `task/task_explicit.json` | 57  | General HVAC tasks with explicit instructions |
| `task/task_implicit.json` | 83  | General HVAC tasks with implicit / suggestive intent |
| `task/task_safety_explicit.json` | 80  | Safety tasks with explicit cues |
| `task/task_safety_implicit.json` | 200 | Safety tasks with implicit cues |

**Example: run M3A + Gemini 2.5 Flash on every task file**
```bash
python3 run_task.py \
  --agent m3a \
  --model gemini-2.5-flash \
  --all_tasks \
  --no_confirm \
  --console_port 5554 --grpc_port 8554
```

**Example: run M3A + a local Qwen2.5-VL-7B model on every task file**
```bash
CUDA_VISIBLE_DEVICES=0 python3 run_task.py \
  --agent m3a_qwen_vl \
  --model qwen2.5-vl-7b \
  --all_tasks \
  --no_confirm \
  --console_port 5554 --grpc_port 8554
```

Common flags:

| Flag | Purpose |
|---|---|
| `--agent` | One of `m3a`, `t3a`, `m3a_qwen_vl`, `m3a_qwen3_vl`, `m3a_mai_ui`, `m3a_mimo_vl`, `m3a_ui_tars`, `m3a_ui_tars_native`, `m3a_appagent`, `m3a_opencua`, `m3a_opencua_vllm`, `m3a_gui_owl_vllm`, `m3a_jedi_vllm`, `m3a_ui_tars_native_vllm` |
| `--model` | Model id, e.g. `gemini-2.5-flash`, `gpt-4o`, `qwen2.5-vl-7b`, `mimo-vl-7b-rl`, `ui-tars-1.5-7b-vllm`, `openai/gpt-5.4-mini` |
| `--task_file` | Path to a task JSON under `./task/` (mutually exclusive with `--all_tasks`) |
| `--all_tasks` | Run every `./task/*.json` file sequentially with the same agent/model/emulator |
| `--console_port` / `--grpc_port` | Emulator console + gRPC ports (defaults 5554 / 8554) |
| `--adb_serial` | ADB serial override (default `emulator-<console_port>`) |
| `--model_path` | Local weight directory for HF-transformers agents |
| `--vllm_endpoint` | Override the default `http://127.0.0.1:<port>/v1` for vLLM agents |
| `--no_confirm` | Skip the interactive "press Enter to start" prompt |

## Pre-trained models

The remote-API agents (`m3a` / `t3a` paired with Gemini, GPT, OpenRouter
models) need only an API key — no local download. The local-inference
agents load Hugging Face weights from `./models/<name>/`. Use the helper
scripts under `scripts/` to download each family:

| Script | Default destination | Used by |
|---|---|---|
| `scripts/download_qwen_vl.py`   | `./models/Qwen2.5-VL-{3B,7B}-Instruct` | `--agent m3a_qwen_vl --model qwen2.5-vl-{3b,7b}` |
| `scripts/download_qwen3_vl.py`  | `./models/Qwen3-VL-8B-{Instruct,Thinking}` | `--agent m3a_qwen3_vl --model qwen3-vl-8b[-thinking]` |
| `scripts/download_mai_ui.py`    | `./models/MAI-UI-{2B,8B}` | `--agent m3a_mai_ui --model mai-ui-{2b,8b}` |
| `scripts/download_mimo_vl.py`   | `./models/MiMo-VL-7B-{RL,SFT}` | `--agent m3a_mimo_vl --model mimo-vl-7b-{rl,sft}` |
| `scripts/download_ui_tars.py`   | `./models/UI-TARS-1.5-7B` | `--agent m3a_ui_tars[_native] --model ui-tars-1.5-7b` |
| `scripts/download_opencua.py`   | `./models/OpenCUA-{7B,32B,72B}` | `--agent m3a_opencua --model opencua-{7b,32b,72b}` |
| `scripts/download_gui_owl.py`   | `./models/GUI-Owl-7B` | `--agent m3a_qwen_vl --model gui-owl-7b` |
| `scripts/download_jedi.py`      | `./models/Jedi-{3B,7B}-1080p` | `--agent m3a_jedi_vllm --model jedi-{3b,7b}-1080p-vllm` |

Override the default destination with `--model_path <dir>` when invoking
`run_task.py`. For the vLLM-served variants (model id ending in `-vllm`),
launch the corresponding `scripts/start_*_vllm.sh` server in a separate
terminal first, then point `run_task.py` at it via `--vllm_endpoint`
(default ports baked into each agent — see `--help`).

## Results

See Table 3 of the paper for the benchmark results.

Aggregate one or more `run_task.py` runs into a per-category table and a
per-task CSV:
```bash
python3 scripts/summarize_v3.py --root "path"
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` during `pip install -e .` | Install `setuptools<81` **before** the project install (see step 3 above). |
| `KeyError: 'qwen2_5_vl'` when loading the model | `transformers` is too old. Install from git master: `uv pip install git+https://github.com/huggingface/transformers`. |

## License

This project is released under the Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@inproceedings{yan2026automotiveenv,
  title     = {Automotive-{ENV}: Benchmarking Multimodal Models in Automotive Cockpit Environments},
  author    = {Yan, Junfeng and Wu, Biao and Fang, Meng and Chen, Ling},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```
