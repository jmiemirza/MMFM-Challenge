import os
import modal
import subprocess
from pathlib import PurePosixPath

GPU_TYPE = os.environ.get("GPU_CONFIG", "a100-80gb")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

MODEL_DIR = "/pretrained"
# MODEL_ID = "zhiqiulin/llava-v1.5-7b-MMMU"
MODEL_ID = os.environ.get("MODEL_ID", "zhiqiulin/llava-v1.5-7b-MMMU-epoch-10")

WORK_DIR = "/repos/MMFM-Challenge"

def download_model_to_image():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        token=os.environ["HF_TOKEN"],
    )
    move_cache()


# Axolotl image hash corresponding to 0.4.0 release (2024-02-14)
AXOLOTL_REGISTRY_SHA = (
    "d5b941ba2293534c01c23202c8fc459fd2a169871fa5e6c45cb00f363d474b6a"
)

base_image = (
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.4")
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.4")  # doesn't work for llava-next
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:sha-eade737")  # works for llava-next, but with errors on huggingface_hub integration
    # modal.Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.2")
    modal.Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .dockerfile_commands("ENTRYPOINT []")
    .dockerfile_commands("WORKDIR /repos")
    .run_commands(
        "git clone https://github.com/jmiemirza/MMFM-Challenge.git /repos/MMFM-Challenge",
        "cd /repos/MMFM-Challenge && pip install -r requirements.txt",
    )
    .run_commands(
        "cd /repos && git clone https://github.com/haotian-liu/LLaVA.git /repos/LLaVA",
        "cd /repos/LLaVA && git checkout ac89962d8fb191f42a0eed965a949c8bb316833a",
        "pip install --upgrade pip",
        "cd /repos/LLaVA && pip install -e .",
    )
    .run_commands(
        "ln -s /repos/LLaVA /repos/MMFM-Challenge/lib/LLaVA",
    )
    .dockerfile_commands(f"WORKDIR {WORK_DIR}")
    .apt_install("poppler-utils", "zip", "unzip")
)

phase_2_image = (
    base_image
    .run_commands(
        "mkdir -p data/raw_datasets",
        "wget -O phase2_data.zip \"https://drive.usercontent.google.com/download?id=1Nnh_5LN6wf_byJvINzf5CXRoLIj6DbEH&export=download&confirm=yes\"",
        "unzip phase2_data -d data/raw_datasets",
        "mv data/raw_datasets/phase2_data/* data/raw_datasets",
        "rm -rf data/raw_datasets/phase2_data",
    )
)

image = (
    phase_2_image
    # >> Option 1:
    .run_commands("bash download_process_merge.sh")
    # >> Option 2:
    # .copy_local_file("download_scripts/docvqa.sh", f"{WORK_DIR}/download_scripts")
    # .copy_local_file("download_scripts/infovqa.sh", f"{WORK_DIR}/download_scripts")
    # .copy_local_file("download_scripts/wtq.sh", f"{WORK_DIR}/download_scripts")
    # .copy_local_file("download_scripts/tabfact.sh", f"{WORK_DIR}/download_scripts")
    # .run_commands(
    #     f"DATASET_DIR={WORK_DIR}/data/raw_datasets \
    #         sh {WORK_DIR}/download_scripts/docvqa.sh & \
    #         sh {WORK_DIR}/download_scripts/infovqa.sh & \
    #         sh {WORK_DIR}/download_scripts/wtq.sh & \
    #         sh {WORK_DIR}/download_scripts/tabfact.sh & \
    #         sh {WORK_DIR}/download_scripts/websrc.sh & \
    #         sh {WORK_DIR}/download_scripts/funsd.sh & \
    #         sh {WORK_DIR}/download_scripts/iconqa.sh & \
    #         sh {WORK_DIR}/download_scripts/textbookqa.sh & \
    #         sh {WORK_DIR}/download_scripts/screen2words.sh & \
    #         sh {WORK_DIR}/download_scripts/wildreceipt.sh & wait",
    # )
    # .run_commands(f"bash process_data.sh {WORK_DIR}/data")
    # .run_commands(f"bash merge_data.sh {WORK_DIR}/data/")
    # >> If funsd fails to get processed.
    # .run_commands(
    #   "apt-get update && apt-get install libgl1",
    #   "python data_preprocessors/funsd.py --input_data_dir {WORK_DIR}/data/raw_datasets/funsd/dataset --main_data_dir {WORK_DIR}/data --out_data_dir {WORK_DIR}/data/processed_data/funsd",
    #   "bash merge_data.sh {WORK_DIR}/data/",
    # )
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("hf-write-secret")],
        timeout=3600,
    )
)

surya_image = (
    phase_2_image
    .pip_install("surya-ocr")
    .run_commands("apt-get update && apt-get install -y python3-opencv")
    .pip_install("opencv-python")
    .run_commands("FLASH_ATTENTION_FORCE_CXX11_ABI=1 FLASH_ATTENTION_FORCE_BUILD=1 pip install torch==2.2.1 flash-attn==2.5.5 --no-build-isolation")
)

INFERENCE_RESULTS_DIR = "/mmfm_inference_results"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    INFERENCE_RESULTS_DIR: modal.Volume.from_name(
        "mmfm-inference-results", create_if_missing=True
    )
}

app = modal.App(image=image)


@app.function(
    cpu=4.0, gpu=GPU_CONFIG, timeout=60*60*3, volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,
)
def run_inference_llava(
    datasets: str = "docvqa,infographicvqa,websrc,wtq,iconqa_fill_in_blank,funsd,iconqa_choose_txt,wildreceipt,textbookqa,tabfact",
    model_path: str = MODEL_ID,
    output_file_name: str = "output_datasets.json",
    data_path: str = "data/processed_data",
    test_file_name: str = "converted_output_test.json",
    eval_llava1_6: bool = False,
):
    print("Running inference on LLaVA...")

    # modal shell modal_llava_inference.py::run_inference_llava

    output_path = f"{INFERENCE_RESULTS_DIR}/{output_file_name}"
    print(f"{output_path = } | {INFERENCE_RESULTS_DIR = }")

    return_code = subprocess.check_call(
        f"""CUDA_VISIBLE_DEVICES=0 python eval_llava.py \
    --output_path  {output_path}\
    --data_path {data_path} \
    --model_path {model_path} \
    --test_file_name {test_file_name} \
    --sub_ds_list {datasets} \
    {"--eval_llava1_6" if eval_llava1_6 else ""}""",
        shell=True,
    )
    print(f"{return_code = }")

    VOLUME_CONFIG[INFERENCE_RESULTS_DIR].commit()

    with open(f"{INFERENCE_RESULTS_DIR}/{output_file_name}") as f:
        return f.read()


surya_app = modal.App(
    "surya",
    image=surya_image,
    volumes=VOLUME_CONFIG,
)


@surya_app.function(cpu=64, gpu=None, timeout=60*60*3, memory=16384)
def run_surya_extraction(
    dataset: str = "mydoc",
    languages: str = "en,ar",
    recognition_batch_size: int = 256,
    detector_batch_size: int = 48,
):
    return_code = subprocess.call(
        f"""RECOGNITION_BATCH_SIZE={recognition_batch_size} \
            DETECTOR_BATCH_SIZE={detector_batch_size} \
            surya_ocr data/raw_datasets/{dataset}/images \
            --langs {languages} \
            --results_dir {INFERENCE_RESULTS_DIR}/surya/{dataset}""",
        shell=True,
    )
    print(f"{return_code = }")

    VOLUME_CONFIG[INFERENCE_RESULTS_DIR].commit()
