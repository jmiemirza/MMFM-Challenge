import os
import modal
import subprocess

GPU_TYPE = os.environ.get("GPU_CONFIG", "a100-80gb")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)
GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

WORK_DIR = "/repos/MMFM-Challenge"

base_image = (
    modal.Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.2")
    .dockerfile_commands("ENTRYPOINT []")
    .dockerfile_commands("WORKDIR /repos")
    # >> Pull repos
    .apt_install("git")
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
    # >> Prepare CWD
    .dockerfile_commands(f"WORKDIR {WORK_DIR}")
    .apt_install("poppler-utils", "zip", "unzip", "wget")
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

surya_image = (
    phase_2_image
    .pip_install("surya-ocr")
    # >> Fixes "ImportError: libGL.so.1"
    .run_commands("apt-get update && apt-get install -y python3-opencv")
    .pip_install("opencv-python")
)

INFERENCE_RESULTS_DIR = "/mmfm_inference_results"
VOLUME_CONFIG: dict[str, modal.Volume] = {
    INFERENCE_RESULTS_DIR: modal.Volume.from_name(
        "mmfm-inference-results", create_if_missing=True
    )
}

surya_app = modal.App(
    "surya",
    image=surya_image,
)


@surya_app.function(
    cpu=20, gpu=GPU_CONFIG, timeout=60*60*3, memory=16384,
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,
)
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
