import os
import time
import warnings
from uuid import uuid4

import modal
import requests

GPU_TYPE = os.environ.get("GPU_CONFIG", "a10g")
GPU_COUNT = os.environ.get("GPU_COUNT", 1)

GPU_CONFIG = f"{GPU_TYPE}:{GPU_COUNT}"

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

MODEL_PATH = "lmms-lab/llama3-llava-next-8b"
# MODEL_PATH = "openbmb/MiniCPM-Llama3-V-2_5"
# MODEL_PATH = "google/paligemma-3b-mix-448"
# MODEL_PATH = "liuhaotian/llava-v1.6-34b"
# MODEL_REVISION = "e7e6a9fd5fd75d44b32987cba51c123338edbede"
TOKENIZER_PATH = "lmms-lab/llama3-llava-next-8b-tokenizer"
# TOKENIZER_PATH = "openbmb/MiniCPM-Llama3-V-2_5"
# TOKENIZER_PATH = "google/paligemma-3b-mix-448"
# TOKENIZER_PATH = "liuhaotian/llava-v1.6-34b-tokenizer"
# MODEL_CHAT_TEMPLATE = "gemma-it"
MODEL_CHAT_TEMPLATE = "llama-3-instruct"


def download_model_to_image():
    import transformers
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_PATH,
        # revision=MODEL_REVISION,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.bin", "*.gguf"],
    )

    # otherwise, this happens on first inference
    transformers.utils.move_cache()


vllm_image = (
    modal.Image.from_registry(  # start from an official NVIDIA CUDA image
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")  # add system dependencies
    .pip_install(  # add sglang and some Python dependencies
        "sglang[all]==0.1.16",
        "ninja",
        "packaging",
        "wheel",
        # "transformers==4.40.2",
        "transformers==4.41.2",
        "torchvision",  # NOTE: new addition
        "datasets",  # NOTE: new addition
    )
    .run_commands(  # add FlashAttention for faster inference using a shell command
        "pip install flash-attn==2.5.8 --no-build-isolation"
    )
    .run_function(  # download the model by running a Python function
        download_model_to_image,
        secrets=[modal.Secret.from_name("hf-write-secret")],
    )
    .pip_install(  # add an optional extra that renders images in the terminal
        "term-image==0.7.1",
    )
)


app = modal.App(
    "app",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-write-secret"),
    ],
)


@app.cls(
    # gpu=GPU_CONFIG,
    timeout=20 * MINUTES,
    container_idle_timeout=20 * MINUTES,
    allow_concurrent_inputs=100,
    image=vllm_image,
)
class Model:
    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        """Starts an SGL runtime to execute inference."""
        import sglang as sgl

        self.runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tokenizer_path=TOKENIZER_PATH,
            tp_size=GPU_COUNT,  # t_ensor p_arallel size, number of GPUs to split the model over
            log_evel=SGL_LOG_LEVEL,
            disable_regex_jump_forward=True,
        )
        self.runtime.endpoint.chat_template = (
            sgl.lang.chat_template.get_chat_template(MODEL_CHAT_TEMPLATE)
        )
        sgl.set_default_backend(self.runtime)

    @modal.web_endpoint(method="POST")
    def generate(self, request: dict):
        import json
        import sglang as sgl
        from outlines.fsm.json_schema import build_regex_from_schema
        from term_image.image import from_file

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")
        print(request)

        image_url = request.get("image_url")
        if image_url is None:
            image_url = "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
        
        json_schema = request.get("json_schema")
        if json_schema is None:
            json_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            }

        image_filename = image_url.split("/")[-1]
        image_path = f"/tmp/{uuid4()}-{image_filename}"
        response = requests.get(image_url)

        response.raise_for_status()

        with open(image_path, "wb") as file:
            file.write(response.content)

        @sgl.function
        def image_qa(s, image_path, question):
            s += sgl.user(sgl.image(image_path) + question)
            s += sgl.assistant(
                sgl.gen(
                    "answer",
                    max_tokens=256,
                    stop="<|eot_id|>",
                    regex=build_regex_from_schema(json.dumps(json_schema)),
                )
            ) 

        question = request.get("question")
        if question is None:
            question = "What is this?"

        state = image_qa.run(
            image_path=image_path, question=question, max_new_tokens=128
        )
        # show the question, image, and response in the terminal for demonstration purposes
        print(
            Colors.BOLD, Colors.GRAY, "Question: ", question, Colors.END, sep=""
        )
        terminal_image = from_file(image_path)
        terminal_image.draw()
        answer = state["answer"]
        print(
            Colors.BOLD,
            Colors.GREEN,
            f"Answer: {answer}",
            Colors.END,
            sep="",
        )
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()


@app.local_entrypoint()
def main(image_url=None, question=None, twice=True):
    model = Model()

    response = requests.post(
        model.generate.web_url,
        json={
            "image_url": image_url,
            "question": question,
        },
    )
    assert response.ok, response.status_code

    if twice:
        # second response is faster, because the Function is already running
        response = requests.post(
            model.generate.web_url,
            json={"image_url": image_url, "question": question},
        )
        assert response.ok, response.status_code


warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
