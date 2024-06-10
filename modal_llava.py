# # Hosting any LLaMA 3 model with Text Generation Inference (TGI)
#
# In this example, we show how to run an optimized inference server using [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
# with performance advantages over standard text generation pipelines including:
# - continuous batching, so multiple generations can take place at the same time on a single container
# - PagedAttention, which applies memory paging to the attention mechanism's key-value cache, increasing throughput
#
# This example deployment, [accessible here](https://modal.chat), can serve LLaMA 3 70B with
# 70 second cold starts, up to 200 tokens/s of throughput, and a per-token latency of 55ms.

# ## Setup
#
# First we import the components we need from `modal`.

import os
import subprocess
from pathlib import Path

from modal import App, Image, Mount, Secret, asgi_app, enter, exit, gpu, method

# Next, we set which model to serve, taking care to specify the GPU configuration required
# to fit the model into VRAM, and the quantization method (`bitsandbytes` or `gptq`) if desired.
# Note that quantization does degrade token generation performance significantly.
#
# Any model supported by TGI can be chosen here.

# MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
# MODEL_ID = "lmms-lab/llama3-llava-next-8b"
MODEL_ID = "llava-hf/llava-v1.6-vicuna-13b-hf"
# MODEL_REVISION = "81ca4500337d94476bda61d84f0c93af67e4495f"
# MODEL_REVISION = "e7e6a9fd5fd75d44b32987cba51c123338edbede"
MODEL_REVISION = "e74a6636922814614b48ec8b7693e5bdd3ea08cf"
# Add `["--quantize", "gptq"]` for TheBloke GPTQ models.
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--revision",
    MODEL_REVISION,
    "--sharded=false",
    "--tokenizer-config-path",
    MODEL_ID,
]

# ## Define a container image
#
# We want to create a Modal image which has the Huggingface model cache pre-populated.
# The benefit of this is that the container no longer has to re-download the model from Huggingface -
# instead, it will take advantage of Modal's internal filesystem for faster cold starts. On
# the largest 70B model, the 135GB model can be loaded in as little as 70 seconds.
#
# ### Download the weights
# We can use the included utilities to download the model weights (and convert to safetensors, if necessary)
# as part of the image build.
#


def download_model():
    subprocess.run(
        [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ],
    )


# ### Image definition
# We’ll start from a Docker Hub image recommended by TGI, and override the default `ENTRYPOINT` for
# Modal to run its own which enables seamless serverless deployments.
#
# Next we run the download step to pre-populate the image with our model weights.
#
# For this step to work on a [gated model](https://github.com/huggingface/text-generation-inference#using-a-private-or-gated-model)
# such as LLaMA 3, the `HF_TOKEN` environment variable must be set.
#
# After [creating a HuggingFace access token](https://huggingface.co/settings/tokens)
# and accepting the [LLaMA 3 license](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct),
# head to the [secrets page](https://modal.com/secrets) to share it with Modal
#
# Finally, we install the `text-generation` client to interface with TGI's Rust webserver over `localhost`.

app = App(
    "example-tgi-" + MODEL_ID.split("/")[-1]
)  # Note: prior to April 2024, "app" was called "stub"

tgi_image = (
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.4")
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.4")  # doesn't work for llava-next
    # Image.from_registry("ghcr.io/huggingface/text-generation-inference:sha-eade737")  # works for llava-next, but with errors on huggingface_hub integration
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:2.0.2")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(
        download_model,
        secrets=[Secret.from_name("hf-write-secret")],
        timeout=3600,
    )
    .pip_install("text-generation", "huggingface_hub==0.23.3")
)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).
# The class syntax is a special representation for a Modal function which splits logic into two parts:
# 1. the `@enter()` function, which runs once per container when it starts up, and
# 2. the `@method()` function, which runs per inference request.
#
# This means the model is loaded into the GPUs, and the backend for TGI is launched just once when each
# container starts, and this state is cached for each subsequent invocation of the function.
# Note that on start-up, we must wait for the Rust webserver to accept connections before considering the
# container ready.
#
# Here, we also
# - specify the secret so the `HUGGING_FACE_HUB_TOKEN` environment variable can be set
# - specify how many A100s we need per container
# - specify that each container is allowed to handle up to 10 inputs (i.e. requests) simultaneously
# - keep idle containers for 10 minutes before spinning down
# - increase the timeout limit


GPU_CONFIG = gpu.H100(count=2)  # 2 H100s


@app.cls(
    secrets=[Secret.from_name("hf-write-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
)
class Model:
    @enter()
    def start_server(self):
        import socket
        import time

        from text_generation import AsyncClient
        # from huggingface_hub import InferenceClient

        self.launcher = subprocess.Popen(
            ["text-generation-launcher"] + LAUNCH_FLAGS,
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            },
        )
        self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
        # self.client = InferenceClient("http://127.0.0.1:8000", timeout=60)
        self.template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        # Poll until webserver at 127.0.0.1:8000 accepts connections before running inputs.
        def webserver_ready():
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                return True
            except (socket.timeout, ConnectionRefusedError):
                # Check if launcher webserving process has exited.
                # If so, a connection can never be made.
                retcode = self.launcher.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"launcher exited unexpectedly with code {retcode}"
                    )
                return False

        while not webserver_ready():
            time.sleep(1.0)

        print("Webserver ready!")

    @exit()
    def terminate_server(self):
        self.launcher.terminate()

    @method()
    async def generate(self, image_bytes_str: str, question: str):
        from text_generation.types import Message
        prompt = self.template.format(user=question)
        result = await self.client.generate(
            prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
        )
        # return result.generated_text
        result = await self.client.chat(
            messages=[
                Message(role="system", content="You are an infographics explainer. You will receive an image as an input and you must answer the user's question based on the image. Be concise and limit responses to at most 3 sentences, preferably one sentence long. Respond in English."),
                Message(
                    role="user",
                    content=[
                        {"type": "image", "image": image_bytes_str},
                        {"type": "text", "text": question},
                    ],
                ),
            ]
        )

        # from huggingface_hub import ChatCompletionOutput
        # result: ChatCompletionOutput = self.client.chat_completion(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": "Whats in this image?"},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
        #                     },
        #                 },
        #             ],
        #         },
        #     ],
        #     seed=42,
        #     max_tokens=100,
        # )
        # print(result)
        # return result.choices[0].message.content

    @method()
    async def generate_stream(self, question: str):
        prompt = self.template.format(user=question)

        async for response in self.client.generate_stream(
            prompt, max_new_tokens=1024, stop_sequences=["<|eot_id|>"]
        ):
            if (
                not response.token.special
                and response.token.text != "<|eot_id|>"
            ):
                yield response.token.text


# ## Run the model
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps) to invoke
# our remote function. You can run this script locally with `modal run text_generation_inference.py`.
@app.local_entrypoint()
def main(prompt: str = None):
    if prompt is None:
        prompt = "Implement a Python function to compute the Fibonacci numbers."
    print(Model().generate.remote(prompt))


# ## Serve the model
# Once we deploy this model with `modal deploy text_generation_inference.py`, we can serve it
# behind an ASGI app front-end. The front-end code (a single file of Alpine.js) is available
# [here](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-frontend/index.html).
#
# You can try our deployment [here](https://modal.chat).

# frontend_path = Path(__file__).parent.parent / "llm-frontend"
# 
# 
# @app.function(
#     mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
#     keep_warm=1,
#     allow_concurrent_inputs=10,
#     timeout=60 * 10,
# )
# @asgi_app(label="llama3")
# def tgi_app():
#     import json
# 
#     import fastapi
#     import fastapi.staticfiles
#     from fastapi.responses import StreamingResponse
# 
#     web_app = fastapi.FastAPI()
# 
#     @web_app.get("/stats")
#     async def stats():
#         stats = await Model().generate_stream.get_current_stats.aio()
#         return {
#             "backlog": stats.backlog,
#             "num_total_runners": stats.num_total_runners,
#             "model": MODEL_ID,
#         }
# 
#     @web_app.get("/completion/{question}")
#     async def completion(question: str):
#         from urllib.parse import unquote
# 
#         async def generate():
#             async for text in Model().generate_stream.remote_gen.aio(
#                 unquote(question)
#             ):
#                 yield f"data: {json.dumps(dict(text=text), ensure_ascii=False)}\n\n"
# 
#         return StreamingResponse(generate(), media_type="text/event-stream")
# 
#     web_app.mount(
#         "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
#     )
#     return web_app


# ## Invoke the model from other apps
# Once the model is deployed, we can invoke inference from other apps, sharing the same pool
# of GPU containers with all other apps we might need.
#
# ```
# $ python
# >>> import modal
# >>> f = modal.Function.lookup("example-tgi-Meta-Llama-3-70B-Instruct", "Model.generate")
# >>> f.remote("What is the story about the fox and grapes?")
# 'The story about the fox and grapes ...
# ```
