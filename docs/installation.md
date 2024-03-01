# Install Instructions
#### Configure the environment for data download and LLaVA inference
```bash
git clone https://us-south.git.cloud.ibm.com/leonidka1/doc-vl-eval.git
conda create -y -n doc-vl-eval python=3.10
conda activate doc-vl-eval
cd doc-vl-eval
pip install -r requirements.txt

# Install the LLaVA dependencies, tested with CUDA 12.1.1
# clone the LLaVA repository in parellel to the doc-vl-eval repository
cd .. && git clone https://github.com/haotian-liu/LLaVA.git
# checkout to an older version of the LLaVA repo which is compatible with the LLaVA 1.5 model
cd LLaVA && git checkout ac89962d8fb191f42a0eed965a949c8bb316833a
# create a symbolic link to the LLaVA repository
cd .. && ln -s LLaVA doc-vl-eval/lib/LLaVA
# environment configuration
cd LLaVA
pip install --upgrade pip
pip install -e .

# optional : Install additional packages for training cases
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

#### Configure the environment for the LLM - Mixtral 8x7b
```bash
# Tested with CUDA 12.1.1
conda create -n mixtral python=3.12 -y
conda activate mixtral
pip install --upgrade pip
pip install transformers==4.38.1
pip3 install torch torchvision torchaudio
pip install bitsandbytes tqdm
```
Note that the Mixtral environment requires a higher version of `transformers` than the LLaVA environment.