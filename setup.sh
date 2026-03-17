#!/bin/bash

export HF_HOME=hf_cache
export HF_TOKEN=your_token_here

module load cuda/12.3

uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
