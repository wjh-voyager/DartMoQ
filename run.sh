#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --nactivated 8 --nexperts 64 --nsamples 64 --quant-scheme a8s8m8
# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --nactivated 16 --nexperts 128 --nsamples 64 --quant-scheme a8s8m84
# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --nactivated 16 --nexperts 128 --nsamples 64 --quant-scheme a8s8m42

# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --nactivated 32 --nexperts 256 --nsamples 64 --quant-scheme a8s8m4222
# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --nactivated 32 --nexperts 256 --nsamples 64 --quant-scheme global

# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 6 --nexperts 64 --nsamples 64 --quant-scheme a8s8m8
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 6 --nexperts 64 --nsamples 64 --quant-scheme a8s4m4
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 12 --nexperts 128 --nsamples 64 --quant-scheme a8s4m84
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme a8s4m8444
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme a8s4m4222
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme a8s8m4222
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 12 --nexperts 128 --nsamples 64 --quant-scheme global
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme global
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --nactivated 48 --nexperts 512 --nsamples 64 --quant-scheme global

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2  --nactivated 6 --nexperts 64 --nsamples 64 --quant-scheme a8s8m8
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --nactivated 12 --nexperts 128 --nsamples 64 --quant-scheme a8s4m42
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme a8s8m4222
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --nactivated 24 --nexperts 256 --nsamples 64 --quant-scheme global
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --nactivated 48 --nexperts 512 --nsamples 64 --quant-scheme global
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --nactivated 96 --nexperts 1024 --nsamples 64 --quant-scheme global

# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2  --nactivated 48 --nexperts 512 --nsamples 64 --quant-scheme global
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2  --nactivated 96 --nexperts 1024 --nsamples 64 --quant-scheme global
