export HF_ENDPOINT=https://hf-mirror.com
cfg1=without_context/starcoder2-7b/greedy
poetry run python ragc/test/inference.py\
    -t completion\
    -o /home/jovyan/work/diplom/EvoCodeBenchPlus/experiments/completions/racg/$cfg1.jsonl\
    -c configs/evocodebench/$cfg1.yml
cfg2=context/CodeLlama-7b-Python-hf/greedy
poetry run python ragc/test/inference.py\
    -t completion\
    -o /home/jovyan/work/diplom/EvoCodeBenchPlus/experiments/completions/racg/$cfg2.jsonl\
    -c configs/evocodebench/$cfg2.yml
cfg3=context/deepseek-coder-6.7b-base/greedy
poetry run python ragc/test/inference.py\
    -t completion\
    -o /home/jovyan/work/diplom/EvoCodeBenchPlus/experiments/completions/racg/$cfg3.jsonl\
    -c configs/evocodebench/$cfg3.yml
cfg4=context/starcoder2-7b/greedy
poetry run python ragc/test/inference.py\
    -t completion\
    -o /home/jovyan/work/diplom/EvoCodeBenchPlus/experiments/completions/racg/$cfg4.jsonl\
    -c configs/evocodebench/$cfg4.yml
