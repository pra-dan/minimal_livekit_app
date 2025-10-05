# python3 -m sglang.launch_server \
# --model-path Qwen/Qwen3-0.6B \
# --mem-fraction-static=0.4 \
# --cuda-graph-max-bs=4 \
# --disable-cuda-graph \
# --tool-call-parser qwen25

python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8880 \
  --mem-fraction-static=0.4 \
  --cuda-graph-max-bs=4 \
  --disable-cuda-graph \
  --tool-call-parser qwen25
