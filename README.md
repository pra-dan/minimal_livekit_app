# RAG App — Minimal Speech Bot
Building for Ubuntu 24, driver versio 550.163.01.

**NOTE: This is not a realtime model-based app**

This is a minimal speech-in/speech-out bot using LiveKit Agents, modeled after `conversify`. It wires STT -> LLM -> TTS using OpenAI-compatible endpoints.


## Prerequisites
- Python 3.10+
- LiveKit Server running (local or cloud)
- An OpenAI-compatible LLM endpoint (e.g., SGLang, vLLM, Ollama w/ openai API)

## Setup
Set local env: 
```
cp .env.example .env.local
```

We will create a common docker image/container for both LLM and app.

```bash
cd rag_app
sudo docker build -t rapp:v0 .
```
But the base image I used, is missing `cudnn` (`find / -name libcudnn_ops_infer.so*` returns nothing!). We need to install it. Inside container
```bash
sudo docker run -it --gpus all --network host docker.io/library/rapp:v0
apt-get update
# try simple installation
apt install nvidia-cudnn
```
if you get "forbidden IP" errors, its probably because the ubuntu repositories are mentioned as http links and not https. We fix this manually.
```bash
# backup
cp /etc/apt/sources.list /etc/apt/sources.list.backup
nano /etc/apt/sources.list
# -> now replace every http with https 
apt update
apt install nvidia-cudnn
find / -name libcudnn_ops_infer.so*
# should return 
# /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8
# /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.2.4
# /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so
# /tmp/nvidia-cudnn/cuda/lib64/libcudnn_ops_infer.so.8
# /tmp/nvidia-cudnn/cuda/lib64/libcudnn_ops_infer.so.8.2.4
# /tmp/nvidia-cudnn/cuda/lib64/libcudnn_ops_infer.so
```
*I wanted to make this part of Dockerfile but editing the sources.list via script would make it harder for future reader (including me).*. I want to commit these changes into an image.

We don't setup open source & local instances for STT or TTS for now.

### Auto (for TTS)
Using [easiest method](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-sarvam) by directly using "Sarvam.ai" for TTS. So I signed-up and got the API-KEY. The plugin installation is handled in `requirements.txt`. Just save the KEY in `.env.local`. My final env file looks like this
```bash
LIVEKIT_API_KEY=APIBJ2NC...4
LIVEKIT_API_SECRET=0fOS0QOJ7...vikNlnyWjsinEOHe9Gv
LIVEKIT_URL=wss://test-os3..z5c.livekit.cloud
ASSEMBLYAI_API_KEY=f1cc818ea2..5d1e2cc36
SARVAM_API_KEY=sk_e5zjorg7_6...mMCMIut
```

### FasterWhisper for STT
<!-- I wanted to use Qwen-Omni to have just one model for TTS+LLM, but not now. For STT, I wanted to use AssemblyAI endpoint or FasterWhisper but as the objective is a quickstart, I will keep them for later. Currently, I use [automatic model selection](https://docs.livekit.io/agents/models/stt/#automatic-model-selection) provided by LiveKit. For custom model, the same doc should be referred. -->
We write a wrapper (`MinimalWhisperSTT`) for the FasterWhisper method such that it inherits the structure of LiveKit's `stt.STT`.

As we use (Faster)Whisper-STT, or any other *non-streaming* models, we need to add a [VAD or Voice Activity Detection](https://github.com/snakers4/silero-vad) model (e.g. `silero` here).

Update `config.yaml` with your endpoints and model names.


Now we'll launch 2 terminals - for LLM serving and main app.

## Run
### SGLang (for LLM) - Terminal 1
Added [instructions](https://docs.sglang.ai/get_started/install.html) for linux to Dockerfile. Build this
```bash
# Launch using
sudo docker run -it --gpus all --network host docker.io/library/rapp:v0
# if docker image exists, use 
# sudo docker run -it --gpus all --network host rapp_min_env:v1
# Inside container
bash scripts/run_llm.sh 
# Runs at http://127.0.0.1:30000
```

Note that `--network host` only works for linux. For other OS, `-p 30000:30000` should work.

Test it using python+curl
```bash
$ python3 test_sglang.py
{'id': '2513443d7e2b480b97f5612772e3561e', 'object': 'chat.completion', 'created': 1759490544, 'model': 'qwen/qwen2.5-0.5b-instruct', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "<think>\nOkay, the user is asking about the capital of France. I know France's capital is Paris. But wait, maybe they're confused with another country? Let me confirm. The capital of France is indeed Paris. No, that's correct. I should state that clearly and maybe explain why it's the capital. Oh, and maybe mention that it's also the most populous city in the world. But the user just asked for the answer, so maybe just state the capital without extra details. Let me check the facts again. Yes, Paris is the capital and most populous city. So the answer is Paris.\n</think>\n\nThe capital of France is **Paris**.", 'reasoning_content': None, 'tool_calls': None}, 'logprobs': None, 'finish_reason': 'stop', 'matched_stop': 151645}], 'usage': {'prompt_tokens': 15, 'total_tokens': 152, 'completion_tokens': 137, 'prompt_tokens_details': None, 'reasoning_tokens': 0}, 'metadata': {'weight_version': 'default'}}
```

<!-- ### Kokoro (for TTS)
```bash
bash /home/prashant/Documents/asmr/rag_app/scripts/run_kokoro.sh
```
For me it runs at `http://0.0.0.0:8880` -->

### LiveKit - Terminal 2
Mounting local code for now 
```bash
sudo docker run -it --gpus all -v /home/prashant/Documents/asmr/rag_app:/root/apps/rag_app --network host rapp_min_env:v1
```

Create a token (example):
```bash
lk token create \
  --api-key devkey --api-secret secret \
  --join --room my-first-room --identity user1 --valid-for 24h
# Returns
#Project URL: http://localhost:7880
#Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTk0MDMxOTIsImlkZW50aXR5IjoidXNlcjEiLCJpc3MiOiJkZXZrZXkiLCJuYW1lIjoidXNlcjEiLCJuYmYiOjE3NTkzMTY3OTIsInN1YiI6InVzZXIxIiwidmlkZW8iOnsicm9vbSI6Im15LWZpcnN0LXJvb20iLCJyb29tSm9pbiI6dHJ1ZX19.NBT7Dtna2T1jC8SdSQAQrXUjxjpE7R1_IqhgOoz1USQ
# Not sure where to use it. Just keep it for now.


```

<!-- Start a local LiveKit server (example):
```bash
conda activate parakeet
livekit-server --dev
``` -->
<!-- # point to VENV's local CUDA 11.8 python lib
export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib
/home/prashant/anaconda3/pkgs/pytorch-1.13.1-py3.7_cuda11.7_cudnn8.5.0_0/lib/python3.7/site-packages/torch/lib/libcudnn_ops_infer.so.8 -->

Start playground
- Sign up on `https://cloud.livekit.io/`
- Next, start playground *room* here `https://agents-playground.livekit.io/`
- Find the room id on right (e.g. `playground-37zS-5BS9`)
- Obtain LIVEKIT credentials: Go to [docs](https://docs.livekit.io/agents/start/voice-ai/) and on any of the code snippets, click on "Reveal API Key and secret". See below SS. Not sure if this is the best way.
![livekit-creds](data/livekit_creds.png) 


Now paste the Livekit credentials in `.env.local`.

## Run
```bash
# conda activate parakeet
# You have 4 options - 
lk cloud auth # Link your LiveKit Cloud project to the CLI
# You will be given a link to authenticate
# Rather using the same container 'rapp'
sudo docker run -it --gpus all --network host rapp_min_env:v1
python3 run.py connect --room=playground-kpLt-B1xk
```

Then visit the LiveKit Agents Playground and join the room to talk with the bot.

## Notes
- LLM uses OpenAI chat completions API. Configure `llm.base_url`, `llm.api_key`, and `llm.model` in `config.yaml`.
- The default model of Whisper sucks. See SS below 
![default_fails](data/def_stt_llm_suck.png)
The one that actually listens and understands is `"deepdml/faster-whisper-large-v3-turbo-ct2"`.