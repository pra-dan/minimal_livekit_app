import logging
import functools
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    JobContext,
    RoomInputOptions,
    RoomOutputOptions,
    cli,
    WorkerOptions,
    Agent,
    stt,
    tts,
    llm,
    ChatContext,
    utils,
)
from livekit.agents.types import NOT_GIVEN, DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit import rtc
from livekit.plugins import silero
from livekit.plugins import sarvam

import openai
import httpx
import yaml
import soundfile as sf
from faster_whisper import WhisperModel
import io

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _ensure_openai_base_url(url: str) -> str:
    # Normalize and ensure single /v1 suffix
    normalized = url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def _to_openai_messages(chat_ctx: ChatContext) -> List[dict]:
    messages: List[dict] = []
    for item in getattr(chat_ctx, "items", []):
        if getattr(item, "type", "message") != "message":
            continue
        role = getattr(item, "role", "user")
        content = getattr(item, "content", [])
        text_parts: List[str] = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, str):
                    text_parts.append(c)
                elif hasattr(c, "text") and isinstance(c.text, str):
                    text_parts.append(c.text)
        elif isinstance(content, str):
            text_parts.append(content)
        text = "\n".join(p for p in text_parts if p)
        messages.append({"role": role, "content": text})
    return messages


class MinimalAgent(Agent):
    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions, allow_interruptions=True)

    async def on_enter(self):
        await self.session.say("Hello! How can I help you today?")

    async def on_exit(self):
        await self.session.say("Goodbye!")


class MinimalWhisperSTT(stt.STT):
    def __init__(self, *, language: Optional[str] = "en", model_size_or_path: str = "base.en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._language = language
        self._model = WhisperModel(model_size_or_path=model_size_or_path, device="auto")
        # logger = logging.getLogger(__name__)
        logger.info(f"STT Whisper model {model_size_or_path}")

    async def _recognize_impl(
        self,
        buffer: "stt.AudioBuffer",  # type: ignore[name-defined]
        *,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        audio_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        audio_array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        segments, _ = self._model.transcribe(audio_array, language=language or self._language, beam_size=1, best_of=1)
        text = " ".join(seg.text.strip() for seg in segments)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text or "", language=language or self._language)],
        )


TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1


# class MinimalOpenAITTS(tts.TTS):
#     def __init__(self, *, model: str, voice: str, base_url: str, api_key: str, speed: float = 1.0):
#         super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=TTS_SAMPLE_RATE, num_channels=TTS_CHANNELS)
#         self._model = model
#         self._voice = voice
#         self._speed = speed
#         self._client = openai.AsyncClient(
#             api_key=api_key,
#             base_url=_ensure_openai_base_url(base_url),
#             http_client=httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)),
#             max_retries=0,
#         )

#     def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
#         return _OpenAITTSStream(self, text, conn_options)


# class _OpenAITTSStream(tts.ChunkedStream):
#     def __init__(self, tts_engine: MinimalOpenAITTS, text: str, conn_options: APIConnectOptions):
#         super().__init__(tts=tts_engine, input_text=text, conn_options=conn_options)
#         self._engine = tts_engine

#     async def _run(self, *_args, **_kwargs):
#         stream = self._engine._client.audio.speech.with_streaming_response.create(
#             input=self.input_text,
#             model=self._engine._model,
#             voice=self._engine._voice,
#             response_format="pcm",
#             speed=self._engine._speed,
#         )
#         request_id = "req"
#         audio_bstream = utils.audio.AudioByteStream(sample_rate=TTS_SAMPLE_RATE, num_channels=TTS_CHANNELS)
#         async with stream as s:
#             async for data in s.iter_bytes():
#                 for frame in audio_bstream.write(data):
#                     self._event_ch.send_nowait(tts.SynthesizedAudio(frame=frame, request_id=request_id))
#             for frame in audio_bstream.flush():
#                 self._event_ch.send_nowait(tts.SynthesizedAudio(frame=frame, request_id=request_id))


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_llm(config: Dict[str, Any]) -> llm.LLM:
    llm_cfg = config["llm"]
    timeout = httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0)
    client = openai.AsyncClient(
        api_key=llm_cfg["api_key"],
        base_url=_ensure_openai_base_url(llm_cfg["base_url"]),
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=timeout, follow_redirects=True),
    )

    class SimpleOpenAILLM(llm.LLM):
        def __init__(self):
            super().__init__()

        def chat(self, *, chat_ctx: ChatContext, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kwargs):
            return _LLMStream(self, model=llm_cfg["model"], client=client, chat_ctx=chat_ctx)

    class _LLMStream(llm.LLMStream):
        def __init__(self, llm, *, model: str, client: openai.AsyncClient, chat_ctx: ChatContext):
            super().__init__(llm, chat_ctx=chat_ctx, tools=[], conn_options=DEFAULT_API_CONNECT_OPTIONS)
            self._model = model
            self._client = client

        async def _run(self) -> None:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=_to_openai_messages(self._chat_ctx),
                stream=True,
            )
            async with stream as s:
                async for chunk in s:
                    for choice in chunk.choices:
                        delta = choice.delta
                        if delta and delta.content:
                            self._event_ch.send_nowait(llm.ChatChunk(id=chunk.id, delta=llm.ChoiceDelta(role="assistant", content=delta.content)))

    return SimpleOpenAILLM()


async def entrypoint(ctx: JobContext, config: Dict[str, Any]):
    await ctx.connect()

    stt_cfg = config.get("stt", {})
    tts_cfg = config.get("tts", {})

    logger.info(f"STT model config: {stt_cfg.get('model')}")
    session = AgentSession(
        llm=build_llm(config),
        stt=MinimalWhisperSTT(language=stt_cfg.get("language", "en"), model_size_or_path=stt_cfg.get("model", "base.en")),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            speaker="anushka",
        ),
        vad=silero.VAD.load(),
        turn_detection=NOT_GIVEN,
    )

    agent = MinimalAgent(instructions=config["agent"]["instructions"]) 

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


def main():
    # Set up logging to file (already set above, but ensure here for entrypoint)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler("rag_app.log"),
            logging.StreamHandler()
        ]
    )
    config = load_config();print(config)
    load_dotenv(config["agent"]["env_file"])  # optional

    worker_options = WorkerOptions(
        entrypoint_fnc=functools.partial(entrypoint, config=config)
    )
    cli.run_app(worker_options)


if __name__ == "__main__":
    main()