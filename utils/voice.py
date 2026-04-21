import asyncio
import json
import threading
import pyaudio
import websockets
import config

# Deepgram WebSocket URL with query params (as per official docs)
# Feature flags go on the URL, body carries only audio frames
WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=en-US"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&interim_results=true"
    "&punctuate=true"
    "&endpointing=300"
    "&utterance_end_ms=1000"
    "&vad_events=true"
)


class VoiceListener:
    def __init__(self):
        self.api_key = config.DEEPGRAM_API_KEY

    def _listen_sync(self) -> str:
        """
        Synchronous worker. Opens a raw WebSocket to Deepgram, streams mic
        audio, and blocks until speech_final is received.
        Returns the full transcript string.
        """
        transcript_parts = []
        is_done = threading.Event()

        async def _run():
            headers = {"Authorization": f"Token {self.api_key}"}

            async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                print("\rListening... (Speak now)", end="", flush=True)

                # --- Mic thread: captures audio and sends to Deepgram ---
                stop_mic = threading.Event()

                def stream_mic():
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024
                    )
                    try:
                        while not stop_mic.is_set():
                            data = stream.read(1024, exception_on_overflow=False)
                            # Never send empty payloads (per docs — terminates connection)
                            if data:
                                asyncio.run_coroutine_threadsafe(ws.send(data), loop)
                    finally:
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                loop = asyncio.get_event_loop()
                mic_thread = threading.Thread(target=stream_mic, daemon=True)
                mic_thread.start()

                # --- Main loop: receive transcription messages ---
                try:
                    async for message in ws:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "Results":
                            try:
                                sentence = data["channel"]["alternatives"][0]["transcript"]
                            except (KeyError, IndexError):
                                continue

                            if not sentence:
                                continue

                            is_final = data.get("is_final", False)
                            speech_final = data.get("speech_final", False)

                            if is_final:
                                transcript_parts.append(sentence)
                                if speech_final:
                                    break  # Done — exit the receive loop
                            else:
                                # Show interim results while user is talking
                                print(f"\rListening: {sentence}...", end="", flush=True)

                        elif msg_type == "UtteranceEnd":
                            # Fallback: if speech_final never fires, end on utterance boundary
                            if transcript_parts:
                                break

                finally:
                    stop_mic.set()
                    mic_thread.join(timeout=2)

        asyncio.run(_run())

        # Clear the "Listening..." line
        print("\r" + " " * 60 + "\r", end="")
        return " ".join(transcript_parts).strip()

    async def get_input(self) -> str:
        """
        Async wrapper — runs the sync listener in a thread pool so the
        asyncio event loop (and your agent) are never blocked.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._listen_sync)
