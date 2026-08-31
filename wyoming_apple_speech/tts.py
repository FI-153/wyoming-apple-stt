"""Siri TTS support: voice discovery, sentence splitting, and the warm worker pool."""

import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Optional

_LOGGER = logging.getLogger(__name__)

# Pipe buffer limit for worker stdout; audio frames can be large.
_STREAM_LIMIT = 8 * 1024 * 1024

# PCM format the Siri engine produces for current neural voices; used for the
# audio envelope when no frame was produced (e.g. empty input). Actual frames
# carry their own format.
SAMPLE_RATE = 48_000
SAMPLE_WIDTH = 2
CHANNELS = 1


@dataclass
class SiriVoice:
    """A system-managed Siri voice as reported by `apple-tts --list-voices`."""

    name: str
    language: str
    type: str
    footprint: str
    gender: str
    version: int
    path: str

    @property
    def voice_id(self) -> str:
        """Unique, stable identifier used as the Wyoming voice name."""
        return f"{self.name}-{self.language}-{self.footprint}"

    @property
    def description(self) -> str:
        """Human-readable description shown in Home Assistant."""
        return f"{self.name.capitalize()} ({self.language}, {self.type}, {self.footprint})"


async def discover_tts_voices(bin_path: str) -> list[SiriVoice]:
    """Query the apple-tts CLI for compatible system Siri voices.

    Runs the binary with --list-voices and parses the JSON array. Returns an
    empty list on any failure so the server can start STT-only.

    Args:
        bin_path: Path to the apple-tts Swift CLI binary.

    Returns:
        List of discovered voices, possibly empty.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            bin_path, "--list-voices",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        if process.returncode != 0:
            _LOGGER.warning(
                "TTS voice discovery failed (exit %s): %s",
                process.returncode,
                stderr.decode().strip() or "(no stderr output)",
            )
            return []
        entries = json.loads(stdout.decode())
        voices = [SiriVoice(**entry) for entry in entries]
        _LOGGER.debug(
            "Discovered %d system Siri voices: %s",
            len(voices),
            [voice.voice_id for voice in voices],
        )
        return voices
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass
        _LOGGER.warning("TTS voice discovery timed out")
    except (json.JSONDecodeError, TypeError, OSError) as exc:
        _LOGGER.warning("TTS voice discovery failed: %s", exc)
    return []


def resolve_voice(
    voices: list[SiriVoice],
    name: Optional[str] = None,
    language: Optional[str] = None,
) -> Optional[SiriVoice]:
    """Pick the voice matching a requested name or language.

    Name matches against the voice id first, then the bare voice name.
    Language matches the full tag first (case-insensitive), then the bare
    language code ("de" matches "de-DE").

    Args:
        voices: Available voices.
        name: Requested voice name/id, if any.
        language: Requested language tag, if any.

    Returns:
        The matching voice, or None when nothing matches.
    """
    if name:
        for voice in voices:
            if voice.voice_id == name:
                return voice
        for voice in voices:
            if voice.name == name:
                return voice

    if language:
        lowered = language.lower()
        for voice in voices:
            if voice.language.lower() == lowered:
                return voice
        short = lowered.split("-")[0]
        for voice in voices:
            if voice.language.lower().split("-")[0] == short:
                return voice

    return None


class SentenceSplitter:
    """Incremental sentence-boundary splitter for streaming text input.

    Text chunks are appended with add(); complete sentences (terminated by
    ., !, ?, or … followed by whitespace) are returned as soon as they are
    available, and finish() flushes whatever remains.
    """

    _BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")

    def __init__(self) -> None:
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        """Append a text chunk and return any newly completed sentences."""
        self._buffer += text
        sentences = []
        while match := self._BOUNDARY.search(self._buffer):
            sentence = self._buffer[: match.end()].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[match.end():]
        return sentences

    def finish(self) -> Optional[str]:
        """Return the remaining partial sentence, if any, and reset."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None


@dataclass
class TtsService:
    """Everything the event handler needs to serve TTS requests."""

    pool: "TtsWorkerPool"
    voices: list[SiriVoice]
    default_voice: SiriVoice
    rate: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    timeout: float = 60.0


@dataclass
class AudioFrame:
    """One chunk of PCM audio produced by a worker."""

    audio: bytes
    rate: int
    width: int
    channels: int


class TtsWorkerError(Exception):
    """Raised when a worker fails to start, dies, or reports a synthesis error."""


class TtsWorker:
    """One apple-tts worker subprocess with a preheated engine.

    The worker is started once, keeps its engines cached for its lifetime,
    and serves one synthesize command at a time over its stdin/stdout pipe.
    """

    def __init__(self, bin_path: str, preload_voice_path: Optional[str] = None) -> None:
        self._bin_path = bin_path
        self._preload_voice_path = preload_voice_path
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

    @property
    def is_alive(self) -> bool:
        """Whether the worker process is running."""
        return self._process is not None and self._process.returncode is None

    async def start(self, timeout: float = 60) -> None:
        """Launch the worker and wait until its engine is preheated.

        Args:
            timeout: Max seconds to wait for the worker's ready signal.

        Raises:
            TtsWorkerError: When the worker cannot be launched or never
                becomes ready.
        """
        command = [self._bin_path]
        if self._preload_voice_path:
            command += ["--preload-voice", self._preload_voice_path]

        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
            )
        except OSError as exc:
            raise TtsWorkerError(f"failed to launch {self._bin_path}: {exc}") from exc

        self._stderr_task = asyncio.create_task(self._drain_stderr())

        try:
            header = await asyncio.wait_for(self._read_header(), timeout=timeout)
        except (asyncio.TimeoutError, TtsWorkerError) as exc:
            await self.stop()
            message = f"worker did not become ready: {exc}"
            if not isinstance(exc, asyncio.TimeoutError):
                # The worker's output pipe broke, i.e. the process died at
                # startup — the signature of the Python binary lacking Full
                # Disk Access (Siri voices load a TCC-protected model cache).
                message += (
                    " — a TTS worker dying at startup usually means the Python"
                    f" running this server ({sys.executable}) lacks Full Disk"
                    " Access. Grant it in System Settings → Privacy & Security"
                    " → Full Disk Access and restart the service; re-grant"
                    " after every Python upgrade (the path changes). See the"
                    " README's TTS section."
                )
            raise TtsWorkerError(message) from exc

        if header.get("type") != "ready":
            await self.stop()
            raise TtsWorkerError(f"unexpected worker greeting: {header}")

        _LOGGER.debug("TTS worker ready (pid %s)", self._process.pid)

    async def synthesize(
        self,
        text: str,
        voice_path: str,
        rate: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        timeout: float = 60,
    ) -> AsyncIterator[AudioFrame]:
        """Synthesize text, yielding PCM frames as the engine produces them.

        Args:
            text: Text to speak.
            voice_path: Absolute path of the voice's .asset directory.
            rate: Speaking rate multiplier.
            pitch: Pitch multiplier.
            volume: Volume multiplier.
            timeout: Max seconds for the whole synthesis.

        Yields:
            AudioFrame chunks in production order.

        Raises:
            TtsWorkerError: When the worker dies or reports an error.
        """
        if not self.is_alive:
            raise TtsWorkerError("worker is not running")
        assert self._process is not None and self._process.stdin is not None

        command = {
            "command": "synthesize",
            "text": text,
            "voice_path": voice_path,
            "rate": rate,
            "pitch": pitch,
            "volume": volume,
        }
        self._process.stdin.write((json.dumps(command) + "\n").encode())
        await self._process.stdin.drain()

        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TtsWorkerError(f"synthesis timed out after {timeout}s")
            try:
                header = await asyncio.wait_for(self._read_header(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TtsWorkerError(f"synthesis timed out after {timeout}s") from exc

            frame_type = header.get("type")
            if frame_type == "audio":
                assert self._process.stdout is not None
                audio = await self._process.stdout.readexactly(int(header["length"]))
                yield AudioFrame(
                    audio=audio,
                    rate=int(header.get("rate", 48000)),
                    width=int(header.get("width", 2)),
                    channels=int(header.get("channels", 1)),
                )
            elif frame_type == "done":
                return
            elif frame_type == "error":
                raise TtsWorkerError(header.get("message", "unknown synthesis error"))
            else:
                raise TtsWorkerError(f"unexpected worker frame: {header}")

    async def stop(self) -> None:
        """Terminate the worker process and clean up."""
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._process is None:
            return
        process, self._process = self._process, None
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _read_header(self) -> dict:
        """Read one JSON header line from the worker's stdout."""
        assert self._process is not None and self._process.stdout is not None
        line = await self._process.stdout.readline()
        if not line:
            raise TtsWorkerError("worker closed its output (process died?)")
        try:
            header: dict = json.loads(line)
            return header
        except json.JSONDecodeError as exc:
            raise TtsWorkerError(f"invalid worker frame: {line[:200]!r}") from exc

    async def _drain_stderr(self) -> None:
        """Forward the worker's stderr lines to the debug log."""
        assert self._process is not None and self._process.stderr is not None
        try:
            while line := await self._process.stderr.readline():
                _LOGGER.debug("%s", line.decode(errors="replace").rstrip())
        except (asyncio.CancelledError, ValueError):
            pass


class TtsWorkerPool:
    """Keeps pre-warmed apple-tts workers so synthesis starts with zero delay.

    The pool maintains idle_target idle workers with preheated engines.
    acquire() hands out a warm worker and immediately begins spawning a
    replacement in the background, so a concurrent second request never
    waits for engine initialization.
    """

    def __init__(
        self,
        bin_path: str,
        preload_voice_path: Optional[str] = None,
        idle_target: int = 1,
    ) -> None:
        self._bin_path = bin_path
        self._preload_voice_path = preload_voice_path
        self._idle_target = max(1, idle_target)
        self._idle: list[TtsWorker] = []
        self._spawning = 0
        self._spawn_tasks: set[asyncio.Task] = set()
        self._closed = False

    async def start(self) -> None:
        """Pre-warm the initial idle workers, waiting until they are ready."""
        workers = await asyncio.gather(
            *(self._spawn_worker() for _ in range(self._idle_target)),
            return_exceptions=True,
        )
        for worker in workers:
            if isinstance(worker, BaseException):
                _LOGGER.warning("Failed to pre-warm TTS worker: %s", worker)
            else:
                self._idle.append(worker)

    async def acquire(self) -> TtsWorker:
        """Take a warm worker, triggering a background replacement spawn.

        Returns:
            A ready worker. The caller must hand it back via release().

        Raises:
            TtsWorkerError: When no worker is available and a fresh spawn fails.
        """
        worker: Optional[TtsWorker] = None
        while self._idle and worker is None:
            candidate = self._idle.pop()
            if candidate.is_alive:
                worker = candidate
            else:
                await candidate.stop()

        self._replenish()

        if worker is None:
            worker = await self._spawn_worker()
        return worker

    async def release(self, worker: TtsWorker) -> None:
        """Return a worker to the pool, or dispose of it if surplus or dead."""
        if self._closed or not worker.is_alive or len(self._idle) >= self._idle_target:
            await worker.stop()
        else:
            self._idle.append(worker)

    async def stop(self) -> None:
        """Shut down all idle workers and cancel pending spawns."""
        self._closed = True
        for task in self._spawn_tasks:
            task.cancel()
        self._spawn_tasks.clear()
        idle, self._idle = self._idle, []
        await asyncio.gather(*(worker.stop() for worker in idle))

    def _replenish(self) -> None:
        """Spawn background workers until the idle target is covered."""
        if self._closed:
            return
        while len(self._idle) + self._spawning < self._idle_target:
            self._spawning += 1
            task = asyncio.create_task(self._spawn_into_pool())
            self._spawn_tasks.add(task)
            task.add_done_callback(self._spawn_tasks.discard)

    async def _spawn_into_pool(self) -> None:
        """Spawn one worker and park it in the idle pool."""
        try:
            worker = await self._spawn_worker()
        except TtsWorkerError as exc:
            _LOGGER.warning("Failed to spawn replacement TTS worker: %s", exc)
            return
        finally:
            self._spawning -= 1

        if self._closed or len(self._idle) >= self._idle_target:
            await worker.stop()
        else:
            self._idle.append(worker)

    async def _spawn_worker(self) -> TtsWorker:
        """Create and start one worker with the preload voice."""
        worker = TtsWorker(self._bin_path, self._preload_voice_path)
        await worker.start()
        return worker
