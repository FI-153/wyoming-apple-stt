"""Tests for the Siri TTS module: splitter, voice discovery, worker, and pool."""

import asyncio
import json
import stat
import sys

import pytest

from wyoming_apple_speech.tts import (
    SentenceSplitter,
    SiriVoice,
    TtsWorker,
    TtsWorkerError,
    TtsWorkerPool,
    discover_tts_voices,
    resolve_voice,
)

# A stand-in for the apple-tts binary: prints ready, then answers every
# synthesize command with two audio frames and a done frame.
FAKE_WORKER = """#!/usr/bin/env python3
import json, sys, os

def frame(header, payload=b""):
    sys.stdout.buffer.write(json.dumps(header).encode() + b"\\n" + payload)
    sys.stdout.buffer.flush()

if "--fail-start" in sys.argv:
    sys.exit(1)

frame({"type": "ready"})
for line in sys.stdin:
    cmd = json.loads(line)
    text = cmd["text"]
    if "explode" in text:
        frame({"type": "error", "message": "engine refused"})
        continue
    if "die" in text:
        os._exit(1)
    payload = text.encode()
    for _ in range(2):
        frame(
            {"type": "audio", "length": len(payload), "rate": 48000, "width": 2, "channels": 1},
            payload,
        )
    frame({"type": "done"})
"""


@pytest.fixture
def fake_worker_bin(tmp_path):
    """Write the fake apple-tts worker script and return its path."""
    path = tmp_path / "fake-apple-tts"
    path.write_text(FAKE_WORKER)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@pytest.fixture
def voices() -> list[SiriVoice]:
    """A small set of discovered voices."""
    return [
        SiriVoice(
            name="helena", language="de-DE", type="neural", footprint="premium",
            gender="female", version=1328, path="/assets/helena.asset",
        ),
        SiriVoice(
            name="aria", language="en-US", type="neural", footprint="premiumhigh",
            gender="female", version=1400, path="/assets/aria.asset",
        ),
    ]


# --- SentenceSplitter ---


def test_splitter_splits_complete_sentences():
    splitter = SentenceSplitter()
    sentences = splitter.add("Hello world. How are you? ")
    assert sentences == ["Hello world.", "How are you?"]


def test_splitter_holds_partial_sentence():
    splitter = SentenceSplitter()
    assert splitter.add("This is not finished") == []
    assert splitter.add(" yet. But this") == ["This is not finished yet."]
    assert splitter.finish() == "But this"


def test_splitter_finish_empty():
    splitter = SentenceSplitter()
    assert splitter.add("Complete. ") == ["Complete."]
    assert splitter.finish() is None


def test_splitter_handles_quotes_and_ellipsis():
    splitter = SentenceSplitter()
    sentences = splitter.add('Er sagte "Hallo!" Dann ging er… Und dann kam er wieder')
    assert sentences == ['Er sagte "Hallo!"', "Dann ging er…"]
    assert splitter.finish() == "Und dann kam er wieder"


def test_splitter_does_not_split_decimal_numbers():
    splitter = SentenceSplitter()
    assert splitter.add("Pi ist 3.14159 und mehr") == []


# --- Voice discovery / resolution ---


def test_voice_id_and_description(voices):
    assert voices[0].voice_id == "helena-de-DE-premium"
    assert "Helena" in voices[0].description
    assert "de-DE" in voices[0].description


async def test_discover_tts_voices(tmp_path):
    entries = [
        {
            "name": "helena", "language": "de-DE", "type": "neural",
            "footprint": "premium", "gender": "female", "version": 1328,
            "path": "/assets/helena.asset",
        }
    ]
    script = tmp_path / "list-voices"
    script.write_text(f"#!/bin/sh\necho '{json.dumps(entries)}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    discovered = await discover_tts_voices(str(script))
    assert len(discovered) == 1
    assert discovered[0].voice_id == "helena-de-DE-premium"


async def test_discover_tts_voices_missing_binary():
    assert await discover_tts_voices("/nonexistent/apple-tts") == []


def test_resolve_voice_by_id(voices):
    assert resolve_voice(voices, name="aria-en-US-premiumhigh") is voices[1]


def test_resolve_voice_by_bare_name(voices):
    assert resolve_voice(voices, name="aria") is voices[1]


def test_resolve_voice_by_language(voices):
    assert resolve_voice(voices, language="en-US") is voices[1]


def test_resolve_voice_by_short_language(voices):
    assert resolve_voice(voices, language="de") is voices[0]


def test_resolve_voice_returns_none_on_no_match(voices):
    assert resolve_voice(voices, name="nope", language="xx") is None


def test_resolve_voice_empty():
    assert resolve_voice([], name="any") is None


# --- TtsWorker ---


async def test_worker_synthesize_streams_frames(fake_worker_bin):
    worker = TtsWorker(fake_worker_bin)
    await worker.start(timeout=10)
    try:
        frames = [
            frame
            async for frame in worker.synthesize(
                text="hello", voice_path="/assets/x.asset", timeout=10
            )
        ]
        assert len(frames) == 2
        assert frames[0].audio == b"hello"
        assert frames[0].rate == 48000
    finally:
        await worker.stop()


async def test_worker_error_frame_raises(fake_worker_bin):
    worker = TtsWorker(fake_worker_bin)
    await worker.start(timeout=10)
    try:
        with pytest.raises(TtsWorkerError, match="engine refused"):
            async for _ in worker.synthesize(
                text="explode", voice_path="/assets/x.asset", timeout=10
            ):
                pass
    finally:
        await worker.stop()


async def test_worker_death_raises(fake_worker_bin):
    worker = TtsWorker(fake_worker_bin)
    await worker.start(timeout=10)
    try:
        with pytest.raises(TtsWorkerError):
            async for _ in worker.synthesize(
                text="die", voice_path="/assets/x.asset", timeout=10
            ):
                pass
    finally:
        await worker.stop()


async def test_worker_start_failure_raises():
    failing = TtsWorker("/nonexistent/apple-tts")
    with pytest.raises(TtsWorkerError):
        await failing.start(timeout=5)


async def test_worker_immediate_death_hints_full_disk_access(tmp_path):
    """A worker dying before its ready line is the Full Disk Access signature:
    the error must name the fix and the exact Python binary to grant."""
    path = tmp_path / "dying-apple-tts"
    path.write_text("#!/bin/sh\nexit 1\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)

    worker = TtsWorker(str(path))
    with pytest.raises(TtsWorkerError, match="Full Disk Access") as excinfo:
        await worker.start(timeout=10)
    assert sys.executable in str(excinfo.value)


# --- TtsWorkerPool ---


async def test_pool_prewarms_idle_worker(fake_worker_bin):
    pool = TtsWorkerPool(fake_worker_bin, idle_target=1)
    await pool.start()
    try:
        assert len(pool._idle) == 1
        assert pool._idle[0].is_alive
    finally:
        await pool.stop()


async def test_pool_acquire_returns_warm_worker_and_replenishes(fake_worker_bin):
    pool = TtsWorkerPool(fake_worker_bin, idle_target=1)
    await pool.start()
    try:
        warm_worker = pool._idle[0]
        worker = await pool.acquire()
        assert worker is warm_worker  # no spawn wait on the request path

        # A replacement spawn was scheduled in the background.
        await asyncio.gather(*pool._spawn_tasks)
        assert len(pool._idle) == 1
        assert pool._idle[0] is not worker

        await pool.release(worker)
        # Pool is full again, so the released worker is disposed.
        assert len(pool._idle) == 1
        assert not worker.is_alive
    finally:
        await pool.stop()


async def test_pool_release_returns_worker_when_idle_low(fake_worker_bin):
    pool = TtsWorkerPool(fake_worker_bin, idle_target=2)
    await pool.start()
    try:
        worker = await pool.acquire()
        second = await pool.acquire()
        idle_before = list(pool._idle)
        await pool.release(worker)
        assert worker in pool._idle or worker in idle_before
        await pool.release(second)
    finally:
        await pool.stop()


async def test_pool_acquire_spawns_when_empty(fake_worker_bin):
    pool = TtsWorkerPool(fake_worker_bin, idle_target=1)
    # No start(): pool is empty, acquire must spawn synchronously.
    worker = await pool.acquire()
    try:
        assert worker.is_alive
    finally:
        await pool.release(worker)
        await pool.stop()


async def test_pool_stop_kills_idle_workers(fake_worker_bin):
    pool = TtsWorkerPool(fake_worker_bin, idle_target=1)
    await pool.start()
    worker = pool._idle[0]
    await pool.stop()
    assert not worker.is_alive
