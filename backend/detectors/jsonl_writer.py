"""Streaming JSONL writer shared by detector runtimes."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

try:
    import orjson  # type: ignore
except Exception:  # pragma: no cover - optional acceleration
    orjson = None


class JsonlWriter:
    def __init__(
        self,
        path: Path | None = None,
        backend: str = "json",
        *,
        output_path: Path | None = None,
        orjson_option: int = 0,
    ) -> None:
        raw_target = output_path if output_path is not None else path
        if raw_target is None:
            raise ValueError("path or output_path is required")
        target = Path(raw_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.backend = str(backend)
        self.orjson_option = int(orjson_option)
        if self.backend == "orjson":
            if orjson is None:
                raise RuntimeError("json-backend=orjson requested but orjson is not installed")
            self._bin = target.open("wb")
            self._txt = None
        else:
            self._txt = target.open("w", encoding="utf-8")
            self._bin = None
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        if self.backend == "orjson":
            self._bin.write(orjson.dumps(record, option=self.orjson_option))  # type: ignore[union-attr]
            self._bin.write(b"\n")  # type: ignore[union-attr]
        else:
            self._txt.write(json.dumps(record, ensure_ascii=False) + "\n")  # type: ignore[union-attr]

    def flush(self) -> None:
        if self.backend == "orjson":
            self._bin.flush()  # type: ignore[union-attr]
        else:
            self._txt.flush()  # type: ignore[union-attr]

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        if self.backend == "orjson":
            self._bin.close()  # type: ignore[union-attr]
        else:
            self._txt.close()  # type: ignore[union-attr]
        self._closed = True


class AsyncJsonlWriter:
    def __init__(
        self,
        path: Path | None = None,
        backend: str = "json",
        queue_size: int = 512,
        *,
        output_path: Path | None = None,
        max_queue: int | None = None,
        orjson_option: int = 0,
    ) -> None:
        raw_target = output_path if output_path is not None else path
        if raw_target is None:
            raise ValueError("path or output_path is required")
        self._target = Path(raw_target)
        self.backend = str(backend)
        self.orjson_option = int(orjson_option)
        size = max_queue if max_queue is not None else queue_size
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max(1, int(size)))
        self._stop = object()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="jsonl_writer", daemon=True)
        self._thread.start()

    def _set_error(self, exc: BaseException) -> None:
        if self._error is None:
            self._error = exc

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("async JSON writer failed") from self._error

    def _worker(self) -> None:
        writer: JsonlWriter | None = None
        try:
            writer = JsonlWriter(self._target, self.backend, orjson_option=self.orjson_option)
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        return
                    writer.write(item)  # type: ignore[arg-type]
                except BaseException as exc:
                    self._set_error(exc)
                    raise
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._set_error(exc)
            self._drain_queue()
        finally:
            if writer is not None:
                writer.close()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def write(self, record: dict[str, Any]) -> None:
        self._raise_if_error()
        self._queue.put(record)
        self._raise_if_error()

    def flush(self) -> None:
        self._queue.join()
        self._raise_if_error()

    def close(self) -> None:
        if self._closed:
            self._raise_if_error()
            return
        if self._error is None:
            self._queue.join()
            self._queue.put(self._stop)
        else:
            self._drain_queue()
            try:
                self._queue.put_nowait(self._stop)
            except queue.Full:
                pass
        self._thread.join()
        self._closed = True
        self._raise_if_error()


def make_jsonl_writer(
    path: Path,
    backend: str,
    *,
    async_writer: bool,
    queue_size: int = 512,
    orjson_option: int = 0,
) -> JsonlWriter | AsyncJsonlWriter:
    if async_writer:
        return AsyncJsonlWriter(path, backend, queue_size=queue_size, orjson_option=orjson_option)
    return JsonlWriter(path, backend, orjson_option=orjson_option)
