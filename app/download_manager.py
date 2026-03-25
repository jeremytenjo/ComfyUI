from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING
from urllib.parse import urlsplit

import aiohttp
from yarl import URL

import folder_paths

if TYPE_CHECKING:
    from server import PromptServer

logger = logging.getLogger(__name__)

ALLOWED_HTTPS_HOSTS = frozenset({
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-eu-1.huggingface.co",
    "civitai.com",
    "api.civitai.com",
})

ALLOWED_EXTENSIONS = frozenset({".safetensors", ".sft"})

MAX_CONCURRENT_DOWNLOADS = 3
MAX_TERMINAL_TASKS = 50
MAX_REDIRECTS = 10

DOWNLOAD_TEMP_SUFFIX = ".download_tmp"
DOWNLOAD_META_SUFFIX = ".download_meta"


class DownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


ACTIVE_STATUSES = frozenset({
    DownloadStatus.PENDING,
    DownloadStatus.DOWNLOADING,
    DownloadStatus.PAUSED,
})

TERMINAL_STATUSES = frozenset({
    DownloadStatus.COMPLETED,
    DownloadStatus.ERROR,
    DownloadStatus.CANCELLED,
})


@dataclass
class DownloadTask:
    id: str
    url: str
    filename: str
    directory: str
    save_path: str
    temp_path: str
    meta_path: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    received_bytes: int = 0
    total_bytes: int = 0
    speed_bytes_per_sec: float = 0.0
    eta_seconds: float = 0.0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    client_id: Optional[str] = None
    _worker: Optional[asyncio.Task] = field(default=None, repr=False)
    _stop_reason: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "filename": self.filename,
            "directory": self.directory,
            "status": self.status.value,
            "progress": self.progress,
            "received_bytes": self.received_bytes,
            "total_bytes": self.total_bytes,
            "speed_bytes_per_sec": self.speed_bytes_per_sec,
            "eta_seconds": self.eta_seconds,
            "error": self.error,
            "created_at": self.created_at,
        }


class DownloadManager:
    def __init__(self, server: PromptServer):
        self.server = server
        self.tasks: dict[str, DownloadTask] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=60)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        workers = [t._worker for t in self.tasks.values() if t._worker and not t._worker.done()]
        for w in workers:
            w.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self._session and not self._session.closed:
            await self._session.close()

    # -- Validation --

    @staticmethod
    def _validate_url(url: str) -> Optional[str]:
        try:
            parts = urlsplit(url)
        except Exception:
            return "Invalid URL"

        if parts.username or parts.password:
            return "Credentials in URL are not allowed"

        host = (parts.hostname or "").lower()
        scheme = parts.scheme.lower()

        if scheme != "https":
            return "Only HTTPS URLs are allowed"

        if host not in ALLOWED_HTTPS_HOSTS:
            return f"Host '{host}' is not in the allowed list"

        if parts.port not in (None, 443):
            return "Custom ports are not allowed for remote downloads"

        return None

    @staticmethod
    def _validate_filename(filename: str) -> Optional[str]:
        if not filename:
            return "Filename must not be empty"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return f"File extension '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        if os.path.sep in filename or (os.path.altsep and os.path.altsep in filename):
            return "Filename must not contain path separators"
        if ".." in filename:
            return "Filename must not contain '..'"
        for ch in filename:
            if ord(ch) < 32:
                return "Filename must not contain control characters"
        return None

    @staticmethod
    def _validate_directory(directory: str) -> Optional[str]:
        if directory not in folder_paths.folder_names_and_paths:
            valid = ', '.join(sorted(folder_paths.folder_names_and_paths.keys()))
            return f"Unknown model directory '{directory}'. Valid directories: {valid}"
        return None

    @staticmethod
    def _resolve_save_path(directory: str, filename: str) -> tuple[str, str, str]:
        """Returns (save_path, temp_path, meta_path) for a download."""
        paths = folder_paths.folder_names_and_paths[directory][0]
        base_dir = paths[0]
        os.makedirs(base_dir, exist_ok=True)

        save_path = os.path.join(base_dir, filename)
        temp_path = save_path + DOWNLOAD_TEMP_SUFFIX
        meta_path = save_path + DOWNLOAD_META_SUFFIX

        real_save = os.path.realpath(save_path)
        real_base = os.path.realpath(base_dir)
        if os.path.commonpath([real_save, real_base]) != real_base:
            raise ValueError("Resolved path escapes the model directory")

        return save_path, temp_path, meta_path

    # -- Sidecar metadata for resume validation --

    @staticmethod
    def _write_meta(meta_path: str, url: str, task_id: str):
        try:
            with open(meta_path, "w") as f:
                json.dump({"url": url, "task_id": task_id}, f)
        except OSError:
            pass

    @staticmethod
    def _read_meta(meta_path: str) -> Optional[dict]:
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _cleanup_files(*paths: str):
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    # -- Task management --

    def _prune_terminal_tasks(self):
        terminal = [
            (tid, t) for tid, t in self.tasks.items()
            if t.status in TERMINAL_STATUSES
        ]
        if len(terminal) > MAX_TERMINAL_TASKS:
            terminal.sort(key=lambda x: x[1].created_at)
            to_remove = len(terminal) - MAX_TERMINAL_TASKS
            for tid, _ in terminal[:to_remove]:
                del self.tasks[tid]

    async def start_download(
        self, url: str, directory: str, filename: str, client_id: Optional[str] = None
    ) -> tuple[Optional[DownloadTask], Optional[str]]:
        err = self._validate_url(url)
        if err:
            return None, err

        err = self._validate_filename(filename)
        if err:
            return None, err

        err = self._validate_directory(directory)
        if err:
            return None, err

        try:
            save_path, temp_path, meta_path = self._resolve_save_path(directory, filename)
        except ValueError as e:
            return None, str(e)

        if os.path.exists(save_path):
            return None, f"File already exists: {directory}/{filename}"

        # Reject duplicate active download by URL
        for task in self.tasks.values():
            if task.url == url and task.status in ACTIVE_STATUSES:
                return None, f"Download already in progress for this URL (id: {task.id})"

        # Reject duplicate active download by destination path (#4)
        for task in self.tasks.values():
            if task.save_path == save_path and task.status in ACTIVE_STATUSES:
                return None, f"Download already in progress for {directory}/{filename} (id: {task.id})"

        # Clean stale temp/meta if no active task owns them (#9)
        existing_meta = self._read_meta(meta_path)
        if existing_meta:
            owning_task = self.tasks.get(existing_meta.get("task_id", ""))
            if not owning_task or owning_task.status in TERMINAL_STATUSES:
                if existing_meta.get("url") != url:
                    self._cleanup_files(temp_path, meta_path)

        task = DownloadTask(
            id=uuid.uuid4().hex[:12],
            url=url,
            filename=filename,
            directory=directory,
            save_path=save_path,
            temp_path=temp_path,
            meta_path=meta_path,
            client_id=client_id,
        )
        self.tasks[task.id] = task
        self._prune_terminal_tasks()

        task._worker = asyncio.create_task(self._run_download(task))
        return task, None

    # -- Redirect-safe fetch (#1, #2, #3) --

    async def _fetch_with_validated_redirects(
        self, session: aiohttp.ClientSession, url: str, headers: dict
    ) -> aiohttp.ClientResponse:
        """Follow redirects manually, validating each hop against the allowlist."""
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            resp = await session.get(current_url, headers=headers, allow_redirects=False)
            if resp.status not in (301, 302, 303, 307, 308):
                return resp

            location = resp.headers.get("Location")
            await resp.release()
            if not location:
                raise ValueError("Redirect without Location header")

            resolved = URL(current_url).join(URL(location))
            current_url = str(resolved)

            # Validate the redirect target host
            parts = urlsplit(current_url)
            host = (parts.hostname or "").lower()
            scheme = parts.scheme.lower()

            if scheme != "https":
                raise ValueError(f"Redirect to non-HTTPS URL: {current_url}")
            if host not in ALLOWED_HTTPS_HOSTS:
                # Allow CDN hosts that HuggingFace/CivitAI commonly redirect to
                raise ValueError(f"Redirect to disallowed host: {host}")

            # 303 means GET with no Range
            if resp.status == 303:
                headers = {k: v for k, v in headers.items() if k.lower() != "range"}

        raise ValueError(f"Too many redirects (>{MAX_REDIRECTS})")

    # -- Download worker --

    async def _run_download(self, task: DownloadTask):
        try:
            async with self._semaphore:
                await self._run_download_inner(task)
        except asyncio.CancelledError:
            if task._stop_reason == "pause":
                task.status = DownloadStatus.PAUSED
                task.speed_bytes_per_sec = 0
                task.eta_seconds = 0
                await self._send_progress(task)
            else:
                task.status = DownloadStatus.CANCELLED
                await self._send_progress(task)
                self._cleanup_files(task.temp_path, task.meta_path)
        except Exception as e:
            task.status = DownloadStatus.ERROR
            task.error = str(e)
            await self._send_progress(task)
            logger.exception("Download error for %s", task.url)

    async def _run_download_inner(self, task: DownloadTask):
        session = await self._get_session()
        headers = {}

        # Resume support with sidecar validation (#9)
        if os.path.exists(task.temp_path):
            meta = self._read_meta(task.meta_path)
            if meta and meta.get("url") == task.url:
                existing_size = os.path.getsize(task.temp_path)
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                    task.received_bytes = existing_size
            else:
                self._cleanup_files(task.temp_path, task.meta_path)

        self._write_meta(task.meta_path, task.url, task.id)
        task.status = DownloadStatus.DOWNLOADING
        await self._send_progress(task)

        resp = await self._fetch_with_validated_redirects(session, task.url, headers)
        try:
            if resp.status == 416:
                content_range = resp.headers.get("Content-Range", "")
                if content_range:
                    total_str = content_range.split("/")[-1]
                    if total_str != "*":
                        total = int(total_str)
                        if task.received_bytes >= total:
                            if not os.path.exists(task.save_path):
                                os.rename(task.temp_path, task.save_path)
                                self._cleanup_files(task.meta_path)
                            task.status = DownloadStatus.COMPLETED
                            task.progress = 1.0
                            task.total_bytes = total
                            await self._send_progress(task)
                            return
                raise ValueError(f"HTTP 416 Range Not Satisfiable")

            if resp.status not in (200, 206):
                task.status = DownloadStatus.ERROR
                task.error = f"HTTP {resp.status}"
                await self._send_progress(task)
                return

            if resp.status == 200:
                task.received_bytes = 0

            content_length = resp.content_length
            if resp.status == 206 and content_length:
                task.total_bytes = task.received_bytes + content_length
            elif resp.status == 200 and content_length:
                task.total_bytes = content_length

            mode = "ab" if resp.status == 206 else "wb"
            speed_window_start = time.monotonic()
            speed_window_bytes = 0
            last_progress_time = 0.0

            with open(task.temp_path, mode) as f:
                async for chunk in resp.content.iter_chunked(1024 * 64):
                    f.write(chunk)
                    task.received_bytes += len(chunk)
                    speed_window_bytes += len(chunk)

                    now = time.monotonic()
                    elapsed = now - speed_window_start
                    if elapsed > 0.5:
                        task.speed_bytes_per_sec = speed_window_bytes / elapsed
                        if task.total_bytes > 0 and task.speed_bytes_per_sec > 0:
                            remaining = task.total_bytes - task.received_bytes
                            task.eta_seconds = remaining / task.speed_bytes_per_sec
                        speed_window_start = now
                        speed_window_bytes = 0

                    if task.total_bytes > 0:
                        task.progress = task.received_bytes / task.total_bytes

                    if now - last_progress_time >= 0.25:
                        await self._send_progress(task)
                        last_progress_time = now
        finally:
            resp.release()

        # Final cancel check before committing (#7)
        if task._stop_reason is not None:
            raise asyncio.CancelledError()

        # Re-check destination before finalizing (#10)
        if os.path.exists(task.save_path):
            task.status = DownloadStatus.ERROR
            task.error = f"Destination file appeared during download: {task.directory}/{task.filename}"
            await self._send_progress(task)
            return

        os.replace(task.temp_path, task.save_path)
        self._cleanup_files(task.meta_path)
        task.status = DownloadStatus.COMPLETED
        task.progress = 1.0
        task.speed_bytes_per_sec = 0
        task.eta_seconds = 0
        await self._send_progress(task)
        logger.info("Download complete: %s/%s", task.directory, task.filename)

    # -- Progress (#8, #14) --

    async def _send_progress(self, task: DownloadTask):
        try:
            self.server.send_sync("download_progress", task.to_dict(), task.client_id)
        except Exception:
            logger.exception("Failed to send download progress event")

    # -- Control operations (#5, #6, #13) --

    def pause_download(self, task_id: str) -> Optional[str]:
        task = self.tasks.get(task_id)
        if not task:
            return "Download not found"
        if task.status not in (DownloadStatus.PENDING, DownloadStatus.DOWNLOADING):
            return f"Cannot pause download in state '{task.status.value}'"
        task._stop_reason = "pause"
        if task._worker and not task._worker.done():
            task._worker.cancel()
        return None

    def resume_download(self, task_id: str) -> Optional[str]:
        task = self.tasks.get(task_id)
        if not task:
            return "Download not found"
        if task.status != DownloadStatus.PAUSED:
            return f"Cannot resume download in state '{task.status.value}'"
        task._stop_reason = None
        task.status = DownloadStatus.PENDING
        task._worker = asyncio.create_task(self._run_download(task))
        return None

    def cancel_download(self, task_id: str) -> Optional[str]:
        task = self.tasks.get(task_id)
        if not task:
            return "Download not found"
        if task.status in TERMINAL_STATUSES:
            return f"Cannot cancel download in state '{task.status.value}'"
        task._stop_reason = "cancel"
        if task._worker and not task._worker.done():
            task._worker.cancel()
        else:
            task.status = DownloadStatus.CANCELLED
            self._cleanup_files(task.temp_path, task.meta_path)
        return None

    # -- Query --

    def get_all_tasks(self, client_id: Optional[str] = None) -> list[dict]:
        tasks = self.tasks.values()
        if client_id is not None:
            tasks = [t for t in tasks if t.client_id == client_id]
        return [t.to_dict() for t in tasks]

    def get_task(self, task_id: str) -> Optional[dict]:
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None
