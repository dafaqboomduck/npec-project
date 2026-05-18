"""Object storage backends.

Provides a unified interface for saving binary artifacts and
receiving back a URL the frontend can fetch.

Two implementations:
- LocalStorageBackend   - writes to disk, returns file:// URI
                          (development only, frontend cannot fetch these)
- MinIOStorageBackend   - uploads to MinIO, returns a presigned HTTPS URL

Backend is selected via the STORAGE_BACKEND environment variable.


MinIO environment variables
---------------------------
MINIO_ENDPOINT      Host:port of the MinIO server.  Default: localhost:9000
MINIO_ROOT_USER    Access key (username).          Default: minioadmin
MINIO_ROOT_PASSWORD    Secret key (password).          Default: minioadmin
MINIO_BUCKET        Bucket name.                    Default: npec-artifacts
MINIO_SECURE        Use HTTPS (true/false).         Default: false
MINIO_PRESIGN_TTL   Presigned URL TTL in seconds.   Default: 604800 (7 days)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Config


STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "minio")

MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ROOT_USER: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_ROOT_PASSWORD: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "npec-artifacts")
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_PRESIGN_TTL: int = int(os.getenv("MINIO_PRESIGN_TTL", str(7 * 24 * 3600)))

# Local fallback base path
LOCAL_STORAGE_BASE = Path(os.getenv("LOCAL_STORAGE_PATH", "./data"))


# Abstract interface


class AbstractStorageBackend(ABC):
    """Interface for storing binary artifacts and retrieving their URLs."""

    @abstractmethod
    def upload(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload `data` and return a URL the frontend can fetch.

        Args:
            object_name: Logical path/name for the object.
            data: Raw bytes to store.
            content_type: MIME type of the data.

        Returns:
            A URL string - presigned HTTPS URL (MinIO) or `file://` URI
            (local). `None` is never returned; callers must wrap in
            try/except and handle failure themselves.

        Raises:
            Exception: On any storage failiure.
        """

    @abstractmethod
    def download(self, uri: str) -> bytes:
        """Fetch the raw bytes stored as `uri`.

        Args:
            uri: The URI returned by a previous `upload()` call.

        Returns:
            Raw bytes of the stored object.

        Raises:
            Exception: On any storage failure.
        """


# Local implementation - writes to disk


class LocalStorageBackend(AbstractStorageBackend):
    """Writes files to a local directory tree."""

    def __init__(self, base_path: Path | None = None) -> None:
        """Writes files to a local directory tree.

        Returns `file://` URIs. The frontend cannot fetch these directly
        over HTTP - use `GET /files/{object_name}` as a proxy, or switch to
        MinIO for a proper HTTP URL.

        Args:
            base_path: Root directory under which object names are resolved.
                Defaults to `LOCAL_STORAGE_BASE`.
        """
        self._base = base_path or LOCAL_STORAGE_BASE

    def upload(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write `data` to `{base}/{object_name}` and return a file:// URI.

        Args:
            object_name: Relative path under the base directory.
            data: Raw bytes to write.
            content_type: Ignored for local storage.

        Returns:
            Absolute `file://` URI of the written file.
        """
        dest = self._base / object_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.debug("LocalStorageBackend: wrote %s", dest)
        return dest.resolve().as_uri()

    def download(self, uri: str) -> bytes:
        """Read bytes from a file:// URI or absolute path.

        Args:
            uri: file:// URI or absolute path.

        Returns:
            Raw bytes of the file.
        """
        path = Path(uri.replace("file://", "")) if uri.startswith("file://") else Path(uri)
        return path.read_bytes()


# MinIO implementation


class MinIOStorageBackend(AbstractStorageBackend):
    """Uploads artifacts to MinIO and returns presigned HTTPS URLs."""

    def __init__(
        self,
        endpoint: str = MINIO_ENDPOINT,
        access_key: str = MINIO_ROOT_USER,
        secret_key: str = MINIO_ROOT_PASSWORD,
        bucket: str = MINIO_BUCKET,
        secure: bool = MINIO_SECURE,
        presign_ttl_seconds: int = MINIO_PRESIGN_TTL,
    ) -> None:
        """Uploads artifacts to MinIO and returns presigned HTTPS URLs.

        The bucket is created automatically on first use if it does not exist.
        The MinIO client is initialised lazily so that import-time failures
        (e.g. `minio` not installed) do not crash the server.

        Args:
            endpoint: MinIO server host:port.
            access_key: MinIO access key.
            secret_key: MinIO secret key.
            bucket: Bucket name.
            secure: Whether to use HTTPS.
            presign_ttl_seconds: Presigned URL time-to-live in seconds.
        """
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._secure = secure
        self._ttl = timedelta(seconds=presign_ttl_seconds)
        self._client = None

    def _get_client(self):
        """Return the MinIO client, initialising it on first call.

        Also ensures the configured bucket exists, creating it if not.

        Returns:
            An initialised `minio.Minio` client.
        """
        if self._client is None:
            from minio import Minio

            self._client = Minio(
                self._endpoint,
                access_key=self._access_key,
                secret_key=self._secret_key,
                secure=self._secure,
            )
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
                logger.info("MinIOStorageBackend: created bucket '%s'", self._bucket)
        return self._client

    def upload(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload `data` to MinIO and return a presigned URL.

        Args:
            object_name: Object key withing the bucket.
            data: Raw bytes to upload.
            content_type: MIME type stored as object metadata.

        Returns:
            A presigned HTTPS URL valid for `presign_ttl_seconds`.
        """
        import io

        client = self._get_client()
        client.put_object(
            bucket_name=self._bucket,
            object_name=object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        url = client.presigned_get_object(
            bucket_name=self._bucket,
            object_name=object_name,
            expires=self._ttl,
        )
        logger.debug("MinIOStorageBackend: uploaded '%s' -> %s", object_name, url[:80])
        return url

    def download(self, uri: str) -> bytes:
        """Fetch object bytes from MinIO.

        Extracts the object name from the presigned URL or plain object key
        and downloads the raw bytes.

        Args:
            uri: Presigned URL or object key previously returned by upload().

        Returns:
            Raw bytes of the stored object.
        """
        from urllib.parse import urlparse

        client = self._get_client()
        # Presigned URLs contain the object key as the URL path
        parsed = urlparse(uri)
        object_name = parsed.path.lstrip("/")
        # Strip bucket name prefix if present in the path
        if object_name.startswith(self._bucket + "/"):
            object_name = object_name[len(self._bucket) + 1 :]

        response = client.get_object(self._bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


# Factory - called once by dependencies


def get_storage_backend() -> AbstractStorageBackend:
    """Return the configured storage backend.

    Controlled by the `STORAGE_BACKEND` environment variable.
    - `local` -> `LocalStorageBackend`
    - `minio` -> `MinIOStorageBackend`

    Returns:
        The active `AbstractStorageBackend` implementation.
    """
    if STORAGE_BACKEND == "minio":
        logger.info("Storage backend: MinIO (%s / bucket=%s)", MINIO_ENDPOINT, MINIO_BUCKET)
        return MinIOStorageBackend()
    logger.info("Storage backend: local (%s)", LOCAL_STORAGE_BASE)
    return LocalStorageBackend()
