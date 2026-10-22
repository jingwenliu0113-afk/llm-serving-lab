"""A throwaway self-signed certificate for the TLS tests (decision 038 §6): openssl writes it into the test's own
temporary directory, which pytest deletes; never a production certificate or key."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def self_signed(directory: Path, *, cn: str = "capserve-test") -> tuple:
    """(certfile, keyfile) of a fresh self-signed certificate for CN / SAN localhost, 127.0.0.1."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    cert, key = directory / "cert.pem", directory / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", f"/CN={cn}",
                    "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(key), "-out", str(cert)],
                   check=True, capture_output=True)
    return str(cert), str(key)
