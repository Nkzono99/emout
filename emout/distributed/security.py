"""TLS helpers for secure ``emout server`` sessions."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .server_state import ensure_private_dir, server_certs_dir, write_private_bytes


@dataclass(frozen=True)
class ClusterSecurityFiles:
    """Filesystem paths for per-session TLS material."""

    certs_dir: Path
    ca_cert: Path
    ca_key: Path
    scheduler_cert: Path
    scheduler_key: Path
    worker_cert: Path
    worker_key: Path
    client_cert: Path
    client_key: Path

    @classmethod
    def for_session(cls, server_name: str) -> ClusterSecurityFiles:
        cert_dir = server_certs_dir(server_name)
        return cls(
            certs_dir=cert_dir,
            ca_cert=cert_dir / "ca-cert.pem",
            ca_key=cert_dir / "ca-key.pem",
            scheduler_cert=cert_dir / "scheduler-cert.pem",
            scheduler_key=cert_dir / "scheduler-key.pem",
            worker_cert=cert_dir / "worker-cert.pem",
            worker_key=cert_dir / "worker-key.pem",
            client_cert=cert_dir / "client-cert.pem",
            client_key=cert_dir / "client-key.pem",
        )

    def all_files(self) -> tuple[Path, ...]:
        return (
            self.ca_cert,
            self.ca_key,
            self.scheduler_cert,
            self.scheduler_key,
            self.worker_cert,
            self.worker_key,
            self.client_cert,
            self.client_key,
        )

    def cluster_kwargs(self) -> dict[str, str]:
        """Return CLI-friendly TLS file paths for scheduler / worker / client."""
        return {
            "ca_file": str(self.ca_cert),
            "scheduler_cert": str(self.scheduler_cert),
            "scheduler_key": str(self.scheduler_key),
            "worker_cert": str(self.worker_cert),
            "worker_key": str(self.worker_key),
            "client_cert": str(self.client_cert),
            "client_key": str(self.client_key),
        }

    def client_state(self) -> dict[str, Any]:
        """Return the subset stored in persisted server state."""
        return {
            "ca_file": str(self.ca_cert),
            "client_cert": str(self.client_cert),
            "client_key": str(self.client_key),
            "require_encryption": True,
        }


def _require_cryptography():
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise RuntimeError(
            "Secure emout servers require the 'cryptography' package. "
            "Install the latest emout dependencies and try again."
        ) from exc

    return x509, hashes, serialization, rsa, ExtendedKeyUsageOID, NameOID


def _write_pem_pair(cert_path: Path, key_path: Path, cert_bytes: bytes, key_bytes: bytes) -> None:
    write_private_bytes(cert_path, cert_bytes)
    write_private_bytes(key_path, key_bytes)


def ensure_cluster_security(server_name: str, scheduler_host: str | None = None) -> ClusterSecurityFiles:
    """Create or reuse per-session TLS credentials."""
    files = ClusterSecurityFiles.for_session(server_name)
    ensure_private_dir(files.certs_dir)

    if all(path.exists() for path in files.all_files()):
        for path in files.all_files():
            path.chmod(0o600)
        return files

    x509, hashes, serialization, rsa, ExtendedKeyUsageOID, NameOID = _require_cryptography()

    def _serialize_key(key: Any) -> bytes:
        return key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def _serialize_cert(cert: Any) -> bytes:
        return cert.public_bytes(serialization.Encoding.PEM)

    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=365)

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "emout dask local ca")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(expires)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key=ca_key, algorithm=hashes.SHA256())
    )

    _write_pem_pair(files.ca_cert, files.ca_key, _serialize_cert(ca_cert), _serialize_key(ca_key))

    san = []
    if scheduler_host:
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(scheduler_host)))
        except ValueError:
            san.append(x509.DNSName(scheduler_host))

    def _issue_cert(common_name: str) -> tuple[bytes, bytes]:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(expires)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
        )
        if san:
            builder = builder.add_extension(x509.SubjectAlternativeName(san), critical=False)
        cert = builder.sign(private_key=ca_key, algorithm=hashes.SHA256())
        return _serialize_cert(cert), _serialize_key(key)

    scheduler_cert, scheduler_key = _issue_cert(f"emout-{server_name}-scheduler")
    worker_cert, worker_key = _issue_cert(f"emout-{server_name}-worker")
    client_cert, client_key = _issue_cert(f"emout-{server_name}-client")

    _write_pem_pair(files.scheduler_cert, files.scheduler_key, scheduler_cert, scheduler_key)
    _write_pem_pair(files.worker_cert, files.worker_key, worker_cert, worker_key)
    _write_pem_pair(files.client_cert, files.client_key, client_cert, client_key)
    return files


def load_client_security_from_state(state: Mapping[str, Any] | None):
    """Build a Dask ``Security`` object from a saved server state."""
    if state is None:
        return None
    if state.get("protocol", "tcp") != "tls":
        return None

    tls = state.get("tls") or {}
    if not tls:
        return None

    from distributed.security import Security

    return Security(
        tls_ca_file=tls.get("ca_file"),
        tls_client_cert=tls.get("client_cert"),
        tls_client_key=tls.get("client_key"),
        require_encryption=bool(tls.get("require_encryption", True)),
    )
