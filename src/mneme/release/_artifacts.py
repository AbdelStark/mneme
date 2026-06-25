"""Release artifact validation for source, wheel, and fixture evidence."""

from __future__ import annotations

import json
import tarfile
import zipfile
from dataclasses import dataclass
from email.message import Message
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from mneme.core import ValidationError
from mneme.eval import validate_report_json

RELEASE_ARTIFACT_REPORT_SCHEMA: Final = "mneme.release_artifact_report.v1"
PACKAGE_NAME: Final = "mneme"
DIST_INFO_PREFIX: Final = "mneme-"

REQUIRED_PROJECT_URLS: Final = (
    "Documentation",
    "Source",
    "Issues",
    "Security",
    "Changelog",
)
REQUIRED_RUNTIME_DEPENDENCIES: Final = ("blake3>=0.4", "numpy>=1.26")
REQUIRED_SDIST_FILES: Final = (
    "CITATION.cff",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "mkdocs.yml",
    "README.md",
    "SECURITY.md",
    "SUPPORT.md",
    "docs/index.md",
    "docs/release/RELEASE_CHECKLIST.md",
    "docs/spec/09-release-and-versioning.md",
    "examples/README.md",
    "pyproject.toml",
    "uv.lock",
)
REQUIRED_WHEEL_SUFFIXES: Final = (
    "mneme/py.typed",
    ".dist-info/METADATA",
    ".dist-info/WHEEL",
    ".dist-info/RECORD",
    ".dist-info/licenses/LICENSE",
)


@dataclass(frozen=True)
class ReleaseArtifactReport:
    """Machine-readable release artifact validation report."""

    ok: bool
    package_name: str
    version: str
    dist_dir: str
    wheel: str | None
    sdist: str | None
    fixture_report: str | None
    installed_version: str | None
    checked: tuple[str, ...]
    errors: tuple[str, ...]
    schema_version: str = RELEASE_ARTIFACT_REPORT_SCHEMA

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable validation report."""

        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "package_name": self.package_name,
            "version": self.version,
            "dist_dir": self.dist_dir,
            "wheel": self.wheel,
            "sdist": self.sdist,
            "fixture_report": self.fixture_report,
            "installed_version": self.installed_version,
            "checked": list(self.checked),
            "errors": list(self.errors),
        }


def validate_release_artifacts(
    dist: str | Path,
    *,
    fixture_report: str | Path | None,
    expected_version: str | None = None,
) -> ReleaseArtifactReport:
    """Validate built package artifacts and fixture evidence for a release."""

    dist_dir = Path(dist)
    errors: list[str] = []
    checked: list[str] = []
    installed_version = _installed_version(PACKAGE_NAME)
    version = expected_version or installed_version
    if version is None:
        version = "unknown"
        errors.append("installed package metadata not found for mneme")

    wheels: list[Path] = []
    sdists: list[Path] = []
    if not dist_dir.is_dir():
        errors.append(f"dist directory not found: {dist_dir}")
    else:
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))

    wheel = _single_artifact(wheels, "wheel", errors)
    sdist = _single_artifact(sdists, "sdist", errors)
    if wheel is not None:
        _validate_wheel(wheel, version=version, errors=errors, checked=checked)
    if sdist is not None:
        _validate_sdist(sdist, version=version, errors=errors, checked=checked)

    fixture_path = Path(fixture_report) if fixture_report is not None else None
    _validate_fixture_report(fixture_path, errors=errors, checked=checked)

    return ReleaseArtifactReport(
        ok=not errors,
        package_name=PACKAGE_NAME,
        version=version,
        dist_dir=str(dist_dir),
        wheel=wheel.name if wheel is not None else None,
        sdist=sdist.name if sdist is not None else None,
        fixture_report=str(fixture_path) if fixture_path is not None else None,
        installed_version=installed_version,
        checked=tuple(checked),
        errors=tuple(errors),
    )


def _installed_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _single_artifact(
    paths: list[Path],
    label: str,
    errors: list[str],
) -> Path | None:
    if len(paths) != 1:
        errors.append(f"expected exactly one {label} artifact, found {len(paths)}")
        return None
    return paths[0]


def _validate_wheel(
    path: Path,
    *,
    version: str,
    errors: list[str],
    checked: list[str],
) -> None:
    if not path.name.startswith(f"{DIST_INFO_PREFIX}{version}-"):
        errors.append(f"wheel filename does not include version {version}: {path.name}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            _reject_generated_python_artifacts(names, "wheel", errors)
            _require_suffixes(names, REQUIRED_WHEEL_SUFFIXES, "wheel", errors)
            metadata_name = _single_name_with_suffix(
                names,
                ".dist-info/METADATA",
                "wheel metadata",
                errors,
            )
            if metadata_name is not None:
                message = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
                _validate_metadata(
                    message,
                    source="wheel",
                    version=version,
                    errors=errors,
                )
                checked.append("wheel metadata")
            checked.append("wheel contents")
    except zipfile.BadZipFile as exc:
        errors.append(f"wheel is not a valid zip archive: {exc}")


def _validate_sdist(
    path: Path,
    *,
    version: str,
    errors: list[str],
    checked: list[str],
) -> None:
    expected_name = f"{PACKAGE_NAME}-{version}.tar.gz"
    if path.name != expected_name:
        errors.append(f"sdist filename must be {expected_name}: {path.name}")
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
            _reject_generated_python_artifacts(names, "sdist", errors)
            root = _sdist_root(names, errors)
            if root is not None:
                for required in REQUIRED_SDIST_FILES:
                    candidate = f"{root}/{required}"
                    if candidate not in names:
                        errors.append(f"sdist missing required file: {required}")
                metadata_name = f"{root}/PKG-INFO"
                member = archive.extractfile(metadata_name)
                if member is None:
                    errors.append("sdist missing PKG-INFO metadata")
                else:
                    message = Parser().parsestr(member.read().decode("utf-8"))
                    _validate_metadata(
                        message,
                        source="sdist",
                        version=version,
                        errors=errors,
                    )
                    checked.append("sdist metadata")
                checked.append("sdist contents")
    except tarfile.TarError as exc:
        errors.append(f"sdist is not a valid tar.gz archive: {exc}")


def _sdist_root(names: set[str], errors: list[str]) -> str | None:
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    if len(roots) != 1:
        errors.append("sdist must contain exactly one root directory")
        return None
    root = next(iter(roots))
    expected = f"{PACKAGE_NAME}-"
    if not root.startswith(expected):
        errors.append(f"sdist root must start with {expected!r}: {root}")
    return root


def _validate_metadata(
    message: Message,
    *,
    source: str,
    version: str,
    errors: list[str],
) -> None:
    if (message.get("Name") or "").lower() != PACKAGE_NAME:
        errors.append(f"{source} metadata Name must be {PACKAGE_NAME!r}")
    if message.get("Version") != version:
        errors.append(f"{source} metadata Version must be {version!r}")
    if message.get("Requires-Python") != ">=3.11":
        errors.append(f"{source} metadata Requires-Python must be >=3.11")
    for dependency in REQUIRED_RUNTIME_DEPENDENCIES:
        if dependency not in _headers(message, "Requires-Dist"):
            errors.append(f"{source} metadata missing dependency: {dependency}")
    project_urls = _headers(message, "Project-URL")
    for label in REQUIRED_PROJECT_URLS:
        if not any(value.startswith(f"{label}, ") for value in project_urls):
            errors.append(f"{source} metadata missing Project-URL: {label}")
    license_files = _headers(message, "License-File")
    if "LICENSE" not in license_files:
        errors.append(f"{source} metadata missing License-File: LICENSE")


def _headers(message: Message, name: str) -> tuple[str, ...]:
    return tuple(str(value) for value in message.get_all(name, []))


def _require_suffixes(
    names: set[str],
    suffixes: tuple[str, ...],
    source: str,
    errors: list[str],
) -> None:
    for suffix in suffixes:
        if not any(name.endswith(suffix) for name in names):
            errors.append(f"{source} missing required path suffix: {suffix}")


def _reject_generated_python_artifacts(
    names: set[str],
    source: str,
    errors: list[str],
) -> None:
    for name in sorted(names):
        parts = name.split("/")
        if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
            errors.append(f"{source} contains generated Python artifact: {name}")


def _single_name_with_suffix(
    names: set[str],
    suffix: str,
    label: str,
    errors: list[str],
) -> str | None:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        errors.append(f"expected exactly one {label}, found {len(matches)}")
        return None
    return matches[0]


def _validate_fixture_report(
    path: Path | None,
    *,
    errors: list[str],
    checked: list[str],
) -> None:
    if path is None:
        errors.append("fixture report path is required")
        return
    try:
        report = validate_report_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        errors.append(f"fixture report is invalid: {exc}")
        return
    if not report.passed:
        errors.append("fixture report did not pass")
    if report.dataset.kind != "fixture":
        errors.append("fixture report dataset kind must be fixture")
    if not report.caveats:
        errors.append("fixture report must preserve caveats")
    checked.append("fixture report")


__all__ = [
    "RELEASE_ARTIFACT_REPORT_SCHEMA",
    "ReleaseArtifactReport",
    "validate_release_artifacts",
]
