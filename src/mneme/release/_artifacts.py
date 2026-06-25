"""Release artifact validation for source, wheel, and fixture evidence."""

from __future__ import annotations

import tarfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from email.message import Message
from email.parser import Parser
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Final

from mneme.core import ValidationError
from mneme.core._json import loads_strict_json
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

    def __post_init__(self) -> None:
        if self.schema_version != RELEASE_ARTIFACT_REPORT_SCHEMA:
            raise ValidationError("unsupported release artifact report schema")
        object.__setattr__(self, "ok", _require_bool(self.ok, "ok"))
        object.__setattr__(
            self,
            "package_name",
            _require_non_empty_string(self.package_name, "package_name"),
        )
        object.__setattr__(
            self, "version", _require_non_empty_string(self.version, "version")
        )
        object.__setattr__(
            self, "dist_dir", _require_non_empty_string(self.dist_dir, "dist_dir")
        )
        object.__setattr__(
            self, "wheel", _optional_non_empty_string(self.wheel, "wheel")
        )
        object.__setattr__(
            self, "sdist", _optional_non_empty_string(self.sdist, "sdist")
        )
        object.__setattr__(
            self,
            "fixture_report",
            _optional_non_empty_string(self.fixture_report, "fixture_report"),
        )
        object.__setattr__(
            self,
            "installed_version",
            _optional_non_empty_string(self.installed_version, "installed_version"),
        )
        object.__setattr__(self, "checked", _string_tuple(self.checked, "checked"))
        object.__setattr__(self, "errors", _string_tuple(self.errors, "errors"))

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


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a bool")
    return value


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_non_empty_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence")
    return tuple(
        _require_non_empty_string(item, f"{field_name} item") for item in value
    )


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
            _reject_unsafe_archive_paths(names, "wheel", errors)
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
            _reject_unsafe_archive_paths(names, "sdist", errors)
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


def _reject_unsafe_archive_paths(
    names: set[str],
    source: str,
    errors: list[str],
) -> None:
    for name in sorted(names):
        parsed = PurePosixPath(name)
        if (
            not name
            or parsed.is_absolute()
            or ".." in parsed.parts
            or "\\" in name
            or ":" in name
            or name.startswith("~")
        ):
            errors.append(f"{source} contains unsafe archive path: {name}")


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
        report = validate_report_json(
            loads_strict_json(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, ValidationError) as exc:
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
