from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from _entrypoint_runner import run_entrypoint

import mneme
from mneme.core import CliExitCode, ValidationError
from mneme.eval import run_fixture_evaluation, write_report_json
from mneme.release import (
    RELEASE_ARTIFACT_REPORT_SCHEMA,
    ReleaseArtifactReport,
    validate_release_artifacts,
)
from mneme.release.validate_artifacts import main as validate_artifacts_main


def test_release_artifact_validator_accepts_valid_artifacts(
    tmp_path: Path,
) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)
    fixture_report = _write_fixture_report(tmp_path)

    report = validate_release_artifacts(
        dist,
        fixture_report=fixture_report,
        expected_version=mneme.__version__,
    )

    assert report.ok, report.errors
    assert report.schema_version == RELEASE_ARTIFACT_REPORT_SCHEMA
    assert report.package_name == "mneme"
    assert report.version == mneme.__version__
    assert "wheel metadata" in report.checked
    assert "sdist contents" in report.checked
    assert "fixture report" in report.checked


def test_release_artifact_validator_requires_fixture_report(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)

    report = validate_release_artifacts(
        dist,
        fixture_report=tmp_path / "missing.json",
        expected_version=mneme.__version__,
    )

    assert not report.ok
    assert any("fixture report is invalid" in error for error in report.errors)


def test_release_artifact_validator_rejects_nonstandard_fixture_json(
    tmp_path: Path,
) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)
    fixture_report = tmp_path / "fixtures.json"
    fixture_report.write_text('{"schema_version": NaN}', encoding="utf-8")

    report = validate_release_artifacts(
        dist,
        fixture_report=fixture_report,
        expected_version=mneme.__version__,
    )

    assert not report.ok
    assert any("fixture report is invalid" in error for error in report.errors)


def test_release_artifact_validator_rejects_generated_python_artifacts(
    tmp_path: Path,
) -> None:
    dist = _write_fake_dist(
        tmp_path,
        version=mneme.__version__,
        wheel_extra_files={"mneme/__pycache__/__init__.cpython-312.pyc": "bytecode\n"},
        sdist_extra_files={
            "src/mneme/__pycache__/_version.cpython-312.pyc": "bytecode\n"
        },
    )
    fixture_report = _write_fixture_report(tmp_path)

    report = validate_release_artifacts(
        dist,
        fixture_report=fixture_report,
        expected_version=mneme.__version__,
    )

    assert not report.ok
    assert any(
        "wheel contains generated Python artifact" in error for error in report.errors
    )
    assert any(
        "sdist contains generated Python artifact" in error for error in report.errors
    )


def test_release_artifact_validator_rejects_unsafe_archive_paths(
    tmp_path: Path,
) -> None:
    dist = _write_fake_dist(
        tmp_path,
        version=mneme.__version__,
        wheel_extra_files={"../escape.txt": "escape\n"},
        sdist_extra_files={"../escape.txt": "escape\n"},
    )
    fixture_report = _write_fixture_report(tmp_path)

    report = validate_release_artifacts(
        dist,
        fixture_report=fixture_report,
        expected_version=mneme.__version__,
    )

    assert not report.ok
    assert any(
        "wheel contains unsafe archive path: ../escape.txt" in error
        for error in report.errors
    )
    assert any(
        f"sdist contains unsafe archive path: mneme-{mneme.__version__}/../escape.txt"
        in error
        for error in report.errors
    )


def test_release_artifact_validation_command_returns_json(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)
    fixture_report = _write_fixture_report(tmp_path)
    output = tmp_path / "release-artifacts.json"

    result = run_entrypoint(
        validate_artifacts_main,
        "--dist",
        dist,
        "--fixture-report",
        fixture_report,
        "--out",
        output,
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == RELEASE_ARTIFACT_REPORT_SCHEMA
    assert payload["ok"] is True
    assert payload["wheel"] == f"mneme-{mneme.__version__}-py3-none-any.whl"
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_release_artifact_validation_command_reports_write_error(
    tmp_path: Path,
) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)
    fixture_report = _write_fixture_report(tmp_path)
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("occupied", encoding="utf-8")

    result = run_entrypoint(
        validate_artifacts_main,
        "--dist",
        dist,
        "--fixture-report",
        fixture_report,
        "--out",
        blocked_parent / "release-artifacts.json",
    )

    assert result.returncode == int(CliExitCode.INTERNAL)
    assert "failed to write release artifact report" in result.stderr


def test_release_artifact_report_constructor_normalizes_sequences() -> None:
    report = ReleaseArtifactReport(
        ok=True,
        package_name="mneme",
        version=mneme.__version__,
        dist_dir="dist",
        wheel=None,
        sdist=None,
        fixture_report=None,
        installed_version=mneme.__version__,
        checked=["wheel metadata", "fixture report"],
        errors=[],
    )

    assert report.checked == ("wheel metadata", "fixture report")
    assert report.errors == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"schema_version": "mneme.release_artifact_report.v2"}, "unsupported release"),
        ({"ok": "yes"}, "ok must be a bool"),
        ({"package_name": object()}, "package_name must be a non-empty string"),
        ({"version": ""}, "version must be a non-empty string"),
        ({"dist_dir": object()}, "dist_dir must be a non-empty string"),
        ({"wheel": ""}, "wheel must be a non-empty string"),
        ({"sdist": object()}, "sdist must be a non-empty string"),
        ({"fixture_report": ""}, "fixture_report must be a non-empty string"),
        ({"installed_version": object()}, "installed_version must be"),
        ({"checked": "wheel metadata"}, "checked must be a sequence"),
        ({"checked": ("",)}, "checked item must be a non-empty string"),
        ({"errors": object()}, "errors must be a sequence"),
        ({"errors": ("",)}, "errors item must be a non-empty string"),
    ),
)
def test_release_artifact_report_constructor_rejects_malformed_fields(
    kwargs: dict[str, object],
    match: str,
) -> None:
    values = _release_report_values()
    values.update(kwargs)

    with pytest.raises(ValidationError, match=match):
        ReleaseArtifactReport(**values)


def _release_report_values() -> dict[str, object]:
    return {
        "ok": True,
        "package_name": "mneme",
        "version": mneme.__version__,
        "dist_dir": "dist",
        "wheel": f"mneme-{mneme.__version__}-py3-none-any.whl",
        "sdist": f"mneme-{mneme.__version__}.tar.gz",
        "fixture_report": "fixtures.json",
        "installed_version": mneme.__version__,
        "checked": ("wheel metadata", "fixture report"),
        "errors": (),
    }


def _write_fixture_report(tmp_path: Path) -> Path:
    path = tmp_path / "fixtures.json"
    report = run_fixture_evaluation(
        seed=0,
        command=("mneme", "eval", "fixtures", "--out", str(path)),
        created_at="2026-06-24T00:00:00Z",
        git_commit="abcdef0",
    )
    write_report_json(report, path)
    return path


def _write_fake_dist(
    tmp_path: Path,
    *,
    version: str,
    wheel_extra_files: dict[str, str] | None = None,
    sdist_extra_files: dict[str, str] | None = None,
) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    metadata = _metadata(version)
    dist_info = f"mneme-{version}.dist-info"
    wheel = dist / f"mneme-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mneme/__init__.py", "")
        archive.writestr("mneme/py.typed", "")
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"{dist_info}/RECORD", "")
        archive.writestr(f"{dist_info}/licenses/LICENSE", "Apache-2.0\n")
        for name, text in (wheel_extra_files or {}).items():
            archive.writestr(name, text)

    root = f"mneme-{version}"
    sdist = dist / f"{root}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        _add_tar_text(archive, f"{root}/PKG-INFO", metadata)
        for name in (
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
        ):
            _add_tar_text(archive, f"{root}/{name}", f"{name}\n")
        for name, text in (sdist_extra_files or {}).items():
            _add_tar_text(archive, f"{root}/{name}", text)
    return dist


def _metadata(version: str) -> str:
    return f"""Metadata-Version: 2.4
Name: mneme
Version: {version}
Summary: Episodic memory and retrieval for latent world models.
Project-URL: Documentation, https://abdelstark.github.io/mneme/
Project-URL: Source, https://github.com/AbdelStark/mneme
Project-URL: Issues, https://github.com/AbdelStark/mneme/issues
Project-URL: Security, https://github.com/AbdelStark/mneme/security/advisories
Project-URL: Changelog, https://github.com/AbdelStark/mneme/releases
License-File: LICENSE
Requires-Python: >=3.11
Requires-Dist: blake3>=0.4
Requires-Dist: numpy>=1.26

"""


def _add_tar_text(archive: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))
