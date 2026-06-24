from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import mneme
from mneme.core import CliExitCode
from mneme.eval import run_fixture_evaluation, write_report_json
from mneme.release import (
    RELEASE_ARTIFACT_REPORT_SCHEMA,
    validate_release_artifacts,
)


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


def test_release_artifact_validation_command_returns_json(tmp_path: Path) -> None:
    dist = _write_fake_dist(tmp_path, version=mneme.__version__)
    fixture_report = _write_fixture_report(tmp_path)
    output = tmp_path / "release-artifacts.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mneme.release.validate_artifacts",
            "--dist",
            str(dist),
            "--fixture-report",
            str(fixture_report),
            "--out",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == int(CliExitCode.SUCCESS), result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == RELEASE_ARTIFACT_REPORT_SCHEMA
    assert payload["ok"] is True
    assert payload["wheel"] == f"mneme-{mneme.__version__}-py3-none-any.whl"
    assert json.loads(output.read_text(encoding="utf-8")) == payload


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


def _write_fake_dist(tmp_path: Path, *, version: str) -> Path:
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

    root = f"mneme-{version}"
    sdist = dist / f"{root}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        _add_tar_text(archive, f"{root}/PKG-INFO", metadata)
        for name in (
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "README.md",
            "SECURITY.md",
            "docs/release/RELEASE_CHECKLIST.md",
            "docs/spec/09-release-and-versioning.md",
            "pyproject.toml",
        ):
            _add_tar_text(archive, f"{root}/{name}", f"{name}\n")
    return dist


def _metadata(version: str) -> str:
    return f"""Metadata-Version: 2.4
Name: mneme
Version: {version}
Summary: Episodic memory and retrieval for latent world models.
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
