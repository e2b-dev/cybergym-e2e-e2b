from __future__ import annotations

import io
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from cybergym_e2b.cli import _parser, _verify_upstream
from cybergym_e2b.config import asset_path
from cybergym_e2b.inventory import build_code_bundle, resolve_task
from cybergym_e2b.runtime import _network, _policy

UPSTREAM = Path("vendor/cybergym-e2e")


def test_patch_keeps_anthropic_return_ahead_of_openai_compatible_branch() -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    payload = build_code_bundle(UPSTREAM, resolved)
    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        utils = archive.extractfile("scripts/utils.py")
        assert utils is not None
        source = utils.read().decode()
    anthropic_return = source.index("return response.content[0].text")
    new_branch = source.index('model_provider == "openai-compatible"')
    assert anthropic_return < new_branch


def test_verify_upstream_rejects_dirty_checkout(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("pinned\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "pin"],
        cwd=repo,
        check=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    monkeypatch.setattr("cybergym_e2b.cli.UPSTREAM_COMMIT", head)

    assert _verify_upstream(repo) == head

    (repo / "tracked.txt").write_text("edited validator\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _verify_upstream(repo)


def _tarball(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_extract_results_rejects_parent_traversal(tmp_path: Path) -> None:
    from cybergym_e2b.runtime import _extract_results

    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"../escape.txt": b"x"}))
    with pytest.raises(RuntimeError, match="unsafe path"):
        _extract_results(archive_path, tmp_path / "sandbox")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_results_extracts_nested_members(tmp_path: Path) -> None:
    from cybergym_e2b.runtime import _extract_results

    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"./agent_output/run/log.txt": b"ok"}))
    _extract_results(archive_path, tmp_path / "sandbox")
    assert (tmp_path / "sandbox" / "agent_output" / "run" / "log.txt").read_bytes() == b"ok"


def test_gemini_cli_is_not_an_accepted_agent() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "curl/arvo_66012", "--agent", "gemini-cli"])


@pytest.mark.parametrize("policy_name", ["network.json", "network-locked.json"])
def test_packaged_policies_accept_any_bedrock_region(policy_name: str) -> None:
    policy = _policy(asset_path(f"policies/{policy_name}"))
    host = "bedrock-mantle.eu-west-1.api.aws"
    network = _network(
        policy,
        phase="runtime",
        egress="restricted",
        hf_token=None,
        model_key="secret",
        non_http=[],
        model_host=host,
    )
    assert host in network["allow_out"]
    assert set(network["rules"]) == {host}


def test_bundle_rejects_missing_install_codex_override(tmp_path: Path) -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    with pytest.raises(FileNotFoundError):
        build_code_bundle(UPSTREAM, resolved, remote_install_codex=tmp_path / "missing.sh")
