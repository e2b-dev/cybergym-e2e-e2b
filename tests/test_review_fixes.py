from __future__ import annotations

import json
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from cybergym_e2b.cli import _parser, _verify_upstream, main
from cybergym_e2b.config import DEFAULT_NETWORK_POLICY, DEFAULT_PATCH_FILE, asset_path
from cybergym_e2b.inventory import build_code_bundle, resolve_task
from cybergym_e2b.runtime import (
    _extract_results,
    _network,
    _policy,
    _require_runnable_policy,
)

UPSTREAM = Path("vendor/cybergym-e2e")
LOCKED_POLICY = asset_path("policies/network-locked.json")


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


def test_patch_carries_context_and_applies_without_unidiff_zero() -> None:
    check = subprocess.run(
        ["git", "apply", "--check", str(DEFAULT_PATCH_FILE.resolve())],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def _git_repo_with_commit(root: Path) -> str:
    (root / "scripts").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "scripts" / "validate.py").write_text("pinned\n")
    subprocess.run(["git", "add", "scripts/validate.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "pin"],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_verify_upstream_rejects_dirty_checkout(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "upstream"
    head = _git_repo_with_commit(repo)
    monkeypatch.setattr("cybergym_e2b.cli.UPSTREAM_COMMIT", head)

    assert _verify_upstream(repo) == head

    # IDE metadata or notes outside the shipped trees must not block every command.
    (repo / "notes.md").write_text("scratch\n")
    assert _verify_upstream(repo) == head

    (repo / "scripts" / "validate.py").write_text("edited validator\n")
    with pytest.raises(RuntimeError, match="dirty") as excinfo:
        _verify_upstream(repo)
    # sync-upstream refuses dirty trees too, so the message must name a remedy that works,
    # and every git step in it must target the vendored checkout, not the caller's repo.
    assert f"git -C {repo} checkout -- ." in str(excinfo.value)
    assert f"git -C {repo} clean -fd" in str(excinfo.value)

    (repo / "scripts" / "validate.py").write_text("pinned\n")
    (repo / "scripts" / "extra.py").write_text("untracked but shipped\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _verify_upstream(repo)


def _tarball(members: dict[str, bytes | Path | tuple[str, str]]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            if isinstance(data, Path):
                info.type = tarfile.SYMTYPE
                info.linkname = str(data)
                archive.addfile(info)
            elif isinstance(data, tuple):
                info.type = tarfile.LNKTYPE
                info.linkname = data[1]
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, BytesIO(data))
    return buffer.getvalue()


def test_extract_results_rejects_parent_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"../escape.txt": b"x"}))
    with pytest.raises(RuntimeError, match="unsafe path"):
        _extract_results(archive_path, tmp_path / "sandbox")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_results_extracts_nested_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(_tarball({"./agent_output/run/log.txt": b"ok"}))
    _extract_results(archive_path, tmp_path / "sandbox")
    assert (tmp_path / "sandbox" / "agent_output" / "run" / "log.txt").read_bytes() == b"ok"


def test_extract_results_drops_unsafe_links_but_keeps_in_tree_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "results.tgz"
    archive_path.write_bytes(
        _tarball(
            {
                "./agent_output/fix.patch": Path("/etc/passwd"),
                "./agent_output/summary.json": b"{}",
                # GNU cp -a followed by tar emits the second hardlink name as LNKTYPE.
                "./agent_output/summary-copy.json": ("hardlink", "./agent_output/summary.json"),
            }
        )
    )
    _extract_results(archive_path, tmp_path / "sandbox")
    out = tmp_path / "sandbox" / "agent_output"
    assert (out / "summary.json").read_bytes() == b"{}"
    assert (out / "summary-copy.json").read_bytes() == b"{}"
    assert not (out / "fix.patch").is_symlink()
    assert not (out / "fix.patch").exists()


def test_gemini_cli_is_not_an_accepted_agent() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "curl/arvo_66012", "--agent", "gemini-cli"])


@pytest.mark.parametrize("policy_path", [DEFAULT_NETWORK_POLICY, LOCKED_POLICY])
def test_packaged_policies_accept_any_bedrock_region(policy_path: Path) -> None:
    policy = _policy(policy_path)
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


def test_packaged_policies_share_host_lists() -> None:
    default = _policy(DEFAULT_NETWORK_POLICY)
    locked = _policy(LOCKED_POLICY)
    for key in ("model_hosts", "artifact_hosts", "registry_hosts"):
        assert default[key] == locked[key], key


def test_locked_policy_refuses_agent_runs_but_allows_smoke() -> None:
    locked = _policy(LOCKED_POLICY)
    with pytest.raises(ValueError, match="agent tooling"):
        _require_runnable_policy(locked, kind="run", egress="policy")
    _require_runnable_policy(locked, kind="smoke", egress="policy")
    default = _policy(DEFAULT_NETWORK_POLICY)
    _require_runnable_policy(default, kind="run", egress="policy")
    _require_runnable_policy(default, kind="run", egress="restricted")


def test_bundle_rejects_missing_install_codex_override(tmp_path: Path) -> None:
    resolved = resolve_task(UPSTREAM, "curl/arvo_66012")
    with pytest.raises(FileNotFoundError):
        build_code_bundle(UPSTREAM, resolved, remote_install_codex=tmp_path / "missing.sh")


def test_cli_validates_asset_overrides_before_doing_work(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "instal_codex.sh"
    code = main(["preflight", "--remote-install-codex", str(missing)])
    assert code == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["type"] == "FileNotFoundError"
    assert "instal_codex.sh" in error["message"]


def test_batch_refuses_unrunnable_policy_before_submitting_work(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    # main() requires the E2B key before dispatching to batch; keep the test hermetic.
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("curl/arvo_66012\n")
    code = main(
        [
            "batch",
            "--kind",
            "run",
            "--tasks-file",
            str(tasks),
            "--network-policy",
            str(LOCKED_POLICY),
        ]
    )
    assert code == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["type"] == "ValueError"
    assert "agent tooling" in error["message"]
