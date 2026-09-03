"""Release policy behavior for the standalone public repository."""

import importlib.util
import subprocess
from pathlib import Path


def _load_public_tree_module():
    path = Path(__file__).parents[1] / "scripts" / "check_public_tree.py"
    module_spec = importlib.util.spec_from_file_location("check_public_tree", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


public_tree = _load_public_tree_module()
find_forbidden_paths = public_tree.find_forbidden_paths


def test_public_tree_policy_rejects_secrets_and_generated_paths():
    tracked = [
        ".env",
        ".env.example",
        ".venv/pyvenv.cfg",
        "README.md",
        "package.egg-info/PKG-INFO",
        "src/cybergym_e2b/__pycache__/runtime.pyc",
        "artifacts/run/result.json",
        "nested/.env.customer",
        "policies/network.json",
        "vendor/cybergym-e2e/tasks/example.toml",
        "docs/validation/report.md",
        "experiments/resource-study/config.json",
        "campaigns/full-run.txt",
        "reports/customer-summary.json",
        "template-manifest.8c8g.json",
        "artifacts/templates/build-ledger.json",
        "credentials.json",
        "tls/client.key",
        "bedrock-campaign-tasks.txt",
        "scripts/analyze_profiles.py",
        "examples/documented-tasks.txt",
    ]

    assert find_forbidden_paths(tracked) == [
        ".env",
        ".venv/pyvenv.cfg",
        "artifacts/run/result.json",
        "artifacts/templates/build-ledger.json",
        "bedrock-campaign-tasks.txt",
        "campaigns/full-run.txt",
        "credentials.json",
        "docs/validation/report.md",
        "experiments/resource-study/config.json",
        "nested/.env.customer",
        "package.egg-info/PKG-INFO",
        "reports/customer-summary.json",
        "scripts/analyze_profiles.py",
        "src/cybergym_e2b/__pycache__/runtime.pyc",
        "template-manifest.8c8g.json",
        "tls/client.key",
        "vendor/cybergym-e2e/tasks/example.toml",
    ]


def test_public_tree_command_validates_an_extracted_tree_without_git(
    tmp_path: Path, capsys
) -> None:
    (tmp_path / "README.md").write_text("public source\n", encoding="utf-8")

    assert public_tree.main(tmp_path) == 0
    assert capsys.readouterr().out == "Public-tree policy passed.\n"

    (tmp_path / "bedrock-campaign-tasks.txt").write_text("task\n", encoding="utf-8")

    assert public_tree.main(tmp_path) == 1
    assert "bedrock-campaign-tasks.txt" in capsys.readouterr().out


def test_gitignore_rejects_root_campaign_lists_and_profile_analyzer(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".gitignore").write_text(
        (Path(__file__).parents[1] / ".gitignore").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "bedrock-campaign-tasks.txt",
            "scripts/analyze_profiles.py",
            "examples/documented-tasks.txt",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository,
    )

    assert completed.returncode == 0
    assert completed.stdout.splitlines() == [
        "bedrock-campaign-tasks.txt",
        "scripts/analyze_profiles.py",
    ]
