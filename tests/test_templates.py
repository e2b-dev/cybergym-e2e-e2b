from __future__ import annotations

import json
from pathlib import Path

import pytest

from cybergym_e2b.cli import _parser
from cybergym_e2b.config import (
    BASE_BUILDER_IMAGES,
    DEFAULT_MANIFEST,
    DEFAULT_NETWORK_POLICY,
    TemplateManifest,
    TemplateRef,
)
from cybergym_e2b.inventory import ResolvedTask
from cybergym_e2b.runtime import RunOptions, _experiment_identity
from cybergym_e2b.templates import build_base_template, verify_template_ref


class _Builder:
    def run_cmd(self, _command: str):
        return self

    def set_ready_cmd(self, _command: str):
        return self


def test_runtime_identity_rejects_a_mutable_project_image(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    TemplateManifest(
        base=TemplateRef(
            name="base-a",
            tag="recipe-aaaaaaaaaaaaaaaa",
            template_id="template-base-a",
            build_id="build-base-1234",
            recipe_sha256="a" * 64,
            images=BASE_BUILDER_IMAGES,
        )
    ).write(manifest_path)
    policy = tmp_path / "network.json"
    policy.write_text("{}", encoding="utf-8")
    upstream = tmp_path / "upstream"
    (upstream / "scripts").mkdir(parents=True)
    task_dir = upstream / "projects" / "curl" / "task"
    task_dir.mkdir(parents=True)
    (task_dir.parent / "project.toml").write_text("", encoding="utf-8")
    resolved = ResolvedTask(
        task="curl/task",
        project="curl",
        task_id="task",
        build_image="example/project:latest",
        runtime_image="example/project:latest",
        repo_to_patch="https://example.invalid/repo",
    )

    with pytest.raises(ValueError, match="digest-locked"):
        _experiment_identity(
            resolved,
            kind="smoke",
            upstream=upstream,
            options=RunOptions(),
            manifest_path=manifest_path,
            network_policy_path=policy,
            patch_file=tmp_path / "patch",
            remote_smoke=tmp_path / "smoke.py",
        )


def test_template_build_is_content_addressed_and_reuses_only_ledger_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requirements = tmp_path / "requirements.lock"
    requirements.write_text(
        "demo==1.2.3 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "build-ledger.json"
    manifest_path = tmp_path / "manifest.json"
    built_names: list[str] = []

    monkeypatch.setattr("cybergym_e2b.templates._docker_builder", lambda _path: _Builder())
    monkeypatch.setattr("cybergym_e2b.templates._pull", lambda builder, _images: builder)

    def build(_builder, name: str, **_kwargs):
        built_names.append(name)
        return type(
            "Build",
            (),
            {
                "template_id": "tpl-cybergym-base",
                "build_id": f"build-{len(built_names):08d}",
            },
        )()

    monkeypatch.setattr("cybergym_e2b.templates.Template.build", build)
    first = build_base_template(
        requirements=requirements,
        name="cybergym-base",
        manifest_path=manifest_path,
        ledger_path=ledger,
    )
    second = build_base_template(
        requirements=requirements,
        name="cybergym-base",
        manifest_path=manifest_path,
        ledger_path=ledger,
    )

    assert len(built_names) == 1
    assert first.base == second.base
    assert first.base.name == "cybergym-base"
    assert first.base.tag == f"recipe-{first.base.recipe_sha256[:16]}"
    assert first.base.reference == f"cybergym-base:recipe-{first.base.recipe_sha256[:16]}"
    assert first.base.template_id == "tpl-cybergym-base"
    assert built_names == [first.base.reference]
    assert json.loads(ledger.read_text(encoding="utf-8"))["records"]

    alternate_namespace = build_base_template(
        requirements=requirements,
        name="another-base",
        manifest_path=manifest_path,
        ledger_path=ledger,
    )
    assert len(built_names) == 2
    assert alternate_namespace.base.name == "another-base"
    assert alternate_namespace.base.reference.startswith("another-base:recipe-")

    requirements.write_text(
        "demo==1.2.4 --hash=sha256:" + "b" * 64 + "\n",
        encoding="utf-8",
    )
    changed = build_base_template(
        requirements=requirements,
        name="cybergym-base",
        manifest_path=manifest_path,
        ledger_path=ledger,
    )
    assert len(built_names) == 3
    assert changed.base.name == first.base.name
    assert changed.base.tag != first.base.tag


def test_template_tag_must_still_point_to_recorded_build(monkeypatch) -> None:
    ref = TemplateRef(
        name="cybergym-base",
        tag="recipe-0123456789abcdef",
        template_id="tpl-cybergym-base",
        build_id="build-base-1234",
        recipe_sha256="0123456789abcdef" + "a" * 48,
        images=BASE_BUILDER_IMAGES,
    )

    monkeypatch.setattr(
        "cybergym_e2b.templates.Template.get_tags",
        lambda _template_id: [
            type(
                "Tag",
                (),
                {"tag": "recipe-0123456789abcdef", "build_id": "build-replaced-9999"},
            )()
        ],
    )

    with pytest.raises(RuntimeError, match="no longer points to recorded build"):
        verify_template_ref(ref)


def test_template_recipe_identity_covers_rendered_build_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cybergym_e2b import templates

    requirements = tmp_path / "requirements.lock"
    requirements.write_text(
        "demo==1.2.3 --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    original_recipe, payload = templates._base_recipe(
        requirements,
        name="base",
        cpu_count=8,
        memory_mb=8192,
        disk_limit_gb=120,
    )
    original_commands = templates._base_commands

    def changed_commands(lock_payload: bytes):
        install, daemon, readiness = original_commands(lock_payload)
        return install + "\necho recipe-change", daemon, readiness

    monkeypatch.setattr(templates, "_base_commands", changed_commands)
    changed_recipe, changed_payload = templates._base_recipe(
        requirements,
        name="base",
        cpu_count=8,
        memory_mb=8192,
        disk_limit_gb=120,
    )

    assert payload == changed_payload
    assert original_recipe["sha256"] != changed_recipe["sha256"]


def test_template_build_rejects_unlocked_python_requirements(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("httpx>=0.28,<0.29\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fully locked"):
        build_base_template(
            requirements=requirements,
            manifest_path=tmp_path / "manifest.json",
            ledger_path=tmp_path / "ledger.json",
        )


def test_cli_defaults_resolve_packaged_runtime_assets_outside_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    args = _parser().parse_args(["run", "curl/arvo_66012"])

    assert args.network_policy == DEFAULT_NETWORK_POLICY
    assert args.network_policy.is_file()
    assert args.patch_file.is_file()
    assert args.remote_smoke.is_file()
    assert args.remote_install_codex.is_file()
    assert Path("artifacts/templates/manifest.json") == DEFAULT_MANIFEST


def test_manifest_rejects_mutable_recorded_images(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    ref = TemplateRef(
        name="base",
        tag="recipe-aaaaaaaaaaaaaaaa",
        template_id="template-base",
        build_id="build-base-1234",
        recipe_sha256="a" * 64,
        images=("ubuntu:22.04",),
    )
    TemplateManifest(base=ref).write(path)

    with pytest.raises(ValueError, match="digest-locked"):
        TemplateManifest.load(path)


def test_upstream_lock_rejects_malformed_pins(tmp_path: Path) -> None:
    from cybergym_e2b.config import load_upstream_lock

    malformed = tmp_path / "upstream.lock.json"
    malformed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": {
                    "repository": "https://example.invalid/upstream.git",
                    "commit": "not-a-commit",
                },
                "dataset": {
                    "repository": "example/dataset",
                    "revision": "a" * 40,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="code commit"):
        load_upstream_lock(malformed)
