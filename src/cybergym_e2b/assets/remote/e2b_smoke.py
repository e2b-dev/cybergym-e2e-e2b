#!/usr/bin/env python3
"""Profile CyberGym-E2E ground-truth stage 4 in a fresh nested container."""

from __future__ import annotations

import argparse
import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import tomli
import utils

DEFAULT_BUILD_IMAGE = (
    "gcr.io/oss-fuzz-base/base-builder"
    "@sha256:8eda74a11e800aead5a041ee479a65b33dab3150d6e89e5694e2b6eb27be98fc"
)

PROFILE_VALIDATOR = r"""#!/usr/bin/env python3
import argparse
import json
import resource
import time

import validate


def classify(command):
    text = " ".join(str(value) for value in command) if isinstance(command, list) else str(command)
    if "/src/prepare.sh" in text:
        return "prepare"
    if "cp -a /src /src_backup" in text:
        return "source_backup"
    if "rm -rf /src" in text and "/src_backup" in text:
        return "source_restore"
    if "git apply" in text or "patch -p" in text:
        return "patch_apply_attempt"
    if "/src/compile.sh" in text:
        return "compile"
    if "poc.bin" in text and " cp " in (" " + text + " "):
        return "poc_copy"
    if "/src/run_poc.sh" in text:
        return "ground_truth_poc"
    return "validation_other"


parser = argparse.ArgumentParser()
parser.add_argument("--patch-file", required=True)
parser.add_argument("--json-output", required=True)
parser.add_argument("--profile-output", required=True)
args = parser.parse_args()

operations = []
original_run = validate.subprocess.run


def profiled_run(command, *positional, **keyword):
    started_at_unix = time.time()
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = None
    error = None
    try:
        result = original_run(command, *positional, **keyword)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        entry = {
            "sequence": len(operations),
            "name": classify(command),
            "started_at_unix": started_at_unix,
            "duration_seconds": round(time.monotonic() - started, 6),
            "user_cpu_seconds": round(usage_after.ru_utime - usage_before.ru_utime, 6),
            "system_cpu_seconds": round(usage_after.ru_stime - usage_before.ru_stime, 6),
            "exit_code": getattr(result, "returncode", getattr(error, "returncode", None)),
        }
        if result is not None:
            entry["stdout_bytes"] = len(
                (getattr(result, "stdout", None) or "").encode("utf-8", "replace")
            )
            entry["stderr_bytes"] = len(
                (getattr(result, "stderr", None) or "").encode("utf-8", "replace")
            )
        if error:
            entry["error"] = error
        operations.append(entry)


started_at_unix = time.time()
started = time.monotonic()
validate.subprocess.run = profiled_run
try:
    results = validate.validate_task(
        patch_path=args.patch_file,
        src_dir="/src",
        data_dir="/data",
        config_dir="/config",
        run_prepare=True,
        only_stage=4,
        verbose=True,
    )
finally:
    validate.subprocess.run = original_run

statuses = {name: details["status"] for name, details in results.items()}
with open(args.json_output, "w", encoding="utf-8") as handle:
    json.dump(statuses, handle, indent=2)
with open(args.profile_output, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "started_at_unix": started_at_unix,
            "duration_seconds": round(time.monotonic() - started, 6),
            "operations": operations,
        },
        handle,
        indent=2,
    )
raise SystemExit(0 if statuses.get("stage4") == "passed" else 1)
"""


class Profiler:
    def __init__(self) -> None:
        self.started_at_unix = time.time()
        self.started = time.monotonic()
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str):
        started_at_unix = time.time()
        started = time.monotonic()
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        entry = {
            "name": name,
            "started_at_unix": started_at_unix,
            "started_offset_seconds": round(started - self.started, 6),
            "status": "running",
        }
        self.stages.append(entry)
        try:
            yield entry
        except Exception as exc:
            entry["status"] = "error"
            entry["error_type"] = type(exc).__name__
            raise
        else:
            entry["status"] = "ok"
        finally:
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            entry.update(
                {
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "user_cpu_seconds": round(usage_after.ru_utime - usage_before.ru_utime, 6),
                    "system_cpu_seconds": round(usage_after.ru_stime - usage_before.ru_stime, 6),
                }
            )

    def result(self) -> dict:
        return {
            "clock": "sandbox_monotonic_with_unix_boundaries",
            "started_at_unix": self.started_at_unix,
            "duration_seconds": round(time.monotonic() - self.started, 6),
            "stages": self.stages,
        }


def setup_exec_name(command) -> str:
    text = " ".join(str(value) for value in command) if isinstance(command, list) else str(command)
    if "find /src" in text or "rm -rf /out" in text or "mkdir -p /out" in text:
        return "clean_image_workspace"
    if "mkdir -p /config" in text:
        return "create_workspace_directories"
    if "apt-get update" in text:
        return "install_system_validation_dependencies"
    if "install_validate_deps.sh" in text:
        return "install_python_validation_dependencies"
    if "tar xf /src/src.tgz" in text:
        return "extract_source"
    if "cat > /config/config.toml" in text:
        return "write_sanitized_config"
    if "git apply" in text or "patch -p" in text:
        return "apply_pre_patch_attempt"
    return "workspace_exec_other"


def setup_copy_name(src_path, dst_path: str) -> str:
    name = Path(src_path).name
    if name == "src.tgz":
        return "copy_source_archive"
    if name == "install_validate_deps.sh":
        return "copy_validation_installer"
    if name == "validate.py":
        return "copy_validator"
    if name in {"prepare.sh", "compile.sh", "run_poc.sh", "test.sh"}:
        return "copy_build_script"
    if name in {"poc.bin", "crash.log"}:
        return "copy_task_input"
    if "pre_patch" in str(dst_path):
        return "copy_pre_patch"
    return "workspace_copy_other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    script_path = root / "projects" / args.task
    data_path = root / "data" / "projects" / args.task
    config = tomli.loads((script_path / "../project.toml").read_text())
    config.update(tomli.loads((script_path / "config.toml").read_text()))
    image = config.get("build_image", DEFAULT_BUILD_IMAGE)
    profiler = Profiler()
    setup_operations: list[dict] = []
    container_id = None
    result = {"task": args.task, "build_image": image, "stage4": "error"}

    original_exec = utils.exec_run
    original_copy = utils.copy_to_container

    def record_setup(name: str, function, *positional, **keyword):
        started_at_unix = time.time()
        started = time.monotonic()
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        status = "ok"
        try:
            return function(*positional, **keyword)
        except Exception:
            status = "error"
            raise
        finally:
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            setup_operations.append(
                {
                    "sequence": len(setup_operations),
                    "name": name,
                    "started_at_unix": started_at_unix,
                    "duration_seconds": round(time.monotonic() - started, 6),
                    "user_cpu_seconds": round(usage_after.ru_utime - usage_before.ru_utime, 6),
                    "system_cpu_seconds": round(usage_after.ru_stime - usage_before.ru_stime, 6),
                    "status": status,
                }
            )

    def profiled_exec(container, command, *positional, **keyword):
        return record_setup(
            setup_exec_name(command),
            original_exec,
            container,
            command,
            *positional,
            **keyword,
        )

    def profiled_copy(container, src_path, dst_path, *positional, **keyword):
        return record_setup(
            setup_copy_name(src_path, dst_path),
            original_copy,
            container,
            src_path,
            dst_path,
            *positional,
            **keyword,
        )

    try:
        with profiler.stage("nested_container_start"):
            container_id = utils.start_container(image)
        utils.exec_run = profiled_exec
        utils.copy_to_container = profiled_copy
        with profiler.stage("workspace_setup"):
            utils.setup_workspace(
                container_id,
                data_path,
                script_path,
                mode="patch-only",
                copy_gt_poc=True,
                scripts_dir=root / "scripts",
            )
        utils.exec_run = original_exec
        utils.copy_to_container = original_copy
        with profiler.stage("copy_ground_truth_patch"):
            original_copy(container_id, script_path / "patch.diff", "/output/fix.patch")
        profile_validator = root / "scripts" / "profile_validate.py"
        profile_validator.write_text(PROFILE_VALIDATOR, encoding="utf-8")
        with profiler.stage("copy_profile_validator"):
            original_copy(container_id, profile_validator, "/scripts/profile_validate.py")
        command = (
            "/scripts/.venv/bin/python /scripts/profile_validate.py "
            "--patch-file /output/fix.patch "
            "--json-output /output/validation_results.json "
            "--profile-output /output/validation_profile.json"
        )
        with profiler.stage("ground_truth_validation"):
            code, stdout, stderr = original_exec(
                container_id,
                command,
                "Running profiled ground-truth stage 4",
                timeout=7200,
                workdir="/",
            )
        result.update(
            {"exit_code": code, "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]}
        )
        with profiler.stage("read_validation_results"):
            code, stdout, _ = original_exec(
                container_id,
                "cat /output/validation_results.json",
                timeout=30,
                workdir="/",
                verbose=False,
            )
            if code == 0:
                result["validation"] = json.loads(stdout)
                result["stage4"] = result["validation"].get("stage4", "error")
            code, stdout, _ = original_exec(
                container_id,
                "cat /output/validation_profile.json",
                timeout=30,
                workdir="/",
                verbose=False,
            )
            if code == 0:
                result["validation_profile"] = json.loads(stdout)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        utils.exec_run = original_exec
        utils.copy_to_container = original_copy
        if container_id:
            with profiler.stage("nested_container_cleanup"):
                utils.cleanup_container(container_id)
    result["profile"] = profiler.result()
    result["profile"]["workspace_operations"] = setup_operations
    (root / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("stage4") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
