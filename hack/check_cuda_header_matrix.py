#!/usr/bin/env python3
"""Scan CUDA headers and compare against HAMI CUDA hook APIs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK_FILE = ROOT / "src/cuda/hook.c"


def strip_c_comments(content: str) -> str:
    content = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    content = re.sub(r"//.*", "", content)
    return content


def extract_hook_apis() -> list[str]:
    content = HOOK_FILE.read_text(encoding="utf-8")
    return re.findall(r'\{\.name\s*=\s*"(cu[^"]+)"\}', strip_c_comments(content))


def discover_include_dir(explicit: str | None) -> Path:
    if explicit:
        include_dir = Path(explicit).expanduser().resolve()
        if not (include_dir / "cuda.h").exists():
            raise FileNotFoundError(f"cuda.h not found under include dir: {include_dir}")
        return include_dir

    candidates: list[Path] = []

    for env_key in ("CUDA_PATH", "CUDA_HOME", "CUDAToolkit_ROOT"):
        env_val = os_env(env_key)
        if env_val:
            candidates.append(Path(env_val) / "include")

    candidates.append(Path("/usr/local/cuda/include"))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*/include"), reverse=True))

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        nvcc_bin = Path(nvcc_path).resolve()
        candidates.append(nvcc_bin.parent.parent / "include")

    seen: set[Path] = set()
    for include_dir in candidates:
        include_dir = include_dir.resolve()
        if include_dir in seen:
            continue
        seen.add(include_dir)
        if (include_dir / "cuda.h").exists():
            return include_dir

    raise FileNotFoundError(
        "Unable to locate CUDA include directory. "
        "Set --include-dir or CUDA_PATH/CUDA_HOME."
    )


def os_env(name: str) -> str | None:
    import os

    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def extract_declared_cuda_driver_apis(include_dir: Path) -> set[str]:
    declared: set[str] = set()
    pattern = re.compile(r"\bCUresult\b[\s\w\*]*?\b(cu[A-Za-z0-9_]+)\s*\(", flags=re.S)
    for header in include_dir.glob("*.h"):
        content = header.read_text(encoding="utf-8", errors="ignore")
        content = strip_c_comments(content)
        for match in pattern.findall(content):
            declared.add(match)
    return declared


def to_report(
    *,
    cuda_version: str,
    include_dir: Path,
    hook_apis: list[str],
    declared_apis: set[str],
) -> dict[str, object]:
    hook_set = set(hook_apis)
    present = sorted(hook_set & declared_apis)
    missing = sorted(hook_set - declared_apis)
    return {
        "cuda_version": cuda_version,
        "include_dir": str(include_dir),
        "hook_api_count": len(hook_set),
        "declared_driver_api_count": len(declared_apis),
        "present_hook_api_count": len(present),
        "missing_hook_api_count": len(missing),
        "missing_hook_apis": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HAMI hook APIs with installed CUDA header declarations."
    )
    parser.add_argument(
        "--cuda-version",
        default="unknown",
        help="CUDA version label for report output.",
    )
    parser.add_argument(
        "--include-dir",
        default=None,
        help="CUDA include directory path. If omitted, auto-discovery is used.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional report output path (JSON).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any hook API is missing from CUDA headers.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        include_dir = discover_include_dir(args.include_dir)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] {exc}")
        return 1

    hook_apis = extract_hook_apis()
    declared_apis = extract_declared_cuda_driver_apis(include_dir)
    report = to_report(
        cuda_version=args.cuda_version,
        include_dir=include_dir,
        hook_apis=hook_apis,
        declared_apis=declared_apis,
    )

    print(
        "[INFO] CUDA header matrix check "
        f"version={report['cuda_version']} include_dir={report['include_dir']}"
    )
    print(
        "[INFO] hook_apis={hook_api_count}, declared_driver_apis={declared_driver_api_count}, "
        "present={present_hook_api_count}, missing={missing_hook_api_count}".format(**report)
    )

    missing = report["missing_hook_apis"]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... (+{len(missing) - 20} more)"
        level = "ERROR" if args.strict else "WARN"
        print(f"[{level}] Hook APIs missing in headers: {preview}{suffix}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[INFO] Report written: {output}")

    if args.strict and report["missing_hook_api_count"] > 0:
        print("[FAIL] CUDA header matrix check failed in strict mode.")
        return 1

    print("[PASS] CUDA header matrix check completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
