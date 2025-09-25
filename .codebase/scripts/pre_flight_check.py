import os
import sys
import subprocess
import re
from typing import List, Set, Dict


def get_changed_files(base_sha: str, head_sha: str) -> List[str]:
    """Gets the list of changed files between two commits."""
    cmd = ["git", "diff", "--name-only", base_sha, head_sha]
    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        files = result.stdout.strip().split("\n")
        return [f for f in files if f]
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.stderr}", file=sys.stderr)
        return ["*"]


def get_required_suite(changed_files: List[str]) -> Dict[str, Set[str]]:
    """
    Determines the required test suite and affected platforms using a platform-aware rule structure.
    Returns a dictionary containing the set of required suites and the set of affected platforms.
    """
    if changed_files == ["*"]:
        print(
            "Warning: git diff failed. Assuming full test suite is required.",
            file=sys.stderr)
        return {
            "suites": {"all", "compile-check"},
            "platforms": {"nvidia", "amd"}
        }

    IGNORE_PATTERNS = r"(\.md|LICENSE|README)$|^(docs/|asset/)|^\.codebase/scripts/(pre_flight_check\.py|check_ci_run\.sh|run_step\.sh)$|^code-format\.sh$"

    PATTERNS = {
        "generic": {
            "Core Change": (
                r"^(csrc/|include/|lib/|shmem/|CMakeLists\.txt|python/setup\.py|patches/)",
                {
                    "unittest", "e2e", "tutorial", "megakernel", "internal",
                    "compile-check"
                },
            ),
            "Model Change": (
                r"^(python/triton_dist/models/)",
                {"e2e"},
            ),
            "e2e env build change": (r"scripts/build_e2e_env\.sh", {"e2e"}),
            "Tutorial Change": (r"tutorials/", {"tutorial"}),
        },
        "nvidia": {
            "Layer Change (Unittest)":
            (r"python/triton_dist/layers/nvidia/(ep_a2a_layer|gemm_allreduce_layer|low_latency_allgather_layer)\.py",
             {"unittest"}),
            "Layer Change (Internal)":
            (r"python/triton_dist/layers/nvidia/p2p\.py", {"internal"}),
            "Layer Change (E2E)":
            (r"python/triton_dist/layers/nvidia/tp_.*\.py", {"e2e"}),
            "Megakernel Change": (r"python/triton_dist/mega_triton_kernel/",
                                  {"megakernel"}),
            "Megakernel Test Change": (
                r"test/nvidia/.*megakernel|run_mega_kernel_test\.sh",
                {"megakernel"},
            ),
            "E2E Test Change": (
                r"test/nvidia/.*(e2e|tp_attn|tp_mlp|tp_moe)|run_e2e_test\.sh",
                {"e2e"},
            ),
            "Unit Test Change":
            (r"test/nvidia/*|run_unittest\.sh|patch_torch_compile|monkey_inductor",
             {"unittest"}),
            "Internal Test Change": (r"run_m10_related_tests\.sh",
                                     {"internal"}),
        },
        "amd": {
            "Layer Change": (r"python/triton_dist/layers/amd/", {"e2e"}),
            "Unit Test Change": (r"test/amd/*|run_unittest\.sh", {"unittest"}),
            "E2E Test Change": (
                r"test/amd/*|run_e2e_test\.sh",
                {"e2e"},
            ),
        }
    }

    required_suites = set()
    affected_platforms = set()

    all_files_ignorable = all(
        re.search(IGNORE_PATTERNS, file) for file in changed_files)
    if all_files_ignorable:
        print(
            "Rule match: All changed files are documentation or CI config. No tests needed.",
            file=sys.stderr)
        return {"suites": set(), "platforms": set()}

    for file in changed_files:
        if re.search(IGNORE_PATTERNS, file):
            print(f"Ignoring file: {file}", file=sys.stderr)
            continue

        matched_any_rule = False
        for platform_name, platform_patterns in PATTERNS.items():
            for rule_name, (pattern, suite) in platform_patterns.items():
                if re.search(pattern, file):
                    log_msg = f"[{platform_name}] Rule match ({rule_name}): '{file}' requires suite(s): {', '.join(suite)}"
                    print(log_msg, file=sys.stderr)
                    required_suites.update(suite)

                    if platform_name in ["nvidia", "amd"]:
                        affected_platforms.add(platform_name)
                    else:  # A generic change affects all platforms
                        affected_platforms.update({"nvidia", "amd"})

                    matched_any_rule = True

        if not matched_any_rule:
            print(
                f"[general] Rule match (Fallback): Unclassified change in '{file}'. Defaulting to all tests for safety.",
                file=sys.stderr)
            required_suites.add("all")

    if not required_suites:
        print("No runnable code changes found after filtering.",
              file=sys.stderr)
        return {"suites": set(), "platforms": set()}

    # If a fallback to "all" happened, it affects all platforms
    if "all" in required_suites:
        affected_platforms.update({"nvidia", "amd"})

    return {"suites": required_suites, "platforms": affected_platforms}


def main():
    platform = os.environ.get("PLATFORM")
    job_type = os.environ.get("JOB_TYPE")
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")

    if not all([platform, job_type]):
        print("Error: Required environment variables must be set.",
              file=sys.stderr)
        sys.exit(1)

    if not base_sha:
        print(
            "Warning: BASE_SHA not set. Falling back to the local 'distributed-main' branch.",
            file=sys.stderr)
        base_sha = run_git_command(["git", "rev-parse", "distributed-main"])

    if not head_sha:
        print(
            "Warning: HEAD_SHA not set. Falling back to the current commit (HEAD).",
            file=sys.stderr)
        head_sha = run_git_command(["git", "rev-parse", "HEAD"])

    print(
        f"--- Analyzing changes for Job Type: '{job_type}' on Platform: '{platform}' ---",
        file=sys.stderr)
    print(f"--- Diffing from base: {base_sha} to head: {head_sha} ---",
          file=sys.stderr)

    changed_files = get_changed_files(base_sha, head_sha)
    if not changed_files:
        print("No changed files detected. Skipping.", file=sys.stderr)
        print("SKIP", end='')
        return

    print("Changed files:\n" + "\n".join(f"- {f}" for f in changed_files),
          file=sys.stderr)

    analysis = get_required_suite(changed_files)
    required_suites = analysis["suites"]
    affected_platforms = analysis["platforms"]

    print("\nAnalysis complete.", file=sys.stderr)
    print(f"-> Required Suites: {required_suites or {'None'}}",
          file=sys.stderr)
    print(f"-> Affected Platforms: {affected_platforms or {'None'}}",
          file=sys.stderr)

    run_job = False
    if not required_suites:
        run_job = False
    else:
        # A job should run if:
        # 1. The current platform is one of the affected platforms.
        # 2. AND the current job type is in the required suites (or 'all' is required).
        # Note: If affected_platforms is empty (e.g. from a tutorial change), we assume
        # it's a generic task that any runner can handle, so the platform check passes.
        platform_match = (platform
                          in affected_platforms) or (not affected_platforms)
        suite_match = ("all" in required_suites) or (job_type
                                                     in required_suites)

        if platform_match and suite_match:
            run_job = True

    if run_job:
        print(f"\nDECISION: PROCEED with job ('{job_type}' on '{platform}').",
              file=sys.stderr)
        print("PROCEED", end='')
    else:
        print(f"\nDECISION: SKIP job ('{job_type}' on '{platform}').",
              file=sys.stderr)
        print("SKIP", end='')


def run_git_command(cmd: List[str]) -> str:
    """Runs a git command and returns its stripped stdout."""
    try:
        print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
