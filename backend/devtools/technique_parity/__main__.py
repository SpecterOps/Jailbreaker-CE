"""Command-line entry point for captured parity evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.techniques.core.redaction import contains_credential_text
from devtools.technique_parity.compare import (
    ParityDifference,
    ParityPolicy,
    ParityReport,
)
from devtools.technique_parity.runtime import (
    ParityArmConfigurationError,
    ParityArmExecutionError,
    ParityRuntime,
    ParityRuntimeFactory,
    default_parity_runtime,
    parity_route_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m devtools.technique_parity",
        description="Resolve and compare legacy and modular technique evidence.",
    )
    parser.add_argument("--technique", required=True, help="Canonical technique ID.")
    parser.add_argument(
        "--legacy-implementation",
        required=True,
        help="Pinned legacy implementation ID.",
    )
    parser.add_argument(
        "--modular-implementation",
        required=True,
        help="Pinned modular implementation ID.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("deterministic", "live"),
        help="Comparison mode.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Redacted JSON report path.")
    parser.add_argument(
        "--legacy-input",
        type=Path,
        help="Captured legacy JSON evidence or live sample array.",
    )
    parser.add_argument(
        "--modular-input",
        type=Path,
        help="Captured modular JSON evidence or live sample array.",
    )
    parser.add_argument(
        "--policy",
        "--policy-file",
        dest="policy",
        type=Path,
        help="Pinned JSON/YAML parity policy (required for live mode).",
    )
    parser.add_argument("--dataset-id", help="Pinned live-comparison dataset ID.")
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Pinned repeats per live experiment cell (default: 1).",
    )
    parser.add_argument(
        "--profile-id",
        action="append",
        default=[],
        metavar="ROLE=PROFILE_ID",
        help="Pinned live model profile; repeat for each logical role.",
    )
    parser.add_argument("--target-profile-id", help="Pinned target profile ID.")
    parser.add_argument("--attacker-profile-id", help="Pinned attacker profile ID.")
    parser.add_argument("--target-judge-profile-id", help="Pinned target_judge profile ID.")
    parser.add_argument(
        "--seed",
        "--random-seed",
        dest="seed",
        type=int,
        default=0,
        help="Pinned random seed supplied to both parity arms (default: 0).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime: ParityRuntime | None = None,
    runtime_factory: ParityRuntimeFactory = default_parity_runtime,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeat_count < 1:
        parser.error("--repeat-count must be a positive integer")

    profiles = _profiles(args, parser)
    if args.mode == "live":
        if args.policy is None:
            parser.error("live mode requires --policy before any model calls or evidence loading")
        if not args.dataset_id:
            parser.error("live mode requires --dataset-id")
        if not profiles:
            parser.error("live mode requires at least one pinned profile ID")

    # This deliberately occurs before either captured arm is opened. A later
    # live runner can call the same entry point without making tolerances
    # observable or mutable after calls begin.
    try:
        policy = (
            ParityPolicy.from_file(
                args.policy,
                require_explicit_success_threshold=args.mode == "live",
            )
            if args.policy is not None
            else ParityPolicy.strict()
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "live":
        try:
            policy.validate_live_configuration()
        except ValueError as exc:
            parser.error(str(exc))
        if policy.dataset_id != args.dataset_id:
            parser.error("--dataset-id must match the pinned live policy")
        if policy.repeat_count != args.repeat_count:
            parser.error("--repeat-count must match the pinned live policy")
        if dict(policy.profile_ids) != profiles:
            parser.error("CLI profile IDs must match the pinned live policy")
    metadata = {
        "technique": args.technique,
        "legacy_implementation": args.legacy_implementation,
        "modular_implementation": args.modular_implementation,
        "dataset_id": args.dataset_id,
        "repeat_count": args.repeat_count,
        "profile_ids": profiles,
        "seed": args.seed,
    }

    if (args.legacy_input is None) != (args.modular_input is None):
        parser.error("--legacy-input and --modular-input must be supplied together")
    try:
        selected_runtime = runtime if runtime is not None else runtime_factory()
        arm_session = selected_runtime.issue_deterministic_arms(
            args.technique,
            legacy_implementation_id=args.legacy_implementation,
            modular_implementation_id=args.modular_implementation,
        )
        legacy_route, modular_route = selected_runtime.deterministic_arm_routes(
            arm_session
        )
    except Exception:  # noqa: BLE001 - emit a safe failed report for unavailable routes
        return _write_failed_report(
            args.output,
            mode=args.mode,
            policy=policy,
            metadata=metadata,
            path="comparison.route_resolution",
            message="one or more exact catalog/router selectors are unavailable",
        )

    metadata = {
        **metadata,
        "resolved_routes": {
            "legacy": parity_route_identity(legacy_route),
            "modular": parity_route_identity(modular_route),
        },
    }
    try:
        selected_runtime.validate_policy_threshold(
            policy,
            arm_session,
            require_declared=args.mode == "live",
        )
    except ValueError:
        return _write_failed_report(
            args.output,
            mode=args.mode,
            policy=policy,
            metadata=metadata,
            path="comparison.route_threshold",
            message="policy success threshold does not match both resolved routes",
        )
    if args.legacy_input is None:
        captures = None
    else:
        try:
            captures = (
                _load_json(args.legacy_input, parser),
                _load_json(args.modular_input, parser),
            )
        except BaseException:
            # ``ArgumentParser.error`` raises SystemExit. The exact route session
            # still must be permanently invalidated before that failure escapes.
            selected_runtime._invalidate_deterministic_arm_session(  # noqa: SLF001
                arm_session
            )
            raise

    try:
        if captures is None and args.mode == "deterministic":
            compared, _ = selected_runtime.execute_and_compare_deterministic(
                arm_session,
                policy=policy,
                dataset_id=args.dataset_id,
                repeat_count=args.repeat_count,
                profile_ids=profiles,
                seed=args.seed,
            )
        elif captures is None:
            compared, _ = selected_runtime.execute_and_compare_live(
                arm_session,
                policy=policy,
                dataset_id=args.dataset_id,
                repeat_count=args.repeat_count,
                profile_ids=profiles,
                seed=args.seed,
            )
        elif args.mode == "deterministic":
            compared = selected_runtime.compare_deterministic_captures(
                captures,
                arm_session,
                policy=policy,
            )
        else:
            compared = selected_runtime.compare_live_captures(
                captures,
                arm_session,
                policy=policy,
            )
    except ParityArmConfigurationError:
        return _write_failed_report(
            args.output,
            mode=args.mode,
            policy=policy,
            metadata=metadata,
            path="comparison.arm_runner",
            message="no generic parity arm runner is registered for the resolved routes",
        )
    except ParityArmExecutionError:
        return _write_failed_report(
            args.output,
            mode=args.mode,
            policy=policy,
            metadata=metadata,
            path="comparison.arm_execution",
            message="a generic parity arm runner failed before comparison",
        )
    except (TypeError, ValueError):
        return _write_failed_report(
            args.output,
            mode=args.mode,
            policy=policy,
            metadata=metadata,
            path="comparison.capture_validation",
            message="arm evidence did not match its resolved selector or evidence contract",
        )
    report = ParityReport(
        mode=compared.mode,
        passed=compared.passed,
        policy=compared.policy,
        differences=compared.differences,
        metrics=compared.metrics,
        metadata=metadata,
    )
    report.write_json(args.output)
    return 0 if report.passed else 1


def _write_failed_report(
    output: Path,
    *,
    mode: str,
    policy: ParityPolicy,
    metadata: Mapping[str, object],
    path: str,
    message: str,
) -> int:
    report = ParityReport(
        mode=mode,
        passed=False,
        policy=policy,
        differences=(
            ParityDifference(
                path=path,
                legacy=None,
                modular=None,
                message=message,
            ),
        ),
        metadata={**metadata, "status": "not_run"},
    )
    report.write_json(output)
    return 1


def _profiles(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, str]:
    profiles: dict[str, str] = {}
    explicit = {
        "target": args.target_profile_id,
        "attacker": args.attacker_profile_id,
        "target_judge": args.target_judge_profile_id,
    }
    for role, profile_id in explicit.items():
        if profile_id:
            profiles[role] = _validate_profile(role, profile_id, parser)
    for item in args.profile_id:
        if "=" not in item:
            parser.error("--profile-id must use ROLE=PROFILE_ID")
        role, profile_id = item.split("=", 1)
        role = role.strip()
        profile_id = profile_id.strip()
        if not role or not profile_id:
            parser.error("--profile-id must use non-empty ROLE=PROFILE_ID values")
        if role in profiles and profiles[role] != profile_id:
            parser.error(f"profile role '{role}' was pinned more than once with different IDs")
        profiles[role] = _validate_profile(role, profile_id, parser)
    return dict(sorted(profiles.items()))


def _validate_profile(
    role: str,
    profile_id: str,
    parser: argparse.ArgumentParser,
) -> str:
    if contains_credential_text(role) or contains_credential_text(profile_id):
        parser.error("profile roles and IDs must not contain credential material")
    return profile_id


def _load_json(path: Path, parser: argparse.ArgumentParser) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"could not load JSON evidence from {path}: {exc}")
    raise AssertionError("argparse parser.error must exit")  # pragma: no cover
if __name__ == "__main__":  # pragma: no cover - exercised through module execution
    raise SystemExit(main())
