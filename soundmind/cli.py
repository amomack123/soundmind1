"""
SoundMind v1 CLI - Argument parsing and dispatch.

Responsibilities:
- Argument parsing
- Dispatch orchestration
- Printing success/errors
- Exit codes

Forbidden:
- No filesystem writes
- No schema logic
- No stage imports
"""

import argparse
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with frozen help text."""
    parser = argparse.ArgumentParser(
        prog="soundmind",
        description="SoundMind v1 command-line interface.",
    )
    
    subparsers = parser.add_subparsers(dest="command")
    
    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Initialize a new SoundMind job workspace.",
        description=(
            "Initialize a new SoundMind job workspace.\n\n"
            "This command creates a deterministic job directory and writes an initial,\n"
            "schema-valid status.json. No audio processing, machine learning, or pipeline\n"
            "stages are executed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--job-id",
        metavar="JOB_ID",
        help="Explicit job identifier to use instead of generating one.",
    )
    run_parser.add_argument(
        "--jobs-root",
        metavar="PATH",
        default="./jobs",
        help="Root directory for job workspaces (default: ./jobs).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned job layout without creating files.",
    )
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """
    Handle the 'run' subcommand.
    
    Orchestrates job creation by delegating to jobs and status_init modules.
    Returns exit code.
    """
    from soundmind.jobs import resolve_job_id, create_workspace, WorkspaceExistsError
    from soundmind.status_init import build_initial_status, validate_status, serialize_status
    
    jobs_root = Path(args.jobs_root)
    job_id = resolve_job_id(args.job_id)
    job_dir = jobs_root / job_id
    status_path = job_dir / "status.json"
    
    # Dry-run: print planned layout and exit
    if args.dry_run:
        print(f"Job ID: {job_id}")
        print(f"Job directory: {job_dir}")
        print(f"Status file: {status_path}")
        return 0
    
    # Create workspace
    try:
        create_workspace(jobs_root, job_id)
    except WorkspaceExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Build and validate status
    status = build_initial_status(job_id)
    errors = validate_status(status)
    if errors:
        print(f"Error: Status validation failed: {errors}", file=sys.stderr)
        return 1
    
    # Write status.json
    status_json = serialize_status(status)
    status_path.write_text(status_json)
    
    print(f"Created job workspace: {job_dir}")
    return 0


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "run":
        exit_code = cmd_run(args)
        sys.exit(exit_code)
