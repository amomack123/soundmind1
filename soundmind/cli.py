"""
SoundMind v1 CLI - Argument parsing and dispatch.

Responsibilities:
- Argument parsing
- Workspace creation and setup
- Input file handling
- Printing success/errors
- Exit codes

Forbidden:
- No schema logic
- No stage imports (Commit 2.5)
- No pipeline execution (Commit 2.5)
"""

import argparse
import shutil
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
            "This command creates a job directory structure, copies the input audio,\n"
            "and prepares the workspace for pipeline execution."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--input",
        metavar="PATH",
        required=True,
        help="Path to input audio file (WAV).",
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
    
    Creates full workspace structure, copies input file, writes metadata.
    Does NOT execute pipeline stages (that's Commit 3).
    Returns exit code.
    """
    from soundmind.jobs import resolve_job_id, create_full_workspace, WorkspaceExistsError
    from soundmind.utils import now_iso, serialize_json
    
    # Validate input file exists
    # NOTE: Input format is not validated here (treated as opaque blob).
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}", file=sys.stderr)
        return 1
    
    jobs_root = Path(args.jobs_root)
    job_id = resolve_job_id(args.job_id)
    job_dir = jobs_root / job_id
    
    # Dry-run: print planned layout and exit
    if args.dry_run:
        if job_dir.exists():
            print(f"Warning: Job workspace already exists: {job_dir}", file=sys.stderr)
        print(f"Job ID: {job_id}")
        print(f"Job directory: {job_dir}")
        print(f"Input file: {input_path}")
        print(f"Directories to create:")
        print(f"  {job_dir}/meta/")
        print(f"  {job_dir}/input/")
        print(f"  {job_dir}/ingest/")
        print(f"  {job_dir}/separation/")
        print(f"  {job_dir}/sqi/")
        print(f"  {job_dir}/diarization/")
        print(f"  {job_dir}/events/")
        print(f"  {job_dir}/rollup/")
        return 0
    
    # Create full workspace structure
    try:
        paths = create_full_workspace(jobs_root, job_id)
    except WorkspaceExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Copy input file to input/original.wav
    original_wav_path = paths["input_dir"] / "original.wav"
    shutil.copy2(input_path, original_wav_path)
    
    # Write input/input.json
    input_json = {
        "original_filename": input_path.name,
        "copied_at": now_iso(),
    }
    input_json_path = paths["input_dir"] / "input.json"
    input_json_path.write_text(serialize_json(input_json))
    
    # Write meta/run.json
    run_json = {
        "job_id": job_id,
        "started_at": now_iso(),
        "cli_args": {
            "input": str(input_path),
            "job_id": args.job_id,  # May be None if auto-generated
            "jobs_root": str(jobs_root),
        },
    }
    run_json_path = paths["meta_dir"] / "run.json"
    run_json_path.write_text(serialize_json(run_json))
    
    # Write job-level status.json (initialized, not yet run)
    job_status = {
        "job_id": job_id,
        "version": "v1",
        "state": "initialized",
        "success": None,
        "stages": {},
        "errors": [],
    }
    status_path = paths["job_dir"] / "status.json"
    status_path.write_text(serialize_json(job_status))
    
    # Build JobContext and run pipeline
    from soundmind.context import JobContext
    from soundmind.pipeline import run_pipeline
    from soundmind.jobs import STAGE_NAMES
    
    ctx = JobContext(
        job_id=job_id,
        job_dir=paths["job_dir"],
        meta_dir=paths["meta_dir"],
        input_wav_path=paths["input_dir"] / "original.wav",
        input_json_path=paths["input_dir"] / "input.json",
        stage_dirs={name: paths["job_dir"] / name for name in STAGE_NAMES},
        run_config={},
    )
    
    # Run pipeline (overwrites initialized status.json)
    success = run_pipeline(ctx)
    
    if success:
        print(f"Pipeline completed successfully: {job_dir}")
        return 0
    else:
        print(f"Pipeline failed: {job_dir}", file=sys.stderr)
        return 1


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

