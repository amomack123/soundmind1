"""
SoundMind v1 JobContext - Pipeline execution context.

Responsibilities:
- Hold all paths and configuration for a job
- Serialization for debugging/logging

Invariants:
- Immutable during pipeline execution
- All paths are absolute
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobContext:
    """Context passed through all pipeline stages."""
    
    job_id: str
    job_dir: Path
    meta_dir: Path
    input_wav_path: Path
    input_json_path: Path
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "job_id": self.job_id,
            "job_dir": str(self.job_dir),
            "meta_dir": str(self.meta_dir),
            "input_wav_path": str(self.input_wav_path),
            "input_json_path": str(self.input_json_path),
            "stage_dirs": {k: str(v) for k, v in self.stage_dirs.items()},
            "run_config": self.run_config,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobContext":
        """
        Deserialize from dictionary.
        
        Note:
            Forward-looking: not used in Commit 3, but included for
            future resumption/serialization workflows.
        """
        return cls(
            job_id=data["job_id"],
            job_dir=Path(data["job_dir"]),
            meta_dir=Path(data["meta_dir"]),
            input_wav_path=Path(data["input_wav_path"]),
            input_json_path=Path(data["input_json_path"]),
            stage_dirs={k: Path(v) for k, v in data["stage_dirs"].items()},
            run_config=data.get("run_config", {}),
        )
