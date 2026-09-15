"""Build or audit offline oracle datasets. Defaults to the 64/16/16 smoke set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections.abc import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft.dataset import DatasetConfig, audit_export, build_dataset, check_export_directory, export_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline oracle SFT dataset construction, no training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "small"), default="smoke")
    parser.add_argument("--audit-only", action="store_true", help="Replay/check existing export without oracle")
    args = parser.parse_args(argv)
    try:
        if not args.audit_only:
            check_export_directory(args.output_dir)
            config = DatasetConfig() if args.preset == "smoke" else DatasetConfig(1024, 128, 128)
            bundle = build_dataset(config)
            export_dataset(bundle, args.output_dir)
        audit = audit_export(args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    summary = {"output_directory": str(args.output_dir.resolve()),
               "splits": {split: {key: value for key, value in info.items() if key != "coordinates"}
                          for split, info in audit["splits"].items()},
               "split_overlap": audit["split_overlap"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
