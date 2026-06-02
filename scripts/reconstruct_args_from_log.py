#!/usr/bin/env python

import argparse
import ast
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct args.json from the printed options dictionary at the top of a run log."
    )
    parser.add_argument("logfile", help="Path to the .out run log")
    parser.add_argument(
        "--output",
        default=None,
        help="Output args.json path; defaults to args.json next to the logfile"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists"
    )
    return parser.parse_args()


def extract_literal_dict(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("No opening '{' found in log")

    depth = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError("Could not find matching '}' for options dictionary")

    return ast.literal_eval(text[start:end])


def main():
    args = parse_args()
    logfile = Path(args.logfile)
    output = Path(args.output) if args.output is not None else logfile.with_name("args.json")

    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; use --force to overwrite")

    options = extract_literal_dict(logfile.read_text())
    output.write_text(json.dumps(options, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
