#!/usr/bin/env python3
"""
Run All Examples

This script runs all Volumatrix examples and reports their status.
"""

import importlib
import os
import time
from pathlib import Path

from logger import setup_logger

log = setup_logger(__name__)


def run_example(example_path):
    """Run a single example and return its status."""
    try:
        log.info(f"{'=' * 60}")
        log.info(f"Running: {example_path.name}")
        log.info(f"{'=' * 60}")

        # Import and run the example
        module_name = f"examples.{example_path.stem}"
        module = importlib.import_module(module_name)

        start_time = time.time()
        module.main()
        end_time = time.time()

        return {
            "name": example_path.name,
            "status": "success",
            "time": end_time - start_time,
        }

    except Exception as e:
        log.error(f"{example_path.name} failed with error: {e}")
        return {"name": example_path.name, "status": "failed", "error": str(e)}


def main():
    """Run all examples and report results."""
    log.info("Volumatrix - Running All Examples")
    log.info("=" * 60)

    # Get all example files
    examples_dir = Path(__file__).parent
    example_files = sorted(
        [
            f
            for f in examples_dir.glob("*.py")
            if f.stem not in ["__init__", "run_all_examples", "log"]
        ]
    )

    # Run each example
    results = []
    for example_file in example_files:
        result = run_example(example_file)
        results.append(result)

    # Print summary
    log.info(f"{'=' * 60}")
    log.info("SUMMARY")
    log.info(f"{'=' * 60}")

    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    total_time = sum(r.get("time", 0) for r in results)

    log.info(f"Successful: {successful}/{total}")
    log.info(f"Total time: {total_time:.2f} seconds")
    log.info(f"Success rate: {successful / total * 100:.1f}%")

    log.info(f"Detailed Results:")
    for result in results:
        status = "✓" if result["status"] == "success" else "✗"
        log.info(f"   {status} {result['name']}")

    if successful == total:
        log.info(f"All examples completed successfully!")
    else:
        log.info(f"Some examples failed. Check the error messages above.")


if __name__ == "__main__":
    main()
