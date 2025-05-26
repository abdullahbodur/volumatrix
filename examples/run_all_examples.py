#!/usr/bin/env python3
"""
Run All Examples

This script runs all Volumatrix examples in sequence. Useful for testing
and demonstrating the full capabilities of the library.
"""

import sys
import time
import importlib.util
from pathlib import Path


def run_example(example_path):
  """Run a single example and return success status."""
  print(f"\n{'=' * 60}")
  print(f"Running: {example_path.name}")
  print(f"{'=' * 60}")

  try:
    # Load and execute the example module
    spec = importlib.util.spec_from_file_location("example", example_path)
    module = importlib.util.module_from_spec(spec)

    start_time = time.time()
    spec.loader.exec_module(module)
    end_time = time.time()

    print(
      f"\n{example_path.name} completed successfully in {end_time - start_time:.2f}s")
    return True

  except Exception as e:
    print(f"\n{example_path.name} failed with error: {e}")
    return False


def main():
  """Run all examples in the examples directory."""
  print("Volumatrix - Running All Examples")
  print("=" * 60)

  # Get the examples directory
  examples_dir = Path(__file__).parent

  # Define the order of examples (recommended learning path)
  example_order = [
      "basic_generation.py",
      "transformations.py",
      "conversions.py",
      "scene_management.py",
      "export_formats.py",
      "batch_processing.py",
      "interactive_visualization.py"
  ]

  # Track results
  results = {}
  total_start_time = time.time()

  # Run each example
  for example_name in example_order:
    example_path = examples_dir / example_name

    if example_path.exists():
      success = run_example(example_path)
      results[example_name] = success
    else:
      print(f"Example not found: {example_name}")
      results[example_name] = False

  total_end_time = time.time()
  total_time = total_end_time - total_start_time

  # Print summary
  print(f"\n{'=' * 60}")
  print("SUMMARY")
  print(f"{'=' * 60}")

  successful = sum(results.values())
  total = len(results)

  print(f"Successful: {successful}/{total}")
  print(f"Total time: {total_time:.2f} seconds")
  print(f"Success rate: {successful / total * 100:.1f}%")

  print(f"\nDetailed Results:")
  for example_name, success in results.items():
    status = "PASS" if success else "FAIL"
    print(f"   {status} {example_name}")

  if successful == total:
    print(f"\nAll examples completed successfully!")
  else:
    print(f"\nSome examples failed. Check the error messages above.")
    sys.exit(1)


if __name__ == "__main__":
  main()
