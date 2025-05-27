#!/usr/bin/env python3
"""
Auto-Visualization Demo

This example demonstrates the automatic visualization capabilities of Volumatrix.
"""

from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def main():
    """Run the auto-visualization demo."""
    log.info("Starting Volumatrix Auto-Visualization Demo")
    log.info("=" * 50)
    # Method 1: Auto-preview with generate()
    log.info("Method 1: Auto-preview with generate()")
    log.debug("Generating cube with automatic visualization...")
    cube = vm.generate("cube")
    log.debug(f"Generated: {cube.name}")
    # Method 2: Using generate_and_show()
    log.info("Method 2: Using generate_and_show()")
    log.debug("Generating sphere with automatic visualization...")
    sphere = vm.generate_and_show("sphere")
    log.debug(f"Generated: {sphere.name}")
    # Method 3: Manual visualization with show()
    log.info("Method 3: Manual visualization with show()")
    log.debug("Generating cylinder and then showing it...")
    cylinder = vm.generate("cylinder")
    vm.show(cylinder)
    log.debug(f"Generated and showed: {cylinder.name}")
    # Method 4: Scene visualization
    log.info("Method 4: Scene visualization")
    log.debug("Creating a scene with multiple objects...")
    scene = vm.Scene(name="AutoVizDemo")
    scene.add(cube, name="Cube", position=[0, 0, 0])
    scene.add(sphere, name="Sphere", position=[3, 0, 0])
    scene.add(cylinder, name="Cylinder", position=[0, 3, 0])
    log.debug(f"Scene created with {len(scene)} objects")
    vm.show(scene)
    log.info("Auto-visualization demo completed successfully!")


if __name__ == "__main__":
    main()
