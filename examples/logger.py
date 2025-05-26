#!/usr/bin/env python3
"""
Simple Logger Module

This module provides a simple logging configuration.
Logging level can be controlled via the DEBUG environment variable:
- DEBUG=0: Only critical errors (default)
- DEBUG=1: Errors
- DEBUG=2: Warnings 
- DEBUG=3: Info
- DEBUG=4: Debug
- DEBUG=5: Detailed debug with timestamps and module info
"""

import os
import logging
from pathlib import Path

# Define logging level mapping
LOG_LEVELS = {
    0: (logging.CRITICAL, '%(levelname)s: %(message)s'),
    1: (logging.ERROR, '%(levelname)s: %(message)s'), 
    2: (logging.WARNING, '%(levelname)s: %(message)s'),
    3: (logging.INFO, '%(levelname)s: %(message)s'),
    4: (logging.DEBUG, '%(levelname)s: %(message)s'),
    5: (logging.DEBUG, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
}

def setup_logger(name=None):
    """Setup and return a configured logger instance.
    
    Args:
        name (str, optional): Logger name. Defaults to module name if None.
    
    Returns:
        logging.Logger: Configured logger instance with appropriate level and formatting.
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.debug("Debug message")
        >>> logger.info("Info message") 
    """
    # Get debug level from environment, defaulting to 0 (CRITICAL only)
    debug_level = min(max(int(os.getenv('DEBUG', 0)), 0), 5)
    
    # Get corresponding log level and format
    level, format_str = LOG_LEVELS[debug_level]
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add handler if needed
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
    
    return logger