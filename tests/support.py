"""Helpers shared by test modules.

`cli()` saves sessions, the profile and the RAG index under paths relative to the current
directory (sessions/, profile.json, data/). A test that runs it from the repository root
leaves those files in the project. Running it from a temporary directory keeps the
repository clean; the original directory is restored afterwards.

Import it with the tests directory on sys.path, as the test modules already do for source/:

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from support import use_temporary_working_directory
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def temporary_working_directory() -> Iterator[Path]:
    """Run the enclosed code from a new empty directory, then come back and delete it."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            yield Path(folder).resolve()
        finally:
            os.chdir(previous)


def use_temporary_working_directory(test: unittest.TestCase) -> Path:
    """From `setUp`: run this test from its own temporary directory. Returns that directory."""
    context = temporary_working_directory()
    path = context.__enter__()
    test.addCleanup(context.__exit__, None, None, None)
    return path


def use_temporary_working_directory_for_module() -> Path:
    """From `setUpModule`: run every test of the module from one temporary directory."""
    context = temporary_working_directory()
    path = context.__enter__()
    unittest.addModuleCleanup(context.__exit__, None, None, None)
    return path
