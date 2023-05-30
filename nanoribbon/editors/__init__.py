"""Nanoribbon AiiDAlab viewers."""

from aiida import load_profile

load_profile()

from .replicate import NanoribbonReplicateEditor

__all__ = ["NanoribbonReplicateEditor"]
