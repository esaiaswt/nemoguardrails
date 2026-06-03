"""
Unit tests for knowledge base folder edge cases.

Tests verify that validate_kb_folder() correctly handles:
- Empty kb/ folder produces empty index (folder_exists=True, doc_count=0)
- Missing kb/ folder disables RAG (folder_exists=False)
- Invalid UTF-8 files are skipped with appropriate logging
- Valid .md files are properly counted

Validates: Requirements 1.4, 1.5, 1.6
"""

import os
import sys
from unittest.mock import patch

import pytest


class TestEmptyKbFolderProducesEmptyIndex:
    """Test that an empty kb/ folder produces an empty index."""

    def test_empty_kb_folder_produces_empty_index(self, tmp_path):
        """Create a temp config dir with an empty kb/ folder.
        Call validate_kb_folder() and verify folder_exists=True, doc_count=0, skipped_files=[].

        Validates: Requirement 1.4
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Create an empty kb/ folder
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()

        result = app.validate_kb_folder(str(tmp_path))

        assert result["folder_exists"] is True
        assert result["doc_count"] == 0
        assert result["skipped_files"] == []


class TestMissingKbFolderDisablesRag:
    """Test that a missing kb/ folder disables RAG."""

    def test_missing_kb_folder_disables_rag(self, tmp_path):
        """Create a temp config dir without a kb/ folder.
        Call validate_kb_folder() and verify folder_exists=False.

        Validates: Requirement 1.5
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # tmp_path exists but has no kb/ subfolder
        result = app.validate_kb_folder(str(tmp_path))

        assert result["folder_exists"] is False
        assert result["doc_count"] == 0
        assert result["skipped_files"] == []


class TestInvalidUtf8FileSkipped:
    """Test that invalid UTF-8 files are skipped with appropriate logging."""

    def test_invalid_utf8_file_skipped(self, tmp_path):
        """Create a temp config dir with kb/ containing a file with invalid UTF-8 bytes.
        Call validate_kb_folder() and verify the file appears in skipped_files.

        Validates: Requirement 1.6
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Create kb/ folder with an invalid UTF-8 file
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()

        # Write invalid UTF-8 bytes to a .md file
        bad_file = kb_dir / "bad_encoding.md"
        bad_file.write_bytes(b"\xff\xfe\x80\x81 invalid utf-8 content \xc3\x28")

        result = app.validate_kb_folder(str(tmp_path))

        assert result["folder_exists"] is True
        assert "bad_encoding.md" in result["skipped_files"]
        assert result["doc_count"] == 0


class TestValidMdFilesCounted:
    """Test that valid .md files are properly counted."""

    def test_valid_md_files_counted(self, tmp_path):
        """Create a temp config dir with kb/ containing valid .md files.
        Verify doc_count matches the number of valid files.

        Validates: Requirements 1.4, 1.5, 1.6
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Create kb/ folder with valid markdown files
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()

        (kb_dir / "doc1.md").write_text("# Document 1\n\nSome content.", encoding="utf-8")
        (kb_dir / "doc2.md").write_text("# Document 2\n\nMore content.", encoding="utf-8")
        (kb_dir / "doc3.md").write_text("# Document 3\n\nEven more.", encoding="utf-8")

        result = app.validate_kb_folder(str(tmp_path))

        assert result["folder_exists"] is True
        assert result["doc_count"] == 3
        assert result["skipped_files"] == []

    def test_mixed_valid_and_invalid_files(self, tmp_path):
        """Verify that valid files are counted even when some files are invalid.

        Validates: Requirement 1.6
        """
        if "app" in sys.modules:
            del sys.modules["app"]

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-testkey12345678"}):
            import app

        # Create kb/ folder with a mix of valid and invalid files
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()

        (kb_dir / "valid1.md").write_text("# Valid\n\nGood content.", encoding="utf-8")
        (kb_dir / "valid2.md").write_text("# Also Valid\n\nMore good content.", encoding="utf-8")
        (kb_dir / "invalid.md").write_bytes(b"\xff\xfe\x80\x81 bad bytes")

        result = app.validate_kb_folder(str(tmp_path))

        assert result["folder_exists"] is True
        assert result["doc_count"] == 2
        assert "invalid.md" in result["skipped_files"]
        assert len(result["skipped_files"]) == 1
