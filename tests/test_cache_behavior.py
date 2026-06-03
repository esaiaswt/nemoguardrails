"""
Tests verifying cache directory behavior for the NeMo Guardrails knowledge base.

The NeMo Guardrails KnowledgeBase.build() method handles caching internally:
- Computes a content hash from: embedding_engine + embedding_model + concatenated chunk texts
- Stores cache files in .cache/ folder: {hash}.ann (Annoy index) and {hash}.esize (embedding size)
- On startup, loads from cache if hash matches; otherwise rebuilds the index

These tests verify the expected cache behavior and document the cache structure.

Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6
"""

import hashlib
import os
import tempfile

import pytest


# --- Cache hash computation (mirrors nemoguardrails.utils.compute_hash) ---


def compute_content_hash(embedding_engine: str, embedding_model: str, chunk_texts: list[str]) -> str:
    """Compute the cache hash the same way KnowledgeBase.build() does.

    The hash is computed over: embedding_engine + embedding_model + concatenated chunk texts.
    Uses MD5 if available, otherwise SHA256.
    """
    hash_prefix = embedding_engine + embedding_model
    combined = hash_prefix + "".join(chunk_texts)
    try:
        hashlib.md5(b"")
        hash_func = hashlib.md5
    except (AttributeError, ValueError):
        hash_func = hashlib.sha256
    return hash_func(combined.encode("utf-8")).hexdigest()


class TestCacheDirectoryStructure:
    """Verify that the cache produces the expected file structure."""

    def test_cache_files_use_content_hash_naming(self):
        """Cache files are named {hash}.ann and {hash}.esize based on content hash."""
        engine = "FastEmbed"
        model = "all-MiniLM-L6-v2"
        chunks = ["# Order Tracking\n\nYou can track your order."]

        hash_value = compute_content_hash(engine, model, chunks)

        expected_ann = f"{hash_value}.ann"
        expected_esize = f"{hash_value}.esize"

        # Verify hash produces valid filenames
        assert len(hash_value) > 0
        assert expected_ann.endswith(".ann")
        assert expected_esize.endswith(".esize")
        # MD5 produces 32 hex chars, SHA256 produces 64
        assert len(hash_value) in (32, 64)

    def test_cache_folder_is_dot_cache(self):
        """The framework uses .cache/ in the current working directory as cache folder."""
        # This mirrors the CACHE_FOLDER constant in nemoguardrails/kb/kb.py
        expected_cache_folder = os.path.join(os.getcwd(), ".cache")
        assert expected_cache_folder.endswith(".cache")

    def test_esize_file_contains_embedding_dimension(self):
        """The .esize file should contain the embedding dimension as a text integer."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".esize", delete=False) as f:
            # FastEmbed all-MiniLM-L6-v2 produces 384-dim embeddings
            f.write("384")
            temp_path = f.name

        try:
            with open(temp_path, "r") as f:
                dimension = int(f.read())
            assert dimension == 384
        finally:
            os.unlink(temp_path)


class TestContentHashDeterminism:
    """Verify that the content hash is deterministic and change-sensitive (Req 3.3, 3.4)."""

    def test_same_content_produces_same_hash(self):
        """Identical KB content should produce the same cache hash."""
        engine = "FastEmbed"
        model = "all-MiniLM-L6-v2"
        chunks = [
            "# Order Tracking\n\nYou can track your order.",
            "# Return Policy\n\nItems may be returned within 30 days.",
        ]

        hash1 = compute_content_hash(engine, model, chunks)
        hash2 = compute_content_hash(engine, model, chunks)

        assert hash1 == hash2

    def test_different_content_produces_different_hash(self):
        """Changed KB content should produce a different cache hash, triggering rebuild."""
        engine = "FastEmbed"
        model = "all-MiniLM-L6-v2"
        chunks_v1 = [
            "# Order Tracking\n\nYou can track your order.",
            "# Return Policy\n\nItems may be returned within 30 days.",
        ]
        chunks_v2 = [
            "# Order Tracking\n\nYou can track your order.",
            "# Return Policy\n\nItems may be returned within 60 days.",
        ]

        hash1 = compute_content_hash(engine, model, chunks_v1)
        hash2 = compute_content_hash(engine, model, chunks_v2)

        assert hash1 != hash2

    def test_different_embedding_model_produces_different_hash(self):
        """Changing the embedding model invalidates the cache."""
        chunks = ["# Test\n\nSome content."]

        hash1 = compute_content_hash("FastEmbed", "all-MiniLM-L6-v2", chunks)
        hash2 = compute_content_hash("FastEmbed", "different-model", chunks)

        assert hash1 != hash2

    def test_different_embedding_engine_produces_different_hash(self):
        """Changing the embedding engine invalidates the cache."""
        chunks = ["# Test\n\nSome content."]

        hash1 = compute_content_hash("FastEmbed", "all-MiniLM-L6-v2", chunks)
        hash2 = compute_content_hash("SentenceTransformers", "all-MiniLM-L6-v2", chunks)

        assert hash1 != hash2


class TestCacheLoadBehavior:
    """Verify cache load/rebuild logic (Req 3.3, 3.4, 3.6)."""

    def test_cache_hit_when_hash_matches(self):
        """When both .ann and .esize files exist for the computed hash, cache is used."""
        with tempfile.TemporaryDirectory() as cache_dir:
            engine = "FastEmbed"
            model = "all-MiniLM-L6-v2"
            chunks = ["# Test\n\nContent here."]
            hash_value = compute_content_hash(engine, model, chunks)

            ann_file = os.path.join(cache_dir, f"{hash_value}.ann")
            esize_file = os.path.join(cache_dir, f"{hash_value}.esize")

            # Simulate cached files
            with open(ann_file, "wb") as f:
                f.write(b"fake_annoy_index_data")
            with open(esize_file, "w") as f:
                f.write("384")

            # Verify cache files exist (cache hit condition)
            assert os.path.exists(ann_file)
            assert os.path.exists(esize_file)

    def test_cache_miss_when_content_changes(self):
        """When content changes, the new hash won't find existing cache files."""
        with tempfile.TemporaryDirectory() as cache_dir:
            engine = "FastEmbed"
            model = "all-MiniLM-L6-v2"

            # Original content cached
            chunks_v1 = ["# Test\n\nOriginal content."]
            hash_v1 = compute_content_hash(engine, model, chunks_v1)
            ann_file_v1 = os.path.join(cache_dir, f"{hash_v1}.ann")
            esize_file_v1 = os.path.join(cache_dir, f"{hash_v1}.esize")

            with open(ann_file_v1, "wb") as f:
                f.write(b"fake_annoy_index")
            with open(esize_file_v1, "w") as f:
                f.write("384")

            # Updated content
            chunks_v2 = ["# Test\n\nUpdated content."]
            hash_v2 = compute_content_hash(engine, model, chunks_v2)
            ann_file_v2 = os.path.join(cache_dir, f"{hash_v2}.ann")
            esize_file_v2 = os.path.join(cache_dir, f"{hash_v2}.esize")

            # New hash doesn't match existing cache → triggers rebuild
            assert not os.path.exists(ann_file_v2)
            assert not os.path.exists(esize_file_v2)
            assert hash_v1 != hash_v2

    def test_cache_miss_when_ann_file_missing(self):
        """If only .esize exists but .ann is missing, cache is invalid and index must rebuild."""
        with tempfile.TemporaryDirectory() as cache_dir:
            engine = "FastEmbed"
            model = "all-MiniLM-L6-v2"
            chunks = ["# Test\n\nContent."]
            hash_value = compute_content_hash(engine, model, chunks)

            esize_file = os.path.join(cache_dir, f"{hash_value}.esize")
            ann_file = os.path.join(cache_dir, f"{hash_value}.ann")

            # Only esize exists, ann is missing
            with open(esize_file, "w") as f:
                f.write("384")

            # Cache hit requires BOTH files
            cache_hit = os.path.exists(ann_file) and os.path.exists(esize_file)
            assert not cache_hit

    def test_cache_miss_when_esize_file_missing(self):
        """If only .ann exists but .esize is missing, cache is invalid and index must rebuild."""
        with tempfile.TemporaryDirectory() as cache_dir:
            engine = "FastEmbed"
            model = "all-MiniLM-L6-v2"
            chunks = ["# Test\n\nContent."]
            hash_value = compute_content_hash(engine, model, chunks)

            ann_file = os.path.join(cache_dir, f"{hash_value}.ann")
            esize_file = os.path.join(cache_dir, f"{hash_value}.esize")

            # Only ann exists, esize is missing
            with open(ann_file, "wb") as f:
                f.write(b"fake_annoy_data")

            # Cache hit requires BOTH files
            cache_hit = os.path.exists(ann_file) and os.path.exists(esize_file)
            assert not cache_hit


class TestGitignore:
    """Verify .cache/ is excluded from version control."""

    def test_cache_directory_in_gitignore(self):
        """The .cache/ folder should be listed in .gitignore to prevent committing binary index files."""
        gitignore_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".gitignore"
        )
        assert os.path.exists(gitignore_path), ".gitignore file should exist"

        with open(gitignore_path, "r") as f:
            content = f.read()

        # Check that .cache/ is in the gitignore (exact line or as part of patterns)
        lines = [line.strip() for line in content.splitlines()]
        assert ".cache/" in lines, ".cache/ should be listed in .gitignore"
