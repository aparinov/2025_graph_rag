#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verification script for collection-based migration."""

import sys


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from qdrant_manager import QdrantManager
        print("  ✓ QdrantManager imported")
    except Exception as e:
        print(f"  ✗ QdrantManager import failed: {e}")
        return False

    try:
        from app.services.document_service import DocumentService
        print("  ✓ DocumentService imported")
    except Exception as e:
        print(f"  ✗ DocumentService import failed: {e}")
        return False

    try:
        from app.services.qa_service import QAService
        print("  ✓ QAService imported")
    except Exception as e:
        print(f"  ✗ QAService import failed: {e}")
        return False

    try:
        from app.ui.gradio_app import create_app
        print("  ✓ Gradio UI imported")
    except Exception as e:
        print(f"  ✗ Gradio UI import failed: {e}")
        return False

    return True


def test_method_signatures():
    """Test that method signatures are correct."""
    print("\nTesting method signatures...")
    try:
        from qdrant_manager import QdrantManager
        import inspect

        # Check QdrantManager methods
        methods = {
            'create_collection': ['collection_name'],
            'get_collections': [],
            'get_collection_type': ['collection_name'],
            'search_multi_collection': ['query', 'collection_names', 'k'],
            'delete_collection': ['collection_name'],
        }

        for method_name, expected_params in methods.items():
            if hasattr(QdrantManager, method_name):
                sig = inspect.signature(getattr(QdrantManager, method_name))
                params = [p for p in sig.parameters.keys() if p != 'self']
                if all(p in params for p in expected_params):
                    print(f"  ✓ {method_name} has correct signature")
                else:
                    print(f"  ✗ {method_name} signature mismatch")
                    print(f"    Expected: {expected_params}")
                    print(f"    Found: {params}")
            else:
                print(f"  ✗ {method_name} not found")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Signature test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility features."""
    print("\nTesting backward compatibility...")
    try:
        from qdrant_manager import QdrantManager
        import inspect

        # Check that search() still accepts session_name (backward compatible)
        sig = inspect.signature(QdrantManager.search)
        params = list(sig.parameters.keys())
        if 'session_name' in params:
            print("  ✓ search() maintains session_name parameter")
        else:
            print("  ✗ search() missing session_name parameter")
            return False

        # Check add_chunks has collection_name as optional
        sig = inspect.signature(QdrantManager.add_chunks)
        params = sig.parameters
        if 'collection_name' in params and params['collection_name'].default is not inspect.Parameter.empty:
            print("  ✓ add_chunks() has optional collection_name")
        else:
            print("  ✗ add_chunks() collection_name not optional")
            return False

        return True
    except Exception as e:
        print(f"  ✗ Backward compatibility test failed: {e}")
        return False


def test_config():
    """Test configuration constants."""
    print("\nTesting configuration...")
    try:
        from app.config import DEFAULT_COLLECTION, COLLECTION_NAME_PATTERN
        print(f"  ✓ DEFAULT_COLLECTION = {DEFAULT_COLLECTION}")
        print(f"  ✓ COLLECTION_NAME_PATTERN = {COLLECTION_NAME_PATTERN}")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Collection-Based Migration Verification")
    print("=" * 60)

    tests = [
        test_imports,
        test_method_signatures,
        test_backward_compatibility,
        test_config,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if all(results):
        print("\n✅ All verification tests passed!")
        print("\nNext steps:")
        print("1. Start Qdrant: docker-compose up -d qdrant")
        print("2. Run application: python3 main.py")
        print("3. Test UI features:")
        print("   - Check legacy sessions appear with '*'")
        print("   - Create new collection")
        print("   - Upload documents to collection")
        print("   - Query multiple collections")
        return 0
    else:
        print("\n❌ Some verification tests failed!")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
