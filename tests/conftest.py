# tests/conftest.py
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_api: marks tests that need API keys")


@pytest.fixture
def mock_api_keys():
    return {"openai": "test-key", "anthropic": "test-key", "google": "test-key"}
