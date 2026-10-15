"""capserve runs only on asyncio (server.py pins loop="asyncio"), so anyio-marked tests run on asyncio only."""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
