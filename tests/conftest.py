"""Test execution is offline; dependency installation is a separate CI step."""

import socket

import pytest


@pytest.fixture(autouse=True)
def forbid_network_during_tests(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Network use is forbidden in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
