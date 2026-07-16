"""Shared pytest configuration. The whole suite is deterministic — no test
makes a live yfinance/network call; data-layer tests either construct a
DataHandler and populate its internal bar dict directly, or monkeypatch
`yf.download` with a fixed synthetic response.
"""
