"""Core cross-cutting utilities: settings and shared API clients.

This package sits below every other project package and must not import from
api/, retrieval/, generation/, ingest/, or evals/ — it is the dependency root.
"""
