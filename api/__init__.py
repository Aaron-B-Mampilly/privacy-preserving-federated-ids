"""FastAPI presentation layer for FedPDA-IDS.

This package is a READ-MOSTLY wrapper around the existing research
modules in `src/fedpda_ids/` -- it imports them, it never reimplements
them, and it never changes a completed experiment's numbers. The one
state-changing operation it exposes (drift-triggered retraining) always
runs under a sandboxed run name so an official checkpoint can never be
overwritten from the UI.
"""
