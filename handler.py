"""Repository-discovery entry point for Runpod's GitHub deployment wizard.

Queue-based Serverless endpoints are detected by scanning the repository's
default branch for ``runpod.serverless.start()``. The production image installs
the real handler at ``/handler.py`` from ``worker/handler.py`` (see the
Dockerfile), so this root shim exists purely so that pre-deploy checks and
repository scanners can identify the handler before any image has been built.

The imports live inside the ``__main__`` guard on purpose: a scanner or an
import of this module must not pull in the worker's heavy dependencies.
"""

from __future__ import annotations


if __name__ == "__main__":
    from worker.handler import handler

    import runpod

    runpod.serverless.start({"handler": handler})
