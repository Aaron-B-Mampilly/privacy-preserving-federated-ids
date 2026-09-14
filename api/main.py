"""FedPDA-IDS API.

Start with:
    uvicorn api.main:app --reload --port 8001

Nothing here trains a model, changes a completed experiment's numbers, or
touches the Prometheus/Grafana stack -- that remains the technical
observability layer and is linked to, not replaced.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import ALLOWED_ORIGINS
from api.routers import dashboard, drift, fl, health, predict, privacy, results

logger = logging.getLogger("fedpda_api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="FedPDA-IDS API",
        description=(
            "Presentation layer over the FedPDA-IDS research codebase. Read-mostly: it serves "
            "completed experiment results and runs real inference against existing checkpoints. "
            "The only state-changing endpoint (drift retraining) always writes to a sandboxed run name."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    for router in (
        health.router,
        dashboard.router,
        results.router,
        fl.router,
        privacy.router,
        drift.router,
        predict.router,
    ):
        app.include_router(router, prefix="/api")

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
        return JSONResponse(
            status_code=422,
            content={
                "error": "Invalid request",
                "detail": f"{field}: {first.get('msg', 'failed validation')}" if field else "Request failed validation",
                "hint": "Check the request shape against /api/docs.",
            },
        )

    @app.exception_handler(FileNotFoundError)
    async def missing_artifact_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Artifact not found",
                "detail": str(exc),
                "hint": "This experiment or checkpoint has not been generated for the selected scope.",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the real traceback server-side; never ship one to the client.
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal error",
                "detail": f"{type(exc).__name__} while handling this request.",
                "hint": "Check the API server logs for the full trace.",
            },
        )

    return app


app = create_app()
