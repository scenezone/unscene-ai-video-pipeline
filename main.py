import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()


def create_app(service_factory=None, store=None, auth_store=None):
    """Start the authenticated local job API and its SQLite-backed worker."""
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from schemas.request import AnalyzeRequest
    from services.auth_store import AuthStore
    from services.job_store import JobConflict, JobStore
    from services.job_worker import JobWorker

    store = store or JobStore(os.getenv("AI_JOB_DB", "data/jobs/jobs.sqlite3"))
    auth_store = auth_store or AuthStore(os.getenv("AI_AUTH_DB", "data/auth/auth.sqlite3"))

    if service_factory is None:
        def service_factory():
            from services.model_registry import ModelRegistry
            from services.pipeline_factory import PipelineFactory
            from services.job_service import JobService
            from services.downloader import Downloader

            registry = ModelRegistry()
            registry.load_models()
            service = JobService(PipelineFactory(registry), registry,
                                 output_dir=os.getenv("AI_OUTPUT_DIR", "outputs/jobs"))
            service.downloader = Downloader(store.path.parent / "assets")
            return service

    worker = JobWorker(store, service_factory)

    @asynccontextmanager
    async def lifespan(app):
        bootstrap = auth_store.ensure_bootstrap_client()
        if bootstrap:
            print("\nInitial API client created. Save this secret; it will not be shown again.")
            print(f"client_id: {bootstrap['client_id']}")
            print(f"client_secret: {bootstrap['client_secret']}")
            print(f"scopes: {' '.join(bootstrap['scopes'])}\n")
        worker.start()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.job_store = store
    app.state.job_worker = worker
    app.state.auth_store = auth_store

    basic_auth = HTTPBasic(auto_error=False)

    def require_scope(scope):
        def authenticate(credentials: HTTPBasicCredentials = Depends(basic_auth)):
            identity = (auth_store.authenticate_client(credentials.username, credentials.password)
                        if credentials else None)
            if identity is None:
                raise HTTPException(status_code=401, detail="Invalid client credentials",
                                    headers={"WWW-Authenticate": "Basic"})
            if scope not in identity["scopes"]:
                raise HTTPException(status_code=403, detail="Insufficient scope")
            return identity
        return authenticate

    @app.get("/healthy")
    def health():
        return {"status": "healthy"}

    @app.post("/v1/analyze/jobs")
    def analyze_job(payload: AnalyzeRequest, client=Depends(require_scope("jobs:write"))):
        try:
            store.submit(payload.model_dump(), owner_client_id=client["client_id"])
        except JobConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "accepted", "jobId": payload.jobId}

    @app.get("/v1/analyze/jobs/{job_id}")
    def job_status(job_id: str, client=Depends(require_scope("jobs:read"))):
        result = store.get(job_id, owner_client_id=client["client_id"])
        if result is None:
            raise HTTPException(status_code=404, detail="Unknown jobId")
        return result

    @app.post("/v1/analyze/jobs/{job_id}/callback")
    def retry_callback(job_id: str, client=Depends(require_scope("callbacks:retry"))):
        try:
            store.retry_callback(job_id, owner_client_id=client["client_id"])
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown jobId") from error
        except JobConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "accepted", "jobId": job_id}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host=os.getenv("AI_API_HOST", "0.0.0.0"),
                port=int(os.getenv("AI_API_PORT", "8000")))
else:
    app = create_app()
