from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from speaches.cjk_post_processor import dict_manager

router = APIRouter(prefix="/v1/domain-dict", tags=["domain-dictionary"])


class LoadDictRequest(BaseModel):
    url: str
    name: str | None = None


class LoadDictResponse(BaseModel):
    status: str
    name: str
    source: str
    entries_loaded: int
    entries_total: int
    duplicates_skipped: int
    load_duration_ms: int
    error: str | None = None


class DictSourceResponse(BaseModel):
    name: str
    path: str
    entries: int
    loaded_at: str
    type: str
    source_url: str | None = None


class SourcesResponse(BaseModel):
    sources: list[DictSourceResponse]
    total_entries: int


class UnloadResponse(BaseModel):
    status: str
    name: str | None = None
    entries_removed: int | None = None
    error: str | None = None


@router.get("")
async def get_domain_dict() -> str:
    """Return the merged content of all loaded domain dictionaries.

    The response is plain text (one word per line, jieba format).
    Clients can cache this and use it for local/fallback segmentation.
    """
    content = dict_manager.export_merged()
    return content


@router.post("/load")
async def load_domain_dict(req: LoadDictRequest) -> LoadDictResponse:
    """Download an external dictionary file and load it into jieba."""
    result = dict_manager.load_url(req.url, name=req.name)
    if result.status == "error":
        raise HTTPException(status_code=400, detail=result.error)
    return LoadDictResponse(
        status=result.status,
        name=result.name,
        source=result.source,
        entries_loaded=result.entries_loaded,
        entries_total=result.entries_total,
        duplicates_skipped=result.duplicates_skipped,
        load_duration_ms=result.load_duration_ms,
        error=result.error,
    )


@router.get("/sources")
async def list_sources() -> SourcesResponse:
    """List all currently loaded dictionary sources."""
    sources = dict_manager.get_sources()
    return SourcesResponse(
        sources=[
            DictSourceResponse(
                name=s.name,
                path=s.path,
                entries=s.entries,
                loaded_at=s.loaded_at,
                type=s.type,
                source_url=s.source_url,
            )
            for s in sources
        ],
        total_entries=dict_manager.get_total_entries(),
    )


@router.delete("/sources/{name}")
async def unload_source(name: str) -> UnloadResponse:
    """Unload a previously loaded external dictionary source."""
    if name == "_builtin":
        raise HTTPException(status_code=400, detail="cannot unload builtin dictionary")

    result = dict_manager.unload(name)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result.get("error", "unknown error"))
    return UnloadResponse(
        status=result["status"],
        name=result.get("name"),
        entries_removed=result.get("entries_removed"),
    )
