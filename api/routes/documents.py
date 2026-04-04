# api/routes/documents.py
import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from api.deps import get_redis, require_api_key
from core.ingestion.pdf_loader import load_pdf_from_bytes
from core.ingestion.summarizer import summarize_document
from core.ingestion.doc_classifier import normalise_doc_type
from core.indexing.pageindex_builder import build_case_pageindex
from db.redis_client import RedisClient
from db.mongo import mongo
from db.postgres import AsyncSessionLocal, Document

router = APIRouter()


@router.post("/{case_id}/documents", status_code=201)
async def upload_document(
    case_id: str,
    file: UploadFile = File(...),
    redis: RedisClient = Depends(get_redis),
    _: str = Depends(require_api_key),
):
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
    raw_bytes = await file.read()

    # Load pages
    pages = load_pdf_from_bytes(raw_bytes)

    # Phase 1: no OCR — text extraction happens via LLM on image in Phase 2
    for page in pages:
        page["text"] = ""

    # Summarise document using filename as fallback
    combined_text = " ".join(p.get("text", "") for p in pages[:3])
    summary = await summarize_document(combined_text or file.filename)
    doc_type = normalise_doc_type(summary.get("doc_type", "other"))

    # Build + cache PageIndex
    await redis.invalidate_case_pageindex(case_id)
    tree = await build_case_pageindex(case_id, doc_id, doc_type, pages)
    await mongo.case_indexes.replace_one(
        {"case_id": case_id, "doc_id": doc_id}, tree, upsert=True
    )
    await redis.set_case_pageindex(case_id, tree)

    # Store document metadata in Postgres
    async with AsyncSessionLocal() as db:
        db.add(Document(
            id=doc_id,
            case_id=case_id,
            doc_type=doc_type,
            filename=file.filename,
            page_count=len(pages),
            storage_path=f"{case_id}/{doc_id}/{file.filename}",
        ))
        await db.commit()

    return {"doc_id": doc_id, "doc_type": doc_type, "pages": len(pages)}
