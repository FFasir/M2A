from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, SiglipModel


TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
MULTIMODAL_MODEL_NAME = "siglip2-base-patch16-384"
MULTIMODAL_CHECKPOINT = "google/siglip2-base-patch16-384"


class EmbeddingRequest(BaseModel):
    model: str
    input: Any | None = None
    messages: list[dict] | None = None
    encoding_format: str | None = None


@dataclass
class ModelBundle:
    text_model: SentenceTransformer
    multimodal_model: SiglipModel
    multimodal_processor: AutoProcessor


def _load_image_from_data_url(url: str) -> Image.Image:
    if "," not in url:
        raise ValueError("Unsupported image URL format")
    _, data = url.split(",", 1)
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_embedding_input(request: EmbeddingRequest) -> list[dict]:
    if request.messages:
        first = request.messages[0]
        content = first.get("content", [])
        if not isinstance(content, list):
            raise ValueError("messages[0].content must be a list")
        return content

    if request.input is None:
        raise ValueError("Missing input")

    if isinstance(request.input, list):
        return [{"type": "text", "text": str(item)} for item in request.input]

    return [{"type": "text", "text": str(request.input)}]


def create_app() -> FastAPI:
    app = FastAPI(title="Local M2A Embedding Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = ModelBundle(
        text_model=SentenceTransformer(TEXT_MODEL_NAME, device=device),
        multimodal_model=SiglipModel.from_pretrained(MULTIMODAL_CHECKPOINT).to(device),
        multimodal_processor=AutoProcessor.from_pretrained(MULTIMODAL_CHECKPOINT),
    )
    bundle.multimodal_model.eval()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": TEXT_MODEL_NAME, "object": "model"},
                {"id": MULTIMODAL_MODEL_NAME, "object": "model"},
            ],
        }

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        try:
            content = _extract_embedding_input(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        model_name = request.model

        if model_name == TEXT_MODEL_NAME:
            texts: list[str] = []
            for item in content:
                if item.get("type") != "text":
                    raise HTTPException(status_code=400, detail="Text model only accepts text input")
                texts.append(item.get("text", ""))

            vectors = bundle.text_model.encode(texts, normalize_embeddings=True).tolist()
            if isinstance(vectors[0], float):
                vectors = [vectors]

            return {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": vector, "index": index}
                    for index, vector in enumerate(vectors)
                ],
                "model": model_name,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        if model_name == MULTIMODAL_MODEL_NAME:
            text_chunks: list[str] = []
            image: Image.Image | None = None

            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    text_chunks.append(item.get("text", ""))
                elif item_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    image = _load_image_from_data_url(image_url)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported content type: {item_type}")

            if image is None and not text_chunks:
                raise HTTPException(status_code=400, detail="Multimodal model needs text or image input")

            with torch.no_grad():
                if image is not None and text_chunks:
                    # Cross-modal text query uses text features in the shared SigLIP space.
                    inputs = bundle.multimodal_processor(
                        text=text_chunks[0],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    vector = bundle.multimodal_model.get_text_features(**inputs)[0]
                elif image is not None:
                    inputs = bundle.multimodal_processor(
                        images=image,
                        return_tensors="pt",
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    vector = bundle.multimodal_model.get_image_features(**inputs)[0]
                else:
                    inputs = bundle.multimodal_processor(
                        text=text_chunks[0],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    vector = bundle.multimodal_model.get_text_features(**inputs)[0]

                vector = torch.nn.functional.normalize(vector, p=2, dim=-1).cpu().tolist()

            return {
                "object": "list",
                "data": [{"object": "embedding", "embedding": vector, "index": 0}],
                "model": model_name,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
