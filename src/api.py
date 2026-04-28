import os
import re
import unicodedata
from io import BytesIO
from typing import Callable, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from PIL import Image, UnidentifiedImageError

from src.ocr_product import OcrProduct
from src.ocr_table import OcrTable


OcrMode = Literal["product", "table"]


class OcrUrlRequest(BaseModel):
    image_url: HttpUrl
    language: str = "eng"
    spell_corrector: bool = False
    show_performance: bool = False


class OcrResponse(BaseModel):
    mode: OcrMode
    text: str
    execution_time: float
    language: str
    spell_corrector: bool


class AnalyzeUrlRequest(BaseModel):
    image_url: HttpUrl
    language: str = "por"
    spell_corrector: bool = False


class AnalyzeResponse(BaseModel):
    product_name: Optional[str]
    product_text: str
    nutrition_text: str
    nutrition_lines: list[str]
    execution_time: float
    language: str
    spell_corrector: bool


def create_app() -> FastAPI:
    app = FastAPI(
        title="OCR Food API",
        description="API local para extrair texto de imagens de produtos e tabelas nutricionais.",
        version="1.0.0",
    )

    cors_origins = [
        origin.strip()
        for origin in os.getenv("OCR_API_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.post("/ocr/{mode}", response_model=OcrResponse)
    async def run_ocr_from_form(
        mode: OcrMode,
        file: Optional[UploadFile] = File(default=None),
        image_url: Optional[str] = Form(default=None),
        language: str = Form(default="eng"),
        spell_corrector: bool = Form(default=False),
        show_performance: bool = Form(default=False),
    ):
        if file is None and not image_url:
            raise HTTPException(
                status_code=400,
                detail="Envie um arquivo no campo 'file' ou uma URL no campo 'image_url'.",
            )

        image_input = image_url if image_url else await _read_upload_as_image(file)
        return await run_in_threadpool(
            _run_ocr,
            mode,
            image_input,
            language,
            spell_corrector,
            show_performance,
        )

    @app.post("/ocr/{mode}/url", response_model=OcrResponse)
    def run_ocr_from_json(mode: OcrMode, payload: OcrUrlRequest):
        return _run_ocr(
            mode=mode,
            image_input=str(payload.image_url),
            language=payload.language,
            spell_corrector=payload.spell_corrector,
            show_performance=payload.show_performance,
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze_food_from_form(
        file: Optional[UploadFile] = File(default=None),
        image_url: Optional[str] = Form(default=None),
        language: str = Form(default="por"),
        spell_corrector: bool = Form(default=False),
    ):
        if file is None and not image_url:
            raise HTTPException(
                status_code=400,
                detail="Envie um arquivo no campo 'file' ou uma URL no campo 'image_url'.",
            )

        image_input = image_url if image_url else await _read_upload_as_image(file)
        return await run_in_threadpool(
            _analyze_food,
            image_input,
            language,
            spell_corrector,
        )

    @app.post("/analyze/url", response_model=AnalyzeResponse)
    def analyze_food_from_json(payload: AnalyzeUrlRequest):
        return _analyze_food(
            image_input=str(payload.image_url),
            language=payload.language,
            spell_corrector=payload.spell_corrector,
        )

    return app


async def _read_upload_as_image(file: Optional[UploadFile]) -> Image.Image:
    if file is None:
        raise HTTPException(status_code=400, detail="Arquivo nao enviado.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        image = Image.open(BytesIO(contents))
        image.load()
        return image
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="Arquivo enviado nao e uma imagem valida.",
        ) from error


def _run_ocr(
    mode: OcrMode,
    image_input,
    language: str,
    spell_corrector: bool,
    show_performance: bool,
) -> OcrResponse:
    ocr_class = _get_ocr_class(mode)

    try:
        result = ocr_class(
            image_input,
            language=language,
            spell_corrector=spell_corrector,
            show_performace=show_performance,
        )
    except (ConnectionError, TypeError, NotImplementedError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Falha ao executar OCR: {error}",
        ) from error

    return OcrResponse(
        mode=mode,
        text=result.text,
        execution_time=result.execution_time,
        language=language,
        spell_corrector=spell_corrector,
    )


def _analyze_food(image_input, language: str, spell_corrector: bool) -> AnalyzeResponse:
    product_input = _copy_image_if_needed(image_input)
    table_input = _copy_image_if_needed(image_input)

    product_result = _run_ocr(
        "product",
        product_input,
        language,
        spell_corrector,
        False,
    )
    table_result = _run_ocr(
        "table",
        table_input,
        language,
        spell_corrector,
        False,
    )

    return AnalyzeResponse(
        product_name=_guess_product_name(product_result.text),
        product_text=product_result.text,
        nutrition_text=table_result.text,
        nutrition_lines=_extract_nutrition_lines(table_result.text),
        execution_time=product_result.execution_time + table_result.execution_time,
        language=language,
        spell_corrector=spell_corrector,
    )


def _copy_image_if_needed(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.copy()
    return image_input


def _guess_product_name(text: str) -> Optional[str]:
    ignored_patterns = [
        r"informacao nutricional",
        r"informacoes nutricionais",
        r"tabela nutricional",
        r"ingredientes",
        r"validade",
        r"lote",
        r"fabricado",
        r"contem",
        r"nao contem",
    ]

    for line in _clean_lines(text):
        normalized = _normalize_text(line)
        if len(line) < 3:
            continue
        if any(re.search(pattern, normalized) for pattern in ignored_patterns):
            continue
        if re.search(r"\d+\s*(g|kg|ml|l)\b", normalized):
            continue
        return line

    return None


def _extract_nutrition_lines(text: str) -> list[str]:
    nutrient_patterns = [
        r"valor energetico",
        r"calorias",
        r"carboidratos?",
        r"proteinas?",
        r"gorduras? totais?",
        r"gorduras? saturadas?",
        r"gorduras? trans",
        r"fibra alimentar",
        r"sodio",
        r"acucares?",
        r"vitaminas?",
        r"calcio",
        r"ferro",
    ]

    lines = []
    for line in _clean_lines(text):
        normalized = _normalize_text(line)
        has_nutrient = any(re.search(pattern, normalized) for pattern in nutrient_patterns)
        has_amount = re.search(r"\d+[\d,.]*\s*(kcal|kj|g|mg|mcg|%)\b", normalized)
        if has_nutrient or has_amount:
            lines.append(line)

    return lines


def _clean_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if re.sub(r"\s+", " ", line).strip()
    ]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _get_ocr_class(mode: OcrMode) -> Callable:
    if mode == "product":
        return OcrProduct
    if mode == "table":
        return OcrTable
    raise HTTPException(status_code=404, detail="Modo de OCR nao encontrado.")


app = create_app()
