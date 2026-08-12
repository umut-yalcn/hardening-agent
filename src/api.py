"""FastAPI servisi.

    uvicorn src.api:app --reload
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import sor
from .controls import BDDK_BASLIK, KONTROLLER
from .freshness import filo_kapsam_ozeti
from .tools import filo_ozeti

app = FastAPI(
    title="Sertlestirme Analiz Ajani",
    description=(
        "Sunucu filosunun sertlestirme durumunu dogal dilde sorgulanabilir kilan "
        "agentic analiz servisi. Bulgular BDDK Bankalarin Bilgi Sistemleri "
        "Yonetmeligi maddelerine baglanir."
    ),
    version="0.1.0",
)


class SoruIstegi(BaseModel):
    soru: str = Field(..., min_length=3, examples=["Hangi BDDK maddesinde en cok sapma var?"])
    dogrula: bool = Field(True, description="Cevabi ayri bir model cagrisiyla denetle.")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"durum": "ayakta", "kontrol_sayisi": len(KONTROLLER)}


@app.get("/kontroller")
def kontroller() -> dict[str, Any]:
    """Kontrol katalogunu ve BDDK eslemesini dondurur."""
    return {
        "kontrol_sayisi": len(KONTROLLER),
        "bddk_maddeleri": {m.value: b for m, b in BDDK_BASLIK.items()},
        "kontroller": [
            {
                "kontrol_id": k.kontrol_id,
                "baslik": k.baslik,
                "kategori": k.kategori,
                "cis_seviyesi": k.seviye,
                "bddk_maddesi": k.bddk.value,
                "risk_agirligi": k.agirlik,
                "kaynak": k.kaynak,
            }
            for k in KONTROLLER
        ],
    }


@app.get("/filo")
def filo() -> dict[str, Any]:
    """Filo ozetini dondurur. Model cagrisi yapmaz."""
    return filo_ozeti.invoke({})


@app.get("/kapsam")
def kapsam() -> dict[str, Any]:
    """Denetim verisinin kapsam ve tazelik ozeti."""
    return filo_kapsam_ozeti()


@app.post("/sor")
def sor_endpoint(istek: SoruIstegi) -> dict[str, Any]:
    """Dogal dilde soru sorar; cevabi kullanilan araclar ve dogrulama ile dondurur."""
    try:
        return sor(istek.soru, dogrula=istek.dogrula)
    except Exception as hata:
        raise HTTPException(status_code=500, detail=f"{type(hata).__name__}: {hata}") from hata
