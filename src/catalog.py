"""Kontrol katalogu uzerinde anlamsal arama (Chroma).

Kullanici "SSH'a parola deneyerek girmeyi zorlastiran ayar" diye soruyor;
katalogda ise kontrolun adi "5.2.5 Ensure SSH MaxAuthTries is set to 3 or less".
61 kontrolun kimligini prompt'a gomup modelin dogru eslemeyi yapmasini ummak
yerine katalogu vektorleyip semantik ariyoruz.

Gercek bir CIS Benchmark'ta yuzlerce kontrol vardir; prompt'a gomme yaklasimi
o olcekte calismaz, bu calisir.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import chromadb
from langchain_core.tools import tool

from .config import CHROMA_PATH, get_embeddings
from .controls import BDDK_BASLIK, KONTROLLER

KOLEKSIYON_ADI = "kontrol_katalogu"


class GoogleEmbeddingFunction:
    """Chroma'nin bekledigi embedding arayuzunu LangChain modeline baglar."""

    def __init__(self) -> None:
        self._model = get_embeddings()

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma imzasi
        return self._model.embed_documents(list(input))

    def name(self) -> str:
        return "google-gemini-embedding"


@lru_cache(maxsize=1)
def _koleksiyon():
    istemci = chromadb.PersistentClient(path=CHROMA_PATH)
    return istemci.get_or_create_collection(
        name=KOLEKSIYON_ADI,
        embedding_function=GoogleEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def katalogu_kur(force: bool = False) -> int:
    """Kontrol katalogunu vektor deposuna yazar."""
    koleksiyon = _koleksiyon()
    if koleksiyon.count() > 0 and not force:
        return koleksiyon.count()

    koleksiyon.upsert(
        ids=[k.kontrol_id for k in KONTROLLER],
        documents=[
            f"{k.kontrol_id} {k.baslik}. {k.aciklama} "
            f"Kategori: {k.kategori}. CIS seviye {k.seviye}. "
            f"BDDK {k.bddk_etiketi}."
            for k in KONTROLLER
        ],
        metadatas=[
            {
                "kontrol_id": k.kontrol_id,
                "baslik": k.baslik,
                "kategori": k.kategori,
                "seviye": k.seviye,
                "bddk_maddesi": k.bddk.value,
                "agirlik": k.agirlik,
                "kaynak": k.kaynak,
            }
            for k in KONTROLLER
        ],
    )
    return koleksiyon.count()


@tool
def kontrol_ara(sorgu: str, top_k: int = 5) -> str:
    """Dogal dilde tarif edilen bir guvenlik konusuna karsilik gelen kontrolleri bulur.

    Kontrol kimligini bilmiyorsan once bunu kullan. Ornek: "parola deneme siniri"
    sorgusu SSH MaxAuthTries kontrolunu, "gunlukler uzak sunucuya gidiyor mu"
    sorgusu loghost kontrolunu dondurur. Kontrol kimligi UYDURMA.

    Args:
        sorgu: Aradigin konunun dogal dildeki tarifi.
        top_k: Dondurulecek kontrol sayisi.
    """
    katalogu_kur()
    sonuc = _koleksiyon().query(query_texts=[sorgu], n_results=max(1, min(top_k, 10)))

    bulunanlar: list[dict[str, Any]] = []
    for belge, meta, uzaklik in zip(
        sonuc["documents"][0], sonuc["metadatas"][0], sonuc["distances"][0]
    ):
        bulunanlar.append(
            {
                "kontrol_id": meta["kontrol_id"],
                "baslik": meta["baslik"],
                "kategori": meta["kategori"],
                "cis_seviyesi": meta["seviye"],
                "bddk_maddesi": meta["bddk_maddesi"],
                "bddk_baslik": BDDK_BASLIK.get(meta["bddk_maddesi"], ""),
                "risk_agirligi": meta["agirlik"],
                "kaynak": meta["kaynak"],
                "aciklama": belge,
                "benzerlik": round(1 - float(uzaklik), 4),
            }
        )

    return json.dumps({"sorgu": sorgu, "bulunan_kontroller": bulunanlar}, ensure_ascii=False)


KATALOG_ARACLARI = [kontrol_ara]
