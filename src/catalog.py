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
import threading
import time
from functools import lru_cache
from typing import Any

import chromadb
from langchain_core.tools import tool

from .config import CHROMA_PATH, get_embeddings
from .controls import BDDK_BASLIK, KONTROLLER

KOLEKSIYON_ADI = "kontrol_katalogu"


#: Embedding cagrisi ag kaynakli gecici hatalarla dusebiliyor (gozlemlenen:
#: "SSL: INVALID_SESSION_ID"). Tek bir gecici hata tum agent kosumunu
#: dusurmemeli - ozellikle canli demoda.
EMBEDDING_DENEME = 4
EMBEDDING_BEKLEME = 2.0


class GoogleEmbeddingFunction:
    """Chroma'nin bekledigi embedding arayuzunu LangChain modeline baglar.

    Chroma (>=1.x) belge ve sorgu icin AYRI cagri yapar: belgeler `__call__`
    uzerinden, sorgular `embed_query` uzerinden gider. Yalnizca `__call__`
    tanimlanirsa arama calisma aninda AttributeError ile duser. Ayrim ayrica
    dogru: embedding modeli sorgu ile belgeyi farkli gorevler olarak kodlar.
    """

    def __init__(self) -> None:
        self._model = get_embeddings()

    @staticmethod
    def _metinler(input: Any) -> list[str]:  # noqa: A002 - Chroma imzasi
        if isinstance(input, str):
            return [input]
        return [str(t) for t in input]

    def _dene(self, fn):
        """Gecici ag hatalarina karsi yeniden dener."""
        son_hata: Exception | None = None
        for deneme in range(EMBEDDING_DENEME):
            try:
                return fn()
            except Exception as hata:  # saglayici hatalarini tek tek ayirmiyoruz
                son_hata = hata
                if deneme < EMBEDDING_DENEME - 1:
                    # Ustel geri cekilme: gozlemlenen SSL kesintileri
                    # ~30 saniyede toparliyor, dogrusal bekleme yetmiyor.
                    time.sleep(EMBEDDING_BEKLEME * (2 ** deneme))
        raise RuntimeError(
            f"Embedding {EMBEDDING_DENEME} denemede alinamadi: {son_hata}"
        ) from son_hata

    def __call__(self, input: Any) -> list[list[float]]:  # noqa: A002 - Chroma imzasi
        metinler = self._metinler(input)
        return self._dene(lambda: self._model.embed_documents(metinler))

    def embed_query(self, input: Any) -> list[list[float]]:  # noqa: A002 - Chroma imzasi
        metinler = self._metinler(input)
        return self._dene(lambda: [self._model.embed_query(t) for t in metinler])

    def name(self) -> str:
        return "google-gemini-embedding"


#: chromadb.PersistentClient, sinif duzeyinde paylasilan bir sozluk kullaniyor
#: ve bu sozluk kilitsiz. LangGraph araclari thread havuzunda kosturdugu icin
#: iki thread ayni anda ilk istemciyi olusturmaya kalkinca yaris olusuyor ve
#: KeyError firliyor. lru_cache bunu engellemiyor: onbellek dolmadan once
#: fonksiyon iki kez cagrilabilir. Olusturmayi kilitliyoruz.
_istemci_kilidi = threading.Lock()


@lru_cache(maxsize=1)
def _koleksiyon():
    with _istemci_kilidi:
        istemci = chromadb.PersistentClient(path=CHROMA_PATH)
        return istemci.get_or_create_collection(
            name=KOLEKSIYON_ADI,
            embedding_function=GoogleEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )


def katalogu_kur(force: bool = False) -> int:
    """Kontrol katalogunu vektor deposuna yazar."""
    koleksiyon = _koleksiyon()

    # Onceden "tek kayit bile varsa katalog tamamdir" varsayiliyordu. Kesilmis
    # bir ilk kurulum ya da degismis controls.py, aramaya hic yansimiyordu -
    # katalog "tek gercek kaynak" olsa da Chroma ondan kalici olarak
    # sapabiliyordu. Artik BEKLENEN kontrol sayisiyla karsilastiriliyor.
    if koleksiyon.count() == len(KONTROLLER) and not force:
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
