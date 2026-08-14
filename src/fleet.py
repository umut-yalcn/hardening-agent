"""Filo verisine erisim katmani.

Denetim sonuclari ile kontrol katalogunu birlestirir. Ustteki her katman
(skorlama, tazelik, ajan araclari) veriyi buradan alir.
"""

from __future__ import annotations

import os
import pathlib
from functools import lru_cache

import pandas as pd

from .controls import BELIRSIZ, KAPSAM_DISI, KESIN_UYUMLU, KESIN_UYUMSUZ, KONTROLLER

KOK = pathlib.Path(__file__).resolve().parents[1]
SUNUCU_YOLU = pathlib.Path(os.getenv("SUNUCU_YOLU", KOK / "data" / "sunucular.csv"))
SONUC_YOLU = pathlib.Path(os.getenv("SONUC_YOLU", KOK / "data" / "denetim_sonuclari.csv"))

_BELIRSIZ = {s.value for s in BELIRSIZ}
_UYUMLU = {s.value for s in KESIN_UYUMLU}
_UYUMSUZ = {s.value for s in KESIN_UYUMSUZ}
_KAPSAM_DISI = {s.value for s in KAPSAM_DISI}


class VeriYok(RuntimeError):
    pass


@lru_cache(maxsize=1)
def sunucular() -> pd.DataFrame:
    """Sunucu envanterini dondurur."""
    if not SUNUCU_YOLU.exists():
        raise VeriYok(
            f"Sunucu envanteri bulunamadi: {SUNUCU_YOLU}\n"
            "Once uret: python scripts/generate_fleet.py"
        )
    # _bool_cevir TEMBEL ice aktariliyor: scoring zaten fleet'i ice aktardigi
    # icin modul duzeyinde import dongu olustururdu.
    from .scoring import _bool_cevir

    df = pd.read_csv(SUNUCU_YOLU)

    # Mukerrer host_id merge'te satirlari COGALTIR: ayni sunucunun bulgulari
    # iki kez sayilir, riski ikiye katlanir, filo geneli siser. Sessizce
    # yanlis sayi uretmektense yuklemeyi durduruyoruz.
    if not df["host_id"].is_unique:
        tekrar = sorted(df.loc[df["host_id"].duplicated(), "host_id"].unique())
        raise VeriYok(f"Envanterde mukerrer host_id var: {tekrar}")

    # internet_erisimi TEK NOKTADA normallestiriliyor. CSV bu kolonu "True",
    # "yes", "1" gibi string tasiyabiliyor ve bool("False") True'dur; arac
    # katmani kolonu ham maskeleme/.sum() ile kullandigi icin kaynakta
    # cozmezsek her tuketici ayri sekilde yanilir.
    if "internet_erisimi" in df.columns:
        df["internet_erisimi"] = df["internet_erisimi"].map(_bool_cevir)
    return df


@lru_cache(maxsize=1)
def sonuclar() -> pd.DataFrame:
    """Denetim sonuclarini kontrol katalogu ile zenginlestirilmis halde dondurur."""
    if not SONUC_YOLU.exists():
        raise VeriYok(
            f"Denetim sonuclari bulunamadi: {SONUC_YOLU}\n"
            "Once uret: python scripts/generate_fleet.py"
        )
    df = pd.read_csv(SONUC_YOLU)

    katalog = pd.DataFrame(
        [
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
        ]
    )

    df = df.merge(katalog, on="kontrol_id", how="left")
    if df["baslik"].isna().any():
        eksik = sorted(df.loc[df["baslik"].isna(), "kontrol_id"].unique())
        raise VeriYok(f"Katalogda olmayan kontrol kimlikleri var: {eksik}")

    # Sonuc durumunu uc anlamli sinifa indirger. Bu ayrim projenin merkezinde:
    # 'belirsiz' hicbir yerde 'uyumlu' sayilmaz.
    df["durum"] = df["sonuc"].map(
        lambda s: "uyumlu" if s in _UYUMLU
        else "uyumsuz" if s in _UYUMSUZ
        else "kapsam_disi" if s in _KAPSAM_DISI
        else "belirsiz"
    )
    return df


@lru_cache(maxsize=1)
def birlesik() -> pd.DataFrame:
    """Denetim sonuclarini sunucu ozellikleriyle birlestirir."""
    df = sonuclar().merge(sunucular(), on="host_id", how="left")

    # Envanterde olmayan host_id, left merge'te tum ozellikleri NaN birakir ve
    # maruziyet carpani 1.0'a (notr) duser: AYNI bulgular 8 kat DUSUK risk
    # gosterir, sunucu siralamanin dibine iner, "internete acik" filtresine
    # takilmaz. Gercek hayatta en olasi veri hatasi budur - CMDB envanteri
    # geride kalir, tarayici yeni makineyi bulur - ve riski oldugundan az
    # gosterdigi icin sessiz kalmamali. Katalogda olmayan kontrol_id zaten
    # hata firlatiyordu; simetrigi eksikti.
    if df["ortam"].isna().any():
        eksik = sorted(df.loc[df["ortam"].isna(), "host_id"].unique())
        raise VeriYok(
            f"Envanterde karsiligi olmayan host_id var: {eksik}. "
            "Bu sunucularin maruziyeti hesaplanamaz; riskleri oldugundan "
            "dusuk gorunurdu."
        )
    return df


def durum_dagilimi(df: pd.DataFrame) -> dict[str, int]:
    """Uyumlu / uyumsuz / belirsiz / kapsam disi sayilarini dondurur."""
    sayim = df["durum"].value_counts().to_dict()
    return {
        "uyumlu": int(sayim.get("uyumlu", 0)),
        "uyumsuz": int(sayim.get("uyumsuz", 0)),
        "belirsiz": int(sayim.get("belirsiz", 0)),
        "kapsam_disi": int(sayim.get("kapsam_disi", 0)),
    }


def uyum_ve_kapsam(df: pd.DataFrame) -> dict[str, float | int | None]:
    """Uyum orani ile kapsam oranini BIRLIKTE dondurur.

    Bu ikisi ayri ayri yanilticidir. Uyum orani yalnizca GOZLEMLENEN sonuclar
    uzerinden hesaplanir; kapsam orani ise ne kadarini gercekten gozlemledigimizi
    soyler. Yuksek uyum + dusuk kapsam, iyi bir durum degil; bilgisiz bir durumdur.
    """
    d = durum_dagilimi(df)
    gozlemlenen = d["uyumlu"] + d["uyumsuz"]
    uygulanabilir = gozlemlenen + d["belirsiz"]

    return {
        "uygulanabilir_kontrol": uygulanabilir,
        "gozlemlenen_kontrol": gozlemlenen,
        "belirsiz_kontrol": d["belirsiz"],
        "uyumlu": d["uyumlu"],
        "uyumsuz": d["uyumsuz"],
        "uyum_orani": round(d["uyumlu"] / gozlemlenen, 4) if gozlemlenen else None,
        "kapsam_orani": round(gozlemlenen / uygulanabilir, 4) if uygulanabilir else None,
    }
