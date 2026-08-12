"""Maruziyet agirlikli risk hesabi.

Iki fikir uzerine kurulu:

1) Ayni bulgu her sunucuda ayni sey degildir.
   Internete acik, hassas veri tutan bir uretim veritabanindaki acik SSH root
   girisi ile izole bir test makinesindeki ayni bulgu ayni riski tasimaz.
   Bulgu sayisi saymak bu farki gormez; maruziyet carpani gorur.

2) Belirsiz sonuc sifir risk degildir.
   'notchecked' bir kontrolun gectigi anlamina gelmez. Ne oldugunu bilmiyoruz.
   Uydurma bir sabit atamak yerine, o kontrolun FILO GENELINDE gozlemlenen
   uyumsuzluk oranini kullaniyoruz: bir kontrol gozlemlendigi yerlerde %40
   kaliyorsa, gozlemlenmedigi yerde beklenen risk tam bulgunun %40'idir.
   Bu bir tahmindir ve rapor edilirken ayri kalem olarak gosterilir.
"""

from __future__ import annotations

import pandas as pd

from .fleet import birlesik

# Maruziyet carpanlari. Carpimsal: etkiler birikir.
INTERNET_CARPANI = 2.2
DESTEGI_BITMIS_CARPANI = 1.4

BOLGE_CARPANI = {"dmz": 1.6, "ic_ag": 1.0, "kisitli": 0.8}
ORTAM_CARPANI = {"uretim": 1.5, "felaket_kurtarma": 1.1, "test": 0.5}
VERI_CARPANI = {"hassas": 1.6, "dahili": 1.0, "genel": 0.8}


def maruziyet_carpani(sunucu: pd.Series | dict) -> float:
    """Bir sunucunun maruziyet carpanini dondurur.

    Taban 1.0. Tum etkenler carpimsal uygulanir; en maruz sunucu en korunakli
    sunucudan yaklasik 10 kat agir sayilir.
    """
    c = 1.0
    c *= ORTAM_CARPANI.get(sunucu["ortam"], 1.0)
    c *= BOLGE_CARPANI.get(sunucu["ag_bolgesi"], 1.0)
    c *= VERI_CARPANI.get(sunucu["veri_siniflandirmasi"], 1.0)
    if sunucu["internet_erisimi"]:
        c *= INTERNET_CARPANI
    if sunucu["destek_durumu"] == "destegi_bitti":
        c *= DESTEGI_BITMIS_CARPANI
    return round(c, 3)


def kontrol_uyumsuzluk_orani(df: pd.DataFrame | None = None) -> pd.Series:
    """Her kontrol icin filo genelinde GOZLEMLENEN uyumsuzluk orani.

    Belirsiz sonuclarin beklenen riskini hesaplarken kullanilir. Yalnizca
    gozlemlenen satirlar paydaya girer - bilinmeyeni bilinmeyenle tahmin etmeyiz.
    """
    if df is None:
        df = birlesik()
    gozlem = df[df["durum"].isin(("uyumlu", "uyumsuz"))]
    if gozlem.empty:
        return pd.Series(dtype=float)
    return gozlem.groupby("kontrol_id")["durum"].apply(
        lambda s: (s == "uyumsuz").mean()
    )


def risk_tablosu(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Her denetim satirina maruziyet carpani ve risk puani ekler.

    Iki ayri risk kalemi uretilir:
      kesin_risk     - gercekten 'fail' donen kontrollerden gelen risk
      belirsiz_risk  - durumu okunamayan kontrollerden gelen BEKLENEN risk
    """
    if df is None:
        df = birlesik()
    df = df.copy()

    df["maruziyet"] = df.apply(maruziyet_carpani, axis=1)

    oran = kontrol_uyumsuzluk_orani(df)
    df["filo_uyumsuzluk_orani"] = df["kontrol_id"].map(oran).fillna(0.0)

    tam_risk = df["agirlik"] * df["maruziyet"]

    df["kesin_risk"] = (df["durum"] == "uyumsuz") * tam_risk
    df["belirsiz_risk"] = (df["durum"] == "belirsiz") * tam_risk * df["filo_uyumsuzluk_orani"]
    df["toplam_risk"] = df["kesin_risk"] + df["belirsiz_risk"]

    return df


def sunucu_riski(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sunucu bazinda risk ozeti, en riskliden baslayarak."""
    r = risk_tablosu(df)

    ozet = r.groupby("host_id").agg(
        ortam=("ortam", "first"),
        rol=("rol", "first"),
        ag_bolgesi=("ag_bolgesi", "first"),
        internet_erisimi=("internet_erisimi", "first"),
        veri_siniflandirmasi=("veri_siniflandirmasi", "first"),
        destek_durumu=("destek_durumu", "first"),
        son_denetim_gun_once=("son_denetim_gun_once", "first"),
        maruziyet=("maruziyet", "first"),
        bulgu_sayisi=("durum", lambda s: int((s == "uyumsuz").sum())),
        belirsiz_sayisi=("durum", lambda s: int((s == "belirsiz").sum())),
        kesin_risk=("kesin_risk", "sum"),
        belirsiz_risk=("belirsiz_risk", "sum"),
        toplam_risk=("toplam_risk", "sum"),
    )

    for kolon in ("kesin_risk", "belirsiz_risk", "toplam_risk"):
        ozet[kolon] = ozet[kolon].round(1)

    return ozet.sort_values("toplam_risk", ascending=False).reset_index()


def bulgu_siralamasi(df: pd.DataFrame | None = None, limit: int = 15) -> pd.DataFrame:
    """Tekil bulgulari maruziyete gore siralar.

    Ciktinin ilk satiri "once bunu duzelt" demektir; bulgu sayisi saymanin
    veremedigi cevap budur.
    """
    r = risk_tablosu(df)
    bulgular = r[r["durum"] == "uyumsuz"].nlargest(limit, "kesin_risk")

    return bulgular[
        [
            "host_id", "kontrol_id", "baslik", "kategori", "seviye",
            "bddk_maddesi", "agirlik", "ortam", "ag_bolgesi",
            "internet_erisimi", "veri_siniflandirmasi", "maruziyet", "kesin_risk",
        ]
    ].round(1).reset_index(drop=True)
