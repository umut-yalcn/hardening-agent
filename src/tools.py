"""Ajanin kullanabilecegi analiz araclari.

Her arac, belirsiz sonuclari uyumlu saymamak zorundadir. Bu kural prompt'ta
degil, arac katmaninda uygulanir: uyum orani hesaplayan her fonksiyon paydaya
yalnizca gozlemlenen sonuclari alir ve kapsam oranini birlikte dondurur.
Ajan isterse bunu gormezden gelemez, cunku sayiyi kendisi hesaplamiyor.
"""

from __future__ import annotations

import math

import pandas as pd
from functools import wraps
from typing import Any

from langchain_core.tools import tool

from .controls import BDDK_BASLIK, BddkMaddesi, KONTROLLER, kontrol_getir
from .fleet import birlesik, sunucular, uyum_ve_kapsam
from .freshness import (
    TAZELIK_ESIGI_GUN,
    filo_kapsam_ozeti,
    sunucu_kapsami,
    yaniltici_temizler,
)
from .scoring import _bool_cevir, bulgu_siralamasi, maruziyet_carpani, sunucu_riski

LISTE_SINIRI = 40

BOYUTLAR = (
    "ortam", "rol", "ag_bolgesi", "isletim_sistemi",
    "destek_durumu", "veri_siniflandirmasi", "kategori", "bddk_maddesi", "seviye",
)


def _json_guvenli(deger: Any) -> Any:
    """NaN ve sonsuzlugu None'a cevirir.

    pandas, "deger yok"u NaN olarak tasir; DataFrame.to_dict("records") bunu
    oldugu gibi disari verir. Python'un json.dumps'i NaN'a izin verip literal
    `NaN` yazar - ama bu GECERLI JSON DEGILDIR. Sonuc: model saglayicisi
    istegin tamamini 400 ile reddediyor ve ajan komple duser.

    Gozlenen hata:
        Invalid JSON payload received. Unexpected token.
        : 61, "uyum_orani": NaN, "kapsam_orani":

    Bu yuzden temizlik arac sinirinda yapiliyor - her cikti buradan gecer.
    """
    # numpy skalerleri ve pandas'in NA'si Python float/int DEGILDIR; onceden
    # bu dallarin hicbirine girmiyor ve json.dumps'ta TypeError uretiyorlardi.
    # Mevcut veriyle tetiklenmiyor ama bozuk/yeni bir hesap uretebilir.
    if deger is pd.NA or deger is pd.NaT:
        return None
    if hasattr(deger, "item") and hasattr(deger, "dtype"):
        try:
            deger = deger.item()   # numpy skaleri -> Python tipi
        except (ValueError, AttributeError):
            return str(deger)
    if isinstance(deger, float):
        return None if (math.isnan(deger) or math.isinf(deger)) else deger
    if isinstance(deger, dict):
        return {k: _json_guvenli(v) for k, v in deger.items()}
    if isinstance(deger, (list, tuple)):
        return [_json_guvenli(v) for v in deger]
    return deger


def _temiz_cikti(fn):
    """Aracin dondurdugu yapiyi JSON-guvenli hale getirir."""

    @wraps(fn)
    def sarmalayici(*args, **kwargs):
        return _json_guvenli(fn(*args, **kwargs))

    return sarmalayici


@tool
@_temiz_cikti
def filo_ozeti() -> dict[str, Any]:
    """Sunucu filosunun genel durumunu dondurur.

    Sunucu sayisi, ortam/bolge dagilimi, uyum ve kapsam oranlari, destegi bitmis
    ve internete acik sunucu sayilari. Analize buradan baslamak iyi bir fikirdir.
    """
    s = sunucular()
    df = birlesik()

    return {
        "sunucu_sayisi": len(s),
        "kontrol_sayisi": len(KONTROLLER),
        "denetim_sonucu_sayisi": len(df),
        "ortam_dagilimi": s["ortam"].value_counts().to_dict(),
        "ag_bolgesi_dagilimi": s["ag_bolgesi"].value_counts().to_dict(),
        "rol_dagilimi": s["rol"].value_counts().to_dict(),
        "internete_acik_sunucu": int(s["internet_erisimi"].sum()),
        "destegi_bitmis_sunucu": int((s["destek_durumu"] == "destegi_bitti").sum()),
        "filo_geneli": uyum_ve_kapsam(df),
        "kapsam": filo_kapsam_ozeti(),
        "not": (
            "uyum_orani yalnizca gozlemlenen sonuclar uzerinden hesaplanir. "
            "kapsam_orani ile birlikte okunmalidir; dusuk kapsamda yuksek uyum "
            "orani sunucunun iyi durumda oldugunu degil, hakkinda bilgi olmadigini "
            "gosterir."
        ),
    }


@tool
@_temiz_cikti
def kontrol_durumu(kontrol_id: str) -> dict[str, Any]:
    """Tek bir kontrolun filo genelindeki durumunu dondurur.

    Args:
        kontrol_id: Kontrol kimligi, ornegin "5.2.12" veya "FIRE-4590".
    """
    try:
        k = kontrol_getir(kontrol_id)
    except KeyError as hata:
        return {"hata": str(hata)}

    df = birlesik()
    alt = df[df["kontrol_id"] == kontrol_id]
    m = uyum_ve_kapsam(alt)

    uyumsuz = alt[alt["durum"] == "uyumsuz"]
    belirsiz = alt[alt["durum"] == "belirsiz"]

    return {
        "kontrol_id": k.kontrol_id,
        "baslik": k.baslik,
        "aciklama": k.aciklama,
        "kategori": k.kategori,
        "cis_seviyesi": k.seviye,
        "kaynak": k.kaynak,
        "bddk": k.bddk_etiketi,
        "risk_agirligi": k.agirlik,
        "olcum": m,
        "uyumsuz_sunucular": uyumsuz["host_id"].tolist()[:40],
        "durumu_bilinmeyen_sunucular": belirsiz["host_id"].tolist()[:40],
        "uyumsuzlarin_internete_acik_olani": int(uyumsuz["internet_erisimi"].sum()),
        "uyumsuzlarin_uretimde_olani": int((uyumsuz["ortam"] == "uretim").sum()),
    }


@tool
@_temiz_cikti
def sunucu_durumu(host_id: str) -> dict[str, Any]:
    """Tek bir sunucunun sertlestirme durumunu dondurur.

    Args:
        host_id: Sunucu kimligi, ornegin "srv-001".
    """
    df = birlesik()
    alt = df[df["host_id"] == host_id]
    if alt.empty:
        return {"hata": f"Bilinmeyen sunucu: '{host_id}'"}

    ilk = alt.iloc[0]
    m = uyum_ve_kapsam(alt)
    gun = int(ilk["son_denetim_gun_once"])

    bulgular = alt[alt["durum"] == "uyumsuz"].nlargest(10, "agirlik")

    return {
        "host_id": host_id,
        "ortam": ilk["ortam"],
        "rol": ilk["rol"],
        "ag_bolgesi": ilk["ag_bolgesi"],
        # bool(nan) True'dur: maruziyet_carpani NaN'i False sayarken burasi
        # True sayiyordu, ayni sunucu iki katmanda ters yorumlaniyordu.
        "internet_erisimi": _bool_cevir(ilk["internet_erisimi"]),
        "veri_siniflandirmasi": ilk["veri_siniflandirmasi"],
        "isletim_sistemi": f"{ilk['isletim_sistemi']} {ilk['os_surum']}",
        "destek_durumu": ilk["destek_durumu"],
        "maruziyet_carpani": maruziyet_carpani(ilk),
        "son_denetim_gun_once": gun,
        "denetim_bayat_mi": gun > TAZELIK_ESIGI_GUN,
        "olcum": m,
        "en_agir_bulgular": bulgular[
            ["kontrol_id", "baslik", "kategori", "bddk_maddesi", "agirlik"]
        ].to_dict("records"),
        "durumu_bilinmeyen_kontrol_sayisi": m["belirsiz_kontrol"],
    }


@tool
@_temiz_cikti
def uyum_kirilimi(boyut: str, yalnizca_internete_acik: bool = False) -> dict[str, Any]:
    """Uyum ve kapsam oranlarini bir boyuta gore kirar.

    Args:
        boyut: Kirilim boyutu. Gecerli degerler: ortam, rol, ag_bolgesi,
            isletim_sistemi, destek_durumu, veri_siniflandirmasi, kategori,
            bddk_maddesi, seviye.
        yalnizca_internete_acik: True ise sadece internete acik sunucular.
    """
    if boyut not in BOYUTLAR:
        return {"hata": f"Gecersiz boyut: '{boyut}'. Secenekler: {', '.join(BOYUTLAR)}"}

    df = birlesik()
    if yalnizca_internete_acik:
        df = df[df["internet_erisimi"]]
        if df.empty:
            return {"hata": "Internete acik sunucu bulunamadi."}

    satirlar = []
    for deger, grup in df.groupby(boyut):
        m = uyum_ve_kapsam(grup)
        satirlar.append(
            {
                boyut: deger,
                "sunucu_sayisi": int(grup["host_id"].nunique()),
                "uyum_orani": m["uyum_orani"],
                "kapsam_orani": m["kapsam_orani"],
                "uyumsuz_kontrol": m["uyumsuz"],
                "belirsiz_kontrol": m["belirsiz_kontrol"],
            }
        )

    satirlar.sort(key=lambda r: (r["uyum_orani"] is None, r["uyum_orani"]))
    return {
        "boyut": boyut,
        "yalnizca_internete_acik": yalnizca_internete_acik,
        "kirilim": satirlar,
    }


@tool
@_temiz_cikti
def bddk_bosluk_analizi(madde: int | None = None) -> dict[str, Any]:
    """BDDK maddesi bazinda uyum bosluklarini dondurur.

    Her kontrol, Bankalarin Bilgi Sistemleri Yonetmeligi'nin ilgili maddesine
    baglidir. Bu arac hangi maddede en fazla sapma oldugunu gosterir.

    Args:
        madde: Tek bir madde numarasi (11, 13, 14, 15, 16). Bos birakilirsa
            tum maddeler dondurulur.
    """
    gecerli = [m.value for m in BddkMaddesi]
    if madde is not None and madde not in gecerli:
        return {"hata": f"Gecersiz madde: {madde}. Secenekler: {gecerli}"}

    df = birlesik()
    maddeler = [BddkMaddesi(madde)] if madde else list(BddkMaddesi)

    sonuc = []
    for m in maddeler:
        alt = df[df["bddk_maddesi"] == m.value]
        olcum = uyum_ve_kapsam(alt)
        uyumsuz = alt[alt["durum"] == "uyumsuz"]

        en_cok = (
            uyumsuz.groupby(["kontrol_id", "baslik"])
            .size()
            .nlargest(5)
            .reset_index(name="uyumsuz_sunucu_sayisi")
            .to_dict("records")
        )

        sonuc.append(
            {
                "madde": m.value,
                "baslik": BDDK_BASLIK[m],
                "kontrol_sayisi": int(alt["kontrol_id"].nunique()),
                "olcum": olcum,
                "etkilenen_sunucu_sayisi": int(uyumsuz["host_id"].nunique()),
                "uretimde_uyumsuz_kontrol": int((uyumsuz["ortam"] == "uretim").sum()),
                "internete_acikta_uyumsuz_kontrol": int(uyumsuz["internet_erisimi"].sum()),
                "en_cok_sapan_kontroller": en_cok,
            }
        )

    sonuc.sort(key=lambda r: (r["olcum"]["uyum_orani"] is None, r["olcum"]["uyum_orani"]))
    return {"kaynak": "Bankalarin Bilgi Sistemleri Yonetmeligi", "maddeler": sonuc}


@tool
@_temiz_cikti
def risk_siralamasi(limit: int = 10, sunucu_bazinda: bool = False) -> dict[str, Any]:
    """Bulgulari maruziyete gore agirliklandirip siralar.

    Ham bulgu sayisi yaniltir: izole bir test makinesindeki 40 bulgu, internete
    acik bir uretim veritabanindaki 3 bulgudan daha az onemli olabilir. Bu arac
    ortam, ag bolgesi, internet erisimi, veri siniflandirmasi ve destek durumunu
    carpimsal bir maruziyet katsayisina cevirip riski ona gore siralar.

    Args:
        limit: Dondurulecek satir sayisi.
        sunucu_bazinda: True ise sunucu bazinda ozet, False ise tekil bulgular.
    """
    limit = max(1, min(int(limit), 50))

    if sunucu_bazinda:
        tablo = sunucu_riski().head(limit)
        return {
            "siralama": "sunucu bazinda, maruziyet agirlikli toplam risk",
            "satirlar": tablo.to_dict("records"),
            "not": (
                "belirsiz_risk, durumu okunamayan kontrollerin BEKLENEN riskidir; "
                "ilgili kontrolun filo genelindeki uyumsuzluk oraniyla tahmin edilir."
            ),
        }

    tablo = bulgu_siralamasi(limit=limit)
    return {
        "siralama": "tekil bulgu bazinda, maruziyet agirlikli",
        "satirlar": tablo.to_dict("records"),
    }


@tool
@_temiz_cikti
def kapsam_raporu(yalnizca_yaniltici: bool = False) -> dict[str, Any]:
    """Denetim verisinin ne kadarinin gercekten okunabildigini raporlar.

    'notchecked', 'error' ve 'unknown' sonuclari uyumlu DEGILDIR; durumu
    bilinmiyordur. Bu arac, dusuk kapsam veya bayat veri yuzunden hakkinda
    hukum verilemeyecek sunuculari cikarir.

    Args:
        yalnizca_yaniltici: True ise sadece "uyumlu gorunen ama bilgisi olmayan"
            sunuculari dondurur - ikili panellerde yesil gorunen tehlikeli grup.
    """
    ozet = filo_kapsam_ozeti()

    if yalnizca_yaniltici:
        tablo = yaniltici_temizler()
        return {
            "ozet": ozet,
            "aciklama": (
                "Bu sunucular yuksek uyum orani gosteriyor ancak haklarinda hukum "
                "vermek icin yeterli veri yok. Ikili bir uyumluluk panelinde yesil "
                "gorunurler; bulgu sayilari dusuktur cunku kontroller kosturulmamistir."
            ),
            "sunucular": tablo.head(25).to_dict("records"),
        }

    tablo = sunucu_kapsami()
    sorunlu = tablo[~tablo["hukum_verilebilir"]].sort_values("kapsam_orani")

    # Boyut kirilimi burada uretiliyor ki ajan sunucu listelerini elle sayarak
    # cikarim yapmak zorunda kalmasin - o yol hem kirilgan hem de kesme
    # sinirina takiliyor.
    kirilim = []
    for ortam, grup in tablo.groupby("ortam"):
        kirilim.append(
            {
                "ortam": ortam,
                "sunucu_sayisi": len(grup),
                "bayat_denetimli": int(grup["bayat"].sum()),
                "yetersiz_kapsamli": int((~grup["yeterli_kapsam"]).sum()),
                "hukum_verilebilir": int(grup["hukum_verilebilir"].sum()),
                "bayat_orani": round(float(grup["bayat"].mean()), 4),
            }
        )
    kirilim.sort(key=lambda r: r["bayat_orani"], reverse=True)

    return {
        "ozet": ozet,
        "ortam_kirilimi": kirilim,
        "hukum_verilemeyen_sunucu_sayisi": len(sorunlu),
        "hukum_verilemeyen_sunucular": sorunlu.head(25).to_dict("records"),
    }


@tool
@_temiz_cikti
def sunucu_listesi(
    ortam: str | None = None,
    rol: str | None = None,
    ag_bolgesi: str | None = None,
    yalnizca_internete_acik: bool = False,
    yalnizca_destegi_bitmis: bool = False,
) -> dict[str, Any]:
    """Envanterden filtreyle sunucu listeler.

    Args:
        ortam: uretim, test veya felaket_kurtarma.
        rol: veritabani, web, uygulama, atlama_sunucusu, yedekleme.
        ag_bolgesi: dmz, ic_ag veya kisitli.
        yalnizca_internete_acik: True ise sadece internete acik olanlar.
        yalnizca_destegi_bitmis: True ise sadece destegi bitmis isletim sistemleri.
            BDDK 16. madde destegi biten sistemlerin kullanimdan kaldirilmasini ister.
    """
    s = sunucular()
    if ortam:
        s = s[s["ortam"] == ortam]
    if rol:
        s = s[s["rol"] == rol]
    if ag_bolgesi:
        s = s[s["ag_bolgesi"] == ag_bolgesi]
    if yalnizca_internete_acik:
        s = s[s["internet_erisimi"]]
    if yalnizca_destegi_bitmis:
        s = s[s["destek_durumu"] == "destegi_bitti"]

    if s.empty:
        return {"sunucu_sayisi": 0, "sunucular": [], "not": "Filtreye uyan sunucu yok."}

    kolonlar = [
        "host_id", "ortam", "rol", "ag_bolgesi", "internet_erisimi",
        "isletim_sistemi", "os_surum", "destek_durumu",
        "veri_siniflandirmasi", "son_denetim_gun_once",
    ]

    # Kesme SESSIZ olmamali. Onceden "sunucu_sayisi: 63" dondurulup yalnizca 40
    # kayit veriliyordu; ajan listeyi tam sanip sayarsa yanlis sonuca varir.
    # Gozlenen kosumda ajan bu yuzden sunucu_listesi'ni 9 kez cagirmak zorunda
    # kaldi ve dogrulama modeli sayilari "uydurulmus" diye isaretledi.
    kesildi = len(s) > LISTE_SINIRI
    cikti: dict[str, Any] = {
        "sunucu_sayisi": len(s),
        "donen_kayit_sayisi": min(len(s), LISTE_SINIRI),
        "kesildi": kesildi,
        "sunucular": s[kolonlar].head(LISTE_SINIRI).to_dict("records"),
    }
    # "not" anahtari KOSULSUZ bulunuyor: yalnizca kesilme halinde eklenince
    # agent alanin varligina guvenemiyor, bazen var bazen yok oluyordu.
    cikti["not"] = ""
    if kesildi:
        cikti["not"] = (
            f"Filtreye {len(s)} sunucu uyuyor ancak yalnizca ilk {LISTE_SINIRI} kaydi "
            "dondurulur. Donen kayitlari SAYARAK sonuc cikarma; toplam icin "
            "sunucu_sayisi alanini kullan, kirilim icin uyum_kirilimi ya da "
            "kapsam_raporu araclarini cagir."
        )
    return cikti


ANALIZ_ARACLARI = [
    filo_ozeti,
    kontrol_durumu,
    sunucu_durumu,
    uyum_kirilimi,
    bddk_bosluk_analizi,
    risk_siralamasi,
    kapsam_raporu,
    sunucu_listesi,
]
