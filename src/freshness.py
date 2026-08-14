"""Denetim verisinin tazeligi ve kapsami.

Projenin ikinci tezi burada: bir sunucu hakkinda bilgi yoksa o sunucu uyumlu
degildir, durumu bilinmiyordur.

Iki ayri korluk kaynagi var ve ikisi de ikili panellerde yesil gorunur:

  KAPSAM BOSLUGU  - denetim kosuldu ama kontrollerin bir kismi okunamadi
                    (notchecked / error / unknown). 'fail' uretmedigi icin
                    bulgu sayisina yansimaz.

  TAZELIK BOSLUGU - denetim uzun suredir hic kosulmadi. Elimizdeki sonuc
                    gecmisteki bir anin fotografidir; bugunu anlatmaz.
"""

from __future__ import annotations

import pandas as pd

from .scoring import _bool_cevir
from .fleet import birlesik, uyum_ve_kapsam

#: Bu esikten eski denetim verisi guncel kabul edilmez.
#: BDDK 15 ve 16. maddeler "duzenli" tarama istiyor; duzenliyi 30 gun tanimliyoruz.
TAZELIK_ESIGI_GUN = 30

#: Bu oranin altinda kapsam, sunucu hakkinda hukum vermek icin yetersizdir.
YETERLI_KAPSAM = 0.85


def sunucu_kapsami(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sunucu bazinda uyum orani, kapsam orani ve tazelik."""
    if df is None:
        df = birlesik()

    satirlar = []
    for host_id, grup in df.groupby("host_id"):
        m = uyum_ve_kapsam(grup)
        # int(NaN) ValueError firlatir ve TUM ajan kosumunu dusururdu:
        # sunucu_kapsami'yi kapsam_raporu cagiriyor, onu da her arac.
        # Bozuk tek bir satir yuzunden analizin tamami olmemeli.
        ham_gun = grup["son_denetim_gun_once"].iloc[0]
        gun = None if pd.isna(ham_gun) else int(ham_gun)
        satirlar.append(
            {
                "host_id": host_id,
                "ortam": grup["ortam"].iloc[0],
                "ag_bolgesi": grup["ag_bolgesi"].iloc[0],
                # bool(nan) True'dur ve bool("False") de True'dur; scoring NaN'i
                # False sayiyordu. Ayni sunucu iki katmanda ters yorumlaniyordu.
                "internet_erisimi": _bool_cevir(grup["internet_erisimi"].iloc[0]),
                "son_denetim_gun_once": gun,
                # Denetim yasi BILINMIYORSA taze SAYILMAZ. Guvenlik
                # tarafinda bilinmeyen, iyi haber degildir: "son taramanin ne
                # zaman oldugunu bilmiyoruz" ile "dun tarandi" ayni sonuca
                # cikamaz. Bu yuzden fail-closed.
                "bayat": True if gun is None else gun > TAZELIK_ESIGI_GUN,
                "uygulanabilir_kontrol": m["uygulanabilir_kontrol"],
                "gozlemlenen_kontrol": m["gozlemlenen_kontrol"],
                "belirsiz_kontrol": m["belirsiz_kontrol"],
                "uyum_orani": m["uyum_orani"],
                "kapsam_orani": m["kapsam_orani"],
            }
        )

    # Bos girdi: pd.DataFrame([]) KOLONSUZ gelir ve sonraki satir
    # KeyError('kapsam_orani') ile patlardi - hem de VeriYok gibi aciklayici
    # bir hata degil, anlasilmaz bir istisnayla. Kolonlar sabitleniyor.
    tablo = pd.DataFrame(satirlar, columns=[
        "host_id", "ortam", "ag_bolgesi", "internet_erisimi",
        "son_denetim_gun_once", "bayat", "uygulanabilir_kontrol",
        "gozlemlenen_kontrol", "belirsiz_kontrol", "uyum_orani", "kapsam_orani",
    ])
    tablo["yeterli_kapsam"] = tablo["kapsam_orani"] >= YETERLI_KAPSAM
    tablo["hukum_verilebilir"] = tablo["yeterli_kapsam"] & ~tablo["bayat"]
    return tablo


def yaniltici_temizler(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uyumlu GORUNEN ama hakkinda yeterli bilgi OLMAYAN sunucular.

    Bu fonksiyon projenin tezini tek basina tasiyor. Buradaki sunucular ikili
    bir uyumluluk panelinde yuksek skorla yesil gorunur, cunku 'fail' sayilari
    dusuktur. Dusuk olmasinin sebebi iyi sertlestirilmis olmalari degil,
    kontrollerin hic kosturulmamis olmasidir.
    """
    tablo = sunucu_kapsami(df)
    suphe = tablo[
        (tablo["uyum_orani"].notna())
        & (tablo["uyum_orani"] >= 0.80)
        & (~tablo["hukum_verilebilir"])
    ].copy()

    suphe["gerekce"] = suphe.apply(
        lambda r: "; ".join(
            filter(
                None,
                [
                    f"kapsam yalnizca %{r['kapsam_orani'] * 100:.0f}"
                    if not r["yeterli_kapsam"] else None,
                    f"{r['son_denetim_gun_once']} gundur denetlenmemis"
                    if r["bayat"] else None,
                ],
            )
        ),
        axis=1,
    )

    return suphe.sort_values("kapsam_orani").reset_index(drop=True)


def filo_kapsam_ozeti(df: pd.DataFrame | None = None) -> dict:
    """Filo genelinde ne kadarini gercekten bildigimizi ozetler."""
    tablo = sunucu_kapsami(df)
    toplam = len(tablo)

    return {
        "sunucu_sayisi": toplam,
        "hukum_verilebilir_sunucu": int(tablo["hukum_verilebilir"].sum()),
        "yetersiz_kapsamli_sunucu": int((~tablo["yeterli_kapsam"]).sum()),
        "bayat_denetimli_sunucu": int(tablo["bayat"].sum()),
        "tazelik_esigi_gun": TAZELIK_ESIGI_GUN,
        "yeterli_kapsam_esigi": YETERLI_KAPSAM,
        "ortalama_kapsam_orani": round(float(tablo["kapsam_orani"].mean()), 4),
        # Yasi bilinmeyen sunucular max()'a girmez; hepsi bilinmiyorsa
        # sayi UYDURULMAZ, None dondurulur.
        "en_bayat_gun": (
            None
            if tablo["son_denetim_gun_once"].dropna().empty
            else int(tablo["son_denetim_gun_once"].dropna().max())
        ),
        "yaniltici_temiz_sunucu": len(yaniltici_temizler(df)),
    }
