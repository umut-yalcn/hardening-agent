"""Analiz katmani testleri.

Hicbiri API anahtari gerektirmez. Test edilen sey modelin davranisi degil,
kod yolundaki kurallardir: belirsiz sonucun uyumlu sayilamamasi, kapsam disi
kontrolun paydaya girmemesi, maruziyetin siralamayi degistirmesi.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.controls import (
    BELIRSIZ,
    KAPSAM_DISI,
    KESIN_UYUMLU,
    KESIN_UYUMSUZ,
    KONTROLLER,
    BddkMaddesi,
    Sonuc,
    kontrol_getir,
    madde_kontrolleri,
)
from src.fleet import birlesik, durum_dagilimi, uyum_ve_kapsam
from src.freshness import TAZELIK_ESIGI_GUN, sunucu_kapsami, yaniltici_temizler
from src.scoring import (
    kontrol_uyumsuzluk_orani,
    maruziyet_carpani,
    risk_tablosu,
    sunucu_riski,
)


# --------------------------------------------------------------------------
class TestKatalog:
    def test_kontrol_kimlikleri_benzersiz(self):
        kimlikler = [k.kontrol_id for k in KONTROLLER]
        assert len(kimlikler) == len(set(kimlikler))

    def test_her_kontrol_bir_bddk_maddesine_bagli(self):
        for k in KONTROLLER:
            assert isinstance(k.bddk, BddkMaddesi)

    def test_tum_bddk_maddeleri_kapsanmis(self):
        for madde in BddkMaddesi:
            assert len(madde_kontrolleri(madde)) > 0, f"MADDE {madde.value} bos"

    def test_agirlik_ve_seviye_araliklari(self):
        for k in KONTROLLER:
            assert 1 <= k.agirlik <= 10
            assert k.seviye in (1, 2)
            assert k.kaynak in ("CIS", "Lynis")

    def test_bilinmeyen_kontrol_hata_verir(self):
        with pytest.raises(KeyError):
            kontrol_getir("9.9.9")

    def test_sonuc_siniflari_ortusmez(self):
        """Bir sonuc durumu ayni anda hem uyumlu hem belirsiz olamaz."""
        gruplar = [KESIN_UYUMLU, KESIN_UYUMSUZ, BELIRSIZ, KAPSAM_DISI]
        hepsi = [s for g in gruplar for s in g]
        assert len(hepsi) == len(set(hepsi))
        assert set(hepsi) == set(Sonuc)


# --------------------------------------------------------------------------
class TestBelirsizlik:
    """Projenin ana kurali: belirsiz sonuc uyumlu degildir."""

    def test_belirsiz_sonuclar_uyumlu_sayilmaz(self):
        df = birlesik()
        for durum in ("notchecked", "error", "unknown"):
            alt = df[df["sonuc"] == durum]
            if not alt.empty:
                assert (alt["durum"] == "belirsiz").all()

    def test_uyum_orani_paydasi_belirsizleri_icermez(self):
        df = pd.DataFrame(
            {
                "durum": ["uyumlu"] * 8 + ["uyumsuz"] * 2 + ["belirsiz"] * 90,
            }
        )
        m = uyum_ve_kapsam(df)
        # 8/10 gozlemlenen -> %80, kapsam 10/100 -> %10
        assert m["uyum_orani"] == 0.8
        assert m["kapsam_orani"] == 0.1

    def test_kapsam_disi_paydaya_girmez(self):
        df = pd.DataFrame({"durum": ["uyumlu"] * 5 + ["kapsam_disi"] * 20})
        m = uyum_ve_kapsam(df)
        assert m["uygulanabilir_kontrol"] == 5
        assert m["uyum_orani"] == 1.0
        assert m["kapsam_orani"] == 1.0

    def test_hicbir_gozlem_yoksa_oran_none(self):
        df = pd.DataFrame({"durum": ["belirsiz"] * 10})
        m = uyum_ve_kapsam(df)
        assert m["uyum_orani"] is None
        assert m["kapsam_orani"] == 0.0

    def test_bozuk_ajanli_sunucu_sifir_bulgu_ama_uyumlu_degil(self):
        """srv-002 senaryosu: hic 'fail' yok, ama hicbir sey de bilinmiyor."""
        df = birlesik()
        alt = df[df["host_id"] == "srv-002"]
        d = durum_dagilimi(alt)
        assert d["uyumsuz"] == 0
        assert d["belirsiz"] > 0
        m = uyum_ve_kapsam(alt)
        assert m["kapsam_orani"] == 0.0


# --------------------------------------------------------------------------
class TestMaruziyet:
    def test_internete_acik_daha_agir(self):
        temel = {
            "ortam": "uretim", "ag_bolgesi": "dmz", "veri_siniflandirmasi": "hassas",
            "internet_erisimi": False, "destek_durumu": "destekleniyor",
        }
        acik = dict(temel, internet_erisimi=True)
        assert maruziyet_carpani(acik) > maruziyet_carpani(temel)

    def test_uretim_testten_agir(self):
        temel = {
            "ag_bolgesi": "ic_ag", "veri_siniflandirmasi": "dahili",
            "internet_erisimi": False, "destek_durumu": "destekleniyor",
        }
        assert maruziyet_carpani(dict(temel, ortam="uretim")) > maruziyet_carpani(
            dict(temel, ortam="test")
        )

    def test_destegi_bitmis_daha_agir(self):
        temel = {
            "ortam": "uretim", "ag_bolgesi": "ic_ag", "veri_siniflandirmasi": "dahili",
            "internet_erisimi": False, "destek_durumu": "destekleniyor",
        }
        eol = dict(temel, destek_durumu="destegi_bitti")
        assert maruziyet_carpani(eol) > maruziyet_carpani(temel)

    def test_en_korunakli_sunucu_carpani_birden_kucuk(self):
        korunakli = {
            "ortam": "test", "ag_bolgesi": "kisitli", "veri_siniflandirmasi": "genel",
            "internet_erisimi": False, "destek_durumu": "destekleniyor",
        }
        assert maruziyet_carpani(korunakli) < 1.0

    def test_siralama_bulgu_sayisindan_farkli(self):
        """Maruziyet agirlikli siralama, ham bulgu sayisiyla ayni sonucu vermemeli."""
        risk = sunucu_riski()
        ust_risk = list(risk.head(5)["host_id"])
        ust_bulgu = list(risk.nlargest(5, "bulgu_sayisi")["host_id"])
        assert ust_risk != ust_bulgu


# --------------------------------------------------------------------------
class TestRiskHesabi:
    def test_uyumlu_satir_risk_uretmez(self):
        r = risk_tablosu()
        assert (r.loc[r["durum"] == "uyumlu", "toplam_risk"] == 0).all()

    def test_kapsam_disi_satir_risk_uretmez(self):
        r = risk_tablosu()
        alt = r[r["durum"] == "kapsam_disi"]
        if not alt.empty:
            assert (alt["toplam_risk"] == 0).all()

    def test_belirsiz_satir_risk_uretir_ama_bulgudan_az(self):
        """Belirsizlik sifir risk degildir; ama kesin bulgu kadar da agir degildir."""
        r = risk_tablosu()
        belirsiz = r[(r["durum"] == "belirsiz") & (r["filo_uyumsuzluk_orani"] > 0)]
        assert not belirsiz.empty
        assert (belirsiz["belirsiz_risk"] > 0).all()
        tam = belirsiz["agirlik"] * belirsiz["maruziyet"]
        assert (belirsiz["belirsiz_risk"] <= tam).all()

    def test_belirsizlik_orani_gozlemden_hesaplanir(self):
        """Uydurma sabit degil, filo genelinde gozlemlenen uyumsuzluk orani."""
        oran = kontrol_uyumsuzluk_orani()
        assert not oran.empty
        assert (oran >= 0).all() and (oran <= 1).all()


# --------------------------------------------------------------------------
class TestTazelik:
    def test_bayat_esigi_uygulanir(self):
        tablo = sunucu_kapsami()
        assert (
            tablo.loc[tablo["son_denetim_gun_once"] > TAZELIK_ESIGI_GUN, "bayat"]
        ).all()
        assert not (
            tablo.loc[tablo["son_denetim_gun_once"] <= TAZELIK_ESIGI_GUN, "bayat"]
        ).any()

    def test_bayat_veri_hukum_verilebilir_sayilmaz(self):
        tablo = sunucu_kapsami()
        assert not tablo.loc[tablo["bayat"], "hukum_verilebilir"].any()

    def test_yaniltici_temizler_bulunur(self):
        """Yuksek uyum orani + yetersiz bilgi kombinasyonu tespit edilmeli."""
        y = yaniltici_temizler()
        assert not y.empty
        assert (y["uyum_orani"] >= 0.80).all()
        assert not y["hukum_verilebilir"].any()
        assert y["gerekce"].str.len().gt(0).all()

    def test_yaniltici_listesi_hukum_verilebilirleri_icermez(self):
        tablo = sunucu_kapsami()
        saglam = set(tablo.loc[tablo["hukum_verilebilir"], "host_id"])
        y = set(yaniltici_temizler()["host_id"])
        assert not (saglam & y)
