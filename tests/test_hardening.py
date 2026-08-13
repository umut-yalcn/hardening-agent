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
    INTERNET_CARPANI,
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

    @pytest.mark.parametrize(
        "deger,acik",
        [
            (True, True), ("True", True), ("true", True), (1, True),
            (False, False), ("False", False), ("false", False),
            ("0", False), ("", False), (0, False), ("hayir", False),
        ],
    )
    def test_internet_erisimi_string_gelse_de_dogru_yorumlanir(self, deger, acik):
        """Regresyon: string 'False' Python'da DOGRU sayilir.

        Duz `if sunucu["internet_erisimi"]:` kullanildiginda CSV'den string
        gelen "False" degeri carpani sessizce 2.2 kat artiriyordu. Sessiz
        olmasi en kotu yani: risk skoru yanlis cikar, hicbir hata gorunmez.
        """
        temel = {
            "ortam": "test", "ag_bolgesi": "ic_ag", "veri_siniflandirmasi": "genel",
            "destek_durumu": "destekleniyor", "internet_erisimi": deger,
        }
        beklenen = 0.4 * (INTERNET_CARPANI if acik else 1.0)
        assert maruziyet_carpani(temel) == pytest.approx(beklenen)

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
class TestJsonGuvenligi:
    """Regresyon: arac ciktisindaki NaN, ajani komple dusuruyordu.

    kapsam_raporu, hic gozlemlenen kontrolu olmayan bir sunucu icin uyum_orani
    olarak pandas NaN donduruyordu. Python'un json.dumps'i NaN'a izin verip
    literal `NaN` yaziyor - ama bu gecerli JSON DEGIL. Model saglayicisi
    istegin tamamini reddetti:

        400 INVALID_ARGUMENT: Invalid JSON payload received. Unexpected token.
        : 61, "uyum_orani": NaN, "kapsam_orani":

    json.dumps hata vermedigi icin onceki testler bunu kacirdi; kati modda
    (allow_nan=False) test etmek gerekiyor.
    """

    ARAC_GIRDILERI = [
        ("filo_ozeti", {}),
        ("kapsam_raporu", {"yalnizca_yaniltici": False}),
        ("kapsam_raporu", {"yalnizca_yaniltici": True}),
        ("risk_siralamasi", {"sunucu_bazinda": True, "limit": 50}),
        ("risk_siralamasi", {"limit": 50}),
        ("sunucu_durumu", {"host_id": "srv-002"}),
        ("sunucu_listesi", {}),
        ("bddk_bosluk_analizi", {}),
        ("uyum_kirilimi", {"boyut": "ortam"}),
        ("kontrol_durumu", {"kontrol_id": "5.2.12"}),
    ]

    @pytest.mark.parametrize("arac_adi,girdi", ARAC_GIRDILERI)
    def test_arac_ciktisi_kati_json(self, arac_adi, girdi):
        import json

        from src.tools import ANALIZ_ARACLARI

        arac = next(t for t in ANALIZ_ARACLARI if t.name == arac_adi)
        json.dumps(arac.invoke(girdi), allow_nan=False)

    def test_temizleyici_nan_ve_sonsuzlugu_cevirir(self):
        from src.tools import _json_guvenli

        girdi = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")}}
        assert _json_guvenli(girdi) == {"a": None, "b": [1.0, None], "c": {"d": None}}

    def test_temizleyici_normal_degerlere_dokunmaz(self):
        from src.tools import _json_guvenli

        girdi = {"a": 1.5, "b": [1, "x", True, None], "c": {"d": 0.0}}
        assert _json_guvenli(girdi) == girdi

    def test_liste_kesmesi_bildirilir(self):
        """Regresyon: sunucu_listesi 63 sayar, 40 kayit dondururdu - sessizce.

        Ajan donen kayitlari sayarak cikarim yaparsa yanilir. Gozlenen kosumda
        ajan bu yuzden araci 9 kez cagirmak zorunda kaldi ve dogrulama modeli
        (dogru olan) sayilari "uydurulmus" diye isaretledi.
        """
        from src.tools import LISTE_SINIRI, sunucu_listesi

        r = sunucu_listesi.invoke({"ortam": "uretim"})
        assert r["sunucu_sayisi"] > LISTE_SINIRI
        assert r["kesildi"] is True
        assert r["donen_kayit_sayisi"] == LISTE_SINIRI
        assert len(r["sunucular"]) == LISTE_SINIRI
        assert "not" in r

    def test_kesilmeyen_listede_uyari_yok(self):
        from src.tools import LISTE_SINIRI, sunucu_listesi

        r = sunucu_listesi.invoke({"ortam": "test"})
        assert r["sunucu_sayisi"] <= LISTE_SINIRI
        assert r["kesildi"] is False
        assert r["donen_kayit_sayisi"] == len(r["sunucular"])

    def test_kapsam_raporu_ortam_kirilimi_verir(self):
        """Ajan sunucu listelerini elle saymak zorunda kalmamali."""
        from src.freshness import sunucu_kapsami
        from src.tools import kapsam_raporu

        r = kapsam_raporu.invoke({"yalnizca_yaniltici": False})
        kirilim = r["ortam_kirilimi"]
        assert kirilim

        tablo = sunucu_kapsami()
        for satir in kirilim:
            grup = tablo[tablo["ortam"] == satir["ortam"]]
            assert satir["sunucu_sayisi"] == len(grup)
            assert satir["bayat_denetimli"] == int(grup["bayat"].sum())

        assert sum(s["bayat_denetimli"] for s in kirilim) == r["ozet"]["bayat_denetimli_sunucu"]

    def test_arac_aciklamalari_korundu(self):
        """Temizleyici dekoratoru @tool'un gordugu docstring'i bozmamali."""
        from src.tools import ANALIZ_ARACLARI

        for t in ANALIZ_ARACLARI:
            assert t.description, f"{t.name} aciklamasini kaybetti"


# --------------------------------------------------------------------------
class TestDayanakKontrolu:
    """Ajan, arac hatasi aldiginda cevabi uydurabiliyor.

    Kardes projede gozlenen gercek davranis: ajan olmayan bir kolon adiyla
    cagri yapti, guard reddetti, ajan duzeltmek yerine sayi uydurdu.
    Dogrulama modeli bunu yakalayabilir ama o bir model cagrisi - dusebilir,
    kota nedeniyle kapatilmis olabilir. Bu sayim deterministik.
    """

    def _tm(self, icerik: str):
        from langchain_core.messages import ToolMessage

        return ToolMessage(content=icerik, tool_call_id="1")

    def test_hata_ciktisi_hatali_sayilir(self):
        from src.agent import _arac_ciktilarini_say

        assert _arac_ciktilarini_say([self._tm('{"hata": "Bilinmeyen kontrol"}')]) == (0, 1)

    def test_basarili_cikti_sayilir(self):
        from src.agent import _arac_ciktilarini_say

        assert _arac_ciktilarini_say([self._tm('{"kontrol_id": "5.2.12"}')]) == (1, 0)

    def test_arac_yoksa_sifir(self):
        from src.agent import _arac_ciktilarini_say

        assert _arac_ciktilarini_say([]) == (0, 0)

    def test_gercek_arac_ciktilariyla_sayim(self):
        """ToolNode dict'i JSON'a ceviriyor; sayim gercek yolda da dogru olmali."""
        from langchain_core.messages import AIMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        from src.agent import _arac_ciktilarini_say
        from src.tools import ANALIZ_ARACLARI

        graf = StateGraph(MessagesState)
        graf.add_node("a", ToolNode(ANALIZ_ARACLARI))
        graf.add_edge(START, "a")
        graf.add_edge("a", END)
        app = graf.compile()

        sonuc = app.invoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "kontrol_durumu", "args": {"kontrol_id": "5.2.12"},
                             "id": "1", "type": "tool_call"},
                            {"name": "kontrol_durumu", "args": {"kontrol_id": "YOK-9999"},
                             "id": "2", "type": "tool_call"},
                        ],
                    )
                ]
            }
        )
        assert _arac_ciktilarini_say(sonuc["messages"]) == (1, 1)


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


# --------------------------------------------------------------------------
class TestBagimsizDenetimBulgulari:
    """Bagimsiz bir denetci ajanin bulduklari."""

    def test_hic_gozlemlenmemis_kontrol_sifir_risk_uretmez(self):
        """P2-A: fillna(0.0) yuzunden, hakkinda hicbir sey bilinmeyen bir
        kontrol TAM UYUMLU gibi puanlaniyordu. Projenin tezinin tam tersi.
        """

        from src.scoring import risk_tablosu

        df = birlesik().copy()
        maske = df["kontrol_id"] == "FIRE-4590"
        df.loc[maske, "sonuc"] = "notchecked"
        df.loc[maske, "durum"] = "belirsiz"

        r = risk_tablosu(df)
        alt = r[r["kontrol_id"] == "FIRE-4590"]
        assert (alt["filo_uyumsuzluk_orani"] > 0).all(), "bilinmeyen kontrol sifir oran aldi"
        assert alt["belirsiz_risk"].sum() > 0, "hic gozlemlenmemis kontrol sifir risk uretti"

    def test_gozlemlenen_kontroller_etkilenmedi(self):
        """Asiri duzeltme olmamali: gozlemlenen kontroller kendi oranini kullanmali."""
        from src.scoring import kontrol_uyumsuzluk_orani, risk_tablosu

        oran = kontrol_uyumsuzluk_orani()
        r = risk_tablosu()
        for kid in ("5.2.12", "PKGS-7392"):
            gercek = r.loc[r["kontrol_id"] == kid, "filo_uyumsuzluk_orani"].iloc[0]
            assert abs(gercek - oran[kid]) < 1e-9

    @pytest.mark.parametrize(
        "girdi,beklenen",
        [
            ({"dogrulandi": "true", "gerekce": "", "sorunlar": []}, True),
            ({"dogrulandi": "FALSE", "gerekce": "", "sorunlar": []}, False),
            ({"dogrulandi": True, "gerekce": "", "sorunlar": []}, True),
            ({"dogrulandi": "belki", "gerekce": "", "sorunlar": []}, None),
            (["liste", "geldi"], None),
            ("duz metin", None),
        ],
    )
    def test_dogrulama_semasi_bicimi_oturtur(self, girdi, beklenen):
        """P2-B: denetci model 'true' (string) donunce CLI KeyError ile cokuyordu."""
        from src.agent import _dogrulama_semasi

        r = _dogrulama_semasi(girdi)
        assert r["dogrulandi"] is beklenen
        assert isinstance(r["sorunlar"], list)
        assert isinstance(r["gerekce"], str)

    def test_cli_eslemesi_hicbir_girdide_patlamaz(self):
        from src.agent import _dogrulama_semasi

        for girdi in ({"dogrulandi": "true"}, {"dogrulandi": None}, ["x"], 42):
            d = _dogrulama_semasi(girdi)
            {True: "DOGRULANDI", False: "SORUNLU", None: "CALISTIRILAMADI"}[d["dogrulandi"]]

    def test_maruziyet_araligi_belgelenen_degerle_uyusuyor(self):
        """P2-C: README '~10 kat' diyordu, gercek ~37 kat."""
        from src.scoring import maruziyet_carpani

        en_maruz = maruziyet_carpani({
            "ortam": "uretim", "ag_bolgesi": "dmz", "veri_siniflandirmasi": "hassas",
            "internet_erisimi": True, "destek_durumu": "destegi_bitti"})
        en_korunakli = maruziyet_carpani({
            "ortam": "test", "ag_bolgesi": "kisitli", "veri_siniflandirmasi": "genel",
            "internet_erisimi": False, "destek_durumu": "destekleniyor"})
        oran = en_maruz / en_korunakli
        assert 35 < oran < 39, f"belgelenen ~37 kat degil: {oran:.1f}"
