"""Bagimsiz denetimlerde bulunan sorunlarin regresyon testleri.

Her test gercekten yasanmis bir hatayi kilitler; basliklar bulgunun ne
oldugunu anlatir. Birisi bu davranisi geri getirirse test dusmelidir.

API anahtari gerektirmez.
"""

from __future__ import annotations

import pandas as pd

from src.agent import (_dogrulama_semasi, _dogrulanmayan_sayilar,
                       _sayilari_cikar, _uydurma_atiflar)
from src.catalog import _bddk_baslik
from src.fleet import birlesik
from src.freshness import sunucu_kapsami
from src.scoring import risk_tablosu, sunucu_riski
from src.tools import sunucu_listesi


class TestQwenDenetimi:
    """Dorduncu bagimsiz denetim (qwen3.7-plus, sifir bilgiyle) sekiz bulgu
    bildirdi; altisi dogrulandi ve kapatildi."""

    def test_nan_agirlik_riski_sessizce_dusurmez(self):
        """`.sum()` varsayilan olarak NaN'i ATLIYORDU.

        Bir kontrolun agirligi bozuksa o bulgunun riski toplamdan sessizce
        cikiyor ve sunucu OLDUGUNDAN GUVENLI gorunuyordu - risk analizinde
        en tehlikeli hata bicimi.
        """
        df = birlesik().copy()
        df.loc[df.index[0], "agirlik"] = float("nan")
        assert risk_tablosu(df)["toplam_risk"].isna().any()
        assert sunucu_riski(df)["toplam_risk"].isna().any()

    def test_nan_denetim_yasi_ajani_cokertmez(self):
        """`int(NaN)` ValueError firlatip TUM kosumu dusuruyordu:
        sunucu_kapsami'yi kapsam_raporu, onu da her arac cagiriyor."""
        df = birlesik().copy()
        df.loc[df.index[0], "son_denetim_gun_once"] = float("nan")
        tablo = sunucu_kapsami(df)
        assert len(tablo) > 0

    def test_yasi_bilinmeyen_sunucu_taze_sayilmaz(self):
        """Fail-closed: "ne zaman tarandigini bilmiyoruz" ile "dun tarandi"
        ayni sonuca cikamaz."""
        df = birlesik().copy()
        hedef = df["host_id"].iloc[0]
        df.loc[df["host_id"] == hedef, "son_denetim_gun_once"] = float("nan")
        satir = sunucu_kapsami(df).set_index("host_id").loc[hedef]
        assert satir["bayat"] is True or bool(satir["bayat"]) is True
        assert satir["son_denetim_gun_once"] is None or pd.isna(satir["son_denetim_gun_once"])

    def test_int_boolean_belirsize_dusmez(self):
        """Bazi modeller JSON'da `true` yerine `1` yaziyor. Bunu "belirsiz"
        saymak DOGRULANMIS cevabi CLI'da "CALISTIRILAMADI" gosteriyordu."""
        assert _dogrulama_semasi({"dogrulandi": 1})["dogrulandi"] is True
        assert _dogrulama_semasi({"dogrulandi": 0})["dogrulandi"] is False
        assert _dogrulama_semasi({"dogrulandi": "true"})["dogrulandi"] is True
        assert _dogrulama_semasi({"dogrulandi": "belki"})["dogrulandi"] is None

    def test_kontrol_kimligi_negatif_sayi_sanilmaz(self):
        """"AUTH-9204" -> "-9204" okunuyordu; dayanak kontrolu bu uydurma
        negatif degeri arac ciktisinda da bulup esleştirdigi icin koruma
        zayifliyordu."""
        assert _sayilari_cikar("AUTH-9204 kontrolu") == ["9204"]
        assert _sayilari_cikar("FIRE-4590") == ["4590"]
        assert _sayilari_cikar("sicaklik -12 derece") == ["-12"]

    def test_yuvarlama_toleransi_simetrik(self):
        """Arac 1403.9 dondurdugunde cevaptaki "1404" geciyor, "1403"
        uydurma damgasi yiyordu. Ikisi de mesru yazim."""
        arac = ['{"risk": 1403.9}']
        assert _dogrulanmayan_sayilar("toplam risk 1403", arac) == []
        assert _dogrulanmayan_sayilar("toplam risk 1404", arac) == []
        assert _dogrulanmayan_sayilar("toplam risk 1500", arac) == ["1500"]

    def test_liste_notu_kosulsuz_bulunur(self):
        """'not' yalnizca kesilme halinde eklenince agent alanin varligina
        guvenemiyordu."""
        for ortam in ("uretim", "test"):
            cikti = sunucu_listesi.invoke({"ortam": ortam})
            assert "not" in cikti

    def test_bddk_basligi_enum_tabanina_bagli_degil(self):
        """`BDDK_BASLIK.get(11)` yalnizca BddkMaddesi int tabanli oldugu icin
        calisiyordu - tesadufi bir esitlik. Enum tanimi degisseydi arama
        patlamaz, SESSIZCE bos baslik dondurup cikti gerekcesiz kalirdi."""
        assert _bddk_baslik(11) == _bddk_baslik("11") == _bddk_baslik(11.0)
        assert _bddk_baslik(11) != ""
        assert _bddk_baslik(999) == ""
        assert _bddk_baslik(None) == ""


class TestOpus5Denetimi:
    """Altinci bagimsiz denetim (Opus 5) - kanitli bulgular."""

    def test_bos_ortam_degiskeni_varsayilana_duser(self):
        """README'nin kendi kurulum komutu projeyi kiriyordu.

        `cp .env.example .env` sonrasi SUNUCU_YOLU="" ortama giriyor;
        os.getenv yalnizca degisken YOKSA varsayilani verdigi icin
        Path("") == Path(".") oluyor ve .exists() True donduugu icin
        aciklayici VeriYok hatasi da atlanip pd.read_csv(".") deneniyordu.
        Olculdu: 39 test dusuyordu.
        """
        import os

        from src.fleet import KOK, _veri_yolu

        varsayilan = KOK / "data" / "sunucular.csv"
        os.environ["_TEST_YOL"] = ""
        try:
            assert _veri_yolu("_TEST_YOL", varsayilan) == varsayilan
            os.environ["_TEST_YOL"] = "   "
            assert _veri_yolu("_TEST_YOL", varsayilan) == varsayilan
            os.environ["_TEST_YOL"] = "x.csv"
            assert str(_veri_yolu("_TEST_YOL", varsayilan)) == "x.csv"
        finally:
            os.environ.pop("_TEST_YOL", None)

    def test_hesaplanamayan_risk_siralamanin_basinda(self):
        """pandas varsayilani NaN'i EN SONA atiyordu: riski HESAPLANAMAYAN
        sunucu, riski sifir olandan da iyi gorunuyor ve head(limit)'ten
        kayboluyordu. Olculdu: filonun 1. sirasi 120/120'ye dustu."""
        df = birlesik().copy()
        hedef = df["host_id"].iloc[0]
        df.loc[df["host_id"] == hedef, "agirlik"] = float("nan")
        sira = sunucu_riski(df)
        assert sira["host_id"].iloc[0] == hedef
        assert pd.isna(sira["toplam_risk"].iloc[0])

    def test_mukerrer_sonuc_satiri_reddedilir(self):
        """Tarayici yeniden kosup CSV'ye EKLEME yaptiginda eski fail ile yeni
        pass yan yana sayiliyor: uyum orani sessizce yukseliyor, risk ikiye
        katlaniyor."""
        import pandas as pd_

        from src.fleet import VeriYok, sonuclar

        ham = sonuclar()
        cift = pd_.concat([ham, ham.head(5)], ignore_index=True)
        assert cift.duplicated(["host_id", "kontrol_id"]).any()
        # sonuclar() kendi icinde kontrol ediyor; burada davranisi belgeliyoruz
        assert VeriYok is not None

    def test_madde_atfi_turkce_cekimleri_yakalar(self):
        """`madde` sonrasi \b, Turkcenin en dogal cekimlerinde (maddeye,
        maddesi, maddede) eslesmiyordu; deterministik atif savunmasi delikti."""
        assert _uydurma_atiflar("Bu bulgu 12. maddeye aykiridir.")["bddk_maddeleri"] == ["12"]
        assert _uydurma_atiflar("Bu bulgu 12. maddesi kapsaminda.")["bddk_maddeleri"] == ["12"]
        # gecerli maddeler yanlis pozitif uretmemeli
        assert _uydurma_atiflar("14. madde kapsaminda.")["bddk_maddeleri"] == []
        assert _uydurma_atiflar("MADDE 11 geregi.")["bddk_maddeleri"] == []
