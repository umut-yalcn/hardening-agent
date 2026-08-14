"""Bagimsiz denetimlerde bulunan sorunlarin regresyon testleri.

Her test gercekten yasanmis bir hatayi kilitler; basliklar bulgunun ne
oldugunu anlatir. Birisi bu davranisi geri getirirse test dusmelidir.

API anahtari gerektirmez.
"""

from __future__ import annotations

import pandas as pd

from src.agent import _dogrulama_semasi, _dogrulanmayan_sayilar, _sayilari_cikar
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
