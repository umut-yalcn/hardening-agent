"""Zorunlu tekrar deneme (duzeltme dongusu) testleri.

Model cagirmadan test edilir: gercek LLM yerine davranisi onceden yazilmis
sahte bir model kullaniliyor. Boylece "ajan hata alinca ne yapar" sorusu
deterministik olarak sinanabiliyor.

API anahtari gerektirmez.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import agent as ajan_modulu
from src.agent import MAKS_DUZELTME, _duzeltici_mesaj, _hata_metinleri, ajan_kur


class SahteModel:
    """Onceden yazilmis cevaplari sirayla dondurur."""

    def __init__(self, cevaplar: list[AIMessage]) -> None:
        self.cevaplar = list(cevaplar)
        self.gorulen_istemler: list[list] = []

    def bind_tools(self, _araclar):
        return self

    def with_retry(self, *args, **kwargs):
        # config.dayanikli() gercek modeli Runnable.with_retry ile sariyor;
        # sahte model bu cagriyi karsilamali.
        return self

    def invoke(self, mesajlar, *args, **kwargs) -> AIMessage:
        self.gorulen_istemler.append(mesajlar)
        if self.cevaplar:
            return self.cevaplar.pop(0)
        return AIMessage(content="Baska sozum yok.")


def _arac_cagrisi(ad: str, args: dict, cid: str = "1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": ad, "args": args, "id": cid, "type": "tool_call"}],
    )


@pytest.fixture
def sahte(monkeypatch):
    def kur(cevaplar):
        model = SahteModel(cevaplar)
        monkeypatch.setattr(ajan_modulu, "get_llm", lambda *a, **k: model)
        return model

    return kur


class TestDuzelticiMesaj:
    def test_bilinmeyen_kontrol_icin_kontrol_ara_der(self):
        mesaj = _duzeltici_mesaj(['"Bilinmeyen kontrol: YOK-9999".'])
        assert "kontrol_ara" in mesaj
        assert "TEKRAR" in mesaj

    def test_bilinmeyen_sunucu_icin_liste_der(self):
        mesaj = _duzeltici_mesaj(["Bilinmeyen sunucu: 'srv-999'"])
        assert "sunucu_listesi" in mesaj

    def test_gecersiz_madde_icin_gecerli_maddeleri_sayar(self):
        mesaj = _duzeltici_mesaj(["Gecersiz madde: 99."])
        assert "11, 13, 14, 15 veya 16" in mesaj

    def test_taninmayan_hata_icin_genel_yonlendirme(self):
        assert "kontrol_ara" in _duzeltici_mesaj(["Beklenmedik durum"])

    def test_hata_metinleri_toplanir(self):
        mesajlar = [
            ToolMessage(content='{"hata": "Bilinmeyen kontrol: X"}', tool_call_id="1"),
            ToolMessage(content='{"kontrol_id": "5.2.12"}', tool_call_id="2"),
        ]
        assert _hata_metinleri(mesajlar) == ["Bilinmeyen kontrol: X"]


class TestDuzeltmeDongusu:
    def test_hatadan_sonra_uydurma_engellenir_ve_tekrar_denenir(self, sahte):
        model = sahte(
            [
                _arac_cagrisi("kontrol_durumu", {"kontrol_id": "SSH-ROOT-1"}),
                AIMessage(content="SSH root girisi 40 sunucuda acik."),
                _arac_cagrisi("kontrol_durumu", {"kontrol_id": "5.2.12"}, cid="2"),
                AIMessage(content="SSH root girisi 32 sunucuda acik."),
            ]
        )

        sonuc = ajan_kur().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )

        assert sonuc["duzeltme_denemesi"] == 1
        assert sonuc["messages"][-1].content == "SSH root girisi 32 sunucuda acik."
        assert any("DUR." in str(m.content) for m in model.gorulen_istemler[-1])

    def test_basarili_cagri_varsa_duzeltmeye_gitmez(self, sahte):
        sahte(
            [
                _arac_cagrisi("kontrol_durumu", {"kontrol_id": "5.2.12"}),
                AIMessage(content="32 sunucuda acik."),
            ]
        )
        sonuc = ajan_kur().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )
        assert sonuc["duzeltme_denemesi"] == 0

    def test_hic_arac_cagrilmadiysa_duzeltmeye_gitmez(self, sahte):
        sahte([AIMessage(content="Merhaba.")])
        sonuc = ajan_kur().invoke(
            {"messages": [HumanMessage(content="merhaba")], "duzeltme_denemesi": 0}
        )
        assert sonuc["duzeltme_denemesi"] == 0

    def test_deneme_siniri_sonsuz_dongude_kilitlenmez(self, sahte):
        inatci = []
        for i in range(12):
            inatci.append(_arac_cagrisi("kontrol_durumu", {"kontrol_id": "YOK"}, cid=str(i)))
            inatci.append(AIMessage(content=f"Uydurma {i}: 40 sunucu."))
        sahte(inatci)

        sonuc = ajan_kur().invoke(
            {"messages": [HumanMessage(content="soru")], "duzeltme_denemesi": 0}
        )
        assert sonuc["duzeltme_denemesi"] == MAKS_DUZELTME
        assert isinstance(sonuc["messages"][-1], AIMessage)

    def test_denemeler_tukenirse_dayanaksiz_isaretlenir(self, sahte):
        inatci = []
        for i in range(12):
            inatci.append(_arac_cagrisi("kontrol_durumu", {"kontrol_id": "YOK"}, cid=str(i)))
            inatci.append(AIMessage(content="40 sunucuda acik."))
        sahte(inatci)

        sonuc = ajan_modulu.sor("soru", dogrula=False)

        assert sonuc["dayanaksiz_cevap"] is True
        assert sonuc["cevap"].startswith("[DAYANAKSIZ CEVAP]")
        assert sonuc["duzeltme_denemesi"] == MAKS_DUZELTME
        assert sonuc["arac_ozeti"]["basarili"] == 0

    def test_duzeltme_sonrasi_basarili_cevap_isaretlenmez(self, sahte):
        sahte(
            [
                _arac_cagrisi("kontrol_durumu", {"kontrol_id": "YOK"}),
                AIMessage(content="Uydurma 40."),
                _arac_cagrisi("kontrol_durumu", {"kontrol_id": "5.2.12"}, cid="2"),
                AIMessage(content="32 sunucuda acik."),
            ]
        )
        sonuc = ajan_modulu.sor("soru", dogrula=False)
        assert sonuc["dayanaksiz_cevap"] is False
        assert sonuc["duzeltme_denemesi"] == 1
        assert sonuc["arac_ozeti"] == {"basarili": 1, "hatali": 1}
