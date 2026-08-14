"""LangGraph tabanli sertlestirme analiz ajani.

Akis:

    soru -> [ajan] --arac cagrisi var mi?--> [araclar] -> [ajan] -> ... -> cevap
                                                                            |
                                                                     [dogrulama]

Ajan hangi araci hangi sirayla cagiracagina kendi karar verir. Cevabi
urettikten sonra ikinci bir model cagrisi, cevaptaki her sayisal iddiayi arac
ciktilariyla karsilastirir. Dogrulama basarisiz olursa bu kullanicidan
gizlenmez; cevabin yanina uyari olarak yazilir.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .catalog import KATALOG_ARACLARI
from .config import dayanikli, get_llm
from .freshness import TAZELIK_ESIGI_GUN, YETERLI_KAPSAM
from .tools import ANALIZ_ARACLARI

TUM_ARACLAR = KATALOG_ARACLARI + ANALIZ_ARACLARI

MAKS_ADIM = 12

SISTEM_PROMPTU = f"""Sen bir banka bilgi guvenligi analistisin. Sunucu filosunun
sertlestirme durumunu inceliyor, sorulari denetim verisine dayanarak yanitliyorsun.

Calisma bicimin:
- Kontrol kimligini bilmiyorsan once kontrol_ara ile ara. Kontrol kimligi UYDURMA.
- Bir sayi soylemeden once onu bir arac cagrisiyla dogrula.
- Birden fazla arac cagrisi gerekiyorsa yap; tek cagriyla yetinme zorunlulugun yok.
- Bulgular celisiyorsa gizleme, celiskiyi soyle.

Bu alanin iki kurali var, ikisi de pazarlik konusu degil:

1) BELIRSIZ SONUC UYUMLU DEGILDIR.
   'notchecked', 'error' ve 'unknown' sonuclari kontrolun gectigi anlamina gelmez;
   durumunun okunamadigi anlamina gelir. Bir uyum orani verirken yaninda mutlaka
   kapsam oranini da ver. Kapsam %{YETERLI_KAPSAM * 100:.0f} altindaysa o sunucu
   ya da grup icin "uyumlu" deme; "hakkinda hukum verilemiyor" de.
   {TAZELIK_ESIGI_GUN} gunden eski denetim verisi de guncel durumu anlatmaz.

2) BULGU SAYISI ONCELIK DEMEK DEGILDIR.
   Izole bir test makinesindeki 40 bulgu, internete acik hassas veri tutan bir
   uretim veritabanindaki 3 bulgudan daha az acil olabilir. Onceliklendirme
   sorulduysa risk_siralamasi aracini kullan; kendi kafandan siralama yapma.

Her bulguyu, ihlal ettigi BDDK maddesine bagla. Bu bir bankacilik denetim
yukumlulugudur; "SSH root girisi acik" demek yerine "SSH root girisi acik -
MADDE 11" de.

Madde BASLIKLARINI uydurma. Bir maddenin basligini yazacaksan arac ciktisinda
gecen basligi aynen kullan. Yanlis duzenleyici atif, eksik atiftan kotudur.

Cevap bicimi:
- Once sonucu soyle. Bulgun ne ise ilk cumlede o olsun.
- Sonra dayanagini ver: hangi kontrol, kac sunucu, hangi oran.
- Kisa tut. Rapor yazmiyorsun, soruyu yanitliyorsun.
- Turkce yaz.
"""

DOGRULAMA_PROMPTU = """Sen bir denetci modelsin. Gorevin, bir analistin verdigi
cevabin arac ciktilariyla desteklenip desteklenmedigini kontrol etmek.

Sana arac ciktilari ve analistin cevabi verilecek. Sunlari kontrol et:
- Cevaptaki her sayi arac ciktilarinda var mi? Uydurulmus sayi var mi?
- Uyum orani verilmisse yaninda kapsam orani da verilmis mi?
- Belirsiz sonuclar yanlislikla "uyumlu" olarak sunulmus mu?
- Kontrol kimlikleri arac ciktilarinda geciyor mu?
- BDDK madde numaralari ve MADDE BASLIKLARI arac ciktilariyla birebir ayni mi?
  Analist bir maddeye arac ciktisinda gecmeyen bir baslik uydurmus olabilir;
  bu ciddi bir hatadir cunku duzenleyici atif yanlis olur.

Yalnizca su JSON'u dondur, baska hicbir sey yazma:
{"dogrulandi": true veya false, "gerekce": "tek cumle", "sorunlar": ["..."]}

Sorun yoksa "sorunlar" bos liste olsun."""


#: Ajan cevabi uydurmaya kalktiginda kac kez geri gonderilecegi.
#: Sinir SART: yoksa arac duser -> model yeniden yazar -> yine duser dongusu olusur.
MAKS_DUZELTME = 2

#: Hata tipine gore ne yapmasi gerektigini SOYLEYEN yonlendirmeler.
#: Genel bir "hatani duzelt" mesaji, somut bir talimat kadar ise yaramiyor.
_DUZELTICI_YONLENDIRME: tuple[tuple[str, str], ...] = (
    (
        "Bilinmeyen kontrol",
        "Kontrol kimligini uydurdun ya da yanlis yazdin. kontrol_ara ile aradigin "
        "konuyu dogal dilde ara, donen kimligi aynen kullanarak TEKRAR cagir.",
    ),
    (
        "Bilinmeyen sunucu",
        "Sunucu kimligi yanlis. sunucu_listesi ile mevcut sunuculari gor ve "
        "dogru host_id ile tekrar dene.",
    ),
    (
        "Gecersiz boyut",
        "Gecerli boyutlardan birini sec ve uyum_kirilimi'ni tekrar cagir.",
    ),
    (
        "Gecersiz madde",
        "BDDK maddesi 11, 13, 14, 15 veya 16 olabilir. Dogru madde ile tekrar dene.",
    ),
    (
        "Internete acik sunucu bulunamadi",
        "Bu filtreyle sonuc yok. Filtreyi kaldirip tekrar dene.",
    ),
)


class SertlestirmeDurumu(MessagesState):
    """Mesajlara ek olarak kac kez duzeltmeye gonderildigini tasir."""

    duzeltme_denemesi: int


def _hata_metinleri(mesajlar: list[Any]) -> list[str]:
    """Arac mesajlarindaki hata aciklamalarini toplar."""
    hatalar = []
    for m in mesajlar:
        if not isinstance(m, ToolMessage):
            continue
        icerik = str(m.content)
        if icerik.startswith(_LANGCHAIN_HATA_ONEKLERI):
            hatalar.append(icerik[:200])
            continue
        try:
            veri = json.loads(icerik)
        except json.JSONDecodeError:
            continue
        if isinstance(veri, dict) and "hata" in veri:
            hatalar.append(str(veri["hata"]))
    return hatalar


def _duzeltici_mesaj(hatalar: list[str]) -> str:
    """Hata tipine gore somut bir yonlendirme uretir."""
    yonergeler = []
    for hata in hatalar:
        for anahtar, yonerge in _DUZELTICI_YONLENDIRME:
            if anahtar in hata and yonerge not in yonergeler:
                yonergeler.append(yonerge)

    if not yonergeler:
        yonergeler.append(
            "Cagrini gozden gecir ve duzelterek tekrar dene; kontrol kimligini "
            "bilmiyorsan once kontrol_ara ile ara."
        )

    return (
        "DUR. Hicbir arac cagrin basarili sonuc dondurmedi, dolayisiyla elinde "
        "hicbir veri yok. Bu durumda sayi veremezsin - verdigin her sayi "
        "uydurma olur.\n\n"
        "Alinan hatalar:\n"
        + "\n".join(f"  - {h}" for h in hatalar[-3:])
        + "\n\nSimdi yapman gereken:\n"
        + "\n".join(f"  - {y}" for y in yonergeler)
        + "\n\nDuzeltemiyorsan cevap uydurma; soruyu neden yanitlayamadigini soyle."
    )


def ajan_kur():
    """Derlenmis LangGraph akisini dondurur."""
    llm = dayanikli(get_llm().bind_tools(TUM_ARACLAR))

    def modeli_cagir(state: SertlestirmeDurumu) -> dict[str, Any]:
        mesajlar = state["messages"]
        # Adim siniri: ajan donguye girerse elindekiyle sonlandirmasini iste.
        if len(mesajlar) > MAKS_ADIM * 2:
            mesajlar = mesajlar + [
                HumanMessage(
                    content="Adim sinirina ulasildi. Simdiye kadar topladigin "
                    "bulgularla cevabini yaz, yeni arac cagrisi yapma."
                )
            ]
            return {"messages": [dayanikli(get_llm()).invoke([SystemMessage(SISTEM_PROMPTU)] + mesajlar)]}

        return {"messages": [llm.invoke([SystemMessage(SISTEM_PROMPTU)] + mesajlar)]}

    def duzeltmeye_gonder(state: SertlestirmeDurumu) -> dict[str, Any]:
        """Ajani, hatayi duzeltip tekrar denemeye zorlar.

        Bu ICSEL bir oz-elestiri degil: modele kendi cevabini degerlendirmesini
        soylemiyoruz. Arac katmaninin urettigi somut, deterministik hata
        mesajini geri veriyoruz. Arastirma bu ayrimda net - modeller dis geri
        bildirim olmadan kendi hatalarini duzeltemiyor, dis geri bildirimle
        duzeltebiliyor. Dogrulama modelinden farki da bu: o bir model cagrisi,
        bu deterministik bir kod yolu.
        """
        hatalar = _hata_metinleri(state["messages"])
        return {
            "messages": [HumanMessage(content=_duzeltici_mesaj(hatalar))],
            "duzeltme_denemesi": state.get("duzeltme_denemesi", 0) + 1,
        }

    def yonlendir(state: SertlestirmeDurumu) -> str:
        mesajlar = state["messages"]
        son = mesajlar[-1]

        # Adim siniri BURADA uygulanir. Onceden yalnizca modele "yeni arac
        # cagirma" deniyordu; model dinlemezse graf arac calistirmaya devam
        # ediyordu. Olculdu: MAKS_ADIM=12 iken 40 arac cagrisi yapildi.
        # Sinir artik dilek degil, yonlendirme karari.
        if len(mesajlar) > MAKS_ADIM * 2 + 4:
            return END

        if getattr(son, "tool_calls", None):
            return "araclar"

        basarili, hatali = _arac_ciktilarini_say(mesajlar)
        deneme = state.get("duzeltme_denemesi", 0)

        if basarili == 0 and deneme < MAKS_DUZELTME:
            # Onceden yalnizca hatali > 0 kosuluna bakiliyordu; ajan HIC arac
            # cagirmadan veri hakkinda iddia yazarsa ne duzeltmeye gonderiliyor
            # ne isaretleniyordu. Selamlama gibi iddiasiz cevaplar engellenmiyor.
            if hatali > 0 or _veri_iddiasi_mi(_metin_cikar(son.content)):
                return "duzeltme"
        return END

    graf = StateGraph(SertlestirmeDurumu)
    graf.add_node("ajan", modeli_cagir)
    graf.add_node("araclar", ToolNode(TUM_ARACLAR))
    graf.add_node("duzeltme", duzeltmeye_gonder)

    graf.add_edge(START, "ajan")
    graf.add_conditional_edges(
        "ajan", yonlendir, {"araclar": "araclar", "duzeltme": "duzeltme", END: END}
    )
    graf.add_edge("araclar", "ajan")
    graf.add_edge("duzeltme", "ajan")

    return graf.compile()


def _metin_cikar(icerik: Any) -> str:
    """Mesaj icerigini duz metne cevirir.

    Saglayicilar icerigi farkli bicimlerde donduruyor: bazilari duz string,
    Gemini 3.x ise blok listesi. Ajan kodu bu farki gormemeli.
    """
    if isinstance(icerik, str):
        return icerik

    if isinstance(icerik, list):
        parcalar: list[str] = []
        for blok in icerik:
            if isinstance(blok, str):
                parcalar.append(blok)
            elif isinstance(blok, dict) and blok.get("type") == "text":
                parcalar.append(blok.get("text", ""))
        if parcalar:
            return "\n".join(p for p in parcalar if p)

    return str(icerik)


#: Modelin "true" yazmasi yaygin bir davranis; JSON booleanı ile string
#: karismasin diye acikca esliyoruz.
_DOGRU_METINLER = {"true", "evet", "yes", "1", "dogru", "doğru"}
_YANLIS_METINLER = {"false", "hayir", "hayır", "no", "0", "yanlis", "yanlış"}


def _dogrulama_semasi(ham: Any) -> dict[str, Any]:
    """Denetci modelin ciktisini guvenli bir bicime oturtur.

    Model JSON uretiyor ama SEMASI garanti degil: `true` yerine `"true"`
    yazabiliyor, sozluk yerine liste dondurebiliyor. Onceden ham cikti oldugu
    gibi dondurulup CLI'da `{True: ..., False: ..., None: ...}[deger]` ile
    eslestiriliyordu; string gelince KeyError, liste gelince AttributeError
    firliyordu - hem de cevap ekrana basildiktan SONRA ve tum kota
    harcanmisken. Canli demoda kotu bir an.
    """
    if not isinstance(ham, dict):
        return {
            "dogrulandi": None,
            "gerekce": f"Denetci model beklenmeyen bicimde yanit verdi ({type(ham).__name__}).",
            "sorunlar": [],
        }

    deger = ham.get("dogrulandi")
    if isinstance(deger, str):
        kucuk = deger.strip().lower()
        deger = True if kucuk in _DOGRU_METINLER else (False if kucuk in _YANLIS_METINLER else None)
    elif isinstance(deger, int):
        # Bazi modeller JSON'da boolean yerine 1/0 yaziyor. Bunu "belirsiz"
        # saymak DOGRULANMIS bir cevabi CLI'da "CALISTIRILAMADI" gosteriyordu.
        deger = bool(deger)
    elif not isinstance(deger, bool):
        deger = None

    sorunlar = ham.get("sorunlar")
    if not isinstance(sorunlar, list):
        sorunlar = [str(sorunlar)] if sorunlar else []

    return {
        "dogrulandi": deger,
        "gerekce": str(ham.get("gerekce", "")),
        "sorunlar": [str(s) for s in sorunlar],
    }


def _dogrula(cevap: str, arac_ciktilari: list[str]) -> dict[str, Any]:
    """Cevabi arac ciktilariyla karsilastirir.

    Ayri bir model cagrisi; ajanin kendi cevabini kendisinin onaylamasi degil.
    """
    if not arac_ciktilari:
        return {
            "dogrulandi": False,
            "gerekce": "Hic arac cagrilmadi; cevap veriye dayanmiyor.",
            "sorunlar": ["arac cagrisi yok"],
        }

    kanit = "\n\n".join(c[:2500] for c in arac_ciktilari[-8:])
    istem = (
        f"ARAC CIKTILARI:\n{kanit}\n\n"
        f"ANALISTIN CEVABI:\n{cevap}\n\n"
        "Yukaridaki talimata gore JSON dondur."
    )

    try:
        yanit = dayanikli(get_llm()).invoke(
            [SystemMessage(DOGRULAMA_PROMPTU), HumanMessage(content=istem)]
        )
        ham = _metin_cikar(yanit.content).strip()
        if ham.startswith("```"):
            ham = ham.split("```")[1].removeprefix("json").strip()
        return _dogrulama_semasi(json.loads(ham))
    except Exception as hata:  # dogrulama coktuyse cevabi bloklamayiz
        return {
            "dogrulandi": None,
            "gerekce": f"Dogrulama calistirilamadi: {type(hata).__name__}",
            "sorunlar": [],
        }


DESTEKSIZ_CEVAP_UYARISI = (
    "[DAYANAKSIZ CEVAP] Bu soruya hicbir arac cagrisi basarili sonuc dondurmedi. "
    "Asagidaki metin veriye dayanmiyor; icindeki sayilara guvenilmemelidir."
)

_SAYI_DESENI = re.compile(r"\d")


#: LangChain, sema dogrulamasi basarisiz oldugunda ya da olmayan bir arac
#: cagrildiginda kendi hata metnini duz string olarak dondurur - bizim
#: {"hata": ...} bicimimizde degil. Bunlar JSON olarak ayristirilamadigi icin
#: onceden BASARILI sayiliyordu; yani ajan yalnizca gecersiz cagrilar yapip
#: cevap uydurdugunda dayanak kontrolu devreye girmiyordu.
_LANGCHAIN_HATA_ONEKLERI = ("Error invoking tool", "Error:")


def _arac_ciktisi_hata_mi(icerik: str) -> bool:
    """Bir arac mesaji hata mi bildiriyor?"""
    if icerik.startswith(_LANGCHAIN_HATA_ONEKLERI):
        return True
    try:
        veri = json.loads(icerik)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(veri, dict) and "hata" in veri


#: Cevaptan sayi cikarmak icin. Binlik ayraci ve ondalik virgul/nokta kabul eder.
_SAYI_YAKALA = re.compile(r"(?<![A-Za-z0-9])-?\d[\d.,]*")
#: Lookbehind sart: "AUTH-9204" gibi kontrol kimliklerinde tire eksi
#: isareti sanilip sayi "-9204" olarak okunuyordu. Dayanak kontrolu o
#: uydurma negatif degeri arac ciktisinda da bulup eslestirdigi icin
#: koruma zayifliyordu.


def _sayilari_cikar(metin: str) -> list[str]:
    """Metindeki sayilari normallestirerek dondurur.

    Bicim farki (1.234,56 / 1234.56 / %12,1) karsilastirmayi bozmasin diye
    ayraclar atilip ondalik noktaya cevrilir.
    """
    sonuc = []
    for ham in _SAYI_YAKALA.findall(metin):
        t = ham.strip(".,")
        if not t:
            continue
        # Turkce bicim: son ayrac ondalik ayracidir
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rindex(",") > t.rindex(".")                 else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", ".") if len(t.split(",")[-1]) != 3 else t.replace(",", "")
        try:
            d = float(t)
        except ValueError:
            continue
        sonuc.append(f"{d:.4f}".rstrip("0").rstrip("."))
    return sonuc


def _dayanakli_degerler(arac_ciktilari: list[str]) -> list[float]:
    """Arac ciktilarinda gecen tum sayilar (yuzde karsiliklariyla birlikte)."""
    degerler: list[float] = []
    for c in arac_ciktilari:
        for s in _sayilari_cikar(c):
            d = float(s)
            degerler.append(d)
            degerler.append(d * 100)   # 0.6811 -> 68.11 olarak da anilabilir
    return degerler


#: Cevapta gecen kontrol kimliklerini yakalar: "5.2.12" (CIS) ve
#: "AUTH-9204" (Lynis) bicimleri.
_KIMLIK_DESENI = re.compile(r"\b(\d+\.\d+\.\d+|[A-Z]{3,5}-\d{4})\b")
#: "MADDE 14" ya da "14. madde" yazimlari.
_MADDE_DESENI = re.compile(r"\bMADDE\s*(\d+)\b|\b(\d+)\.\s*madde\b", re.IGNORECASE)


def _uydurma_atiflar(cevap: str) -> dict[str, list[str]]:
    """Cevaptaki kontrol kimliklerini ve BDDK maddelerini KATALOGA dogrular.

    Sayi dayanagi yalnizca SAYILARI denetliyordu. Oysa projenin iki manshet
    iddiasi "kontrol kimlikleri uydurulmuyor" ve "her bulgu BDDK maddesine
    baglaniyor". Bunlarin tek savunmasi sistem prompt'u ve denetci modeldi;
    README'nin kendi durustluk notu denetci modelin guvenilmez oldugunu ve
    kota bitince hic calismayabilecegini soyluyor. Yanlis duzenleyici atif,
    projenin kendi ifadesiyle eksik atiftan kotudur - o yuzden bu kontrol
    modele degil, katalogun kendisine dayaniyor.
    """
    from .controls import BDDK_BASLIK, KONTROLLER

    gecerli_kimlik = {k.kontrol_id for k in KONTROLLER}
    gecerli_madde = {str(m.value) for m in BDDK_BASLIK}

    kimlikler = sorted({
        k for k in _KIMLIK_DESENI.findall(cevap or "") if k not in gecerli_kimlik
    })
    maddeler = sorted({
        (a or b) for a, b in _MADDE_DESENI.findall(cevap or "")
        if (a or b) not in gecerli_madde
    })
    return {"kontrol_kimlikleri": kimlikler, "bddk_maddeleri": maddeler}


def _dogrulanmayan_sayilar(cevap: str, arac_ciktilari: list[str]) -> list[str]:
    """Cevapta gecip arac ciktilarinda BULUNMAYAN sayilar.

    Bagimsiz denetim su acigi gosterdi: dayanak kontrolu yalnizca "basarili bir
    arac cagrisi var mi" diye bakiyordu. Agent list_columns cagirip ardindan
    "temerrut orani %98,7" dediginde cevap dayanakli sayiliyordu - alakasiz tek
    bir basarili cagri, cevaptaki TUM sayilara sinirsiz dayanak sagliyordu.

    Kontrol deterministik: model cagrisi yok, kota yemiyor, keyfi yanlis pozitif
    uretmiyor. YUVARLAMAYA TOLERANSLI - agent 1403.42'yi "1403" diye yazabilir;
    cevaptaki sayinin ondalik hassasiyetinde eslesme araniyor.

    Turetilmis degerler (agent'in iki sayidan hesapladigi oran) dogal olarak
    eslesmeyebilir; bu yuzden ENGELLEYICI degil RAPORLAYICI.
    """
    if not cevap:
        return []

    dayanak = _dayanakli_degerler(arac_ciktilari)
    dogrulanmayan = []
    for ham in _sayilari_cikar(cevap):
        deger = float(ham)
        if abs(deger) < 10:   # tek/iki haneli sayilar gurultu uretir
            continue
        basamak = len(ham.split(".")[1]) if "." in ham else 0
        # round() ile tam esitlik ASIMETRIKTI: arac 1403.9 dondurdugunde
        # cevaptaki "1404" (yuvarlama) geciyor, "1403" (kirpma) uydurma
        # damgasi yiyordu. Ikisi de mesru yazim; son basamak genisliginde
        # tolerans araniyor.
        tolerans = 10.0 ** (-basamak)
        if any(abs(d - deger) < tolerans for d in dayanak):
            continue
        dogrulanmayan.append(ham)
    return dogrulanmayan


@lru_cache(maxsize=1)
def _alan_sozlugu() -> tuple[str, ...]:
    """Kontrol kimlikleri, kategoriler ve filo terimleri."""
    from .controls import KONTROLLER
    from .fleet import sunucular

    terimler = {k.kontrol_id.lower() for k in KONTROLLER}
    terimler |= {k.kategori for k in KONTROLLER}
    terimler |= {"uyum orani", "kapsam orani", "sunucu", "kontrol", "bulgu", "madde"}
    try:
        s = sunucular()
        for kol in ("ortam", "rol", "ag_bolgesi", "destek_durumu", "veri_siniflandirmasi"):
            terimler |= {str(v).lower() for v in s[kol].unique()}
    except Exception:
        pass
    return tuple(sorted(terimler))


def _veri_iddiasi_mi(cevap: str) -> bool:
    """Cevap, veri hakkinda bir iddia iceriyor mu?

    Selamlama engellenmemeli; ama "en riskli ortam uretimdir" gibi RAKAMSIZ bir
    veri iddiasi da dayanaksiz kalmamali.
    """
    if _SAYI_DESENI.search(cevap):
        return True
    kucuk = cevap.lower()
    for t in _alan_sozlugu():
        kacis = re.escape(t)
        desen = (rf"(?<![a-z0-9ğüşıöç]){kacis}" if len(t) >= 4
                 else rf"(?<![a-z0-9ğüşıöç]){kacis}(?![a-z0-9ğüşıöç])")
        if re.search(desen, kucuk):
            return True
    return False


def _arac_ciktilarini_say(mesajlar: list[Any]) -> tuple[int, int]:
    """Kac arac cagrisinin basarili, kacinin hata dondurdugunu sayar.

    Araclar hatalarini istisna yerine {"hata": ...} olarak donduruyor - ajanin
    plan degistirebilmesi icin. Ama bu, hatanin sessizce yutulabilmesi demek:
    ajan hatayi gorup duzeltmek yerine cevabi uydurabilir. Kardes projede
    gozlenen davranis tam olarak buydu. Dogrulama modeli bunu yakalayabilir
    ama modele bagli; bu sayim deterministik.
    """
    basarili = hatali = 0
    for m in mesajlar:
        if not isinstance(m, ToolMessage):
            continue
        if _arac_ciktisi_hata_mi(str(m.content)):
            hatali += 1
        else:
            basarili += 1
    return basarili, hatali


def sor(soru: str, dogrula: bool = True) -> dict[str, Any]:
    """Bir soruyu uctan uca calistirir."""
    ajan = ajan_kur()
    sonuc = ajan.invoke(
        {"messages": [HumanMessage(content=soru)], "duzeltme_denemesi": 0}
    )

    son: AIMessage = sonuc["messages"][-1]
    cevap = _metin_cikar(son.content)

    arac_cagrilari = [
        {"arac": tc["name"], "girdi": tc["args"]}
        for m in sonuc["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    arac_ciktilari = [
        str(m.content) for m in sonuc["messages"] if isinstance(m, ToolMessage)
    ]

    basarili, hatali = _arac_ciktilarini_say(sonuc["messages"])

    # Kod yolunda dayanak kontrolu. Dogrulama modeli de bunu yakalayabilir ama
    # o bir model cagrisi - dusebilir, yanilabilir, kota nedeniyle kapatilmis
    # olabilir. Bu kontrol her kosulda calisir.
    #
    # Yalnizca "basarili bir arac cagrisi var mi" diye bakmak yetmiyor: alakasiz
    # tek bir basarili cagri, cevaptaki TUM sayilara dayanak saglayabiliyor.
    # O yuzden cevaptaki sayilar arac ciktilariyla karsilastiriliyor.
    dogrulanmayan = _dogrulanmayan_sayilar(cevap, arac_ciktilari)
    cevap_sayilari = [x for x in _sayilari_cikar(cevap) if abs(float(x)) >= 10]
    hicbiri_dayanakli_degil = bool(cevap_sayilari) and len(dogrulanmayan) == len(cevap_sayilari)

    dayanaksiz = (basarili == 0 and _veri_iddiasi_mi(cevap)) or hicbiri_dayanakli_degil
    if dayanaksiz:
        cevap = f"{DESTEKSIZ_CEVAP_UYARISI}\n\n{cevap}"

    uydurma = _uydurma_atiflar(cevap)

    cikti = {
        "soru": soru,
        "cevap": cevap,
        "kullanilan_araclar": arac_cagrilari,
        "adim_sayisi": len(sonuc["messages"]),
        "arac_ozeti": {"basarili": basarili, "hatali": hatali},
        "duzeltme_denemesi": sonuc.get("duzeltme_denemesi", 0),
        "dayanaksiz_cevap": dayanaksiz,
        "dogrulanmayan_sayilar": dogrulanmayan,
        "katalogda_olmayan_kimlikler": uydurma["kontrol_kimlikleri"],
        "gecersiz_bddk_maddeleri": uydurma["bddk_maddeleri"],
    }

    if dogrula:
        cikti["dogrulama"] = _dogrula(cevap, arac_ciktilari)

    return cikti
