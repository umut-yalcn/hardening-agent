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
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .catalog import KATALOG_ARACLARI
from .config import get_llm
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


def ajan_kur():
    """Derlenmis LangGraph akisini dondurur."""
    llm = get_llm().bind_tools(TUM_ARACLAR)

    def modeli_cagir(state: MessagesState) -> dict[str, Any]:
        mesajlar = state["messages"]
        # Adim siniri: ajan donguye girerse elindekiyle sonlandirmasini iste.
        if len(mesajlar) > MAKS_ADIM * 2:
            mesajlar = mesajlar + [
                HumanMessage(
                    content="Adim sinirina ulasildi. Simdiye kadar topladigin "
                    "bulgularla cevabini yaz, yeni arac cagrisi yapma."
                )
            ]
            return {"messages": [get_llm().invoke([SystemMessage(SISTEM_PROMPTU)] + mesajlar)]}

        return {"messages": [llm.invoke([SystemMessage(SISTEM_PROMPTU)] + mesajlar)]}

    graf = StateGraph(MessagesState)
    graf.add_node("ajan", modeli_cagir)
    graf.add_node("araclar", ToolNode(TUM_ARACLAR))

    graf.add_edge(START, "ajan")
    graf.add_conditional_edges("ajan", tools_condition, {"tools": "araclar", END: END})
    graf.add_edge("araclar", "ajan")

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
        yanit = get_llm().invoke(
            [SystemMessage(DOGRULAMA_PROMPTU), HumanMessage(content=istem)]
        )
        ham = _metin_cikar(yanit.content).strip()
        if ham.startswith("```"):
            ham = ham.split("```")[1].removeprefix("json").strip()
        return json.loads(ham)
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
        icerik = m.content
        veri: Any = icerik
        if isinstance(icerik, str):
            try:
                veri = json.loads(icerik)
            except json.JSONDecodeError:
                veri = icerik
        if isinstance(veri, dict) and "hata" in veri:
            hatali += 1
        else:
            basarili += 1
    return basarili, hatali


def sor(soru: str, dogrula: bool = True) -> dict[str, Any]:
    """Bir soruyu uctan uca calistirir."""
    ajan = ajan_kur()
    sonuc = ajan.invoke({"messages": [HumanMessage(content=soru)]})

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
    # olabilir. Bu sayim her kosulda calisir.
    dayanaksiz = basarili == 0 and bool(_SAYI_DESENI.search(cevap))
    if dayanaksiz:
        cevap = f"{DESTEKSIZ_CEVAP_UYARISI}\n\n{cevap}"

    cikti = {
        "soru": soru,
        "cevap": cevap,
        "kullanilan_araclar": arac_cagrilari,
        "adim_sayisi": len(sonuc["messages"]),
        "arac_ozeti": {"basarili": basarili, "hatali": hatali},
        "dayanaksiz_cevap": dayanaksiz,
    }

    if dogrula:
        cikti["dogrulama"] = _dogrula(cevap, arac_ciktilari)

    return cikti
