"""Analiz katmanini API anahtari OLMADAN gosterir.

Model cagrisi yapmaz; araclari dogrudan calistirir. Projenin iki tezi burada
veri uzerinde gorulebilir:

  - Bulgu sayisiyla siralamak yanlis onceligi one cikarir.
  - Belirsiz sonuclari uyumlu saymak, hakkinda bilgi olmayan sunuculari yesil gosterir.

    python scripts/demo_posture.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.controls import BDDK_BASLIK, BddkMaddesi  # noqa: E402
from src.fleet import birlesik, uyum_ve_kapsam  # noqa: E402
from src.freshness import filo_kapsam_ozeti, yaniltici_temizler  # noqa: E402
from src.scoring import bulgu_siralamasi, sunucu_riski  # noqa: E402


def baslik(no: int, ad: str) -> None:
    print(f"\n{no}. {ad}")
    print("   " + "-" * (len(ad) + 2))


def main() -> None:
    print("=" * 74)
    print("  SERTLESTIRME ANALIZ KATMANI  -  API anahtari gerektirmez")
    print("=" * 74)

    df = birlesik()
    risk = sunucu_riski()

    # ------------------------------------------------------------------
    baslik(1, "FILO OZETI")
    m = uyum_ve_kapsam(df)
    print(f"   Sunucu                : {df['host_id'].nunique()}")
    print(f"   Denetim sonucu        : {len(df)}")
    print(f"   Uyum orani            : %{m['uyum_orani'] * 100:.1f}"
          f"  ({m['uyumlu']}/{m['gozlemlenen_kontrol']} gozlemlenen kontrol)")
    print(f"   Kapsam orani          : %{m['kapsam_orani'] * 100:.1f}"
          f"  ({m['belirsiz_kontrol']} kontrolun durumu okunamadi)")
    print("   NOT: uyum orani yalnizca gozlemlenen sonuclar uzerinden hesaplandi.")

    # ------------------------------------------------------------------
    baslik(2, "ONCELIKLENDIRME  -  bulgu sayisi vs maruziyet")
    print("   Ayni filo, iki farkli siralama.\n")

    print("   (a) HAM BULGU SAYISINA GORE:")
    for r in risk.nlargest(4, "bulgu_sayisi").itertuples():
        print(f"       {r.host_id}  {r.bulgu_sayisi:>3} bulgu   "
              f"{r.ortam:<17} {r.ag_bolgesi:<7} internet={str(r.internet_erisimi):<5}")

    print("\n   (b) MARUZIYET AGIRLIKLI RISKE GORE:")
    for r in risk.head(4).itertuples():
        print(f"       {r.host_id}  {r.bulgu_sayisi:>3} bulgu   "
              f"{r.ortam:<17} {r.ag_bolgesi:<7} internet={str(r.internet_erisimi):<5}"
              f" carpan={r.maruziyet:.1f}  risk={r.toplam_risk:.0f}")

    ust_naive = set(risk.nlargest(4, "bulgu_sayisi")["host_id"])
    ust_risk = set(risk.head(4)["host_id"])
    print(f"\n   Iki listenin ortak sunucusu: {len(ust_naive & ust_risk)}/4")
    print("   Bulgu sayisi saymak, riskin nerede oldugunu soylemiyor.")

    # ------------------------------------------------------------------
    baslik(3, "EN ACIL TEKIL BULGULAR  -  maruziyete gore")
    for b in bulgu_siralamasi(limit=5).itertuples():
        print(f"       {b.host_id}  {b.kontrol_id:<10} MADDE {b.bddk_maddesi}  "
              f"risk={b.kesin_risk:>6.0f}")
        print(f"                  {b.baslik[:66]}")

    # ------------------------------------------------------------------
    baslik(4, "DURUSTLUK  -  bilinmeyen, uyumlu degildir")
    ozet = filo_kapsam_ozeti()
    print(f"   Hakkinda hukum verilebilen sunucu : {ozet['hukum_verilebilir_sunucu']}"
          f"/{ozet['sunucu_sayisi']}")
    print(f"   Kapsami yetersiz                  : {ozet['yetersiz_kapsamli_sunucu']}")
    print(f"   Denetimi bayat (>{ozet['tazelik_esigi_gun']} gun)          : "
          f"{ozet['bayat_denetimli_sunucu']}")
    print(f"   'Temiz gorunen ama bilinmeyen'    : {ozet['yaniltici_temiz_sunucu']}\n")

    print("   Ikili bir uyumluluk panelinde YESIL gorunecek sunucular:")
    for y in yaniltici_temizler().head(4).itertuples():
        print(f"       {y.host_id}  uyum=%{y.uyum_orani * 100:>5.1f}  "
              f"kapsam=%{y.kapsam_orani * 100:>5.1f}   {y.gerekce}")
    print("\n   Bulgu sayilari dusuk oldugu icin degil; kontroller kosturulmadigi icin.")

    # ------------------------------------------------------------------
    baslik(5, "SENARYO  -  denetim ajani bozuk sunucu")
    alt = df[df["host_id"] == "srv-002"]
    d = alt["durum"].value_counts().to_dict()
    print("   srv-002 (uretim / uygulama)")
    print(f"       uyumsuz bulgu sayisi : {d.get('uyumsuz', 0)}")
    print(f"       durumu okunamayan    : {d.get('belirsiz', 0)}")
    print("       Sifir bulgu. Bu sunucu uyumlu DEGIL; hakkinda hicbir sey bilinmiyor.")

    baslik(6, "SENARYO  -  az bulgu, yuksek maruziyet")
    s1 = risk[risk["host_id"] == "srv-001"].iloc[0]
    sira = int(risk.index[risk["host_id"] == "srv-001"][0]) + 1
    naive = risk.sort_values("bulgu_sayisi", ascending=False).reset_index(drop=True)
    sira_naive = int(naive.index[naive["host_id"] == "srv-001"][0]) + 1
    print(f"   srv-001 ({s1['ortam']} / {s1['rol']} / {s1['ag_bolgesi']}, "
          f"internet={s1['internet_erisimi']}, veri={s1['veri_siniflandirmasi']})")
    print(f"       bulgu sayisi     : {s1['bulgu_sayisi']}")
    print(f"       maruziyet carpani: {s1['maruziyet']}")
    print(f"       bulgu sayisi siralamasinda : {sira_naive}. sirada")
    print(f"       risk siralamasinda         : {sira}. sirada")

    # ------------------------------------------------------------------
    baslik(7, "BDDK MADDE BAZINDA UYUM")
    for madde in BddkMaddesi:
        alt = df[df["bddk_maddesi"] == madde.value]
        mm = uyum_ve_kapsam(alt)
        print(f"   MADDE {madde.value:<3} {BDDK_BASLIK[madde]:<38} "
              f"uyum=%{mm['uyum_orani'] * 100:>5.1f}  kapsam=%{mm['kapsam_orani'] * 100:>5.1f}")

    print("\n" + "=" * 74)
    print("  Hicbir sayi model tarafindan uretilmedi. Hepsi denetim verisinden.")
    print("=" * 74)


if __name__ == "__main__":
    main()
