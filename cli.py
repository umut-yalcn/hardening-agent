"""Komut satiri arayuzu.

    python cli.py "DMZ'deki uretim sunucularinda en kritik acik ne?"
"""

from __future__ import annotations

import sys

from src.agent import sor

# Windows konsolu varsayilan olarak cp1254 kullaniyor; Turkce karakterler bozuluyor.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    soru = " ".join(sys.argv[1:])
    sonuc = sor(soru)

    print(f"\nSORU: {sonuc['soru']}\n")
    print(sonuc["cevap"])

    if sonuc.get("duzeltme_denemesi"):
        print(f"\nDuzeltmeye geri gonderme: {sonuc['duzeltme_denemesi']} kez "
              "(dayanaksiz cevap yazmaya kalkisti)")

    ozet = sonuc.get("arac_ozeti", {})
    araclar = sonuc["kullanilan_araclar"]
    print(f"\n--- {len(araclar)} arac cagrisi "
          f"({ozet.get('basarili', 0)} basarili, {ozet.get('hatali', 0)} hatali) ---")
    for a in araclar:
        print(f"  {a['arac']}({a['girdi']})")

    # Deterministik uyarilar, DENETCI MODELDEN ONCE basiliyor: bunlar katalogun
    # kendisine dayaniyor, kota bitse de calisiyorlar. Hesaplanip kullaniciya
    # gosterilmemeleri korumayi etkisiz kiliyordu.
    sayilar = sonuc.get("dogrulanmayan_sayilar") or []
    kimlikler = sonuc.get("katalogda_olmayan_kimlikler") or []
    maddeler = sonuc.get("gecersiz_bddk_maddeleri") or []
    if sayilar or kimlikler or maddeler:
        print("\n--- Dayanak uyarilari (deterministik) ---")
        if kimlikler:
            print(f"  ! Katalogda OLMAYAN kontrol kimligi: {', '.join(kimlikler)}")
        if maddeler:
            print(f"  ! Gecersiz BDDK maddesi: {', '.join(maddeler)}")
        if sayilar:
            print(f"  ! Arac ciktisinda bulunamayan sayi: {', '.join(sayilar)}")

    d = sonuc.get("dogrulama")
    if d:
        durum = {True: "DOGRULANDI", False: "SORUNLU", None: "CALISTIRILAMADI"}[
            d.get("dogrulandi")
        ]
        print(f"\n--- Dogrulama: {durum} ---")
        print(f"  {d.get('gerekce', '')}")
        for sorun in d.get("sorunlar", []):
            print(f"  ! {sorun}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
