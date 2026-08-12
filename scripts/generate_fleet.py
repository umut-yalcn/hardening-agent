"""Sentetik sunucu filosu ve denetim sonucu ureteci.

Veri rastgele degildir. Iki gercek olguyu tasimasi icin kuruldu:

1) Sertlestirme olgunlugu ortamla iliskilidir. Uretim ve kisitli bolge daha iyi
   sertlestirilmis, test ortami daha gevsektir. Yani ham bulgu sayisi test
   ortamini one cikarir - oysa risk orada degildir.

2) Bazi sunucularda denetim ajani bozuktur. Bu sunucular 'fail' uretmedikleri
   icin ikili panellerde TEMIZ gorunur. Gercekte durumlari bilinmiyordur.

Bu iki olgu, projenin iki tezini (maruziyet onceligi ve 'bilinmeyen != uyumlu')
veri uzerinde gosterilebilir kilar.

Kullanim:
    python scripts/generate_fleet.py --hosts 120
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.controls import KONTROLLER, Sonuc  # noqa: E402

VERI_DIZINI = pathlib.Path(__file__).resolve().parents[1] / "data"

ORTAMLAR = ("uretim", "test", "felaket_kurtarma")
ORTAM_AGIRLIK = (0.50, 0.30, 0.20)

ROLLER = ("veritabani", "web", "uygulama", "atlama_sunucusu", "yedekleme")
ROL_AGIRLIK = (0.22, 0.24, 0.30, 0.10, 0.14)

BOLGELER = ("dmz", "ic_ag", "kisitli")

ISLETIM_SISTEMLERI = (
    ("Ubuntu", "22.04 LTS", "destekleniyor"),
    ("Ubuntu", "20.04 LTS", "destekleniyor"),
    ("RHEL", "9", "destekleniyor"),
    ("RHEL", "8", "destekleniyor"),
    ("CentOS", "7", "destegi_bitti"),
    ("Ubuntu", "18.04 LTS", "destegi_bitti"),
)
OS_AGIRLIK = (0.30, 0.24, 0.16, 0.16, 0.09, 0.05)

# Ortalama sertlestirme olgunlugu. Test ortami bilerek dusuk.
ORTAM_OLGUNLUK = {"uretim": 0.74, "felaket_kurtarma": 0.63, "test": 0.42}
BOLGE_OLGUNLUK = {"kisitli": 0.10, "ic_ag": 0.0, "dmz": -0.04}

# Denetim ajani bozuk olan sunucu orani. Bu sunucular 'notchecked' uretir.
BOZUK_AJAN_ORANI = 0.12


def _rol_bolge(rng: np.random.Generator, rol: str) -> str:
    """Rol ile ag bolgesi arasinda gercekci bir iliski kurar."""
    if rol == "web":
        return rng.choice(BOLGELER, p=[0.62, 0.33, 0.05])
    if rol == "veritabani":
        return rng.choice(BOLGELER, p=[0.04, 0.36, 0.60])
    if rol == "atlama_sunucusu":
        return rng.choice(BOLGELER, p=[0.55, 0.40, 0.05])
    if rol == "yedekleme":
        return rng.choice(BOLGELER, p=[0.02, 0.45, 0.53])
    return rng.choice(BOLGELER, p=[0.15, 0.65, 0.20])


def _veri_sinifi(rng: np.random.Generator, rol: str, ortam: str) -> str:
    """BDDK 14. madde hassas verinin ozel agda tutulmasini istiyor."""
    if ortam == "test":
        return rng.choice(("dahili", "genel"), p=[0.7, 0.3])
    if rol in ("veritabani", "yedekleme"):
        return rng.choice(("hassas", "dahili"), p=[0.82, 0.18])
    return rng.choice(("hassas", "dahili", "genel"), p=[0.25, 0.55, 0.20])


def sunuculari_uret(n: int, rng: np.random.Generator) -> pd.DataFrame:
    kayitlar = []
    for i in range(n):
        ortam = rng.choice(ORTAMLAR, p=ORTAM_AGIRLIK)
        rol = rng.choice(ROLLER, p=ROL_AGIRLIK)
        bolge = _rol_bolge(rng, rol)

        os_idx = rng.choice(len(ISLETIM_SISTEMLERI), p=OS_AGIRLIK)
        os_ad, os_surum, destek = ISLETIM_SISTEMLERI[os_idx]

        # DMZ'de olmak internete acik olmayi garanti etmez; guvenlik duvari olabilir.
        if bolge == "dmz":
            internet = bool(rng.random() < 0.78)
        elif bolge == "ic_ag":
            internet = bool(rng.random() < 0.08)
        else:
            internet = False

        # Denetim tazeligi: cogu sunucu duzenli taraniyor, bir kismi unutulmus.
        if rng.random() < 0.18:
            denetim_gun = int(rng.integers(45, 210))
        else:
            denetim_gun = int(rng.integers(0, 30))

        olgunluk = ORTAM_OLGUNLUK[ortam] + BOLGE_OLGUNLUK[bolge]
        olgunluk += float(rng.normal(0, 0.09))
        olgunluk = float(np.clip(olgunluk, 0.05, 0.95))

        kayitlar.append(
            {
                "host_id": f"srv-{i + 1:03d}",
                "ortam": ortam,
                "rol": rol,
                "ag_bolgesi": bolge,
                "internet_erisimi": internet,
                "isletim_sistemi": os_ad,
                "os_surum": os_surum,
                "destek_durumu": destek,
                "veri_siniflandirmasi": _veri_sinifi(rng, rol, ortam),
                "son_denetim_gun_once": denetim_gun,
                "denetim_ajani_saglikli": bool(rng.random() > BOZUK_AJAN_ORANI),
                "olgunluk": olgunluk,
            }
        )
    return pd.DataFrame(kayitlar)


def _uygulanabilir_mi(rng: np.random.Generator, kategori: str, rol: str) -> bool:
    """Her kontrol her sunucuya uygulanmaz; XCCDF bunu notapplicable ile isaretler."""
    if kategori == "depolama" and rol not in ("yedekleme", "veritabani"):
        return rng.random() > 0.55
    if kategori == "donanim" and rol != "atlama_sunucusu":
        return rng.random() > 0.40
    if kategori == "zararli_yazilim" and rol == "atlama_sunucusu":
        return rng.random() > 0.25
    return True


def sonuclari_uret(sunucular: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    satirlar = []
    for host in sunucular.itertuples():
        for k in KONTROLLER:
            if not _uygulanabilir_mi(rng, k.kategori, host.rol):
                sonuc = Sonuc.NOTAPPLICABLE
            elif not host.denetim_ajani_saglikli:
                # Bozuk ajan: cogu kontrol hic kosturulmuyor. 'fail' uretmedigi
                # icin ikili panellerde bu sunucu temiz gorunur.
                r = rng.random()
                sonuc = (
                    Sonuc.NOTCHECKED if r < 0.62
                    else Sonuc.ERROR if r < 0.78
                    else (Sonuc.PASS if rng.random() < host.olgunluk else Sonuc.FAIL)
                )
            else:
                r = rng.random()
                if r < 0.020:
                    sonuc = Sonuc.ERROR
                elif r < 0.035:
                    sonuc = Sonuc.UNKNOWN
                else:
                    # Seviye 2 kontrolleri daha nadir uygulanir; agir kontroller
                    # daha sik ihmal edilir.
                    p_pass = host.olgunluk + (0.14 if k.seviye == 1 else -0.10)
                    p_pass -= 0.015 * (k.agirlik - 5)
                    p_pass = float(np.clip(p_pass, 0.04, 0.97))
                    sonuc = Sonuc.PASS if rng.random() < p_pass else Sonuc.FAIL

            satirlar.append(
                {
                    "host_id": host.host_id,
                    "kontrol_id": k.kontrol_id,
                    "sonuc": sonuc.value,
                    "olcum_gun_once": host.son_denetim_gun_once,
                }
            )
    return pd.DataFrame(satirlar)


def senaryolari_yerlestir(
    sunucular: pd.DataFrame, sonuclar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uc senaryoyu determinist olarak yerlestirir.

    Demo'nun rastgeleligin insafina kalmamasi icin. Uculu de gercek hayatta
    siklikla gorulen durumlar.
    """
    s = sunucular.set_index("host_id")

    # 1) Az bulgusu olan ama son derece maruz bir uretim veritabani.
    #    Ham bulgu sayisiyla siralamada dibe duser, maruziyetle tepeye cikar.
    hedef = "srv-001"
    s.loc[hedef, ["ortam", "rol", "ag_bolgesi", "internet_erisimi"]] = [
        "uretim", "veritabani", "dmz", True,
    ]
    s.loc[hedef, ["veri_siniflandirmasi", "destek_durumu"]] = ["hassas", "destekleniyor"]
    s.loc[hedef, "denetim_ajani_saglikli"] = True
    s.loc[hedef, "son_denetim_gun_once"] = 3
    maske = sonuclar["host_id"] == hedef
    sonuclar.loc[maske, "sonuc"] = Sonuc.PASS.value
    sonuclar.loc[maske, "olcum_gun_once"] = 3
    for kid in ("5.2.12", "FIRE-4590", "PKGS-7392"):
        sonuclar.loc[maske & (sonuclar["kontrol_id"] == kid), "sonuc"] = Sonuc.FAIL.value

    # 2) Denetim ajani bozuk sunucu: 'fail' yok, ama durumu okunamiyor.
    hedef = "srv-002"
    s.loc[hedef, ["ortam", "rol", "ag_bolgesi", "internet_erisimi"]] = [
        "uretim", "uygulama", "ic_ag", False,
    ]
    s.loc[hedef, "denetim_ajani_saglikli"] = False
    s.loc[hedef, "son_denetim_gun_once"] = 6
    maske = sonuclar["host_id"] == hedef
    sonuclar.loc[maske, "sonuc"] = Sonuc.NOTCHECKED.value
    sonuclar.loc[maske, "olcum_gun_once"] = 6

    # 3) Uzun suredir denetlenmemis, destegi bitmis sunucu.
    hedef = "srv-003"
    s.loc[hedef, ["isletim_sistemi", "os_surum", "destek_durumu"]] = [
        "CentOS", "7", "destegi_bitti",
    ]
    s.loc[hedef, ["ortam", "ag_bolgesi", "internet_erisimi"]] = ["uretim", "dmz", True]
    s.loc[hedef, "son_denetim_gun_once"] = 187
    sonuclar.loc[sonuclar["host_id"] == hedef, "olcum_gun_once"] = 187

    return s.reset_index(), sonuclar


def main() -> None:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--hosts", type=int, default=120)
    ayristirici.add_argument("--seed", type=int, default=42)
    args = ayristirici.parse_args()

    rng = np.random.default_rng(args.seed)

    sunucular = sunuculari_uret(args.hosts, rng)
    sonuclar = sonuclari_uret(sunucular, rng)
    sunucular, sonuclar = senaryolari_yerlestir(sunucular, sonuclar)

    sunucular = sunucular.drop(columns=["olgunluk"])

    VERI_DIZINI.mkdir(exist_ok=True)
    sunucular.to_csv(VERI_DIZINI / "sunucular.csv", index=False)
    sonuclar.to_csv(VERI_DIZINI / "denetim_sonuclari.csv", index=False)

    dagilim = sonuclar["sonuc"].value_counts()
    print(f"Sunucu           : {len(sunucular)}")
    print(f"Kontrol          : {len(KONTROLLER)}")
    print(f"Denetim sonucu   : {len(sonuclar)}")
    print("\nSonuc dagilimi:")
    for durum, adet in dagilim.items():
        print(f"  {durum:<15} {adet:>6}  ({adet / len(sonuclar):.1%})")
    print(f"\nDestegi bitmis sunucu : {(sunucular['destek_durumu'] == 'destegi_bitti').sum()}")
    print(f"Internete acik sunucu : {sunucular['internet_erisimi'].sum()}")
    print(f"30 gunden bayat       : {(sunucular['son_denetim_gun_once'] > 30).sum()}")
    print(f"\nYazildi: {VERI_DIZINI / 'sunucular.csv'}")
    print(f"Yazildi: {VERI_DIZINI / 'denetim_sonuclari.csv'}")


if __name__ == "__main__":
    main()
