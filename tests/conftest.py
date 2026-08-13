"""Testler icin veri hazirligi.

Taze bir klonda `data/` bostur - filo verisi depoya konulmaz, uretilir. README
"pytest tests/ -q" komutunun API anahtari olmadan calistigini soyluyor; bunun
TAZE KLONDA da dogru olmasi gerekiyor.

Onceden oyle degildi: klonlayan biri once pytest calistirdiginda 28 test
VeriYok ile dusuyordu. Veri uretimi artik burada, toplama asamasindan once
yapiliyor.

Uretec tohumlu; var olan veriye dokunulmaz, yalnizca eksikse uretilir.
Sunucu sayisi READMEdeki sayilarla ayni olsun diye 120'de sabit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
SUNUCULAR = KOK / "data" / "sunucular.csv"
SONUCLAR = KOK / "data" / "denetim_sonuclari.csv"
URETEC = KOK / "scripts" / "generate_fleet.py"
SUNUCU_SAYISI = 120


def pytest_configure(config) -> None:
    """Veri yoksa uretir. Testler toplanmadan once calisir."""
    if SUNUCULAR.exists() and SONUCLAR.exists():
        return

    print(f"\n[conftest] Filo verisi bulunamadi, uretiliyor ({SUNUCU_SAYISI} sunucu)")
    sonuc = subprocess.run(
        [sys.executable, str(URETEC), "--hosts", str(SUNUCU_SAYISI)],
        cwd=KOK,
        capture_output=True,
        text=True,
    )
    if sonuc.returncode != 0:
        raise RuntimeError(
            f"Test verisi uretilemedi.\n"
            f"Komut: python scripts/generate_fleet.py --hosts {SUNUCU_SAYISI}\n"
            f"{sonuc.stderr}"
        )
    print(f"[conftest] Hazir: {SUNUCULAR.name}, {SONUCLAR.name}")
