"""Sertlestirme kontrol katalogu ve BDDK eslemesi.

Tek gercek kaynak. Filo verisi de, ajan araclari da bu katalogu okur.

Kontrol kimlikleri UYDURULMAMISTIR:
  - CIS 5.2.x kimlikleri ve basliklari CIS Ubuntu Linux Benchmark'tan birebir alindi.
  - Lynis test kimlikleri (AUTH-9204, SSH-7408, ...) CISOfy/lynis tests.db'den alindi.

Dogrulanmis basligi olmayan bolumler icin CIS numarasi uydurmak yerine Lynis
kimligi kullanildi; boylece her satir disaridan kontrol edilebilir kalir.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Sonuc(str, Enum):
    """XCCDF kural sonuc durumlari.

    Paneller bunlari genellikle gecti/kaldi ikilisine indirger ve 'notchecked'i
    yesil gosterir. Standart bu ayrimi yapiyor; biz de yapiyoruz.
    """

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    UNKNOWN = "unknown"
    NOTCHECKED = "notchecked"
    NOTAPPLICABLE = "notapplicable"


#: Kontrolun gercekten saglandigi tek durum.
KESIN_UYUMLU = frozenset({Sonuc.PASS})

#: Kontrolun saglanmadigi kesin olan durum.
KESIN_UYUMSUZ = frozenset({Sonuc.FAIL})

#: Durumu OKUNAMAYAN sonuclar. Bunlar uyumlu DEGILDIR; bilinmiyordur.
BELIRSIZ = frozenset({Sonuc.ERROR, Sonuc.UNKNOWN, Sonuc.NOTCHECKED})

#: Sunucuya uygulanamayan kontrol. Paydaya girmez.
KAPSAM_DISI = frozenset({Sonuc.NOTAPPLICABLE})


class BddkMaddesi(int, Enum):
    """Bankalarin Bilgi Sistemleri Yonetmeligi (yururluk 2020-07-01) maddeleri."""

    KIMLIK_ERISIM = 11
    IZ_KAYITLARI = 13
    AG_GUVENLIGI = 14
    GUVENLIK_YAPILANDIRMASI = 15
    ACIK_YAMA = 16


BDDK_BASLIK = {
    BddkMaddesi.KIMLIK_ERISIM: "Kimlik dogrulama ve islem guvenligi",
    BddkMaddesi.IZ_KAYITLARI: "Iz kayitlari",
    BddkMaddesi.AG_GUVENLIGI: "Ag guvenligi",
    BddkMaddesi.GUVENLIK_YAPILANDIRMASI: "Guvenlik yapilandirmasi yonetimi",
    BddkMaddesi.ACIK_YAMA: "Guvenlik aciklari ve yama yonetimi",
}


@dataclass(frozen=True)
class Kontrol:
    kontrol_id: str
    baslik: str
    aciklama: str
    kategori: str
    seviye: int          # CIS profil seviyesi: 1 temel, 2 derinlemesine savunma
    bddk: BddkMaddesi
    agirlik: int         # 1-10, kontrolun kendi basina tasidigi risk
    kaynak: str          # "CIS" | "Lynis"

    @property
    def bddk_etiketi(self) -> str:
        return f"MADDE {self.bddk.value} - {BDDK_BASLIK[self.bddk]}"


def _c(kontrol_id, baslik, aciklama, kategori, seviye, bddk, agirlik, kaynak):
    return Kontrol(kontrol_id, baslik, aciklama, kategori, seviye, bddk, agirlik, kaynak)


# --- SSH: CIS Ubuntu Linux Benchmark bolum 5.2, basliklar birebir -------------
_SSH = (
    _c("5.2.2", "Ensure SSH LogLevel is appropriate",
       "SSH gunluk seviyesi yetersizse yetkisiz erisim denemesi iz birakmaz.",
       "ssh", 1, BddkMaddesi.IZ_KAYITLARI, 5, "CIS"),
    _c("5.2.3", "Ensure SSH X11Forwarding is disabled",
       "X11 yonlendirme acikken oturum uzerinden goruntu trafigi tasinabilir.",
       "ssh", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 4, "CIS"),
    _c("5.2.4", "Ensure SSH X11UseLocalhost is enabled",
       "X11 soketi yerel arayuze baglanmazsa aga acilir.",
       "ssh", 2, BddkMaddesi.AG_GUVENLIGI, 3, "CIS"),
    _c("5.2.5", "Ensure SSH MaxAuthTries is set to 3 or less",
       "Deneme siniri yoksa parola deneme saldirisi maliyetsiz hale gelir. "
       "BDDK 11. madde basarisiz giris denemelerinin sinirlanmasini istiyor.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 7, "CIS"),
    _c("5.2.6", "Ensure SSH MaxSessions is set to 10 or less",
       "Tek baglanti uzerinden acilabilecek oturum sayisi sinirlanmali.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 4, "CIS"),
    _c("5.2.7", "Ensure SSH MaxStartups is configured",
       "Kimlik dogrulanmamis eszamanli baglanti sayisi sinirlanmali.",
       "ssh", 1, BddkMaddesi.AG_GUVENLIGI, 4, "CIS"),
    _c("5.2.9", "Ensure SSH PermitEmptyPasswords is disabled",
       "Bos parolayla SSH girisi kimlik dogrulamayi tamamen gecersiz kilar.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 9, "CIS"),
    _c("5.2.10", "Ensure SSH PermitUserEnvironment is disabled",
       "Kullanici ortam degiskeni enjekte ederek yetki yukseltebilir.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 6, "CIS"),
    _c("5.2.11", "Ensure SSH IgnoreRhosts is enabled",
       "Rhosts tabanli guven iliskisi kimlik dogrulamayi atlatir.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 6, "CIS"),
    _c("5.2.12", "Ensure SSH PermitRootLogin is disabled",
       "Dogrudan root girisi hem sorumlulugu izlenemez kilar hem de tek adimda "
       "tam yetki verir. BDDK 11. madde gorevler ayriligini zorunlu tutuyor.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 10, "CIS"),
    _c("5.2.13", "Ensure SSH HostbasedAuthentication is disabled",
       "Makine tabanli guven, kullanici kimligini dogrulamadan erisim verir.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 6, "CIS"),
    _c("5.2.14", "Ensure only strong Ciphers are used",
       "Zayif sifreleme algoritmalari oturum trafigini cozulebilir kilar.",
       "ssh", 1, BddkMaddesi.AG_GUVENLIGI, 7, "CIS"),
    _c("5.2.15", "Ensure only strong MAC algorithms are used",
       "Zayif MAC algoritmasi trafik butunlugunu garanti etmez.",
       "ssh", 1, BddkMaddesi.AG_GUVENLIGI, 6, "CIS"),
    _c("5.2.16", "Ensure only strong Key Exchange algorithms are used",
       "Zayif anahtar degisimi oturum anahtarini ele gecirilebilir kilar.",
       "ssh", 1, BddkMaddesi.AG_GUVENLIGI, 7, "CIS"),
    _c("5.2.17", "Ensure SSH LoginGraceTime is configured",
       "Uzun bekleme suresi acik ama dogrulanmamis baglantilari biriktirir.",
       "ssh", 1, BddkMaddesi.AG_GUVENLIGI, 4, "CIS"),
    _c("5.2.18", "Ensure SSH ClientAliveInterval and ClientAliveCountMax are configured",
       "Bosta kalan oturum kapatilmazsa acik terminal devralinabilir. "
       "BDDK 11. madde hareketsiz oturumlarin sonlandirilmasini istiyor.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 6, "CIS"),
    _c("5.2.23", "Ensure SSH AllowTcpForwarding is disabled",
       "TCP yonlendirme, sunucuyu ic aga acilan bir tunele cevirir.",
       "ssh", 2, BddkMaddesi.AG_GUVENLIGI, 7, "CIS"),
    _c("5.2.24", "Ensure SSH AllowAgentForwarding is disabled",
       "Ajan yonlendirme, ele gecirilen sunucudan anahtarin kullanilmasina izin verir.",
       "ssh", 2, BddkMaddesi.KIMLIK_ERISIM, 6, "CIS"),
    _c("5.2.25", "Ensure SSH AllowStreamLocalForwarding is disabled",
       "Soket yonlendirme yerel servisleri disari acar.",
       "ssh", 2, BddkMaddesi.AG_GUVENLIGI, 5, "CIS"),
)

# --- Lynis test kimlikleri: CISOfy/lynis tests.db ----------------------------
_LYNIS = (
    # Kimlik ve erisim -> MADDE 11
    _c("AUTH-9204", "Check users with an UID of zero",
       "UID 0 tasiyan ikinci bir hesap, root esdegeri gizli yetki demektir.",
       "kimlik", 1, BddkMaddesi.KIMLIK_ERISIM, 9, "Lynis"),
    _c("AUTH-9283", "Detect accounts lacking passwords",
       "Parolasiz hesap kimlik dogrulamayi tamamen devre disi birakir.",
       "kimlik", 1, BddkMaddesi.KIMLIK_ERISIM, 10, "Lynis"),
    _c("AUTH-9229", "Verify password hashing algorithms",
       "Zayif ozet algoritmasi calinan parola veritabanini cozulebilir kilar.",
       "kimlik", 1, BddkMaddesi.KIMLIK_ERISIM, 8, "Lynis"),
    _c("AUTH-9286", "Review user password aging settings",
       "Parola yaslandirma tanimli degilse ayricalikli hesaplar suresiz kalir.",
       "kimlik", 1, BddkMaddesi.KIMLIK_ERISIM, 5, "Lynis"),
    _c("AUTH-9228", "Check password file consistency with pwck",
       "Bozuk parola dosyasi hesap yonetimini ongorulemez kilar.",
       "kimlik", 2, BddkMaddesi.KIMLIK_ERISIM, 4, "Lynis"),
    _c("AUTH-9328", "Check default umask values",
       "Genis umask yeni dosyalari varsayilan olarak okunabilir birakir.",
       "kimlik", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("SSH-7440", "Validate AllowUsers and AllowGroups restrictions",
       "SSH erisimi kullanici/grup bazinda kisitlanmamissa herkes deneyebilir.",
       "ssh", 1, BddkMaddesi.KIMLIK_ERISIM, 6, "Lynis"),
    _c("SSH-7402", "Check for running SSH daemon",
       "SSH servisinin gerekmedigi sunucuda calismasi gereksiz yuzey acar.",
       "ssh", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 4, "Lynis"),

    # Iz kayitlari -> MADDE 13
    _c("ACCT-9628", "Check for auditd",
       "auditd yoksa sistem cagrisi duzeyinde iz kaydi uretilmez. BDDK 13. madde "
       "hassas veri erisim kayitlarinin 5 yil saklanmasini zorunlu kiliyor.",
       "denetim", 1, BddkMaddesi.IZ_KAYITLARI, 9, "Lynis"),
    _c("ACCT-9630", "Check for auditd rules",
       "auditd kural olmadan calisiyorsa kayit uretmez; servis ayakta gorunur.",
       "denetim", 1, BddkMaddesi.IZ_KAYITLARI, 8, "Lynis"),
    _c("ACCT-9622", "Review Linux accounting capabilities",
       "Surec muhasebesi kapaliysa kullanici eylemleri geriye donuk izlenemez.",
       "denetim", 2, BddkMaddesi.IZ_KAYITLARI, 5, "Lynis"),
    _c("LOGG-2130", "Check for running syslog daemon",
       "Merkezi gunlukleme servisi yoksa kayitlar uretilmez.",
       "gunlukleme", 1, BddkMaddesi.IZ_KAYITLARI, 8, "Lynis"),
    _c("LOGG-2230", "Check for running RSyslog daemon",
       "RSyslog kapaliysa gunluk aktarimi durur.",
       "gunlukleme", 1, BddkMaddesi.IZ_KAYITLARI, 6, "Lynis"),
    _c("LOGG-2152", "Checking loghost",
       "Gunlukler uzak bir toplayiciya gonderilmiyorsa sunucu ele gecirildiginde "
       "iz kayitlari da saldirganin elindedir. Butunluk garantisi kalmaz.",
       "gunlukleme", 1, BddkMaddesi.IZ_KAYITLARI, 9, "Lynis"),
    _c("LOGG-2146", "Checking logrotate.conf and logrotate.d",
       "Dondurme yapilandirmasi yoksa gunlukler ya diski doldurur ya da kaybolur.",
       "gunlukleme", 1, BddkMaddesi.IZ_KAYITLARI, 5, "Lynis"),
    _c("TIME-3104", "Check for running NTP daemon or client",
       "Saat senkron degilse iz kayitlarinin zaman damgasi delil degeri tasimaz.",
       "zaman", 1, BddkMaddesi.IZ_KAYITLARI, 7, "Lynis"),
    _c("TIME-3170", "Check configuration files",
       "NTP yapilandirmasi eksikse saat kaymasi fark edilmez.",
       "zaman", 2, BddkMaddesi.IZ_KAYITLARI, 4, "Lynis"),

    # Ag guvenligi -> MADDE 14
    _c("FIRE-4590", "Check firewall status",
       "Guvenlik duvari kapaliysa ag katmani savunmasi yoktur. BDDK 14. madde "
       "katmanli ag mimarisi ve guvenlik duvari kullanimini zorunlu kiliyor.",
       "guvenlik_duvari", 1, BddkMaddesi.AG_GUVENLIGI, 10, "Lynis"),
    _c("FIRE-4512", "Check iptables for empty ruleset",
       "Kural seti bos bir guvenlik duvari calisiyor gorunur ama hicbir sey filtrelemez.",
       "guvenlik_duvari", 1, BddkMaddesi.AG_GUVENLIGI, 9, "Lynis"),
    _c("FIRE-4502", "Check iptables kernel module",
       "Modul yuklenmemisse kurallar uygulanmaz.",
       "guvenlik_duvari", 2, BddkMaddesi.AG_GUVENLIGI, 6, "Lynis"),
    _c("FIRE-4536", "Assess nftables status",
       "nftables yapilandirmasi tutarsizsa filtreleme ongorulemez.",
       "guvenlik_duvari", 2, BddkMaddesi.AG_GUVENLIGI, 5, "Lynis"),
    _c("NETW-3012", "Check listening ports",
       "Gereksiz dinleyen port dogrudan saldiri yuzeyidir. BDDK 15. madde yalnizca "
       "ihtiyac duyulan servislerin portlarinin acik olmasini istiyor.",
       "ag", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 8, "Lynis"),
    _c("NETW-3200", "Determine available network protocols",
       "Kullanilmayan protokoller devre disi birakilmali.",
       "ag", 2, BddkMaddesi.AG_GUVENLIGI, 5, "Lynis"),
    _c("NETW-3004", "Search available network interfaces",
       "Beklenmeyen arayuz, belgelenmemis bir ag yolu anlamina gelir.",
       "ag", 2, BddkMaddesi.AG_GUVENLIGI, 5, "Lynis"),
    _c("STRG-1926", "Checking NFS exports",
       "Genis NFS paylasimi dosya sistemini aga acar.",
       "depolama", 1, BddkMaddesi.AG_GUVENLIGI, 7, "Lynis"),
    _c("STRG-1920", "Checking NFS daemon",
       "Gereksiz NFS servisi calisiyorsa kapatilmali.",
       "depolama", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 4, "Lynis"),

    # Guvenlik yapilandirmasi -> MADDE 15
    _c("KRNL-6000", "Check sysctl key pairs in scan profile",
       "Cekirdek parametreleri sertlestirilmemisse ag yigini varsayilan halde kalir.",
       "cekirdek", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 7, "Lynis"),
    _c("KRNL-5820", "Checking core dumps configuration",
       "Core dump acikken bellek icerigi diske yazilir; parola ve anahtar sizabilir.",
       "cekirdek", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 6, "Lynis"),
    _c("KRNL-5726", "Checking Linux loaded kernel modules",
       "Beklenmeyen cekirdek modulu, denetlenmemis kod calisiyor demektir.",
       "cekirdek", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 6, "Lynis"),
    _c("FILE-6310", "Checking /tmp, /home and /var directory",
       "Ayri bolume alinmamis dizinler dolum ve yetki sorunlari yaratir.",
       "dosya_sistemi", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("FILE-6362", "Verify /tmp sticky bit protection",
       "Sticky bit yoksa kullanicilar birbirinin gecici dosyalarini silebilir.",
       "dosya_sistemi", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("FILE-7524", "Perform file permissions check",
       "Genis dosya izinleri yetki yukseltmenin en yaygin yoludur.",
       "dosya_sistemi", 1, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 7, "Lynis"),
    _c("HRDN-7220", "Check if one or more compilers are installed",
       "Uretim sunucusunda derleyici, saldirganin arac uretmesini kolaylastirir.",
       "sertlestirme", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("HRDN-7230", "Check for malware scanner",
       "Zararli yazilim taramasi yoksa bulasma fark edilmez.",
       "sertlestirme", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("MALW-3290", "Presence of malware scanner",
       "BDDK 15. madde uygulama beyaz listesi ve zararli yazilim korumasi istiyor.",
       "zararli_yazilim", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 5, "Lynis"),
    _c("BOOT-5121", "Check for GRUB boot loader presence",
       "Onyukleyici parolasiz ise fiziksel erisimde tek adimda root alinir.",
       "onyukleme", 2, BddkMaddesi.KIMLIK_ERISIM, 6, "Lynis"),
    _c("BOOT-5264", "Execute systemd-analyze security check",
       "Servislerin yalitim ayarlari zayifsa tek servis ihlali tum sisteme yayilir.",
       "onyukleme", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 6, "Lynis"),
    _c("USB-3000", "Check for presence of USBGuard",
       "USB cihaz kontrolu yoksa fiziksel erisimle veri cikarilabilir.",
       "donanim", 2, BddkMaddesi.GUVENLIK_YAPILANDIRMASI, 4, "Lynis"),

    # Acik ve yama yonetimi -> MADDE 16
    _c("PKGS-7392", "Review Debian/Ubuntu security updates",
       "Uygulanmamis guvenlik guncellemesi bilinen aciklari acik birakir. BDDK 16. "
       "madde duzenli zafiyet taramasi ve yama yonetimi zorunlu kiliyor.",
       "yama", 1, BddkMaddesi.ACIK_YAMA, 10, "Lynis"),
    _c("PKGS-7410", "Count installed kernel packages",
       "Eski cekirdek paketleri sistemde kalirsa yanlislikla onyuklenebilir.",
       "yama", 2, BddkMaddesi.ACIK_YAMA, 4, "Lynis"),
    _c("PKGS-7345", "Querying dpkg",
       "Paket envanteri cikarilamiyorsa yama durumu degerlendirilemez. "
       "BDDK 16. madde yama yonetimi istiyor; envanter cikmadan yama "
       "durumu degerlendirilemez.",
       "yama", 1, BddkMaddesi.ACIK_YAMA, 6, "Lynis"),
    _c("KRNL-5695", "Determine Linux kernel version and release",
       "Cekirdek surumu bilinmeden zafiyet eslemesi yapilamaz.",
       "yama", 1, BddkMaddesi.ACIK_YAMA, 5, "Lynis"),
)

KONTROLLER: tuple[Kontrol, ...] = _SSH + _LYNIS

KONTROL_INDEKS: dict[str, Kontrol] = {k.kontrol_id: k for k in KONTROLLER}

def kontrol_getir(kontrol_id: str) -> Kontrol:
    """Kontrolu kimligine gore dondurur."""
    try:
        return KONTROL_INDEKS[kontrol_id]
    except KeyError:
        raise KeyError(
            f"Bilinmeyen kontrol: '{kontrol_id}'. Katalogda {len(KONTROLLER)} kontrol var."
        ) from None


def madde_kontrolleri(madde: BddkMaddesi) -> tuple[Kontrol, ...]:
    """Bir BDDK maddesine bagli kontrolleri dondurur."""
    return tuple(k for k in KONTROLLER if k.bddk is madde)
