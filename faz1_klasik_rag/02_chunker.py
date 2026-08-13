# -*- coding: utf-8 -*-
"""
Task 1.2 - Chunking
--------------------
Bu script, 01_pdf_loader.py ile cikardigimiz ham metni anlamli parcalara
(chunk) boler. Iki farkli yontemi karsilastirmali olarak gosterir:

1) Karakter-Bazli Splitter (basit, format bilmez)
2) Madde-Bazli (Header-Aware) Chunking (akilli, KVKK gibi metinler icin) -
   uzun maddeler alt-chunklara bolunur, her chunk'a madde_no + chunk_index +
   dogru sayfa_no metadatasi eklenir.

Kullanim:
    python 02_chunker.py ../data/documents/ornek.pdf
"""

import sys
import os
import re
import json
import importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# --- 01_pdf_loader.py'yi import ediyoruz ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_loader_path = os.path.join(_current_dir, "01_pdf_loader.py")
_spec = importlib.util.spec_from_file_location("pdf_loader_module", _loader_path)
_pdf_loader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pdf_loader_module)

pdf_metnini_cikar = _pdf_loader_module.pdf_metnini_cikar
PdfOkumaHatasi = _pdf_loader_module.PdfOkumaHatasi


# ============================================================
# Turkce'ye ozgu karakterleri \u kacis dizileriyle tanimliyoruz.
# Boylece dosya hangi ortamdan gecerse gecsin (editor, Windows konsolu,
# kopyala-yapistir) bu karakterler ASCII kacis kodlari oldugu icin
# BOZULAMAZ - encoding sorunlarina karsi en saglam yontem budur.
# ============================================================
_C_CEDILLA = "\u00c7"   # Ç
_I_DOTTED = "\u0130"    # İ
_S_CEDILLA = "\u015e"   # Ş
_EN_DASH = "\u2013"     # –

GECICI_KELIMESI = f"GE{_C_CEDILLA}{_I_DOTTED}C{_I_DOTTED}"          # "GEÇİCİ"
BASLIK_GIRIS_ETIKETI = f"BA{_S_CEDILLA}LIK/G{_I_DOTTED}R{_I_DOTTED}{_S_CEDILLA}"  # "BAŞLIK/GİRİŞ"


# ============================================================
# Sayfa haritasi: hangi karakter araligi hangi sayfaya ait
# ============================================================

def tam_metin_ve_sayfa_haritasi_olustur(sayfalar: list[dict]):
    tam_metin_parcalari = []
    sayfa_araliklari = []  # [(baslangic, bitis, sayfa_no), ...]
    imlec = 0

    for sayfa in sayfalar:
        parca = sayfa["metin"]
        baslangic = imlec
        tam_metin_parcalari.append(parca)
        imlec += len(parca)
        tam_metin_parcalari.append("\n")
        imlec += 1
        sayfa_araliklari.append((baslangic, imlec, sayfa["sayfa_no"]))

    return "".join(tam_metin_parcalari), sayfa_araliklari


def sayfa_bul(offset: int, sayfa_araliklari: list[tuple]) -> int:
    for baslangic, bitis, sayfa_no in sayfa_araliklari:
        if baslangic <= offset < bitis:
            return sayfa_no
    return sayfa_araliklari[-1][2] if sayfa_araliklari else 1


# ============================================================
# YONTEM 1: Karakter-Bazli Splitter (offset takipli)
# ============================================================

def kapanis_baslik_temizle(metin: str) -> tuple:
    """
    KVKK gibi Turkce kanun metinlerinde her maddenin BASLIGI, "MADDE N"
    ifadesinden ONCE gelir (orn. "Veri güvenliğine ilişkin yükümlülükler
    \nMADDE 12- ..."). Bizim madde-bazli chunking'imiz "MADDE N" ifadesinden
    tam olarak boldugu icin, N'in basligi yanlislikla N-1'in chunk'inin
    SONUNA yapisip kaliyordu (orn. MADDE 12'nin basligi MADDE 11'in
    sonuna sizmisti).

    Bu fonksiyon, bir chunk'in EN SONUNDAKI, kisa ve noktalama isareti
    ile bitmeyen satir(lar)i (baslik olma ihtimali yuksek) kirpar.
    Sinirlari (offsetleri) DEGISTIRMEDEN, sadece metnin sonundaki
    "gurultuyu" temizler - bu, sinir kaydirma yontemine gore çok daha
    az riskli çunku BASLIK/GIRIS gibi diger chunklarin yapisini bozma
    ihtimali yok (guvenlik kontrolu asagida).

    DUZELTME (gercek testte bulunan hata - MADDE 11 ornegi): Onceden bu
    fonksiyon kirpilan basligi SADECE SILIYORDU - "İlgili kişinin
    hakları" gibi, SONRAKI maddenin (MADDE 11) EN GUCLU semantik ipucu
    olan bir baslik, hem MADDE 10'un sonundan silinip hem MADDE 11'e
    HIC eklenmeden TAMAMEN KAYBOLUYORDU. Bu da MADDE 11'in "İlgili
    kişinin hakları nelerdir?" gibi bir soruya embedding aramasinda
    ZAYIF eslesmesine (baska, konuyla daha az ilgili maddelerin bile
    onune gecmesine) yol aciyordu.

    Simdi bu fonksiyon, kirpilan metni de (temizlenmis_metin, kirpilan_baslik)
    ikilisi olarak dondurur - cagiran kod (madde_bazli_split) bu basligi
    ATMAK yerine SONRAKI maddenin basina EKLEMELI.
    """
    orijinal = metin
    satirlar = metin.split("\n")
    kirpilan_satirlar = []

    while len(satirlar) >= 2:
        son_satir = satirlar[-1].strip()
        if son_satir and len(son_satir) < 80 and son_satir[-1] not in ",.;:":
            kirpilan_satirlar.insert(0, satirlar.pop())
        else:
            break

    sonuc = "\n".join(satirlar).rstrip()
    kirpilan_baslik = "\n".join(kirpilan_satirlar).strip()

    # GUVENLIK KONTROLU: Eger kirpma sonucu chunk neredeyse tamamen
    # bosaldiysa (orn. BASLIK/GIRIS chunk'inda oldugu gibi, dokuman
    # basligi + BOLUM basligi + madde basligi UST USTE gelip hicbiri
    # noktalama icermedigi icin TUMU kirpilabilir), bu buyuk ihtimalle
    # GERCEK icerigi de yanlislikla sildigimiz anlamina gelir - bu
    # durumda hicbir sey yapmadan ORIJINALI geri donuyoruz (kirpilan
    # baslik da bos donuyor, cunku aslinda hicbir sey kirpilmamis sayilir).
    if len(sonuc) < 20:
        return orijinal, ""

    return sonuc, kirpilan_baslik


def karakter_bazli_split_ofsetli(metin: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Metni sabit karakter uzunluguna gore boler. Her donen chunk, girdi
    metni icindeki gercek baslangic ofsetini de tasir - boylece bu
    fonksiyon bir madde'nin alt-parcalarini bolmek icin kullanildiginda,
    her alt-parcanin gercekte hangi sayfada basladigini hesaplayabiliriz.

    Donen deger: [{"metin": str, "ofset": int}, ...]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size pozitif olmali")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap, 0 ile chunk_size arasinda olmali")

    # Her paragrafin (bos olmayan satirin) metin icindeki gercek
    # baslangic ofsetini hesapla
    paragraf_bilgileri = []  # (paragraf_metni, baslangic_ofset)
    imlec = 0
    for satir in metin.split("\n"):
        temiz = satir.strip()
        if temiz:
            ic_ofset = satir.find(temiz)
            paragraf_bilgileri.append((temiz, imlec + ic_ofset))
        imlec += len(satir) + 1  # +1: split ile kaybolan "\n"

    def kelime_sinirindan_kes(uzun_metin: str, kesim_noktasi: int) -> int:
        """Kesim noktasindan once en yakin bosluga gider (kelimeyi bolmemek icin)."""
        bosluk_konumu = uzun_metin.rfind(" ", 0, kesim_noktasi)
        if bosluk_konumu <= 0:
            return kesim_noktasi  # bosluk yok, care yapilamiyor (nadir)
        return bosluk_konumu

    def overlap_metnini_al(metin: str, uzunluk: int) -> str:
        """
        Metnin son 'uzunluk' karakterini alir ama bir kelimenin
        ortasindan baslamamasi icin ilk boslugu bulup ondan sonrasini
        dondurur (orn. "bilgilendirilmeye" -> "ilmeye" degil, tam kelime).
        """
        if not metin or uzunluk <= 0:
            return ""
        ham = metin[-uzunluk:]
        ilk_bosluk = ham.find(" ")
        if ilk_bosluk == -1:
            return ""  # tek uzun kelime - overlap eklemeye degmez
        return ham[ilk_bosluk + 1:]

    chunklar = []
    mevcut_chunk = ""
    mevcut_chunk_ofset = None

    for paragraf, paragraf_ofset in paragraf_bilgileri:
        if len(mevcut_chunk) + len(paragraf) + 1 <= chunk_size:
            if mevcut_chunk_ofset is None:
                mevcut_chunk_ofset = paragraf_ofset
            mevcut_chunk = (mevcut_chunk + "\n" + paragraf).strip() if mevcut_chunk else paragraf
            continue

        if mevcut_chunk:
            chunklar.append({"metin": mevcut_chunk, "ofset": mevcut_chunk_ofset})

        overlap_metni = overlap_metnini_al(mevcut_chunk, chunk_overlap)
        # KRITIK DUZELTME: Overlap metnini yeni paragrafla BOSLUKLA degil
        # YENI SATIRLA birlestiriyoruz. Nedeni: eger overlap tam olarak bir
        # bendin ("d) ...") sonuna denk gelirse ve sonraki paragraf bir
        # sonraki bendin ("e) ...") basiysa, bunlari boslukla birlestirmek
        # ikisini TEK SATIRA yapistiriyordu - bu da bentleri_cikar
        # fonksiyonunun (06_llm_answer.py, satir-bazli calisir) "e)"yi yeni
        # bir bent olarak degil, "d)"nin devami olarak algilamasina yol
        # aciyordu. Yeni satirla birlestirmek, orijinal PDF'teki satir
        # yapisini koruyor.
        #
        # OFSET DUZELTMESI (gercek testte bulunan hata): overlap_metni,
        # ESKI mevcut_chunk'in SONUNDAN alinan bir parca - yani metnin
        # paragraf_ofset'ten DAHA ERKEN bir noktasindan basliyor. Onceden
        # burada mevcut_chunk_ofset = paragraf_ofset yaziliyordu, bu da
        # kaydedilen ofsetin chunk'in GERCEK basladigi yerden (overlap'in
        # basladigi yerden) ILERIDE gostermesine yol aciyordu - bu da
        # (nadir durumda, overlap bir sayfa sinirina denk gelirse) YANLIS
        # sayfa numarasi atanmasina sebep olabiliyordu. Simdi overlap'in
        # ESKI chunk icindeki gercek konumundan hareketle dogru ofseti
        # hesapliyoruz.
        if overlap_metni:
            overlap_baslangic_ofset = mevcut_chunk_ofset + (len(mevcut_chunk) - len(overlap_metni))
            yeni_chunk_ham = overlap_metni + "\n" + paragraf
            # strip() ile bastan silinecek bosluk miktarini ofsete ekliyoruz
            silinen_bas = len(yeni_chunk_ham) - len(yeni_chunk_ham.lstrip())
            mevcut_chunk = yeni_chunk_ham.strip()
            mevcut_chunk_ofset = overlap_baslangic_ofset + silinen_bas
        else:
            mevcut_chunk = paragraf
            mevcut_chunk_ofset = paragraf_ofset

        while len(mevcut_chunk) > chunk_size:
            kesim = kelime_sinirindan_kes(mevcut_chunk, chunk_size)
            chunklar.append({
                "metin": mevcut_chunk[:kesim].strip(),
                "ofset": mevcut_chunk_ofset
            })

            # KRITIK DUZELTME: Ilerlemeyi ham karakter sayisiyla degil,
            # kelime sinirina hizalayarak hesapliyoruz. Aksi halde overlap
            # bir kelimenin ortasindan baslayip onu ikiye bolebilir
            # (orn. "Islenen" -> "Isl" (onceki chunk'ta) + "en" (yeni
            # chunk'in basinda, "Isl" oneki olmadan) - bu gercek bir
            # bug'di, kelime_sinirina_git ile onleniyor).
            hedef_pozisyon = max(kesim - chunk_overlap, 1)
            bosluk_konumu = mevcut_chunk.find(" ", hedef_pozisyon)
            if bosluk_konumu == -1:
                ilerleme = max(hedef_pozisyon, 1)
            else:
                ilerleme = bosluk_konumu + 1

            mevcut_chunk_ofset += ilerleme
            mevcut_chunk = mevcut_chunk[ilerleme:]

            # strip() basdaki bosluklari silebilir; bu silinen karakter
            # sayisini da ofsete eklemezsek, ofset gercek konumdan kayar
            silinen_bas = len(mevcut_chunk) - len(mevcut_chunk.lstrip())
            mevcut_chunk_ofset += silinen_bas
            mevcut_chunk = mevcut_chunk.strip()

    if mevcut_chunk:
        chunklar.append({"metin": mevcut_chunk, "ofset": mevcut_chunk_ofset})

    return chunklar


def karakter_bazli_split(metin: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Geriye-uyumlu basit arayuz: sadece metinleri dondurur (ofsetsiz)."""
    return [c["metin"] for c in karakter_bazli_split_ofsetli(metin, chunk_size, chunk_overlap)]


# ============================================================
# YONTEM 2: Madde-Bazli (Header-Aware) Chunking
# ============================================================

MADDE_DESENI = re.compile(
    r"(?m)(?=^(?:" + GECICI_KELIMESI + r"\s+)?MADDE\s+\d+\s*[-" + _EN_DASH + r".]?)"
)

UZUN_MADDE_ESIGI = 800  # bu karakterden uzun maddeler alt-chunklara bolunur


def madde_bazli_split(tam_metin: str, sayfa_araliklari: list[tuple]) -> list[dict]:
    """
    Metni MADDE basliklarina gore boler. Uzun maddeler kendi icinde
    alt-chunklara bolunur ve HER alt-chunk kendi gercek ofsetine gore
    dogru sayfa numarasini alir (onceki versiyondaki hata: tum alt-chunklara
    maddenin BASLANGIC sayfasi atanıyordu, bu duzeltildi).
    """
    sinirlar = [m.start() for m in MADDE_DESENI.finditer(tam_metin)]

    if not sinirlar or sinirlar[0] != 0:
        sinirlar = [0] + sinirlar
    sinirlar.append(len(tam_metin))

    ham_maddeler = []
    onceki_kirpilan_baslik = ""  # bir onceki maddeden "sizip" gelen baslik
    for i in range(len(sinirlar) - 1):
        baslangic_ofset = sinirlar[i]
        bitis_ofset = sinirlar[i + 1]
        parca = tam_metin[baslangic_ofset:bitis_ofset].strip()
        if not parca:
            continue

        # Bir sonraki maddenin basligi (varsa) bu maddenin sonuna
        # sizmis olabilir - temizliyoruz (bkz. kapanis_baslik_temizle
        # fonksiyonunun docstring'i).
        parca, kirpilan_baslik = kapanis_baslik_temizle(parca)

        # ONEMLI SIRALAMA: Ofset VE madde_no tespitini, baslik EKLENMEDEN
        # ONCEKI (orijinal, sadece bu maddeye ait) parca uzerinden
        # yapiyoruz - iki nedenden: (1) asagida eklenecek
        # "onceki_kirpilan_baslik" metni bu maddenin SINIRLARI icinde
        # bulunamaz (find() basarisiz olur), (2) madde_no regex'i
        # metnin EN BASINDA "MADDE N" arar - eger baslik ONCE eklenirse
        # metin artik "MADDE N" ile degil baslikla basladigi icin
        # eslesme BASARISIZ olur (madde yanlislikla BASLIK/GIRIS
        # etiketine dusuyordu - gercek testte bulundu).
        ic_ofset = tam_metin[baslangic_ofset:bitis_ofset].find(parca)
        gercek_baslangic_ofset = baslangic_ofset + ic_ofset

        eslesme = re.match(r"((?:" + GECICI_KELIMESI + r"\s+)?MADDE\s+\d+)", parca)
        madde_no = eslesme.group(1) if eslesme else BASLIK_GIRIS_ETIKETI

        # DUZELTME: Bir ONCEKI maddeden sizip BURAYA kirpilmis olan
        # baslik varsa (orn. "İlgili kişinin hakları" -> MADDE 11'in
        # basligi), bu maddenin (dogru sahibinin) BASINA ekliyoruz -
        # artik ATILMIYOR, dogru yere TASINIYOR. Bu, madde_no tespiti
        # YAPILDIKTAN SONRA eklenir (yukaridaki not).
        if onceki_kirpilan_baslik:
            parca = onceki_kirpilan_baslik + "\n" + parca
        onceki_kirpilan_baslik = kirpilan_baslik

        ham_maddeler.append({
            "madde_no": madde_no,
            "baslangic_ofset": gercek_baslangic_ofset,
            "metin": parca
        })

    sonuc = []
    for madde in ham_maddeler:
        if len(madde["metin"]) <= UZUN_MADDE_ESIGI:
            sonuc.append({
                "madde_no": madde["madde_no"],
                "chunk_index": 0,
                "sayfa_no": sayfa_bul(madde["baslangic_ofset"], sayfa_araliklari),
                "metin": madde["metin"]
            })
        else:
            alt_chunklar = karakter_bazli_split_ofsetli(
                madde["metin"], chunk_size=UZUN_MADDE_ESIGI, chunk_overlap=80
            )
            for idx, alt_chunk in enumerate(alt_chunklar):
                # Alt-chunk'in madde icindeki goreli ofseti + maddenin
                # tam metindeki mutlak baslangici = gercek mutlak ofset
                mutlak_ofset = madde["baslangic_ofset"] + alt_chunk["ofset"]
                sonuc.append({
                    "madde_no": madde["madde_no"],
                    "chunk_index": idx,
                    "sayfa_no": sayfa_bul(mutlak_ofset, sayfa_araliklari),
                    "metin": alt_chunk["metin"]
                })

    return sonuc


# ============================================================
# Ana calisma blogu
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 02_chunker.py <pdf_dosya_yolu>")
        sys.exit(1)

    pdf_yolu = sys.argv[1]

    try:
        sayfalar = pdf_metnini_cikar(pdf_yolu)
    except PdfOkumaHatasi as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    tam_metin, sayfa_araliklari = tam_metin_ve_sayfa_haritasi_olustur(sayfalar)

    print("=" * 60)
    print("YONTEM 1: Karakter-Bazli Splitter")
    print("=" * 60)
    basit_chunklar = karakter_bazli_split(tam_metin, chunk_size=500, chunk_overlap=50)
    print(f"Toplam chunk sayisi: {len(basit_chunklar)}")
    print("\n--- Ornek: 3. chunk ---")
    if len(basit_chunklar) > 2:
        print(basit_chunklar[2][:300])
    print()

    print("=" * 60)
    print("YONTEM 2: Madde-Bazli (Header-Aware) Chunking")
    print("=" * 60)
    madde_chunklari = madde_bazli_split(tam_metin, sayfa_araliklari)
    print(f"Toplam chunk sayisi: {len(madde_chunklari)}")

    uzun_bolunmus = [c for c in madde_chunklari if c["chunk_index"] > 0]
    print(f"Alt-chunk'a bolunmus uzun madde parcasi sayisi: {len(uzun_bolunmus)}")

    print("\n--- Ornek: MADDE 5 chunk'i (metadata dahil) ---")
    for c in madde_chunklari:
        if c["madde_no"] == "MADDE 5":
            print(f"madde_no={c['madde_no']} | chunk_index={c['chunk_index']} | sayfa_no={c['sayfa_no']}")
            print(c["metin"][:400])
            break

    print("\n--- Ornek: " + GECICI_KELIMESI + " MADDE var mi, dogru yakalaniyor mu? ---")
    gecici_var = [c for c in madde_chunklari if GECICI_KELIMESI in c["madde_no"]]
    print(f"Bulunan {GECICI_KELIMESI} MADDE sayisi: {len(gecici_var)}")
    if gecici_var:
        print(f"Ilk ornek: {gecici_var[0]['madde_no']}")

    print("\n--- Ornek: coklu sayfaya yayilan uzun madde var mi? ---")
    for madde_no_ara in set(c["madde_no"] for c in madde_chunklari):
        parcalar = [c for c in madde_chunklari if c["madde_no"] == madde_no_ara]
        sayfalar_set = set(c["sayfa_no"] for c in parcalar)
        if len(parcalar) > 1 and len(sayfalar_set) > 1:
            print(f"{madde_no_ara}: {len(parcalar)} alt-chunk, sayfalar: {sorted(sayfalar_set)}")
            break
    else:
        print("(Bu ornekte birden fazla sayfaya yayilan bolunmus madde bulunamadi)")

    print("\n" + "=" * 60)
    print("KARSILASTIRMA")
    print("=" * 60)
    print(f"Karakter-bazli : {len(basit_chunklar)} chunk")
    print(f"Madde-bazli    : {len(madde_chunklari)} chunk (madde + sayfa metadatali)")

    # ============================================================
    # Chunk'lari data/processed/ klasorune JSON olarak kaydet.
    # Boylece hem sonucu bir dosyada gorebiliriz, hem de Task 1.3
    # (Embedding) bu dosyayi dogrudan okuyup her seferinde PDF'i
    # yeniden okuyup chunklamak zorunda kalmaz.
    # ============================================================
    cikti_klasoru = os.path.join(_current_dir, "..", "data", "processed")
    os.makedirs(cikti_klasoru, exist_ok=True)

    madde_bazli_yolu = os.path.join(cikti_klasoru, "chunks_madde_bazli.json")
    with open(madde_bazli_yolu, "w", encoding="utf-8") as f:
        json.dump(madde_chunklari, f, ensure_ascii=False, indent=2)

    karakter_bazli_yolu = os.path.join(cikti_klasoru, "chunks_karakter_bazli.json")
    with open(karakter_bazli_yolu, "w", encoding="utf-8") as f:
        json.dump(basit_chunklar, f, ensure_ascii=False, indent=2)

    print(f"\n\u2705 Chunk'lar kaydedildi:")
    print(f"   {os.path.normpath(madde_bazli_yolu)}")
    print(f"   {os.path.normpath(karakter_bazli_yolu)}")

    print("\nBir sonraki adim: Embedding (Task 1.3)")
