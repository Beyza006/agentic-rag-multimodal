"""
Task 1.1 - PDF Yukleme (PDF Loading)
--------------------------------------
Bu script, bir PDF dosyasini okuyup icindeki metni cikarir.
Kullanilan kutuphane: PyMuPDF (fitz)

Kullanim:
    python 01_pdf_loader.py data/ornek.pdf
"""

import os
import sys
import fitz  # PyMuPDF


# Bir sayfa "neredeyse bos" sayilsin diye belirledigimiz esik deger.
# Taranmis (scanned) sayfalarda genelde hic metin katmani olmaz,
# bu yuzden cok dusuk bir karakter sayisi "burada muhtemelen OCR gerekiyor"
# anlamina gelir.
BOS_SAYFA_ESIGI = 20


class PdfOkumaHatasi(Exception):
    """PDF acilamadigi veya okunamadigi durumlarda firlatilan ozel hata."""
    pass


def pdf_metnini_cikar(pdf_yolu: str) -> list[dict]:
    """
    Verilen PDF dosyasindaki her sayfanin metnini cikarir.

    Donen deger: her biri {"sayfa_no": int, "metin": str, "taranmis_olabilir": bool}
    seklinde bir sozluk listesi. Sayfa numarasini sakliyoruz ki ileride
    "bu bilgi hangi sayfadan geldi" diye kaynak gosterebilelim
    (Faz 5 - Kaynak Gosterme icin onemli).

    Hatalar:
        PdfOkumaHatasi: dosya bulunamadi, PDF degil, sifreli veya bozuksa firlatilir.
    """
    if not os.path.isfile(pdf_yolu):
        raise PdfOkumaHatasi(f"Dosya bulunamadi: '{pdf_yolu}'")

    try:
        dokuman = fitz.open(pdf_yolu)
    except Exception as hata:
        raise PdfOkumaHatasi(f"PDF acilamadi (bozuk veya PDF formatinda degil): {hata}")

    if dokuman.is_encrypted:
        dokuman.close()
        raise PdfOkumaHatasi(
            "PDF sifreli/korumali. Once sifreyi kaldirmaniz ya da "
            "dogru sifreyle acmaniz gerekiyor."
        )

    if dokuman.page_count == 0:
        dokuman.close()
        raise PdfOkumaHatasi("PDF'de hic sayfa yok (bos dosya).")

    sayfalar = []
    for sayfa_no, sayfa in enumerate(dokuman, start=1):
        metin = sayfa.get_text().strip()
        sayfalar.append({
            "sayfa_no": sayfa_no,
            "metin": metin,
            # Karakter sayisi cok dusukse, bu sayfa muhtemelen taranmis
            # bir goruntudur (metin katmani yok) ve OCR gerektirir.
            "taranmis_olabilir": len(metin) < BOS_SAYFA_ESIGI
        })

    dokuman.close()
    return sayfalar


def ozet_yazdir(sayfalar: list[dict]):
    """Cikarilan metnin kisa bir ozetini terminale yazdirir (kontrol amacli)."""
    toplam_karakter = sum(len(s["metin"]) for s in sayfalar)
    taranmis_sayfalar = [s["sayfa_no"] for s in sayfalar if s["taranmis_olabilir"]]

    print(f"\nToplam sayfa sayisi: {len(sayfalar)}")
    print(f"Toplam karakter sayisi: {toplam_karakter}")

    if taranmis_sayfalar:
        print(
            f"\n⚠️  UYARI: {len(taranmis_sayfalar)} sayfada neredeyse hic metin "
            f"bulunamadi (sayfa no: {taranmis_sayfalar}). "
            f"Bu sayfalar taranmis (scanned) goruntu olabilir ve OCR "
            f"gerektirebilir (bkz. Faz 3.5 - OCR)."
        )

    print("\n--- Ilk sayfanin ilk 500 karakteri (kontrol icin) ---")
    if sayfalar:
        print(sayfalar[0]["metin"][:500] or "(bu sayfada metin bulunamadi)")
    print("--- ---\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 01_pdf_loader.py <pdf_dosya_yolu>")
        print("Ornek:    python 01_pdf_loader.py data/ornek.pdf")
        sys.exit(1)

    pdf_yolu = sys.argv[1]

    print(f"'{pdf_yolu}' okunuyor...")

    try:
        sayfalar = pdf_metnini_cikar(pdf_yolu)
    except PdfOkumaHatasi as hata:
        print(f"\n❌ HATA: {hata}")
        sys.exit(1)

    ozet_yazdir(sayfalar)
    print("PDF basariyla okundu. Bir sonraki adim: Chunking (Task 1.2)")
