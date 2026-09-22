# -*- coding: utf-8 -*-
"""
Task 3.5 - OCR (Optik Karakter Tanima)
------------------------------------------
Bu script, taranmis (scanned) PDF sayfalarindaki veya goruntulerdeki
YAZIYI, gercek/kopyalanabilir METNE cevirir.

NEDEN GEREKLI: Task 1.1'de (01_pdf_loader.py), eger bir PDF sayfasinda
cok az metin varsa ("< 20 karakter"), bu sayfayi "taranmis_olabilir"
diye ISARETLIYORDUK ama gercek bir cozum SUNMUYORDUK - sadece bir uyari
veriyorduk. Bu script, o eksik halkayi tamamliyor: eger bir sayfa
gercekten taranmis (yani fitz/PyMuPDF metin KATMANI bulamiyor) ise,
o sayfayi GORSEL olarak isleyip OCR ile metni "okuyoruz".

Kullanilan kutuphane: PaddleOCR
- Arastirma sonucunda (bkz. research-notes.md), PaddleOCR'in EasyOCR'a
  gore daha yuksek dogruluk sagladigi, hatta yakin tarihli bir akademik
  calismada VLM'imiz olan Qwen2.5-VL ile ESIT dogrulukta ama COK DAHA
  HIZLI (50 kat) oldugu belirlendi.
- Turkce dahil 80'den fazla dili destekliyor.

Kullanim:
    python 11_ocr.py <gorsel_veya_pdf_yolu> [sayfa_no (PDF ise)]
    Ornek 1 (gorsel): python 11_ocr.py ../data/images/ornek_sayfa21.png
    Ornek 2 (PDF):    python 11_ocr.py ../data/documents/ornek.pdf 21
"""

import sys
import os
import re
import importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_current_dir = os.path.dirname(os.path.abspath(__file__))


def _modul_yukle(dosya_yolu, modul_adi):
    spec = importlib.util.spec_from_file_location(modul_adi, dosya_yolu)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


# TASK 3.3'teki PDF->goruntu donusum fonksiyonunu tekrar yazmiyoruz,
# oldugu gibi kullaniyoruz (girdinin PDF olma ihtimaline karsi).
_pdf_gorsel = _modul_yukle(os.path.join(_current_dir, "09_pdf_gorsel.py"), "pdf_gorsel_module")

# PDF sayfalari gorsele cevrildiginde kaydedilecegi klasor (Task 3.3'teki
# ile ayni klasor - data/images) - bu tanim eksikti, eklendi.
CIKTI_KLASORU = os.path.join(_current_dir, "..", "data", "images")

_ocr_motoru = None  # Model agir oldugu icin sadece BIR KEZ yuklenip tekrar kullanilacak


def ocr_motorunu_yukle():
    """
    PaddleOCR modelini yukler. Bu islem biraz zaman alabilir (ilk
    calistirmada model dosyalari internetten inecek) - bu yuzden
    modeli GLOBAL bir degiskende (_ocr_motoru) saklayip, ayni
    calistirmada birden fazla goruntu islenirse tekrar tekrar
    YUKLENMESINI onluyoruz.
    """
    global _ocr_motoru
    if _ocr_motoru is None:
        from paddleocr import PaddleOCR
        print("PaddleOCR modeli yukleniyor (ilk calistirmada biraz surebilir)...")
        # lang='tr': Turkce diline ozel tanima modeli kullanilir
        # Windows sistemlerinde oneDNN (mkldnn) kaynaklı olası çökme (crash) sorunlarını önlemek için donanım hızlandırma devre dışı bırakılmıştır.
def gorselden_metin_cikar(gorsel_yolu: str) -> str:
    """
    Bir goruntu dosyasindaki TUM yaziyi OCR ile okuyup, satir satir
    birlestirilmis bir metin olarak dondurur.
    """
    if not os.path.isfile(gorsel_yolu):
        raise FileNotFoundError(f"Gorsel dosyasi bulunamadi: '{gorsel_yolu}'")

    motor = ocr_motorunu_yukle()
    sonuc = motor.predict(gorsel_yolu)

    if not sonuc:
        return "(Bu goruntude hicbir metin tespit edilemedi.)"

    ilk_sayfa = sonuc[0]

    # ONEMLI DEGISIKLIK: PaddleOCR 3.x'te sonuc formati DEGISTI. Eski
    # surumde (2.x) sonuc, her biri [konum_kutusu, (metin, skor)] seklinde
    # elemanlardan olusan bir listeydi. Yeni surumde (3.x), sonuc bir
    # "sonuc nesnesi" dondurur ve bu nesne, dict gibi erisilebilen
    # 'rec_texts' adinda bir alan icinde TUM metin satirlarinin listesini
    # tutuyor. Once bu YENI formati deniyoruz, olmazsa (ilerideki bir
    # surum degisikligine karsi) ESKI formata da geri donebiliyoruz.
    metinler = None
    try:
        metinler = ilk_sayfa["rec_texts"]
    except (KeyError, TypeError):
        pass

    if metinler is None:
        try:
            # Bazi PaddleOCR surumlerinde sonuc, .json niteligi
            # uzerinden erisilen bir sozluk icinde geliyor olabilir.
            metinler = ilk_sayfa.json["res"]["rec_texts"]
        except Exception:
            pass

    if metinler is None:
        # Hicbir bilinen format eslesmedi - ESKI (2.x) format olabilir
        try:
            metinler = [eleman[1][0] for eleman in ilk_sayfa]
        except Exception:
            raise RuntimeError(
                f"OCR sonuc formati taninamadi (PaddleOCR surumu degismis "
                f"olabilir). Ham veri turu: {type(ilk_sayfa)}"
            )

    if not metinler:
        return "(Bu goruntude hicbir metin tespit edilemedi.)"

    return "\n".join(metinler)


def html_tablolarini_okunabilir_metne_cevir(markdown_metni: str) -> str:
    """
    PP-StructureV3'un urettigi Markdown ciktisinda, tablolar HAM HTML
    etiketleri (<table><tr><td>...) ile gomulu geliyor - bu, kullanicinin
    ekranda/terminalde dogrudan okuyabilecegi bir format DEGIL (henuz
    Gradio/web arayuzumuz olmadigi icin, bkz. Faz 5).

    Bu fonksiyon, metindeki HER HTML tablo blogunu bulup, satir/sutun
    bilgisini KORUYARAK (hangi hucrenin hangi satira ait oldugu
    kaybolmadan) duz, "|" ile ayrilmis okunabilir bir metne cevirir.
    Tablo OLMAYAN kisimlar (basliklar, paragraflar) oldugu gibi kalir.
    """
    from bs4 import BeautifulSoup

    def _tabloyu_metne_cevir(eslesme):
        html_parcasi = eslesme.group(0)
        soup = BeautifulSoup(html_parcasi, "html.parser")
        satir_metinleri = []
        for satir in soup.find_all("tr"):
            hucreler = [h.get_text(strip=True) for h in satir.find_all(["td", "th"])]
            satir_metinleri.append(" | ".join(hucreler))
        return "\n".join(satir_metinleri)

    tablolari_cevrilmis = re.sub(r"<table.*?</table>", _tabloyu_metne_cevir, markdown_metni, flags=re.DOTALL)

    # Tablolari cevirdikten sonra, geriye kalan SARMALAYICI etiketleri de
    # (orn. <div style="...">, <html>, <body>, </div> gibi) temizliyoruz -
    # bunlar tablo disinda kalip ekranda cirkin/anlasilmaz gorunuyordu.
    # Tablo icerigi artik "|" ile ayrilmis duz metin oldugu icin (hicbir
    # "<" veya ">" karakteri icermiyor), bu genel temizlik onu ETKILEMEZ,
    # sadece kalan bos sarmalayici etiketleri siler.
    temizlenmis = re.sub(r"<[^>]+>", "", tablolari_cevrilmis)

    return temizlenmis.strip()


_yapi_motoru = None  # PP-StructureV3 modeli - ayri, daha agir bir model


def gorselden_tablo_yapili_metin_cikar(gorsel_yolu: str) -> str:
    """
    TASK 3.5 GELISMIS SURUM: gorselden_metin_cikar'in aksine, bu fonksiyon
    SADECE duz metin degil, TABLO YAPISINI (hangi hucre hangi satir/sutuna
    ait) da koruyarak bilgi cikarir. PP-StructureV3 (PaddleOCR'in belge
    ayrıştırma - "document parsing" - hatti) kullanilir.

    Sonuc, bir Markdown dosyasi olarak kaydedilir - tablolar Markdown/HTML
    tablo formatinda temsil edilir, boylece "hangi sayi hangi sutuna ait"
    bilgisi KAYBOLMAZ (gorselden_metin_cikar'daki en buyuk eksiklik buydu).
    """
    global _yapi_motoru

    if not os.path.isfile(gorsel_yolu):
        raise FileNotFoundError(f"Gorsel dosyasi bulunamadi: '{gorsel_yolu}'")

    if _yapi_motoru is None:
        from paddleocr import PPStructureV3
        print("PP-StructureV3 modeli yukleniyor (ilk calistirmada model dosyalari inebilir, biraz surebilir)...")
        # Yapısal ayrıştırma işleminde (PPStructure) istikrarı korumak adına mkldnn devre dışı bırakılmıştır.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim (duz metin):  python 11_ocr.py <gorsel_veya_pdf_yolu> [sayfa_no]")
        print("Kullanim (tablo yapisi korunarak): python 11_ocr.py <gorsel_veya_pdf_yolu> [sayfa_no] yapi")
        print("Ornek 1 (gorsel): python 11_ocr.py ../data/images/ornek_sayfa21.png")
        print("Ornek 2 (PDF):    python 11_ocr.py ../data/documents/ornek.pdf 21")
        print("Ornek 3 (yapi):   python 11_ocr.py ../data/documents/ornek.pdf 21 yapi")
        sys.exit(1)

    dosya_yolu = sys.argv[1]
    uzanti = os.path.splitext(dosya_yolu)[1].lower()

    # "yapi" kelimesi son argument olarak verildiyse, tablo-yapili modu sec
    yapi_modu = sys.argv[-1].lower() == "yapi"
    argumanlar = sys.argv[:-1] if yapi_modu else sys.argv

    if uzanti == ".pdf":
        if len(argumanlar) < 3:
            print("PDF girdisi icin sayfa numarasi da gerekli.")
            sys.exit(1)
        sayfa_no = int(argumanlar[2])
        print(f"'{dosya_yolu}' dosyasinin {sayfa_no}. sayfasi goruntuye ceviriliyor...")
        try:
            gorsel_yolu = _pdf_gorsel.pdf_sayfasini_gorsele_cevir(dosya_yolu, sayfa_no)
        except (FileNotFoundError, ValueError) as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)
        print(f"\u2705 Sayfa goruntusu kaydedildi: {gorsel_yolu}")
    else:
        gorsel_yolu = dosya_yolu

    print(f"'{gorsel_yolu}' uzerinde OCR calistiriliyor ({'tablo yapisi korunarak' if yapi_modu else 'duz metin'})...")

    try:
        if yapi_modu:
            metin = gorselden_tablo_yapili_metin_cikar(gorsel_yolu)
        else:
            metin = gorselden_metin_cikar(gorsel_yolu)
    except FileNotFoundError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("OCR ILE CIKARILAN METIN" + (" (TABLO YAPISI KORUNARAK)" if yapi_modu else ""))
    print("=" * 60)
    print(metin)

    print("\n\U0001F50D Task 3.5 - OCR ile metin cikarimi tamamlandi!")
