# -*- coding: utf-8 -*-
"""
Task 3.3 - PDF Icindeki Gorseller (Metin ve Gorselleri Birlikte Yorumlama)
------------------------------------------------------------------------------
Bu script, bir PDF sayfasini GORSELE cevirip VLM ile analiz eder.

Neden boyle bir yaklasim gerekti: KVKK PDF'imizde gercek anlamda "gomulu
resim/grafik" YOK - kadro tablosu bile aslinda duz METIN olarak PDF'in
icinde duruyor (Task 1.1'de fitz ile metin olarak cikarmistik). Ama bazi
karmasik yerlesimli sayfalarda (tablolar gibi), duz metin cikarimi
YAPININ (hangi sayi hangi sutuna ait) bozulmasina yol acabilir.

Cozum: PDF sayfasini oldugu gibi (gorsel olarak) bir PNG'ye "fotograf
cekercesine" donusturup, bunu VLM'e vermek - boylece VLM, tablo
yapisini GORSEL olarak (bir insanin ekrana bakip okumasi gibi) yorumlar.
Bu, Task 3.1'de senin elle yaptigin ekran goruntusu almayi PROGRAMATIK
hale getiriyor.

Kullanilan kutuphane: PyMuPDF (fitz) - Task 1.1'de PDF metni cikarmak
icin kullandigimiz ayni kutuphane, burada sayfayi goruntuye cevirmek
icin kullaniliyor (page.get_pixmap()).

Kullanim:
    python 09_pdf_gorsel.py <pdf_yolu> <sayfa_no> ["<soru>"]
    Ornek: python 09_pdf_gorsel.py ../data/documents/ornek.pdf 21 "Bu tabloyu özetle"
"""

import sys
import os
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


# Task 3.1/3.2'deki VLM fonksiyonlarini tekrar yazmiyoruz, oldugu gibi kullaniyoruz
_vision = _modul_yukle(os.path.join(_current_dir, "08_vision.py"), "vision_module")

# Cikti gorsellerinin kaydedilecegi klasor (data/images ile ayni mantik)
CIKTI_KLASORU = os.path.join(_current_dir, "..", "data", "images")

# Cozunurluk carpani: PDF sayfalari varsayilan olarak dusuk cozunurlukte
# render edilir (72 DPI), bu da VLM'in metni okumasini zorlastirabilir.
# 2x carpan ~144 DPI'ye denk gelir - metin okunabilirligi ile islem
# suresi/boyutu arasinda makul bir denge (3x/216 DPI, CPU'da VLM'in
# islem suresini onemli olcude uzattigi icin 2x'e dusuruldu).
COZUNURLUK_CARPANI = 2.0


def pdf_sayfasini_gorsele_cevir(pdf_yolu: str, sayfa_no: int) -> str:
    """
    Bir PDF dosyasinin belirtilen sayfasini (1-indeksli, kullanicinin
    goreceği gibi) yuksek cozunurluklu bir PNG dosyasina cevirir ve
    kaydedilen dosyanin yolunu dondurur.
    """
    import fitz  # PyMuPDF - Task 1.1'de de kullandigimiz kutuphane

    if not os.path.isfile(pdf_yolu):
        raise FileNotFoundError(f"PDF dosyasi bulunamadi: '{pdf_yolu}'")

    dokuman = fitz.open(pdf_yolu)
    toplam_sayfa = dokuman.page_count  # kapatmadan ONCE degiskene kaydediyoruz

    if sayfa_no < 1 or sayfa_no > toplam_sayfa:
        dokuman.close()
        raise ValueError(
            f"Gecersiz sayfa numarasi: {sayfa_no}. "
            f"PDF'te toplam {toplam_sayfa} sayfa var (1-{toplam_sayfa} arasi olmali)."
        )

    sayfa = dokuman[sayfa_no - 1]  # fitz 0-indeksli calisir, kullanici 1-indeksli dusunuyor
    matris = fitz.Matrix(COZUNURLUK_CARPANI, COZUNURLUK_CARPANI)
    pixmap = sayfa.get_pixmap(matrix=matris)

    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    pdf_adi = os.path.splitext(os.path.basename(pdf_yolu))[0]
    cikti_yolu = os.path.join(CIKTI_KLASORU, f"{pdf_adi}_sayfa{sayfa_no}.png")
    pixmap.save(cikti_yolu)

    dokuman.close()
    return cikti_yolu


def tablo_odakli_soru_olustur(soru: str) -> str:
    """
    TASK 3.3 IYILESTIRMESI (v4 - sadelestirmeye geri donus): Ust uste
    eklenen kurallar (aciklamali ornekler, "EN ONEMLI KURAL" basliklari
    vb.) modelin cevabini gittikce daha mekanik/madde-isaretli bir hale
    getirdigi, dogal akiciligini kaybettirdigi gozlemlendi (gercek
    kullanici testinde bulundu). Cozum: Task 3.3'te EN BASINDA iyi
    calisan SADE prompt'a geri donuyoruz, uzerine SADECE tarih formati
    duzeltmesini (kanitlanmis bir gercek sorun) ekliyoruz - baska hicbir
    ek kural/ornek eklemiyoruz. "Az ama etkili talimat", "cok ama
    cakisan talimat"tan daha iyi sonuc veriyor.
    """
    return f"""Bu görsel, bir PDF sayfasının görüntüsüdür ve muhtemelen bir
TABLO içeriyor. Tabloyu SATIR SATIR, DİKKATLİCE oku. Her satırdaki
bilgileri BİRBİRİNE KARIŞTIRMADAN, AYRI AYRI listele.

HÜCRE İÇİ LİSTE KURALI (ÇOK ÖNEMLİ): Eğer bir hücrenin içinde "19, 20, 21, 25" veya "6, 9, 18, Geçici Madde 3" gibi BİRDEN FAZLA madde/sayı varsa, bunların HİÇBİRİNİ ATLAMA. Hepsini tam ve eksiksiz bir şekilde, aralarına virgül koyarak yaz.

SÜTUN EŞLEŞTİRMESİ ÇOK ÖNEMLİ: Her değeri DOĞRU sütun başlığıyla eşleştir.
Özellikle her satırın EN SOLDAKİ hücresi (ilk sütun) o satırın kimlik/numara bilgisidir; bu hücreyi diğer sütunlarla KARIŞTIRMA.

TARİH KURALI (KESİN KURAL): Görselde gördüğün tarihleri (örneğin 1/6/2024 veya 5/12/2017) KESİNLİKLE gördüğün sırayla, olduğu gibi yaz. Gün ve ay yerlerini ASLA değiştirme! (Örn: 1/6/2024 görüyorsan, 6.01.2024 DİYE UYDURMA, doğrudan 1/6/2024 yaz veya 1 Haziran 2024 olarak çevir).

CEVABIN YAPISI (ÇOK ÖNEMLİ - bu yapıya AYNEN uy):
1. ÖNCE tablodaki TÜM satırları tek tek, her satırın tüm sütunlarıyla
   (kanun/karar numarası, değiştirdiği maddeler, yürürlük tarihi)
   birlikte sırayla açıkla. Hiçbir satırı atlama.
2. SONRA "Özetle:" veya benzeri bir geçişle, kullanıcının sorduğu
   SPESİFİK soruya doğrudan ve net bir cevap ver.

Soru: {soru}"""


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Kullanim: python 09_pdf_gorsel.py <pdf_yolu> <sayfa_no> [\"<soru>\"]")
        print("Ornek:    python 09_pdf_gorsel.py ../data/documents/ornek.pdf 21 \"Bu tabloyu özetle\"")
        sys.exit(1)

    pdf_yolu = sys.argv[1]
    try:
        sayfa_no = int(sys.argv[2])
    except ValueError:
        print(f"\n\u274c HATA: Sayfa numarasi bir tam sayi olmali, '{sys.argv[2]}' degil.")
        sys.exit(1)

    soru = sys.argv[3] if len(sys.argv) >= 4 else None

    print(f"'{pdf_yolu}' dosyasinin {sayfa_no}. sayfasi goruntuye ceviriliyor...")

    try:
        gorsel_yolu = pdf_sayfasini_gorsele_cevir(pdf_yolu, sayfa_no)
    except (FileNotFoundError, ValueError) as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print(f"\u2705 Sayfa goruntusu kaydedildi: {gorsel_yolu}")
    print(f"'{_vision.OLLAMA_VISION_MODEL}' modeline gonderiliyor...")

    try:
        if soru:
            cevap = _vision.gorsel_sorusu_sor(gorsel_yolu, tablo_odakli_soru_olustur(soru))
        else:
            cevap = _vision.resmi_aciklama_uret(gorsel_yolu)
    except RuntimeError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"PDF: {pdf_yolu} | SAYFA: {sayfa_no}")
    print(f"SORU: {soru if soru else '(genel açıklama istendi)'}")
    print("=" * 60)
    print(f"\nCEVAP:\n{cevap}")

    print("\n\U0001F4C4 Task 3.3 - PDF sayfasi gorsel olarak analiz edildi!")
