# Görüntülerin çözünürlük sorunlarını ve metin okunaklılığını artırmak amacıyla uygulanan iyileştirmeler.
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
