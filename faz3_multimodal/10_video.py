# -*- coding: utf-8 -*-
"""
Task 3.4 - Video + Frame + VLM (Basit Surum)
------------------------------------------------
Bu script, Faz 3'un "ana demo"su olan video soru-cevap ozelliginin BASIT
halini gerceklestirir:

    Video -> Belirli bir andaki kareyi (frame) cikar -> VLM'e gonder -> Cevap

Bu, roadmap'te tanimlanan iki seviyeden ILKI (basit hali). Daha gelismis
"Temporal Reasoning" (birden fazla kareyi birlikte degerlendirme) surumu,
bu basit surum saglam calistiktan sonra eklenecek.

Kullanilan kutuphane: OpenCV (cv2) - videodan belirli bir saniyedeki
kareyi goruntu (resim) olarak cikarmak icin. Bu, Task 3.3'te PDF
sayfasini goruntuye cevirmekle AYNI MANTIK - farkli bir kaynaktan
(video yerine PDF) baslangic noktasi, ama sonunda ikisi de bir goruntu
uretiyor ve o goruntu VLM'e gonderiliyor.

Kullanim:
    python 10_video.py <video_yolu> <saniye> ["<soru>"]
    Ornek: python 10_video.py video.mp4 15 "Bu anda ne oluyor?"
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


# Task 3.1/3.2'deki VLM fonksiyonlarini (gorsel_sorusu_sor, resmi_aciklama_uret,
# OLLAMA_VISION_MODEL) tekrar yazmiyoruz, oldugu gibi kullaniyoruz - tipki
# Task 3.3'teki 09_pdf_gorsel.py'nin yaptigi gibi.
_vision = _modul_yukle(os.path.join(_current_dir, "08_vision.py"), "vision_module")

CIKTI_KLASORU = os.path.join(_current_dir, "..", "data", "images")


def video_karesini_cikar(video_yolu: str, saniye: float) -> str:
    """
    Bir video dosyasinin belirtilen saniyesindeki kareyi (frame) bir PNG
    goruntusune cevirir ve kaydedilen dosyanin yolunu dondurur.

    OpenCV mantigi: Videolar "frame" (kare) dizisinden olusur, her
    saniyede kac kare oldugu FPS (Frames Per Second) ile belirlenir.
    Istenen saniyeye denk gelen kare numarasini bulmak icin:
        kare_numarasi = istenen_saniye * FPS
    """
    import cv2

    if not os.path.isfile(video_yolu):
        raise FileNotFoundError(f"Video dosyasi bulunamadi: '{video_yolu}'")

    yakalayici = cv2.VideoCapture(video_yolu)

    if not yakalayici.isOpened():
        raise RuntimeError(
            f"Video acilamadi: '{video_yolu}'. Dosyanin bozuk olmadigindan "
            f"veya desteklenen bir formatta (mp4, avi vb.) oldugundan emin olun."
        )

    fps = yakalayici.get(cv2.CAP_PROP_FPS)
    toplam_kare = yakalayici.get(cv2.CAP_PROP_FRAME_COUNT)
    video_suresi_saniye = toplam_kare / fps if fps > 0 else 0

    if saniye < 0 or saniye > video_suresi_saniye:
        yakalayici.release()
        raise ValueError(
            f"Gecersiz saniye: {saniye}. Video suresi yaklasik "
            f"{video_suresi_saniye:.1f} saniye (0-{video_suresi_saniye:.1f} arasi olmali)."
        )

    hedef_kare_no = int(saniye * fps)
    yakalayici.set(cv2.CAP_PROP_POS_FRAMES, hedef_kare_no)

    basarili, kare = yakalayici.read()
    yakalayici.release()

    if not basarili:
        raise RuntimeError(
            f"'{saniye}'. saniyedeki kare okunamadi - video dosyasi bozuk olabilir."
        )

    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    video_adi = os.path.splitext(os.path.basename(video_yolu))[0]
    cikti_yolu = os.path.join(CIKTI_KLASORU, f"{video_adi}_saniye{saniye}.png")

    # OpenCV goruntuleri BGR (Blue-Green-Red) formatinda tutar, ama bu
    # cv2.imwrite icin sorun degil - cv2.imwrite BGR'yi doğrudan PNG'ye
    # yazabilir (donusum gerekmez, sadece VLM'e gonderirken degil, ekranda
    # GORUNTULERKEN RGB/BGR farki onemli olur - burada sadece dosyaya
    # kaydediyoruz, sorun yok).
    cv2.imwrite(cikti_yolu, kare)

    return cikti_yolu


def video_karelerini_cikar(video_yolu: str, kare_sayisi: int = 5) -> list:
    """
    TASK 3.4 GELISMIS SURUM (Temporal Reasoning): Videoyu bastan sona
    esit araliklarla bolup, "kare_sayisi" kadar kareyi cikarir - orn.
    kare_sayisi=5 ise, videonun %0, %25, %50, %75, %100 noktalarindan
    birer kare alir. Bu, videonun ZAMAN ICINDEKI akisini temsil eden
    bir "ozet dizisi" olusturur.

    video_karesini_cikar'dan (tek kare) farki: bu fonksiyon bir LISTE
    donduruyor, boylece bu kareler birlikte VLM'e gonderilip "zamanla
    ne degisti" diye sorulabiliyor.
    """
    import cv2

    if not os.path.isfile(video_yolu):
        raise FileNotFoundError(f"Video dosyasi bulunamadi: '{video_yolu}'")

    yakalayici = cv2.VideoCapture(video_yolu)
    if not yakalayici.isOpened():
        raise RuntimeError(f"Video acilamadi: '{video_yolu}'.")

    fps = yakalayici.get(cv2.CAP_PROP_FPS)
    toplam_kare = yakalayici.get(cv2.CAP_PROP_FRAME_COUNT)
    video_suresi_saniye = toplam_kare / fps if fps > 0 else 0

    os.makedirs(CIKTI_KLASORU, exist_ok=True)
    video_adi = os.path.splitext(os.path.basename(video_yolu))[0]

    kare_yollari = []
    for i in range(kare_sayisi):
        # 0, 1, 2, ..., kare_sayisi-1 -> 0/(N-1), 1/(N-1), ..., 1 araligina yay
        oran = i / (kare_sayisi - 1) if kare_sayisi > 1 else 0
        hedef_saniye = oran * video_suresi_saniye
        hedef_kare_no = int(hedef_saniye * fps)
        # SINIRLAMA: Son kare tam video suresine denk gelince, hesaplanan
        # kare numarasi VIDEODAKI SON GECERLI INDEKSI (toplam_kare - 1)
        # asabiliyor (orn. 50 kareli bir videoda indeksler 0-49'dur, ama
        # yuvarlama 50'yi hedefleyebilir) - bu, o karenin OKUNAMAMASINA
        # (basarisiz olmasina) yol aciyordu. min() ile bunu engelliyoruz.
        hedef_kare_no = min(hedef_kare_no, int(toplam_kare) - 1)

        yakalayici.set(cv2.CAP_PROP_POS_FRAMES, hedef_kare_no)
        basarili, kare = yakalayici.read()
        if not basarili:
            continue

        cikti_yolu = os.path.join(CIKTI_KLASORU, f"{video_adi}_temporal_{i}.png")
        cv2.imwrite(cikti_yolu, kare)
        kare_yollari.append(cikti_yolu)

    yakalayici.release()
    return kare_yollari


def video_zaman_akisini_sor(video_yolu: str, soru: str, kare_sayisi: int = 5) -> str:
    """
    TASK 3.4 GELISMIS SURUM (DUZELTILMIS MIMARI): Ollama'nin /api/generate
    uc noktasi, arastirma sonucunda (Ollama'nin kendi GitHub deposu,
    issue #8513) SADECE TEK GORSEL destekledigi ortaya cikti - ilk
    denemede "birden fazla gorseli tek istekte gonderme" varsayimimiz
    400 Bad Request hatasina yol acti.

    DOGRU MIMARI (roadmap'in aslinda tarif ettigi sekilde):
        1) HER KAREYI AYRI AYRI VLM'e gonderip kisa bir aciklama al
        2) Bu aciklamalari, ZAMAN SIRASIYLA etiketleyerek birlestir
        3) Bu birlesik metni METIN modeline (gemma2:9b) gonderip,
           "zaman icinde ne degisti/ne oldu" diye SENTEZ yaptir

    Bu, VLM'in gorme gucunu, LLM'in metin akil yurutme gucuyle
    birlestiren dogru bir "iki-modelli" tasarim - Task 1.6'da
    ogrendigimiz "her modele kendi guclu oldugu, dar bir gorev ver"
    prensibinin bir baska uygulamasi.
    """
    kare_yollari = video_karelerini_cikar(video_yolu, kare_sayisi)

    if not kare_yollari:
        raise RuntimeError("Videodan hic kare cikarilamadi.")

    # ADIM 1: Her kareyi AYRI AYRI, kisa bir aciklama icin VLM'e gonder
    kare_aciklamalari = []
    for i, kare_yolu in enumerate(kare_yollari):
        print(f"  [{i+1}/{len(kare_yollari)}] kare analiz ediliyor...")
        aciklama = _vision.gorsel_sorusu_sor(
            kare_yolu,
            "Bu görselde ne görüyorsun? Kısaca (1-2 cümle) anlat."
        )
        print(f"      -> {aciklama}")  # DOGRULAMA ICIN: ham VLM aciklamasini goster
        kare_aciklamalari.append(aciklama)

    # ADIM 2: Aciklamalari zaman sirasiyla etiketleyerek birlestir
    zaman_etiketleri = ["BAŞLANGIÇ"] + [f"ORTA-{i}" for i in range(1, len(kare_yollari) - 1)] + ["SON"]
    if len(kare_yollari) == 1:
        zaman_etiketleri = ["TEK KARE"]

    birlesik_metin = "\n\n".join(
        f"[{etiket}]: {aciklama}"
        for etiket, aciklama in zip(zaman_etiketleri, kare_aciklamalari)
    )

    # ADIM 3: Metin modeline (gemma2:9b) gonderip sentez/akil yurutme yaptir
    sentez_prompt = f"""Aşağıda bir videodan ZAMAN SIRASIYLA alınmış kare
açıklamaları var (BAŞLANGIÇ'tan SON'a doğru). Bu açıklamaları kullanarak
zaman içinde neyin değiştiğini/ne olduğunu anlat ve aşağıdaki soruyu cevapla.

ÇOK ÖNEMLİ: SADECE aşağıda verilen kare açıklamalarında GEÇEN bilgileri
kullan. Açıklamalarda bahsedilmeyen hiçbir detayı (nesne, aktivite,
kişi sayısı vb.) UYDURMA/EKLEME - eğer açıklamalar kısa/sınırlıysa,
cevabın da o ölçüde kısa/sınırlı olsun. Emin olmadığın bir şeyi tahmin
ederek yazma.

Kare açıklamaları:
{birlesik_metin}

Soru: {soru}

Cevap:"""

    return _metin_modeliyle_sentezle(sentez_prompt)


def _metin_modeliyle_sentezle(prompt: str) -> str:
    """gemma2:9b'yi (metin modelimiz) kullanarak kare aciklamalarini sentezler."""
    import requests

    try:
        yanit = requests.post(
            _vision.OLLAMA_API_URL,
            json={"model": _vision.OLLAMA_METIN_MODEL, "prompt": prompt, "stream": False},
            timeout=400
        )
        yanit.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Ollama'ya baglanilamadi.")
    except requests.exceptions.HTTPError as hata:
        raise RuntimeError(f"Ollama modeli calistirilirken hata olustu: {hata}")

    return yanit.json()["response"]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Kullanim (tek an):     python 10_video.py <video_yolu> <saniye> [\"<soru>\"]")
        print("Kullanim (zaman akisi): python 10_video.py <video_yolu> zaman \"<soru>\"")
        print("Ornek 1: python 10_video.py video.mp4 15 \"Bu anda ne oluyor?\"")
        print("Ornek 2: python 10_video.py video.mp4 zaman \"Videoda zaman içinde ne değişti?\"")
        sys.exit(1)

    video_yolu = sys.argv[1]

    # TASK 3.4 GELISMIS MOD: ikinci parametre "zaman" ise, tek kare yerine
    # coklu kare + zamansal akil yurutme moduna geciyoruz.
    if sys.argv[2].lower() == "zaman":
        if len(sys.argv) < 4:
            print("Zaman akisi modu icin bir soru gerekli.")
            print("Ornek: python 10_video.py video.mp4 zaman \"Videoda ne oluyor?\"")
            sys.exit(1)

        soru = sys.argv[3]
        print(f"'{video_yolu}' videosundan zaman icinde yayilan kareler cikariliyor...")

        try:
            cevap = video_zaman_akisini_sor(video_yolu, soru)
        except (FileNotFoundError, RuntimeError) as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"VIDEO: {video_yolu} | MOD: Zaman Akisi (Temporal Reasoning)")
        print(f"SORU: {soru}")
        print("=" * 60)
        print(f"\nCEVAP:\n{cevap}")
        print("\n\U0001F3AC Task 3.4 (gelismis) - Video zaman akisi VLM ile analiz edildi!")
        sys.exit(0)

    # TASK 3.4 BASIT MOD: tek bir saniyedeki kareyi analiz et
    try:
        saniye = float(sys.argv[2])
    except ValueError:
        print(f"\n\u274c HATA: Saniye bir sayi olmali, '{sys.argv[2]}' degil.")
        sys.exit(1)

    soru = sys.argv[3] if len(sys.argv) >= 4 else None

    print(f"'{video_yolu}' videosunun {saniye}. saniyesindeki kare cikariliyor...")

    try:
        kare_yolu = video_karesini_cikar(video_yolu, saniye)
    except (FileNotFoundError, RuntimeError, ValueError) as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print(f"\u2705 Kare kaydedildi: {kare_yolu}")
    print(f"'{_vision.OLLAMA_VISION_MODEL}' modeline gonderiliyor...")

    try:
        if soru:
            cevap = _vision.gorsel_sorusu_sor(kare_yolu, soru)
        else:
            cevap = _vision.resmi_aciklama_uret(kare_yolu)
    except RuntimeError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"VIDEO: {video_yolu} | SANIYE: {saniye}")
    print(f"SORU: {soru if soru else '(genel açıklama istendi)'}")
    print("=" * 60)
    print(f"\nCEVAP:\n{cevap}")

    print("\n\U0001F3AC Task 3.4 - Video karesi VLM ile analiz edildi!")
