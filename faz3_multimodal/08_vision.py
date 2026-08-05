# -*- coding: utf-8 -*-
"""
Task 3.1 - Vision Language Model (VLM) ile Gorsel Analiz
------------------------------------------------------------
Bu script, Faz 3'un ilk adimi: bir gorseli (resim dosyasi) alip, VLM'e
(Ollama uzerinden calisan qwen2.5vl:3b) gonderip soruyu cevaplatir.

Kullanilan model: qwen2.5vl:3b (Ollama uzerinden, yerel)
- Alibaba'nin Qwen2.5-VL ailesinden, 3 milyar parametreli, dusuk kaynakli
  cihazlar (GPU'suz, orta seviye RAM) icin uygun kucuk bir VLM.
- Ozellikle DOKUMAN ve TABLO/GRAFIK iceren gorsellerde LLaVA gibi diger
  kucuk VLM'lerden daha iyi performans gosterdigi belirlendi (Task 3.1
  arastirmasi, bkz. research-notes.md).
- Proje roadmap'inin kendi notuna gore: "model performansi degerlendirme
  kriteri degil, odak agent/RAG mantiginda" - yani amacimiz en iyi VLM'i
  bulmak degil, agent'in gorsel sorulari doğru yonlendirip kullanabildigini
  gostermek.

Ollama'nin gorsel (vision) destegi: Metin promptlarindan farkli olarak,
gorsel iceren istekler icin API'ye "images" alaninda BASE64 formatinda
kodlanmis resim gonderilir. Model bu resmi "gorup" soruyu cevaplar.

Kullanim:
    python 08_vision.py <resim_yolu> "<soru>"
    Ornek: python 08_vision.py ornek.jpg "Bu görselde ne görüyorsun?"
"""

import sys
import os
import base64

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OLLAMA_VISION_MODEL = "qwen2.5vl:7b"
OLLAMA_METIN_MODEL = "gemma2:9b"  # yazim kontrolu icin - 06_llm_answer.py'deki ile ayni
OLLAMA_API_URL = "http://localhost:11434/api/generate"


def resmi_base64_cevir(resim_yolu: str) -> str:
    """
    Bir resim dosyasini okuyup BASE64 metnine cevirir - Ollama'nin
    vision API'si resimleri bu formatta bekliyor (dosyanin kendisini
    degil, metne kodlanmis halini).
    """
    if not os.path.isfile(resim_yolu):
        raise FileNotFoundError(f"Resim dosyasi bulunamadi: '{resim_yolu}'")

    with open(resim_yolu, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def gorsel_sorusu_sor(resim_yolu: str, soru: str) -> str:
    """
    Bir resmi ve soruyu Ollama'nin vision-capable modeline gonderir,
    modelin cevabini dondurur.
    """
    import requests

    base64_resim = resmi_base64_cevir(resim_yolu)

    try:
        yanit = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_VISION_MODEL,
                "prompt": soru,
                "images": [base64_resim],  # Ollama vision API'sinin bekledigi alan
                "stream": False
            },
            timeout=400  # gorsel isleme metne gore daha uzun surebilir, ozellikle CPU'da
        )
        yanit.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama'ya baglanilamadi. Ollama'nin bilgisayarinizda calistigindan "
            "emin olun. Kontrol icin terminalde 'ollama list' calistirabilirsiniz."
        )
    except requests.exceptions.HTTPError as hata:
        raise RuntimeError(
            f"Ollama modeli calistirilirken hata olustu: {hata}. "
            f"'{OLLAMA_VISION_MODEL}' modelinin indirilmis oldugundan emin olun "
            f"('ollama pull {OLLAMA_VISION_MODEL}')."
        )

    return yanit.json()["response"]


VARSAYILAN_ACIKLAMA_PROMPTU = (
    "Bu görseli detaylıca açıkla: görselde neler var, hangi nesneler/"
    "kişiler/metinler görünüyor, genel bağlam nedir? Türkçe ve akıcı bir "
    "şekilde anlat."
)


def resmi_aciklama_uret(resim_yolu: str) -> str:
    """
    TASK 3.2 - Gorsel Aciklama Uretme: Kullanici ozel bir soru sormadan,
    sadece bir gorsel yukleyip "bu ne?" turunden genel bir aciklama
    isteyebilmesi icin. Ayni gorsel_sorusu_sor fonksiyonunu, sabit/
    varsayilan bir "detayli acikla" promptuyla cagirir - boylece
    kullanicinin her seferinde soru yazmasina gerek kalmaz.
    """
    return gorsel_sorusu_sor(resim_yolu, VARSAYILAN_ACIKLAMA_PROMPTU)


def yazim_kontrolunden_gecir(vlm_ham_cevap: str) -> str:
    """
    VLM'lerin (ozellikle kucuk/3B modellerin) gorseldeki metni okurken
    yaptigi kucuk karakter tanima hatalarini (orn. "andicerek" ->
    "andigerek" gibi c/g karismasi) duzeltmek icin, ham cevabi metin
    tabanli modelimize (gemma2:9b - zaten Task 1.6'da kullaniyoruz)
    bir "yazim kontrolu" turu icin gonderiyoruz.

    Bu, iki modelin GUCLU yanlarini birlestiren bir yaklasim: VLM
    goruntuyu "goruyor" ama karakter duzeyinde kucuk hatalar yapabiliyor;
    metin modeli goruntuyu gormuyor ama Turkce dilbilgisi/yazim
    kurallarinda VLM'den daha guclu, bu yuzden belirgin yazim hatalarini
    yakalayip duzeltebiliyor.
    """
    prompt = f"""Aşağıdaki metin, bir görsel analiz modelinden (VLM) geldi.
Bu tür modeller görseldeki metni okurken bazen küçük karakter tanıma
hataları yapabilir (örn. "ç" yerine "ğ" yazmak gibi).

GÖREV: Bu metindeki BARİZ yazım/karakter hatalarını düzelt. SAYILARI,
TARİHLERİ, MADDE NUMARALARINI VEYA İÇERİĞİ DEĞİŞTİRME - sadece açıkça
yanlış yazılmış Türkçe kelimeleri düzelt. Emin olmadığın bir kelimeyi
OLDUĞU GİBİ bırak, tahmin ederek değiştirme.

Metin:
{vlm_ham_cevap}

Düzeltilmiş metin (SADECE düzeltilmiş metni yaz, başka hiçbir açıklama ekleme):"""

    try:
        import requests
        yanit = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_METIN_MODEL, "prompt": prompt, "stream": False},
            timeout=180
        )
        yanit.raise_for_status()
        return yanit.json()["response"].strip()
    except Exception:
        # Yazim kontrolu basarisiz olursa (orn. gemma2:9b kurulu degilse),
        # sessizce ham cevaba geri don - bu adim OPSIYONEL bir iyilestirme,
        # basarisiz olmasi asil VLM cevabini engellememeli.
        return vlm_ham_cevap


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 08_vision.py <resim_yolu> [\"<soru>\"]")
        print("Ornek 1 (soru-cevap):    python 08_vision.py ornek.jpg \"Bu görselde ne görüyorsun?\"")
        print("Ornek 2 (genel aciklama, TASK 3.2): python 08_vision.py ornek.jpg")
        sys.exit(1)

    resim_yolu = sys.argv[1]
    # TASK 3.2: soru verilmezse, varsayilan "detayli acikla" moduna gec
    soru = sys.argv[2] if len(sys.argv) >= 3 else VARSAYILAN_ACIKLAMA_PROMPTU
    aciklama_modu = len(sys.argv) < 3

    if aciklama_modu:
        print(f"Soru verilmedi -> GENEL ACIKLAMA modunda calisiyor (Task 3.2).")

    print(f"'{resim_yolu}' okunuyor ve '{OLLAMA_VISION_MODEL}' modeline gonderiliyor...")

    try:
        cevap = gorsel_sorusu_sor(resim_yolu, soru)
    except (FileNotFoundError, RuntimeError) as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    # NOT (05.08 - gecici geri alma): yazim_kontrolunden_gecir adimi
    # denendi ama Task 2.5'teki Self-Correction sorunuyla AYNI nedenden
    # (duzeltmeyi yapan model KAYNAGI - burada gorseli - gormedigi icin
    # kor tahmin yapiyor) BAZEN dogru metni BOZDUGU icin devre disi
    # birakildi (orn. dogru yazilmis "hükümlerin" kelimesini yanlislikla
    # "hükmüllerin" yapmisti). Fonksiyon SILINMEDI, ileride daha buyuk
    # bir model ile ya da farkli bir yontemle (orn. VLM'e tekrar sorup
    # karsilastirma) tekrar denenebilir.
    # cevap = yazim_kontrolunden_gecir(cevap)

    print("\n" + "=" * 60)
    print("SORU: (genel açıklama istendi)" if aciklama_modu else f"SORU: {soru}")
    print("=" * 60)
    print(f"\nCEVAP:\n{cevap}")

    print("\n\U0001F5BC\uFE0F  Task 3.1/3.2 - VLM ile gorsel analiz tamamlandi!")
