# -*- coding: utf-8 -*-
"""
Faz 6 - Otomatik Degerlendirme (Evaluation)
------------------------------------------------
Bu script, su ana kadar TEK TEK elle test ettigimiz sorulari, KALICI ve
TEKRARLANABILIR bir test setine donusturur. Her soru icin BEKLENEN
anahtar ifadeleri (ve varsa YASAKLI/halusinasyon ifadelerini) tanimlar,
sistemi otomatik olarak calistirir ve sonuclari bir rapor halinde sunar.

NEDEN ANAHTAR KELIME KONTROLU (LLM-tabanli degerlendirme DEGIL):
Proje boyunca (Task 2.5 - Self-Correction, VLM halusinasyonlari)
kucuk yerel modellerin KENDI CIKTISINI degerlendirmesinin guvenilir
olmadigi tekrar tekrar gozlemlendi. Anahtar kelime kontrolu ise
DETERMINISTIK ve TEKRARLANABILIR - ayni test setini 10 kez calistirsan
(LLM'in ana cevap uretimindeki dogal rastgeleligi haric) HEP AYNI
degerlendirme sonucunu alirsin. Bu, projenin bastan beri izledigi
"kod tabanli, garantili" felsefesiyle (bkz. kod_tabanli_nihai_cevap_olustur)
tam ortusuyor.

Kullanim:
    python 15_degerlendirme.py           # tum test setini calistirir
    python 15_degerlendirme.py metin     # sadece metin (doc+web) testleri
    python 15_degerlendirme.py gorsel    # sadece gorsel testleri
    python 15_degerlendirme.py video     # sadece video testleri
"""

import sys
import os
import time
import importlib.util
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_current_dir = os.path.dirname(os.path.abspath(__file__))
_faz3_dir = os.path.join(_current_dir, "..", "faz3_multimodal")
_faz4_dir = os.path.join(_current_dir, "..", "faz4_hybrid_rag")


def _modul_yukle(dosya_yolu, modul_adi):
    spec = importlib.util.spec_from_file_location(modul_adi, dosya_yolu)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


_vision = _modul_yukle(os.path.join(_faz3_dir, "08_vision.py"), "vision_module")
_video = _modul_yukle(os.path.join(_faz3_dir, "10_video.py"), "video_module")
_pdf_gorsel = _modul_yukle(os.path.join(_faz3_dir, "09_pdf_gorsel.py"), "pdf_gorsel_module")
_hybrid_agent = _modul_yukle(os.path.join(_faz4_dir, "13_hybrid_agent.py"), "hybrid_agent_module")

_agent = _hybrid_agent.hybrid_agent_olustur()

# Ornek gorsel/video dosyalarinin proje icindeki standart yollari
_ORNEK_TABLO_PDF = os.path.join(_current_dir, "..", "data", "documents", "ornek.pdf")
_ORNEK_TABLO_GORSEL_SAYFA = 21
_ORNEK_VIDEO = os.path.join(_current_dir, "..", "data", "videos", "test_video.mp4")


# ============================================================
# TEST SETI - her soru icin: soru, beklenen kaynak turu (varsa),
# beklenen (olmasi GEREKEN) ifadeler, yasakli (olmamasi GEREKEN,
# bilinen halusinasyon) ifadeler.
# ============================================================

TEST_SETI = [
    # ---- METIN: DOKUMAN (RAG) sorulari ----
    {
        "id": "doc_01_acik_riza_sartlari",
        "tur": "metin",
        "soru": "Açık rıza olmadan kişisel veri ne zaman işlenebilir?",
        "beklenen_kaynak_turu": "rag",
        "beklenen_ifadeler": [
            "MADDE 5",
            "Kanunlarda açıkça öngörülmesi",
            "beden bütünlüğünün korunması",
            "sözleşmenin kurulması",
            "hukuki yükümlülüğünü",
            "alenileştirilmiş",
            "meşru menfaatleri",
        ],
        "yasakli_ifadeler": [],
    },
    {
        "id": "doc_02_hukuka_aykiri_kaydetme_cezasi",
        "tur": "metin",
        "soru": "Birinin rızası olmadan verilerini hukuka aykırı şekilde kaydeden kişiye ne tür bir ceza verilir?",
        "beklenen_kaynak_turu": None,  # rag VEYA web olabilir, ikisi de hukuken savunulabilir - zorunlu kontrol degil
        "beklenen_ifadeler": [],
        # Anahtar kelime tabanlı doğrulama: Doğru maddenin (TCK atıfları vb.) seçildiğinin teyidi.
]


# ============================================================
# CALISTIRMA VE DEGERLENDIRME MANTIGI
# ============================================================

def _metin_sorusunu_calistir(test: dict) -> dict:
    baslangic_state = {
        "soru": test["soru"], "arac_karari": "", "cevap": "", "kaynaklar": [],
        "gecmis": [], "arama_sorgusu": "", "ham_sonuc": {}, "rag_yeterli": True,
        "web_sonuclari": [], "kaynak_turu": "",
    }
    t0 = time.time()
    try:
        sonuc = _agent.invoke(baslangic_state)
        return {
            "cevap": sonuc.get("cevap", ""),
            "kaynak_turu": sonuc.get("kaynak_turu", ""),
            "sure_sn": round(time.time() - t0, 1),
            "hata": None,
        }
    except Exception as hata:
        return {"cevap": "", "kaynak_turu": "", "sure_sn": round(time.time() - t0, 1), "hata": str(hata)}


def _metin_coklu_tur_sorusunu_calistir(test: dict) -> dict:
    """
    Task 2.4 - Session Memory'yi GERCEKTEN test eder: birden fazla soruyu
    SIRAYLA, bir onceki turun cevabini "gecmis" olarak besleyerek calistirir.
    Sadece SON turun cevabi degerlendirilir (ilk soru(lar) sadece baglam
    olusturmak icin var).
    """
    gecmis = []
    son_cevap = ""
    t0 = time.time()
    try:
        for soru in test["sorular"]:
            baslangic_state = {
                "soru": soru, "arac_karari": "", "cevap": "", "kaynaklar": [],
                "gecmis": gecmis, "arama_sorgusu": "", "ham_sonuc": {}, "rag_yeterli": True,
                "web_sonuclari": [], "kaynak_turu": "",
            }
            sonuc = _agent.invoke(baslangic_state)
            son_cevap = sonuc.get("cevap", "")
            gecmis = gecmis + [(soru, son_cevap)]
        return {"cevap": son_cevap, "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": None}
    except Exception as hata:
        return {"cevap": son_cevap, "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": str(hata)}


def _gorsel_pdf_sorusunu_calistir(test: dict) -> dict:
    t0 = time.time()
    try:
        gorsel_yolu = _pdf_gorsel.pdf_sayfasini_gorsele_cevir(_ORNEK_TABLO_PDF, test["sayfa_no"])
        zenginlestirilmis = _pdf_gorsel.tablo_odakli_soru_olustur(test["soru"])
        cevap = _vision.gorsel_sorusu_sor(gorsel_yolu, zenginlestirilmis)
        return {"cevap": cevap, "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": None}
    except Exception as hata:
        return {"cevap": "", "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": str(hata)}


def _video_sorusunu_calistir(test: dict) -> dict:
    t0 = time.time()
    try:
        cevap = _video.video_zaman_akisini_sor(_ORNEK_VIDEO, test["soru"])
        return {"cevap": cevap, "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": None}
    except Exception as hata:
        return {"cevap": "", "kaynak_turu": None, "sure_sn": round(time.time() - t0, 1), "hata": str(hata)}


def testi_calistir(test: dict) -> dict:
    """Tek bir test kaydini calistirip, sonucu SKORLAR (basarili/basarisiz)."""
    if test["tur"] == "metin":
        sonuc = _metin_sorusunu_calistir(test)
    elif test["tur"] == "metin_coklu_tur":
        sonuc = _metin_coklu_tur_sorusunu_calistir(test)
    elif test["tur"] == "gorsel_pdf":
        sonuc = _gorsel_pdf_sorusunu_calistir(test)
    elif test["tur"] == "video":
        sonuc = _video_sorusunu_calistir(test)
    else:
        raise ValueError(f"Bilinmeyen test turu: {test['tur']}")

    if sonuc["hata"]:
        return {**test, **sonuc, "basarili": False, "detay": f"HATA: {sonuc['hata']}"}

    cevap_kucuk = sonuc["cevap"].lower()
    eksik_ifadeler = [
        ifade for ifade in test.get("beklenen_ifadeler", [])
        if ifade.lower() not in cevap_kucuk
    ]
    bulunan_yasakli = [
        ifade for ifade in test.get("yasakli_ifadeler", [])
        if ifade.lower() in cevap_kucuk
    ]

    # ESNEK kontrol: "beklenen_ifadeler_esnek" varsa, bu listedeki
    # ifadelerden EN AZ BIRININ gecmesi yeterli (hepsinin degil) - orn.
    # "hapis" VEYA "para cezası" - ikisi de dogru kabul edilen belirsiz
    # sorularda kullanilir.
    esnek_basarisiz = False
    if "beklenen_ifadeler_esnek" in test:
        esnek_liste = test["beklenen_ifadeler_esnek"]
        esnek_basarisiz = not any(ifade.lower() in cevap_kucuk for ifade in esnek_liste)

    kaynak_turu_dogru = True
    beklenen_kt = test.get("beklenen_kaynak_turu")
    if beklenen_kt is not None:
        kaynak_turu_dogru = sonuc.get("kaynak_turu") == beklenen_kt

    basarili = not eksik_ifadeler and not bulunan_yasakli and kaynak_turu_dogru and not esnek_basarisiz

    detaylar = []
    if eksik_ifadeler:
        detaylar.append(f"EKSIK ifadeler: {eksik_ifadeler}")
    if bulunan_yasakli:
        detaylar.append(f"YASAKLI ifade bulundu (halusinasyon olabilir): {bulunan_yasakli}")
    if esnek_basarisiz:
        detaylar.append(f"Esnek ifadelerden HICBIRI gecmedi: {test['beklenen_ifadeler_esnek']}")
    if not kaynak_turu_dogru:
        detaylar.append(f"Kaynak turu yanlis: beklenen={beklenen_kt}, gelen={sonuc.get('kaynak_turu')}")

    return {**test, **sonuc, "basarili": basarili, "detay": " | ".join(detaylar) if detaylar else "OK"}


def rapor_olustur(sonuclar: list) -> str:
    """Sonuclari okunabilir bir Markdown raporuna cevirir."""
    toplam = len(sonuclar)
    basarili_sayisi = sum(1 for s in sonuclar if s["basarili"])

    satirlar = [
        "# Faz 6 - Otomatik Degerlendirme Raporu",
        "",
        f"**Toplam test:** {toplam}  |  **Basarili:** {basarili_sayisi}  |  **Basari orani:** {basarili_sayisi/toplam*100:.0f}%",
        "",
        "| ID | Tur | Sonuc | Sure (sn) | Detay |",
        "|---|---|---|---|---|",
    ]
    for s in sonuclar:
        simge = "✅" if s["basarili"] else "❌"
        satirlar.append(f"| {s['id']} | {s['tur']} | {simge} | {s['sure_sn']} | {s['detay']} |")

    satirlar.append("")
    satirlar.append("## Detayli Cevaplar")
    for s in sonuclar:
        satirlar.append(f"\n### {s['id']} ({'✅' if s['basarili'] else '❌'})")
        if "sorular" in s:
            satirlar.append(f"**Sorular (sırayla):** {' → '.join(s['sorular'])}")
        else:
            satirlar.append(f"**Soru:** {s['soru']}")
        satirlar.append(f"\n**Cevap:**\n```\n{s['cevap'][:800]}\n```")

    return "\n".join(satirlar)


if __name__ == "__main__":
    filtre = sys.argv[1] if len(sys.argv) > 1 else None

    if filtre == "metin":
        secili_testler = [t for t in TEST_SETI if t["tur"] in ("metin", "metin_coklu_tur")]
    elif filtre == "gorsel":
        secili_testler = [t for t in TEST_SETI if t["tur"] == "gorsel_pdf"]
    elif filtre == "video":
        secili_testler = [t for t in TEST_SETI if t["tur"] == "video"]
    else:
        secili_testler = TEST_SETI

    print(f"Toplam {len(secili_testler)} test calistirilacak...\n")

    cikti_yolu = os.path.join(_current_dir, "degerlendirme_raporu.md")

    sonuclar = []
    for i, test in enumerate(secili_testler, start=1):
        print(f"[{i}/{len(secili_testler)}] {test['id']} çalıştırılıyor...")
        sonuc = testi_calistir(test)
        sonuclar.append(sonuc)
        print(f"  -> {'✅ BAŞARILI' if sonuc['basarili'] else '❌ BAŞARISIZ'} ({sonuc['sure_sn']}sn) - {sonuc['detay']}")

        # DUZELTME: Basarisiz testlerde CEVABIN TAMAMINI da terminale
        # yazdiriyoruz - "hangi kelime eksik" bilgisi tek basina yetersiz,
        # gercekten mi eksik yoksa model FARKLI bir kelimeyle mi ifade
        # etmis ayirt etmek icin ham cevabi gormek gerekiyor.
        if not sonuc["basarili"] and sonuc["cevap"]:
            print(f"     --- HAM CEVAP ---\n     {sonuc['cevap'][:500]}\n     --- --- ---")

        # DUZELTME: Rapor artik HER TEST SONRASI (sadece en sonda degil)
        # diske yeniden yaziliyor - boylece script yarida kesilse (PC
        # sorunu, hata vb.) bile O ANA KADAR olan sonuclar KAYBOLMAZ.
        rapor = rapor_olustur(sonuclar)
        with open(cikti_yolu, "w", encoding="utf-8") as f:
            f.write(rapor)

    print("\n" + "=" * 60)
    basarili_sayisi = sum(1 for s in sonuclar if s["basarili"])
    print(f"SONUÇ: {basarili_sayisi}/{len(sonuclar)} test başarılı")
    print(f"Detaylı rapor kaydedildi: {cikti_yolu}")
    print("=" * 60)
