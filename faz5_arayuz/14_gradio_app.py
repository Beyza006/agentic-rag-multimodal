# -*- coding: utf-8 -*-
"""
Task 5.1 - Gradio Arayuzu (Birlesik Modern Sohbet)
------------------------------------------------------
Bu script, Faz 5'in ilk adimi: su ana kadar TERMINAL uzerinden calistirdigimiz
her seyi (Hybrid RAG agent - Faz 4, VLM gorsel/video analiz - Faz 3) TEK BIR
sohbet kutusunda birlestirir - tipki Claude.ai/ChatGPT gibi modern AI
sohbet uygulamalarinin calisma bicimine benzer sekilde.

Kullanilan bilesen: gr.MultimodalTextbox - Gradio'nun, hem METIN hem
DOSYA (gorsel/video) EKLEME ozelligini TEK bir giris kutusunda birlestiren
ozel bileseni. Bu bilesen kendi icinde:
  - Bir "+" dosya ekleme dugmesi
  - Yazi yazilinca aktiflesen bir GONDER OKU
iceriyor - yani bu iki ozelligi bizim ayrica kodlamamiza gerek yok,
bilesenin kendisi zaten bu sekilde tasarlanmis.

Yonlendirme mantigi (kullanici ne gonderirse):
  - SADECE METIN         -> Faz 4 Hybrid RAG agent'i (dokuman + web)
  - GORSEL (soru ile/siz) -> Faz 3 VLM (gorsel_sorusu_sor / resmi_aciklama_uret)
  - VIDEO (soru ile/siz)  -> Faz 3 VLM + Faz 3 video kare cikarma

Kullanim:
    python 14_gradio_app.py
"""

import sys
import os
import re
import mimetypes
import importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import gradio as gr

_current_dir = os.path.dirname(os.path.abspath(__file__))
_faz3_dir = os.path.join(_current_dir, "..", "faz3_multimodal")
_faz4_dir = os.path.join(_current_dir, "..", "faz4_hybrid_rag")


def _modul_yukle(dosya_yolu, modul_adi):
    spec = importlib.util.spec_from_file_location(modul_adi, dosya_yolu)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


# Faz 3 ve Faz 4'te yazdigimiz fonksiyonlari tekrar yazmiyoruz, oldugu
# gibi kullaniyoruz.
_vision = _modul_yukle(os.path.join(_faz3_dir, "08_vision.py"), "vision_module")
_video = _modul_yukle(os.path.join(_faz3_dir, "10_video.py"), "video_module")
_pdf_gorsel = _modul_yukle(os.path.join(_faz3_dir, "09_pdf_gorsel.py"), "pdf_gorsel_module")
_hybrid_agent = _modul_yukle(os.path.join(_faz4_dir, "13_hybrid_agent.py"), "hybrid_agent_module")

_agent = _hybrid_agent.hybrid_agent_olustur()

_KAYNAK_ETIKETLERI = {
    "rag": "\U0001F4C4 Doküman (KVKK)",
    "web": "\U0001F310 Web Arama",
    "bulunamadi": "\u2753 Bulunamadı",
}


def _dosya_video_mu(yol: str) -> bool:
    tur, _ = mimetypes.guess_type(yol)
    return bool(tur) and tur.startswith("video")


def _dosya_gorsel_mi(yol: str) -> bool:
    tur, _ = mimetypes.guess_type(yol)
    return bool(tur) and tur.startswith("image")





def _markdown_liste_temizle(metin: str) -> str:
    """
    VLM'in urettigi Markdown listelerinde bazen "*" isareti ile asil
    icerik arasina gereksiz bos satirlar giriyor (orn. "* \n\n6. Maddesi"),
    bu da Gradio'da BOS madde isaretleri + kopuk satirlar seklinde
    goruntuleniyordu (gercek testte gorüldü). Bu fonksiyon:
      1. "*" den sonra gelen bos satirlari kaldirip icerigi ayni satira ceker
      2. Ust uste birden fazla bos satiri tek bos satira indirir
      3. VLM'in bazen ayni cumleyi ARKA ARKAYA birden fazla kez tekrar
         etmesini (bilinen bir "takilma" hatasi, gercek testte 3 kez
         tekrarlanan bir cumle gorüldü) temizler - 06_llm_answer.py'deki
         cevabi_temizle'nin ayni mantigi, VLM cevaplarina da uygulaniyor.
    """
    # "* " sonrasi bir veya daha fazla bos satir varsa, iceriği yukari cek
    metin = re.sub(r"([*\-])[ \t]*\n\s*\n\s*", r"\1 ", metin)
    # "* " sonrasi tek satir sonu varsa da ayni satira cek
    metin = re.sub(r"([*\-])[ \t]*\n[ \t]*(?=\S)", r"\1 ", metin)
    # 3+ ardisik satir sonunu 2'ye indir
    metin = re.sub(r"\n{3,}", "\n\n", metin)

    # Ardisik tekrarlanan cumle/ifadeleri temizle (VLM "takilma" hatasi)
    tekrar_deseni = re.compile(r"(\b.{15,150}?[.!?])\s+\1", re.IGNORECASE)
    onceki_hal = None
    while onceki_hal != metin:
        onceki_hal = metin
        metin = tekrar_deseni.sub(r"\1", metin, count=1)

    return metin.strip()


def _gorsel_soru_tablo_ile_mi_ilgili(soru: str) -> bool:
    """
    Kullanicinin sorusu bir TABLO/belge okuma sorusu mu, yoksa genel bir
    gorsel sorusu mu (manzara, nesne vb.) oldugunu, sorudaki anahtar
    kelimelere bakarak tahmin eder. Tablo-odakli talimat sadece gercekten
    tablo/belge sorularinda kullanilmali - manzara fotografina "sutunlari
    dogru esitle" demek anlamsiz olurdu.
    """
    tablo_kelimeleri = ["tablo", "sütun", "sutun", "satır", "satir", "liste",
                         "belge", "madde", "kanun", "hücre", "hucre"]
    soru_kucuk = soru.lower()
    return any(k in soru_kucuk for k in tablo_kelimeleri)


# Genel (tablo olmayan) gorsel sorularina eklenecek anti-halusinasyon
# talimati - VLM'in "olmayan bina/otobus" gibi seyler uydurmasini
# engellemek icin (gercek testte gorsel ve video cevaplarinda bu tur
# halusinasyonlar gozlemlendi).
_GORSEL_ANTI_HALUSINASYON = (
    "\n\nÖNEMLİ: Sadece görselde AÇIKÇA gördüğün şeyleri anlat. "
    "Emin olmadığın, belirsiz veya seçemediğin hiçbir nesneyi/detayı "
    "(olmayan binalar, yapılar, araçlar gibi) UYDURMA. Görselde bir şeyi "
    "net seçemiyorsan, onu hiç belirtme - tahminde bulunma."
)


def _metin_modeline_sor(prompt: str) -> str:
    import requests
    try:
        yanit = requests.post(
            _vision.OLLAMA_API_URL,
            json={"model": _vision.OLLAMA_METIN_MODEL, "prompt": prompt, "stream": False},
            timeout=400
        )
        yanit.raise_for_status()
        return yanit.json()["response"]
    except Exception as e:
        return f"\u26a0\ufe0f Metin modeli hatasi: {e}"


def _gorsel_video_ile_cevapla(dosya_yolu: str, soru: str) -> str:
    """
    Faz 3'teki VLM/video/OCR fonksiyonlarini kullanarak yuklenen dosyayi
    analiz eder.
    
    YENI MIMARI (Chain-of-Thought): Tablo odakli sorularda VLM'e dogrudan soru sormak yerine,
    1. Once Qwen2.5-VL ile tabloyu sorusuz, pur pruzsuz bir Markdown'a kopyalatiyoruz.
    2. Elde edilen bu eksiksiz Markdown tablosunu, sorumuzla birlikte 
       guclu metin LLM'ine (Gemma2) yollariz. 
       (PaddleOCR kaldirildi, cunku karakter atlama sorunlari yasatti)
    """
    try:
        if _dosya_video_mu(dosya_yolu):
            if soru:
                # kare_sayisi=5 KULLANILIYOR (varsayilan) - terminal
                # testlerimizle (Task 3.4) TUTARLI olmasi icin.
                cevap = _video.video_zaman_akisini_sor(dosya_yolu, soru)
            else:
                kare_yolu = _video.video_karesini_cikar(dosya_yolu, 0)
                cevap = _vision.resmi_aciklama_uret(kare_yolu)
        else:
            if soru:
                if _gorsel_soru_tablo_ile_mi_ilgili(soru):
                    # YENI MIMARI: VLM Markdown Cikarimi (Adim 1)
                    vlm_markdown_prompt = (
                        "Sen bir tablo okuma uzmanısın. Bu görseldeki tabloyu hiçbir harfini, "
                        "sayısını (örn. 27), maddesini (örn. Geçici Madde) veya tarihini atlamadan "
                        "TAM VE EKSİKSİZ bir Markdown tablosuna çevir. Tabloda ne görüyorsan "
                        "birebir aynısını yaz. Soruyu cevaplamaya veya ekstra açıklama yapmaya çalışma, "
                        "sadece Markdown tablosunu ver.\n\n"
                        "TARİH KURALI (ÇOK ÖNEMLİ): Bu Türkçe bir belgedir, tarihler GÜN/AY/YIL "
                        "formatındadır (İngilizce'deki AY/GÜN formatı DEĞİLDİR). Örneğin '5/12/2017' "
                        "gördüğünde bunu '12/05/2017' gibi AY/GÜN sırasına ÇEVİRME - ilk sayı GÜN'dür, "
                        "gördüğün sırayı ASLA değiştirme."
                    )
                    ocr_metni = _vision.gorsel_sorusu_sor(dosya_yolu, vlm_markdown_prompt)
                    
                    # LLM Sentezleme (Adim 2)
                    llm_prompt = f"""Aşağıda Qwen-VL modeli tarafından görselden büyük bir özenle çıkarılmış, eksiksiz bir Markdown tablosu bulunmaktadır.
Bu tablo verisini dikkatlice inceleyerek kullanıcının sorusuna en yüksek kalitede, eksiksiz bir yanıt ver.

ÖNEMLİ KURALLAR:
1. Tabloda yer alan birden fazla madde, sayı veya numara varsa ("19, 20, 21" veya "Geçici Madde 2" gibi), hiçbirini atlama, hepsini belirterek detaylıca anlat.
2. Soruyu yanıtlarken tabloda geçen tarihleri OLDUĞU GİBİ, gördüğün sırayla kullan - gün/ay yerlerini değiştirme veya yeniden yorumlama.
3. Tabloda OLMAYAN hiçbir bilgiyi (madde, tarih, isim vb.) asla uydurma.

--- ÇIKARILAN TABLO VERİSİ BAŞLANGICI ---
{ocr_metni}
--- ÇIKARILAN TABLO VERİSİ BİTİŞİ ---

Kullanıcının Sorusu: {soru}
"""
                    cevap = _metin_modeline_sor(llm_prompt)
                else:
                    zenginlestirilmis_soru = soru + _GORSEL_ANTI_HALUSINASYON
                    cevap = _vision.gorsel_sorusu_sor(dosya_yolu, zenginlestirilmis_soru)
            else:
                cevap = _vision.resmi_aciklama_uret(dosya_yolu)
    except Exception as hata:
        return f"\u26a0\ufe0f Hata: {hata}"

    return _markdown_liste_temizle(cevap)


def sohbet_fonksiyonu(mesaj_verisi: dict, gecmis_gradio: list, oturum_gecmisi: list):
    """
    ANA YONLENDIRME FONKSIYONU (YENI: STREAMING VE LOG DESTEKLI)
    Kullanicinin UI'da bekleme hissini azaltmak icin 'yield' ile adim adim
    Durum guncellemeleri (Log Paneli) ekrana yansitilir. Asla kalite 
    dusurulmez, ayni LangGraph mimarisi 'invoke' yerine 'stream' ile cagirilir.
    """
    import gradio as gr
    
    metin = (mesaj_verisi.get("text") or "").strip()
    dosyalar = mesaj_verisi.get("files") or []

    if not metin and not dosyalar:
        yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)
        return

    # Kullanicinin mesajlarini gecmise ekle
    for dosya_yolu in dosyalar:
        gecmis_gradio.append({"role": "user", "content": {"path": dosya_yolu}})
    if metin:
        gecmis_gradio.append({"role": "user", "content": metin})

    # Asistanin bos bir yanit balonu olusturulur (Streaming baslangici)
    gecmis_gradio.append({"role": "assistant", "content": "🔄 _Sistem başlatılıyor..._"})
    yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)

    if dosyalar:
        # GORSEL/VIDEO YOLU (Faz 3) - Streaming desteksiz, tek adimli yurutme
        gecmis_gradio[-1]["content"] = "👁️ _Görsel/Video yapay zeka tarafından analiz ediliyor (Lütfen bekleyin)..._"
        yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)
        
        cevap = _gorsel_video_ile_cevapla(dosyalar[0], metin if metin else None)
        cevap = cevap.replace("\n", "\n\n").replace("\n\n\n\n", "\n\n")
        
        gecmis_gradio[-1]["content"] = cevap
        yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)
        return

    # SADECE METIN YOLU (Faz 4 - Hybrid RAG Streaming)
    baslangic_state = {
        "soru": metin,
        "arac_karari": "",
        "cevap": "",
        "kaynaklar": [],
        "gecmis": oturum_gecmisi,
        "arama_sorgusu": "",
        "ham_sonuc": {},
        "rag_yeterli": True,
        "web_sonuclari": [],
        "kaynak_turu": "",
    }

    loglar = []
    try:
        # Agent.invoke yerine stream() kullaniyoruz (LangGraph)
        for state in _agent.stream(baslangic_state):
            for dugum_adi, dugum_verisi in state.items():
                if dugum_adi == "karar": loglar.append("- 🧠 Kullanıcı sorusunun amacı analiz edildi.")
                elif dugum_adi == "rag": loglar.append("- 🔍 Vektör veritabanı (KVKK) taranıyor...")
                elif dugum_adi == "yeterlilik": loglar.append("- ⚖️ Dokümandaki bilgilerin soruyu karşılayıp karşılamadığı kontrol ediliyor...")
                elif dugum_adi == "web": loglar.append("- 🌐 Doküman bilgisi yetersiz bulundu, internette geniş çaplı araştırma (Tavily) yapılıyor...")
                elif dugum_adi in ["hibrit_cevap", "cevap_uret", "dogrudan"]:
                    loglar.append("- ✍️ Elde edilen verilerle nihai cevap sentezleniyor...")

                # Guncel loglari gecici olarak ekrana bas (Streaming hissiyati)
                guncel_log_metni = "⏳ **İşlem Devam Ediyor:**\n\n" + "\n".join(loglar)
                gecmis_gradio[-1]["content"] = guncel_log_metni
                yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)
                
                # Cevap uretildiyse (Grafin sonuna gelindiyse)
                if "cevap" in dugum_verisi and (dugum_adi in ["cevap_uret", "hibrit_cevap", "dogrudan"]):
                    cevap = dugum_verisi.get("cevap", "")
                    kaynak_turu = dugum_verisi.get("kaynak_turu", "")
                    kaynaklar = dugum_verisi.get("kaynaklar", [])
                    
                    kaynak_metni = ""
                    if kaynaklar:
                        etiket = _KAYNAK_ETIKETLERI.get(kaynak_turu, kaynak_turu)
                        kaynak_satirlari = "\n".join(f"- {k}" for k in kaynaklar)
                        kaynak_metni = f"\n\n---\n**Kaynak türü:** {etiket}\n\n**Kullanılan kaynaklar:**\n{kaynak_satirlari}"

                    # Task 5.4 - Log Paneli: Basarili calisan asamalari goster
                    log_paneli_html = "\n\n<details><summary>🔍 Ajan İşlem Logları (Tıkla)</summary>\n\n" + "\n".join(loglar) + "\n\n</details>"
                    
                    tam_cevap = cevap + kaynak_metni + log_paneli_html
                    tam_cevap = tam_cevap.replace("\n", "\n\n").replace("\n\n\n\n", "\n\n")
                    
                    oturum_gecmisi = oturum_gecmisi + [(metin, cevap)]
                    gecmis_gradio[-1]["content"] = tam_cevap
                    yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)
                    
    except Exception as hata:
        gecmis_gradio[-1]["content"] = f"\u26a0\ufe0f Ajan Yürütme Hatası: {hata}"
        yield gecmis_gradio, oturum_gecmisi, gr.MultimodalTextbox(value=None)


def sohbeti_temizle():
    return [], []


# ============================================================
# GORSEL TASARIM
# ============================================================

_TEMA = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    # ACIK MOD degerleri
    body_background_fill="#eef0f6",
    block_background_fill="#ffffff",
    block_border_color="rgba(0,0,0,0.06)",
    body_text_color="#1e1f26",
    # KOYU MOD degerleri (senin eklendigin tasarim buraya tasindi)
    body_background_fill_dark="#0b0f19",
    block_background_fill_dark="#151b28",
    block_border_color_dark="rgba(255,255,255,0.05)",
    body_text_color_dark="#f8fafc",
    block_border_width="1px",
    block_radius="24px",
    button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #a855f7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5 0%, #9333ea 100%)",
    button_primary_text_color="white",
)

_OZEL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
}

/* ACIK MOD (varsayilan) */
body, .gradio-container {
    background: #eef0f6 !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
    color: #1e1f26 !important;
}

/* KOYU MOD - tema_btn'e tiklaninca <body>'e eklenen "dark" sinifi
   burada devreye giriyor. Senin eklendigin ozgun koyu tasarim
   burada korunuyor. */
.dark body, .dark .gradio-container {
    background: #0b0f19 !important;
    color: #f8fafc !important;
}

.gradio-container {
    max-width: 850px !important;
    margin: 0 auto !important;
    padding-top: 30px !important;
}

#baslik-satiri {
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 10px 0 5px 0;
    text-align: center;
    margin-bottom: 5px;
}

#tema-buton {
    position: absolute;
    right: 6px;
    top: 4px;
    min-width: 40px !important;
    max-width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    font-size: 1.05rem;
    border: 1px solid rgba(0,0,0,0.08) !important;
    background: white !important;
}
.dark #tema-buton {
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: #151b28 !important;
}

#baslik-metni h3 {
    font-size: 2.2rem;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.03em;
    font-family: 'Outfit', sans-serif;
}

#alt-baslik {
    color: #8b8f9c;
    font-size: 1rem;
    font-weight: 300;
    text-align: center;
    margin: 0 0 25px 0;
    letter-spacing: 0.5px;
}
.dark #alt-baslik { color: #94a3b8; }

#sohbet-kutusu {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 6px;
}
.dark #sohbet-kutusu {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Gradio'nun varsayilan "kuyruk/isleme suresi" durum yazisini gizle -
   bu, tekniksel bir hata ayiklama detayi, gercek bir AI sohbet
   uygulamasinda gorunmez. */
.message .timer, .message-wrap .timer,
[class*="pending"] .timer {
    display: none !important;
}

/* Sohbet balonlari */
.message-wrap.user .message {
    background: var(--primary-gradient) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
    font-size: 0.95rem;
}

.message-wrap.bot .message {
    background: rgba(0, 0, 0, 0.03) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 20px 20px 20px 4px !important;
    color: #1e1f26 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
    font-size: 0.95rem;
}
.dark .message-wrap.bot .message {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #e2e8f0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

#giris-kutusu {
    border-radius: 24px !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    background: #ffffff !important;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.05) !important;
    margin-top: 20px !important;
    transition: all 0.3s ease;
}
.dark #giris-kutusu {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background: rgba(21, 27, 40, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.3) !important;
}

#giris-kutusu:focus-within {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.2) !important;
}

#giris-kutusu textarea {
    color: inherit !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
}

#temizle-satiri {
    display: flex;
    justify-content: center;
    margin-top: 15px;
}

#temizle-btn {
    box-shadow: none !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    background: rgba(0,0,0,0.02) !important;
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    border-radius: 20px !important;
    padding: 5px 15px !important;
    transition: all 0.3s ease;
}
.dark #temizle-btn {
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.03) !important;
    color: #94a3b8 !important;
}

#temizle-btn:hover {
    color: #4f46e5 !important;
}
.dark #temizle-btn:hover {
    color: white !important;
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.2) !important;
}

footer { display: none !important; }
.gap { gap: 0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
.dark ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); }
"""

_TEMA_DEGISTIR_JS = """
() => {
    document.body.classList.toggle('dark');
}
"""


# ============================================================
# ARAYUZ TASARIMI
# ============================================================

with gr.Blocks(title="KVKK AI Asistanı") as demo:
    with gr.Row(elem_id="baslik-satiri"):
        gr.Markdown("### KVKK AI Asistanı", elem_id="baslik-metni")
        tema_btn = gr.Button("\U0001F313", elem_id="tema-buton", size="sm")
        
    gr.Markdown(
        "Gelişmiş RAG • Hibrit Arama • Multimodal Vizyon Analizi",
        elem_id="alt-baslik",
    )

    chatbot = gr.Chatbot(
        label=None,
        show_label=False,
        height=550,
        elem_id="sohbet-kutusu",
        avatar_images=(None, None),
        placeholder=(
            "<div style='text-align: center; margin-top: 80px; opacity: 0.8;'>"
            "<h1 style='font-size: 3rem; margin-bottom: 10px;'>👋</h1>"
            "<h2 style='font-size: 1.5rem; font-weight: 600; margin-bottom: 15px; color: #f8fafc; font-family: Outfit;'>KVKK Asistanı'na Hoş Geldiniz</h2>"
            "<p style='font-size: 1rem; color: #94a3b8; max-width: 500px; margin: 0 auto; line-height: 1.6; font-family: Inter;'>"
            "KVKK ile ilgili sorular sorabilir, <b>web'de arama</b> yapmamı isteyebilir "
            "veya aşağıdaki <b style='color:#a855f7;'>+</b> simgesinden görsel/video "
            "yükleyerek analiz etmemi sağlayabilirsiniz.</p>"
            "</div>"
        ),
    )
    oturum_state = gr.State([])  # Task 2.4 - Session Memory

    giris_kutusu = gr.MultimodalTextbox(
        show_label=False,
        placeholder="Mesajınızı yazın veya dosya ekleyin...",
        sources=["upload"],
        file_types=["image", "video"],
        elem_id="giris-kutusu",
        container=False,
        submit_btn=True,
    )

    with gr.Row(elem_id="temizle-satiri"):
        temizle_btn = gr.Button("🗑️ Sohbeti Temizle", elem_id="temizle-btn", size="sm")

    giris_kutusu.submit(
        sohbet_fonksiyonu,
        inputs=[giris_kutusu, chatbot, oturum_state],
        outputs=[chatbot, oturum_state, giris_kutusu],
    )
    temizle_btn.click(sohbeti_temizle, outputs=[chatbot, oturum_state])
    tema_btn.click(None, js=_TEMA_DEGISTIR_JS)


if __name__ == "__main__":
    demo.launch(theme=_TEMA, css=_OZEL_CSS)
