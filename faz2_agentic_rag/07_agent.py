# -*- coding: utf-8 -*-
"""
Task 2.3 - LangGraph ile Agent Workflow
------------------------------------------
Bu script, Faz 1'de kurdugumuz SABIT workflow'u (PDF -> chunk -> embed ->
ara -> cevap uret, her zaman ayni sirada calisan) bir ADIM ileri tasiyip
gercek bir AGENT'a donusturur.

Faz 1 ile Faz 2 arasindaki fark (Task 2.1'de ogrendigimiz kavram):
- Faz 1: WORKFLOW - her soru icin ayni sabit adimlar calisir.
- Faz 2: AGENT - sistem, soruya bakip HANGI aracin cagrilacagina
  CALISMA ZAMANINDA (runtime) kendisi karar verir.

Bu ilk agent'ta SADECE 2 dugum var (basit tutmak icin):
  1) karar_dugumu: soru KVKK ile mi ilgili, alakasiz mi - LLM'e sorarak
     karar verir (bu, Task 2.1'de ogrendigimiz "Tool Calling" mantiginin
     basitlestirilmis bir versiyonu - LLM burada "hangi arac" kararini
     yapisal bir sekilde veriyor).
  2) rag_dugumu VEYA dogrudan_cevap_dugumu: karara gore CALISTIRILAN dugum.

LangGraph kavramlari (Task 2.3):
- State: dugumler arasinda paylasilan, okunup yazilabilen ortak veri yapisi.
- Node: bir is birimini temsil eden fonksiyon (State al, State guncelle).
- Edge: iki dugum arasindaki sabit gecis.
- Conditional Edge: bir dugumden CIKISTA, State'teki bir degere gore HANGI
  dugume gidilecegine karar veren dinamik gecis - bu, agent'in "karar
  verme" yetenegini saglayan asil mekanizma.

Kullanim:
    python 07_agent.py "Kişisel veriler ne zaman açık rıza olmadan işlenebilir?"
    python 07_agent.py "Bugün hava nasıl?"
"""

import sys
import os
import re
import importlib.util
from typing import TypedDict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from langgraph.graph import StateGraph, END


# --- Faz 1'deki modulleri import ediyoruz (dosya adlari rakamla basladigi
# icin importlib ile dosya yolundan yukluyoruz, ayni Faz 1'deki gibi) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_faz1_dir = os.path.join(_current_dir, "..", "faz1_klasik_rag")


def _modul_yukle(dosya_adi, modul_adi):
    yol = os.path.join(_faz1_dir, dosya_adi)
    spec = importlib.util.spec_from_file_location(modul_adi, yol)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


_retriever = _modul_yukle("05_retriever.py", "retriever_module")
_llm_answer = _modul_yukle("06_llm_answer.py", "llm_answer_module")


# ============================================================
# STATE: Agent'in dugumler arasinda tasidigi ortak veri yapisi
# ============================================================

class AgentState(TypedDict):
    soru: str            # kullanicinin girdisi
    arac_karari: str      # "rag" veya "dogrudan" - karar dugumunun cikardigi sonuc
    cevap: str            # nihai cevap
    kaynaklar: list       # kullanilan madde/sayfa referanslari (varsa)
    gecmis: list          # [(soru, cevap), ...] - Task 2.4: SESSION MEMORY
                          # Bu oturum boyunca sorulan onceki soru-cevaplari
                          # tutar. Programi kapatinca kaybolur (kalici degil) -
                          # bu yuzden "Session Memory", "Persistent Memory" degil.
    yansitma: str         # Task 2.5: REFLECTION - agent'in kendi cevabini
                          # kendi kendine degerlendirdigi kisa notu tutar.
    onerilen_madde: str   # Task 2.5: SELF-CORRECTION - reflection'in "bu
                          # madde daha uygun olurdu" dedigi madde numarasi.
    arama_sorgusu: str    # rag_dugumu'nde kullanilan (Memory ile
                          # zenginlestirilmis) sorgu - duzelt_dugumu'nde
                          # tekrar kullanilir.
    ham_sonuc: dict        # retriever'dan donen HAM sonuc (documents +
                          # metadatas) - duzelt_dugumu, YENIDEN arama
                          # yapmadan bu veriyi kullanarak farkli bir
                          # maddeyle cevap uretebilsin diye saklanir.


# ============================================================
# NODE 1: Karar dugumu - "Tool Calling" mantiginin basitlestirilmis hali
# ============================================================

def karar_dugumu(state: AgentState) -> dict:
    """
    Soruyu LLM'e gonderip "KVKK ile ilgili mi, alakasiz mi" diye siniflandirir.

    TASK 2.4 GUNCELLEMESI: Artik SADECE mevcut soruya degil, gecmis
    (Session Memory) varsa ONA da bakarak karar veriyor. Boylece "peki
    cezasi nedir?" gibi, tek basina belirsiz duran ama bir onceki KVKK
    sorusunun DEVAMI olan bir soru da dogru siniflandirilabiliyor -
    hafizasiz bir agent bunu yanlislikla "ALAKASIZ" sanabilirdi.
    """
    soru = state["soru"]
    gecmis = state.get("gecmis", [])

    baglam_metni = ""
    if gecmis:
        son_soru, son_cevap = gecmis[-1]
        baglam_metni = f"""
ONCEKI KONUSMA BAGLAMI (varsa yeni soru buna gore degerlendirilmeli):
Onceki soru: {son_soru}
Onceki cevabin ozeti: {son_cevap[:200]}
"""

    prompt = f"""Aşağıdaki soru, Kişisel Verilerin Korunması Kanunu (KVKK) veya
kişisel veri koruma konusuyla mı ilgili, yoksa tamamen ALAKASIZ bir konu mu
(örn. hava durumu, yemek tarifi, spor vb.)?
{baglam_metni}
Yeni soru: {soru}

ÖNEMLİ: Eğer yeni soru, önceki konuşmanın DEVAMI niteliğindeyse (örn.
"peki cezası nedir?", "bu durumda ne olur?" gibi, önceki KVKK konusuna
atıfta bulunan takip sorularıysa), bunu da İLGİLİ say.

SADECE tek bir kelime yaz: "ILGILI" veya "ALAKASIZ". Başka hiçbir şey yazma."""

    yanit = _llm_answer.ollama_ile_cevap_uret(prompt).strip().upper()

    if "ALAKASIZ" in yanit:
        karar = "dogrudan"
    else:
        karar = "rag"

    print(f"[Agent karari] Soru '{'KVKK ile ilgili' if karar == 'rag' else 'alakasiz'}' olarak siniflandirildi -> '{karar}' dugumune yonlendiriliyor.")

    return {"arac_karari": karar}


def yonlendirme_fonksiyonu(state: AgentState) -> str:
    """
    LangGraph'in CONDITIONAL EDGE mekanizmasi bu fonksiyonu cagirir ve
    donen string'e gore bir sonraki dugumu secer (asagidaki add_conditional_edges
    cagrisindaki sozluk eslesmesine bakar).
    """
    return state["arac_karari"]


# ============================================================
# NODE 2a: RAG dugumu - Faz 1'in tum pipeline'ini calistirir
# ============================================================

def sorguyu_baglamla_zenginlestir(soru: str, gecmis: list) -> str:
    """
    TASK 2.4: Eger kullanicinin sorusu bir onceki konusmaya atifta
    bulunuyorsa (orn. "peki cezasi nedir?" - "ceza" kelimesi tek basina
    hangi konudaki cezayi kastettigini belli etmez), bu soruyu ARAMA
    ICIN daha isabetli, BAGIMSIZ bir sorguya donusturuyoruz. Bu,
    gercek RAG sistemlerinde "conversational query rewriting" (query
    condensation) olarak bilinen, yaygin kullanilan bir tekniktir.

    Eger gecmis yoksa, soru oldugu gibi kullanilir - bu adim sadece
    takip sorularinda devreye girer.
    """
    if not gecmis:
        return soru

    son_soru, _ = gecmis[-1]
    prompt = f"""Önceki soru: {son_soru}
Yeni soru: {soru}

Yeni soru, önceki sorunun devamı/takibi niteliğindeyse, bu iki soruyu
birleştirerek TEK BAŞINA anlaşılır, bağımsız bir arama sorgusu oluştur.

ÇOK ÖNEMLİ - TERMİNOLOJİYİ KORU: Yeni sorudaki ÖZEL/SPESİFİK kelimeleri
(örneğin "hukuka aykırı", "kaydetme", "aydınlatma yükümlülüğü", "veri
güvenliği" gibi hukuki/teknik terimleri) OLDUĞU GİBİ KORU - bunları
daha genel/soyut bir ifadeyle DEĞİŞTİRME veya sadeleştirme. Kullanıcının
kullandığı SPESİFİK kelimeler, doğru kaynağı bulmak için kritiktir;
bunları kaybetmek yanlış/eksik sonuçlara yol açar.

Eğer yeni soru zaten kendi başına anlaşılırsa, sadece yeni soruyu
aynen döndür.

SADECE oluşturulan sorguyu yaz, başka hiçbir açıklama ekleme."""

    return _llm_answer.ollama_ile_cevap_uret(prompt).strip()

    return birlesik


def belirtilen_madde_icin_cevap_uret(arama_sorgusu: str, sonuc: dict, madde_no: str) -> str:
    """
    Verilen bir madde numarasi icin (sonuc icindeki ilgili chunk'lardan)
    cevap uretir. rag_dugumu (ilk cevap) ve duzelt_dugumu (Task 2.5 -
    Self-Correction, farkli bir maddeyle YENIDEN uretim) tarafindan
    ORTAK olarak kullanilir - boylece ayni mantik iki yerde tekrar
    yazilmiyor.
    """
    ilgili_kayitlar = sorted(
        [
            (doc, meta) for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0])
            if meta["madde_no"] == madde_no
        ],
        key=lambda x: x[1]["chunk_index"]
    )

    # DUZELTME: chunklari_ortusmeyi_temizleyerek_birlestir + bentleri_cikar
    # ikilisi artik SADECE 06_llm_answer.py'de TANIMLI (tek kaynak) -
    # burada sadece cagiriyoruz, boylece iki kopyanin birbirinden
    # SENKRONSUZ kalma riski ortadan kalkiyor.
    birlesik_metin = _llm_answer.chunklari_ortusmeyi_temizleyerek_birlestir(ilgili_kayitlar)
    bentler = _llm_answer.bentleri_cikar(birlesik_metin)

    if len(bentler) >= 2:
        cevap = _llm_answer.kod_tabanli_nihai_cevap_olustur(arama_sorgusu, madde_no, bentler)
    else:
        liste_prompti = _llm_answer.eksiksiz_liste_prompti_olustur(arama_sorgusu, sonuc)
        eksiksiz_liste = _llm_answer.ollama_ile_cevap_uret(liste_prompti)
        akici_prompt = _llm_answer.akici_hale_getir_prompti_olustur(arama_sorgusu, eksiksiz_liste)
        cevap = _llm_answer.ollama_ile_cevap_uret(akici_prompt)

    return _llm_answer.cevabi_temizle(cevap)


# ============================================================
# YENI: LLM-based Reranking - Task 2.5 duzeltmesi
# ============================================================

def en_uygun_maddeyi_sec(arama_sorgusu: str, sonuc: dict) -> str:
    """
    TASK 2.5 DUZELTMESI: Retrieval sonucundaki TEKIL maddeleri LLM'e
    sunup soruya en uygun olanini sectiriyoruz (Reranking).

    NEDEN GEREKLI: Embedding benzerligi bazen yanlis maddeyi birinci
    siraya koyabiliyor. Ornegin "cezasi nedir?" sorusunda "acik riza"
    kelimesi MADDE 5'te cok gectiqi icin cosine similarity onu yukari
    cikariyor, ama cevap aslinda MADDE 17/18'de (cezai hukumler).
    Bu adim, retrieval'dan sonra ama cevap uretiminden ONCE devreye
    girerek dogru maddeyi seciyor.

    NEDEN SELF-CORRECTION YERINE BU: "5 madde arasindan soruya en
    uygununu sec" (siniflandirma gorevi) kucuk modeller icin,
    "urettigin cevabi degerlendir" (meta-bilissel gorev) den cok
    daha kolay ve GUVENILIR bir gorevdir. Ayrica yanlis cevap uretip
    sonra duzeltmek yerine, bastan dogru maddeyi secmek 1 LLM
    cagrisi tasarruf eder.
    """
    # Tekil maddeleri ve ilk chunk ozetlerini cikar
    tekil_maddeler = {}
    for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0]):
        madde = meta["madde_no"]
        if madde not in tekil_maddeler:
            tekil_maddeler[madde] = doc[:200]  # Ilk 200 karakter ozet olarak

    if len(tekil_maddeler) <= 1:
        # Tek madde varsa reranking'e gerek yok
        return list(tekil_maddeler.keys())[0]

    madde_listesi = "\n".join(
        f"- {madde}: {ozet}..." for madde, ozet in tekil_maddeler.items()
    )

    prompt = f"""Aşağıdaki KVKK maddeleri arama sonucunda bulundu. Soruyu en
DOĞRUDAN ve en SPESİFİK şekilde cevaplayan TEK maddeyi seç.

Soru: {arama_sorgusu}

Bulunan maddeler:
{madde_listesi}

ÖNEMLİ: Sorudaki ANAHTAR kavrama (ör. "ceza" soruluyorsa cezai hükümler
maddesi, "haklar" soruluyorsa haklar maddesi) odaklan. Sadece genel olarak
konuyla ilişkili olan değil, soruyu DOĞRUDAN cevaplayan maddeyi seç.

SADECE madde adını yaz (ör. "MADDE 18"), başka hiçbir şey yazma."""

    yanit = _llm_answer.ollama_ile_cevap_uret(prompt).strip().upper()

    # LLM'in cevabinan madde numarasini cikar
    eslesme = re.search(r"MADDE\s+\d+", yanit)
    if eslesme:
        secilen = eslesme.group(0)
        # Sadece GERCEKTEN sonuclarda olan bir maddeyi kabul et
        if secilen in tekil_maddeler:
            ilk_madde = list(tekil_maddeler.keys())[0]
            if secilen != ilk_madde:
                print(f"[Reranking] Embedding siralamasi degistirildi: "
                      f"{ilk_madde} -> {secilen}")
            return secilen

    # Guvenli fallback: LLM gecersiz cevap verdiyse ilk sonucu kullan
    print(f"[Reranking] LLM gecerli bir madde secemedi, ilk sonuc kullaniliyor.")
    return list(tekil_maddeler.keys())[0]


def madde_parcalarini_genislet_hedefli(koleksiyon, sonuc: dict, hedef_madde: str) -> dict:
    """
    madde_parcalarini_genislet ile ayni mantik, ama HER ZAMAN
    en_alakali_madde (ilk sonuc) yerine belirtilen hedef_madde icin
    genisletir. Bu, reranking farkli bir madde sectiginde o maddenin
    tum chunklarinin da sonuca eklenmesini saglar.
    """
    if not sonuc["documents"][0]:
        return sonuc

    tum_parcalar = koleksiyon.get(
        where={"madde_no": hedef_madde},
        include=["documents", "metadatas"]
    )

    mevcut_idler = set()
    for meta in sonuc["metadatas"][0]:
        mevcut_idler.add((meta["madde_no"], meta["chunk_index"]))

    for doc, meta in zip(tum_parcalar["documents"], tum_parcalar["metadatas"]):
        anahtar = (meta["madde_no"], meta["chunk_index"])
        if anahtar not in mevcut_idler:
            sonuc["documents"][0].append(doc)
            sonuc["metadatas"][0].append(meta)
            mevcut_idler.add(anahtar)

    return sonuc


# ============================================================
# GUNCELLENMIS rag_dugumu - Reranking eklenmis hali
# ============================================================

def rag_dugumu(state: AgentState) -> dict:
    """
    Faz 1'de kurdugumuz retriever + generation pipeline'ini oldugu gibi
    kullanir - Task 2.3'un amaci Faz 1'i YENIDEN YAZMAK degil, onu bir
    "arac" (tool) olarak agent'a BAGLAMAK.

    TASK 2.5 GUNCELLEMESI: Artik ilk siraya koru koruye guvenmek yerine,
    LLM-based reranking ile soruya en uygun maddeyi secip, O maddenin
    chunklarini genisletip, O maddeyle cevap uretiyor.
    """
    soru = state["soru"]
    gecmis = state.get("gecmis", [])

    # TASK 2.4: Takip sorularinda, aramayi gecmis baglamla zenginlestir
    arama_sorgusu = sorguyu_baglamla_zenginlestir(soru, gecmis)
    if arama_sorgusu != soru:
        print(f"[Memory] Takip sorusu tespit edildi, arama sorgusu zenginlestirildi: '{arama_sorgusu}'")

    koleksiyon = _llm_answer._retriever.koleksiyonu_ac()
    embedding_modeli = _llm_answer._retriever.modeli_yukle()
    sonuc = _llm_answer._retriever.ara(embedding_modeli, koleksiyon, arama_sorgusu, top_k=_llm_answer.TOP_K)

    if not sonuc["documents"][0]:
        return {
            "cevap": "Bu soruyla ilgili dokümanlarda bir bilgi bulunamadı.",
            "kaynaklar": []
        }

    # --- TASK 2.5 DEGISIKLIK: Reranking + hedefli genisletme ---
    # ONCE reranking ile en uygun maddeyi sec, SONRA o maddenin
    # chunklarini genislet. Eski kod: once genislet (yanlis madde
    # icin), sonra kor koruye ilk sonucu al.
    en_alakali_madde_no = en_uygun_maddeyi_sec(arama_sorgusu, sonuc)
    sonuc = madde_parcalarini_genislet_hedefli(koleksiyon, sonuc, en_alakali_madde_no)
    # --- TASK 2.5 DEGISIKLIK SONU ---

    cevap = belirtilen_madde_icin_cevap_uret(arama_sorgusu, sonuc, en_alakali_madde_no)

    # Kaynaklar listesinde tekrar olmasin diye dedup ediyoruz (sirayi
    # koruyarak) - madde_parcalarini_genislet_hedefli bazen ayni chunk'i
    # birden fazla eklemis olabiliyor, bu da "Kullanilan kaynaklar"
    # listesinde ayni maddenin birden fazla kez gorunmesine yol aciyordu.
    kaynaklar = []
    for meta in sonuc["metadatas"][0]:
        kaynak_metni = f"{meta['madde_no']} (sayfa {meta['sayfa_no']})"
        if kaynak_metni not in kaynaklar:
            kaynaklar.append(kaynak_metni)

    return {
        "cevap": cevap,
        "kaynaklar": kaynaklar,
        "arama_sorgusu": arama_sorgusu,
        "ham_sonuc": sonuc
    }


# ============================================================
# NODE 3: Yansitma (Reflection) dugumu - Task 2.5
# ============================================================

def yansitma_dugumu(state: AgentState) -> dict:
    """
    TASK 2.5 - REFLECTION: Agent, kendi urettigi cevabi kendi kendine
    degerlendirir. Bu ilk versiyon SADECE GOZLEMLER, henuz DUZELTME
    yapmaz (Self-Correction bir sonraki adimda eklenecek).

    Neden gerekli: rag_dugumu, arama sonucunda bulunan BIRDEN FAZLA
    maddeden sadece EN YUKSEK BENZERLIK SKORLU olani kullanarak cevap
    uretiyor. Ama bazen soru, retrieval'in bulup KULLANMADIGI baska bir
    maddeyi (orn. "ceza" sorusu icin MADDE 17/18) daha dogru
    cevaplayabilir. Bu dugum, tam olarak bunu kontrol ediyor: "cevap,
    bulunan maddeler arasinda GERCEKTEN en dogrusunu mu kullandi?"
    """
    soru = state["soru"]
    cevap = state["cevap"]
    kaynaklar = state.get("kaynaklar", [])

    if not kaynaklar:
        # Dogrudan-cevap yolundan geldiyse (RAG hic calismadiysa)
        # yansitmaya gerek yok.
        return {"yansitma": "", "onerilen_madde": ""}

    # Kaynaklardaki TEKIL (tekrarsiz) madde numaralarini cikar
    tekil_maddeler = []
    for k in kaynaklar:
        madde_adi = k.split(" (sayfa")[0]
        if madde_adi not in tekil_maddeler:
            tekil_maddeler.append(madde_adi)

    prompt = f"""Bir soru-cevap sistemi su cevabi uretti. Bu cevabin
GERCEKTEN dogru olup olmadigini degerlendir.

Soru: {soru}

Uretilen cevap: {cevap}

Arama sirasinda bulunan TUM ilgili maddeler (cevap bunlardan sadece
BIRINI kullanmis olabilir): {", ".join(tekil_maddeler)}

Degerlendirmen gereken soru: Uretilen cevap, yukaridaki listede bulunan
DIGER maddelerden biri kullanilsaydi SORUYA DAHA DOGRU/DAHA ILGILI bir
cevap verilebilir miydi? (orn. "ceza" sorulan bir soruda "haklar"
maddesinin kullanilmasi gibi bir uyumsuzluk var mi?)

SADECE su iki formattan birini kullanarak KISA cevap ver:
"UYGUN: <neden uygun oldugunun 1 cumlelik aciklamasi>"
veya
"UYGUN DEGIL: <hangi maddenin daha uygun olabilecegi ve neden>" """

    yansitma_sonucu = _llm_answer.ollama_ile_cevap_uret(prompt).strip()

    onerilen_madde = ""
    if yansitma_sonucu.upper().startswith("UYGUN DEGIL") or yansitma_sonucu.upper().startswith("UYGUN DEĞİL"):
        # Onerilen maddeyi metinden regex ile cikar (orn. "MADDE 18")
        eslesme = re.search(r"MADDE\s+\d+", yansitma_sonucu, re.IGNORECASE)
        if eslesme:
            aday_madde = eslesme.group(0).upper().replace("MADDE", "MADDE")
            # Sadece GERCEKTEN kaynaklarda bulunan bir maddeyi onerelim -
            # LLM'in var olmayan bir madde uydurmasina karsi guvenlik
            if aday_madde in tekil_maddeler:
                onerilen_madde = aday_madde

        if onerilen_madde:
            print(f"[Reflection] \u26a0\ufe0f  Agent kendi cevabini yetersiz buldu, '{onerilen_madde}' ile duzeltmeyi deneyecek: {yansitma_sonucu}")
        else:
            print(f"[Reflection] \u26a0\ufe0f  Agent kendi cevabini yetersiz buldu ama guvenilir bir alternatif bulamadi: {yansitma_sonucu}")
    else:
        print(f"[Reflection] \u2705 Agent cevabini kontrol etti, uygun buldu.")

    return {"yansitma": yansitma_sonucu, "onerilen_madde": onerilen_madde}


def yansitma_sonrasi_yonlendirme(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: Reflection'dan sonra, eger guvenilir bir alternatif
    madde onerildiyse "duzelt" dugumune, aksi halde direkt bitis (END)e
    gidiyoruz. Bu, Self-Correction'in NE ZAMAN devreye girecegine karar
    veren mekanizma.
    """
    if state.get("onerilen_madde"):
        return "duzelt"
    return "bitir"


def duzelt_dugumu(state: AgentState) -> dict:
    """
    TASK 2.5 - SELF-CORRECTION: Reflection, mevcut cevabin yetersiz
    oldugunu ve BASKA bir maddenin (onerilen_madde) daha uygun oldugunu
    tespit ettiyse, bu dugum cevabi O MADDEYLE YENIDEN uretir.

    Onemli: YENIDEN embedding/arama YAPMIYORUZ - zaten Task 1.5'te
    bulunan (ham_sonuc'ta saklanan) chunk'lari kullanarak, sadece HANGI
    maddenin bentlerini kullandigimizi degistiriyoruz. Bu hem hizli hem
    de "hayali" bir maddeye atif yapma riskini ortadan kaldiriyor (cunku
    onerilen_madde zaten yansitma_dugumu'nde kaynaklarda dogrulanmisti).

    NOT: Sonsuz donguyu onlemek icin SADECE 1 kez duzeltme yapiyoruz -
    duzelt_dugumu'nden sonra tekrar yansitma'ya DONMUYORUZ, direkt bitiyor.
    """
    onerilen_madde = state["onerilen_madde"]
    arama_sorgusu = state.get("arama_sorgusu") or state["soru"]
    sonuc = state["ham_sonuc"]

    print(f"[Self-Correction] '{onerilen_madde}' kullanilarak cevap yeniden uretiliyor...")

    yeni_cevap = belirtilen_madde_icin_cevap_uret(arama_sorgusu, sonuc, onerilen_madde)

    return {
        "cevap": yeni_cevap,
        "yansitma": state["yansitma"] + f" [Self-Correction ile '{onerilen_madde}' kullanilarak duzeltildi.]"
    }


# ============================================================
# NODE 2b: Dogrudan cevap dugumu - RAG'a hic gerek olmadigi durumlar icin
# ============================================================

def dogrudan_cevap_dugumu(state: AgentState) -> dict:
    """
    Soru KVKK ile alakasizsa, hicbir arama/embedding islemi yapmadan
    (gereksiz maliyetten kacinarak) kullaniciyi bilgilendiren sabit bir
    cevap doner. Bu, agent'in "her zaman RAG calistir" yerine "gerekliyse
    calistir" mantiginin somut faydasini gosteriyor.
    """
    return {
        "cevap": (
            "Bu sistem şu an sadece Kişisel Verilerin Korunması Kanunu (KVKK) "
            "ile ilgili sorulara cevap verebiliyor. Başka bir konuda "
            "yardımcı olamıyorum."
        ),
        "kaynaklar": []
    }


# ============================================================
# GRAPH: State/Node/Edge/Conditional Edge'leri birlestirip agent'i kuruyoruz
# ============================================================

def agent_olustur():
    graph = StateGraph(AgentState)

    graph.add_node("karar", karar_dugumu)
    graph.add_node("rag", rag_dugumu)
    graph.add_node("yansitma", yansitma_dugumu)
    graph.add_node("duzelt", duzelt_dugumu)
    graph.add_node("dogrudan", dogrudan_cevap_dugumu)

    graph.set_entry_point("karar")

    # CONDITIONAL EDGE: karar dugumunden cikista, yonlendirme_fonksiyonu'nun
    # dondurdugu degere gore "rag" ya da "dogrudan" dugumune gidilir.
    graph.add_conditional_edges(
        "karar",
        yonlendirme_fonksiyonu,
        {"rag": "rag", "dogrudan": "dogrudan"}
    )

    graph.add_edge("rag", "yansitma")

    # NOT (23.07 - gecici geri alma): Self-Correction (duzelt dugumu)
    # bazen Reflection'in KENDISI yanlis degerlendirme yapip DOGRU bir
    # cevabi YANLIS bir maddeyle degistirebildigi icin (9B modelin
    # kendi-kendini-degerlendirme guvenilirlik siniri) SIMDILIK devre
    # disi birakildi. duzelt_dugumu fonksiyonu ve altyapisi KODDA DURUYOR
    # (silinmedi) - ileride ya modeli buyuterek ya da Reflection promptunu
    # daha temkinli hale getirerek tekrar denenecek. Su an icin Reflection
    # SADECE GOZLEMLIYOR, cevabi hic degistirmiyor.
    #
    # ANCAK: Reranking (en_uygun_maddeyi_sec) eklenmesiyle asil sorun
    # (yanlis madde secimi) CEVAP URETIMINDEN ONCE cozuluyor - bu yuzden
    # Self-Correction'a ihtiyac buyuk olcude azaldi.
    graph.add_edge("yansitma", END)

    graph.add_edge("duzelt", END)  # kullanilmiyor ama graph gecerliligi icin duruyor
    graph.add_edge("dogrudan", END)

    return graph.compile()


if __name__ == "__main__":
    agent = agent_olustur()
    gecmis = []  # Task 2.4: Session Memory - program calistigi surece hatirlanir

    print("=" * 60)
    print("KVKK Agent - Sohbet Modu (Task 2.4: Session Memory ile)")
    print("Cikmak icin 'q' veya 'çıkış' yazabilirsiniz.")
    print("=" * 60)

    # Eger komut satirindan dogrudan bir soru verildiyse (eski kullanim
    # sekliyle geriye-uyumluluk), onu ilk soru olarak isle, sonra sohbete
    # devam et.
    ilk_soru = sys.argv[1] if len(sys.argv) >= 2 else None

    while True:
        if ilk_soru is not None:
            soru = ilk_soru
            ilk_soru = None
        else:
            soru = input("\nSoru: ").strip()

        if soru.lower() in ("q", "çıkış", "cikis", "exit"):
            print("Görüşürüz!")
            break
        if not soru:
            continue

        baslangic_state = {
            "soru": soru,
            "arac_karari": "",
            "cevap": "",
            "kaynaklar": [],
            "gecmis": gecmis,
            "yansitma": "",
            "onerilen_madde": "",
            "arama_sorgusu": "",
            "ham_sonuc": {}
        }
        sonuc_state = agent.invoke(baslangic_state)

        print("\n" + "=" * 60)
        print(f"SORU: {soru}")
        print("=" * 60)
        print(f"\nCEVAP:\n{sonuc_state['cevap']}")

        if sonuc_state["kaynaklar"]:
            print("\n" + "-" * 60)
            print("Kullanilan kaynaklar:")
            for k in sonuc_state["kaynaklar"]:
                print(f"  - {k}")

        # Task 2.4: Bu tur, bir sonraki turda hatirlanmak uzere gecmise eklenir
        gecmis.append((soru, sonuc_state["cevap"]))

        print("\n\U0001F916 Task 2.4 - Memory ile agent yaniti tamamlandi!")