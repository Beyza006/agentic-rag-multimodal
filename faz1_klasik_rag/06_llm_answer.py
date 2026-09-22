# -*- coding: utf-8 -*-
"""
Task 1.6 - LLM ile Cevap Uretme
---------------------------------
Bu script, Faz 1'in son adimi: Task 1.5'teki retriever'i kullanarak
alakali chunk'lari bulur, bunlari bir LLM'e (Ollama uzerinden calisan
yerel bir model) "baglam" (context) olarak verir ve LLM'in bu baglama
dayanarak DOGAL, AKICI bir Turkce cevap uretmesini saglar.

Kullanilan LLM: Ollama uzerinden calisan yerel bir model (varsayilan:
gemma2:9b). Model adi asagida tek bir degiskende (OLLAMA_MODEL) tutuluyor.

Kullanim:
    python 06_llm_answer.py "Kişisel veriler ne zaman açık rıza olmadan işlenebilir?"
"""

import sys
import os
import re
import importlib.util

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OLLAMA_MODEL = "gemma2:9b"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

TOP_K = 10


_current_dir = os.path.dirname(os.path.abspath(__file__))
_retriever_yolu = os.path.join(_current_dir, "05_retriever.py")
_spec = importlib.util.spec_from_file_location("retriever_module", _retriever_yolu)
_retriever = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retriever)


def bentleri_cikar(metin: str) -> list:
    """
    Ham chunk metninde "a) ...", "b) ...", "ç) ..." gibi harfli bentleri
    REGEX ile (LLM'e sormadan) kod tarafinda cikarir.
    """
    satirlar = metin.split("\n")
    bent_deseni = re.compile(r"^([a-zçğıöşü])\)\s*(.*)$")
    # KVKK'da harfli bentlerden ("a) ... b) ...") SONRA genelde numarali
    # fikralar ("(2) ...", "(3) ..." gibi) gelir - bunlar ARTIK bir onceki
    # bendin devami DEGIL, tamamen farkli/bagimsiz paragraflardir. Bu
    # deseni tanimadan, bentleri_cikar TUM sonraki metni (sonraki fikralar
    # dahil) son bendin icine yanlislikla ekliyordu (orn. MADDE 18/d
    # bendine fikra (2),(3),(4)'un de sizmasi gibi).
    fikra_deseni = re.compile(r"^\(\d+\)")
    bentler = []
    mevcut_harf = None
    mevcut_metin = []

    for satir in satirlar:
        satir_temiz = satir.strip()
        eslesme = bent_deseni.match(satir_temiz)
        if eslesme:
            if mevcut_harf:
                bentler.append((mevcut_harf, " ".join(mevcut_metin).strip()))
            mevcut_harf = eslesme.group(1)
            mevcut_metin = [eslesme.group(2)]
        elif fikra_deseni.match(satir_temiz):
            # Numarali bir fikraya gecildi - bent toplamayi burada
            # KESIN olarak durduruyoruz (bu fikra, ayri bir yapi).
            if mevcut_harf:
                bentler.append((mevcut_harf, " ".join(mevcut_metin).strip()))
            mevcut_harf = None
            mevcut_metin = []
        elif mevcut_harf:
            mevcut_metin.append(satir_temiz)

    if mevcut_harf:
        bentler.append((mevcut_harf, " ".join(mevcut_metin).strip()))

    # Son bentteki kapanis fazlaligini (orn. "haklarina sahiptir") temizle
    if bentler:
        son_harf, son_icerik = bentler[-1]
        kapanis_deseni = re.compile(
            r",?\s*(haklarına sahiptir|hâlinde mümkündür|halinde mümkündür|"
            r"şarttır|zorunludur)\.?\s*$",
            re.IGNORECASE
        )
        temizlenmis = kapanis_deseni.sub("", son_icerik).rstrip(", ").strip()
        if temizlenmis and temizlenmis != son_icerik:
            bentler[-1] = (son_harf, temizlenmis + ".")

    return bentler


def giris_cumlesi_uret(soru: str, madde_no: str, bent_sayisi: int) -> str:
    """Sadece TEK bir giris cumlesi istiyoruz - liste yazmasini istemiyoruz."""
    prompt = f"""Soru: {soru}

Bu soruya, {madde_no} kapsamında {bent_sayisi} farklı alt bent/durum ile cevap
verilecek. SADECE bu soruya uygun, 1 CÜMLELİK bir giriş yaz (örn. soru ceza soruyorsa
"İlgili cezai hükümler aşağıdaki gibidir:", şart soruyorsa "Bu şartlar şunlardır:" tarzı). 
Liste YAZMA, sadece giriş cümlesini yaz, başka hiçbir şey ekleme.

ÇOK ÖNEMLİ: Sana verilmeyen hiçbir bilgiyi (kanun numarası, tarih, madde
dışında ek referans vb.) UYDURMA/EKLEME - sadece "{madde_no}" ifadesini
kullanabilirsin, başka hiçbir sayı veya isim EKLEME."""
    return ollama_ile_cevap_uret(prompt).strip()


def onsel_tanim_metnini_cikar(metin: str) -> str:
    """
    DUZELTME (gercek testte bulunan hata - MADDE 6 ornegi): Bir maddenin
    BIRDEN FAZLA fikrasi olabilir - bazilari harfli bent (a,b,c...)
    icerir, bazilari (orn. bir TANIM cumlesi) hic bent icermeden tek
    basina anlamli bir paragraftir. kod_tabanli_nihai_cevap_olustur
    SADECE bentleri kullandigi icin, boyle bir "tanim fikrasi" (orn.
    MADDE 6/1: "Kişilerin ırkı, etnik kökeni... özel nitelikli kişisel
    veridir.") SESSIZCE ATLANIYORDU - bu da "X nedir?" turu tanim
    sorularinda, sistemin yanlislikla SADECE istisnalari (bentli fikra)
    cevap olarak vermesine, asil TANIMI hic göstermemesine yol acti.

    Bu fonksiyon, İLK harfli bent baslamadan ONCE gelen metni yakalar.
    Eger bu metin yeterince UZUNSA (80+ karakter - kisa bir madde
    basligini/numarasini yanlislikla "tanim" sanmamak icin), anlamli
    bir tanim/aciklama paragrafi oldugu varsayilir ve dondurulur.
    Kisa ise (orn. MADDE 18'in "Bu Kanunun;" gibi bir baslangici),
    bos string doner - boylece zaten iyi calisan maddelerin cevabina
    gereksiz kirinti eklenmez.
    """
    satirlar = metin.split("\n")
    bent_deseni = re.compile(r"^([a-zçğıöşü])\)\s*(.*)$")
    # Belge hiyerarşisini belirten (Örn. "İKİNCİ BÖLÜM") yapısal başlıkların, içerik (önsel) metne dahil edilmesini engellemek için ilgili satırları filtreliyoruz.
    bolum_baslik_deseni = re.compile(
        r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+BÖLÜM$",
        re.IGNORECASE
    )

    onsel_satirlar = []
    for satir in satirlar:
        satir_temiz = satir.strip()
        if bent_deseni.match(satir_temiz):
            break
        if satir_temiz and not bolum_baslik_deseni.match(satir_temiz):
            onsel_satirlar.append(satir_temiz)

    onsel_metin = " ".join(onsel_satirlar).strip()

    if len(onsel_metin) < 80:
        return ""

    return onsel_metin


def kod_tabanli_nihai_cevap_olustur(soru: str, madde_no: str, bentler: list, onsel_metin: str = "") -> str:
    """
    Nihai cevabi TAMAMEN kod tarafinda, deterministik olarak birlestirir.
    LLM'den sadece giris cumlesini aliyoruz (dar, basit bir gorev); bent
    icerikleri dogrudan kaynak metinden (regex ile cikarilmis, garantili
    dogru) geliyor.

    Not: Kullanıcı arayüzlerinde (Gradio vb.) Markdown formatının doğru render edilebilmesi için satır birleşimlerinde çift satır atlama ("\n\n") karakterleri kullanılmaktadır. Bu durum, metin bloklarının tek paragraf altında rastgele birleşmesini önler.

    onsel_metin Parametresi: Madde bentlerinden önce yer alan yapısal açıklama fıkraları, giriş cümlesinden hemen sonra, asıl bentlerden ise önce eklenir; böylelikle tam bağlam korunmuş olur.
    """
    giris = giris_cumlesi_uret(soru, madde_no, len(bentler))

    satirlar = [giris]
    if onsel_metin:
        satirlar.append(onsel_metin)
    satirlar.append("")
    for harf, icerik in bentler:
        satirlar.append(f"{madde_no}/{harf}) {icerik}")

    return "\n\n".join(satirlar)


def eksiksiz_liste_prompti_olustur(soru: str, sonuc: dict) -> str:
    """
    DURUM 2 (bentsiz, tekil bilgi) icin - LLM'e ham bilgiyi ciktirtiyoruz.
    """
    baglam_parcalari = []
    for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0]):
        baglam_parcalari.append(f"[{meta['madde_no']}]\n{doc}")
    baglam = "\n\n".join(baglam_parcalari)

    return f"""Aşağıdaki kanun maddesi, kullanıcının sorusuna cevap vermek üzere veri tabanından özel olarak seçilmiştir.
Kullanıcının sorusunda geçen kelimeler metinde birebir geçmese bile (örneğin soru özel bir ihlali sorarken metin genel olarak 'suçlar' diyorsa), bu metin hukuken o sorunun cevabıdır.
Görevin, metindeki bilgiyi/kuralı doğrudan özetleyerek çıkarmaktır. "Bilgi yok" DEME.

--- KANUN MADDESİ ---
{baglam}
--- --- ---

Soru: {soru}

Çıkarılan Bilgi (Doğrudan bilgi ver):"""


def akici_hale_getir_prompti_olustur(soru: str, eksiksiz_liste: str) -> str:
    """
    DURUM 2 icin ham bilgiyi akici, dogru bir cevaba cevirir.
    """
    return f"""Aşağıda bir soruya verilmiş, ham bir bilgi var. Bu bilgiyi
kullanarak NİHAİ cevabı şu KURALLARA göre oluştur:

- Soruyu DOĞRUDAN ve TAM cümlelerle cevapla - liste/madde satırı YAPMA,
  sadece 1-3 cümlelik akıcı bir paragraf yaz.
- Kaynaktaki İLGİLİ EK BAĞLAMI da (bu madde/kurum/kavram ne hakkında)
  cevaba doğal şekilde dahil et ki cevap tek başına anlamlı ve tam olsun.
- Madde referansını cümle içinde doğal şekilde geç (örn. "İlgili maddeye göre..."), ayrı bir satır/liste maddesi olarak YAZMA.
- Cevabın SONUNA madde numarasını tekrar eden ayrı bir satır veya
  tırnak içinde alıntı EKLEME - madde referansı sadece cümle içinde,
  bir kez geçsin, tekrar etme.
- HİÇBİR YILDIZ (*) İŞARETİ KULLANMA.

Soru: {soru}

Ham bilgi:
{eksiksiz_liste}

Nihai cevap (yıldız/markdown YOK, dogrudan cevapla basla):"""


def madde_parcalarini_genislet(koleksiyon, sonuc: dict) -> dict:
    """
    En alakali sonucun ait oldugu maddenin TUM alt-chunklarini ChromaDB'den
    ekstra olarak ceker - boylece bolunmus bir madde bile TAM icerigiyle
    LLM'e ulasir.
    """
    if not sonuc["documents"][0]:
        return sonuc

    en_alakali_madde = sonuc["metadatas"][0][0]["madde_no"]

    tum_parcalar = koleksiyon.get(
        where={"madde_no": en_alakali_madde},
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


def cevabi_temizle(cevap: str) -> str:
    """
    Modelin bazen talimata ragmen ekledigi, gereksiz/tekrarci bir kapanis
    ifadesini ve kekeleme tarzi tekrarlari programatik olarak temizler.
    """
    tum_madde_sayisi = len(re.findall(r"\bMADDE\s+\d+\b", cevap, re.IGNORECASE))

    if tum_madde_sayisi == 1:
        desen = re.compile(r"(\.\s+|\n\s*)(MADDE\s+\d+\b.*)", re.IGNORECASE | re.DOTALL)
        eslesme = desen.search(cevap)
        if eslesme and eslesme.start() > 0:
            cevap = cevap[:eslesme.start()].rstrip()

    cevap = re.sub(r",?\s*haklarına sahiptir\.?", ".", cevap, flags=re.IGNORECASE)

    # EK KURAL: Metnin en SONUNDA, tek basina duran (tirnak icinde veya
    # cikinti) bir "MADDE X" ifadesi varsa, metindeki TOPLAM MADDE sayisi
    # ne olursa olsun (1 veya daha fazla) bunu temizle. Yukaridaki kural
    # sadece "toplam 1 MADDE varsa" calisiyor, ama cevap icinde MADDE
    # zaten bir kez mesru olarak gecip SONRA bir kez daha (bu sefer
    # anlamsiz, yalniz basina) tekrar edebiliyor - bu kural, sayidan
    # bagimsiz olarak SADECE bu ozel "yalniz kalan kirintiyi" hedefliyor.
    # re.IGNORECASE ile "MADDE 17" / "Madde 17" farki gozetmiyoruz.
    cevap = re.sub(r'\s*["\']\s*MADDE\s+\d+\s*["\']\s*$', "", cevap, flags=re.IGNORECASE)
    cevap = re.sub(r'(?<=[.!?])\s*MADDE\s+\d+\s*$', "", cevap, flags=re.IGNORECASE)

    # DUZELTME (gercek testte bulunan kozmetik sorun - doc_13 ornegi):
    # Model bazen, cevabin ic kisminda ZATEN soyledigi bir cumleyi,
    # metnin en SONUNDA bir kez daha, bu sefer TIRNAK ICINDE tekrar
    # ediyordu (orn. "...otuz gün içinde sonuçlandırır. [...]
    # "Veri sorumlusu... otuz gün içinde sonuçlandırır.""). Onceki
    # tekrar_deseni (asagida) SADECE BITISIK (araya baska metin
    # girmeyen) ve 100 karakterden KISA tekrarlari yakaliyordu - bu
    # durumda ARADA farkli bir cumle var VE tekrar 100 karakterden
    # UZUN, o yuzden yakalanmiyordu. Bu yeni kural, metnin EN SONUNDAKI
    # tirnakli bir ifadeyi, onun ilk birkaç kelimesi metnin BASKA bir
    # yerinde de geciyorsa (yani gercekten bir TEKRAR ise), kaldırır.
    son_tirnakli_desen = re.compile(r'\s*["\']([^"\']{20,400})["\']\.?\s*$')
    eslesme = son_tirnakli_desen.search(cevap)
    if eslesme:
        tirnak_icerigi = eslesme.group(1).strip()
        metnin_geri_kalani = cevap[:eslesme.start()]
        tirnak_basi = tirnak_icerigi[:40].lower()
        if tirnak_basi and tirnak_basi in metnin_geri_kalani.lower():
            cevap = metnin_geri_kalani.rstrip()

    tekrar_deseni = re.compile(r"(\b.{15,100}?)\s+\1", re.IGNORECASE)
    onceki_hal = None
    while onceki_hal != cevap:
        onceki_hal = cevap
        cevap = tekrar_deseni.sub(r"\1", cevap, count=1)

    cevap = cevap.replace("*", "")
    cevap = re.sub(r"\.{2,}", ".", cevap)
    cevap = re.sub(r"\s+\.", ".", cevap)

    return cevap.strip()


def chunklari_ortusmeyi_temizleyerek_birlestir(ilgili_kayitlar: list) -> str:
    """
    DUZELTME (gercek testte bulunan hata): Ayni maddeye ait, chunk_index
    sirasina gore siralanmis parcalari TEK bir metinde birlestirir - ama
    chunking sirasinda (Task 1.2) EKLENEN kasitli ORTUSMEYI (overlap,
    ~80 karakter) de dedupe ederek.

    NEDEN GEREKLI: Onceki yaklasim, bentleri_cikar'i HER PARCAYA AYRI AYRI
    uyguluyordu (overlap tekrarini onlemek icin). Ama bu, bir bendin
    (orn. MADDE 18/d) TAM OLARAK chunk sinirina denk gelmesi durumunda
    ciddi bir soruna yol aciyordu: bendin ilk parcasi bir chunk'ta YARIM
    kaliyor, devam metni ise SONRAKI chunk'ta "d)" harfi OLMADAN
    basladigi icin (overlap nedeniyle) bentleri_cikar tarafindan HICBIR
    bende ait olarak TANINMIYOR ve SESSIZCE KAYBOLUYORDU (gercek testte
    "50.000 Türk lirasından... idari para cezası verilir" ifadesinin
    tamamen dustugu gozlemlendi).

    COZUM: Once TUM parcalari, aralarindaki ortusen (duplicate) metni
    TESPIT EDIP SADECE BIR KEZ sayarak TEK bir surekli metinde
    birlestiriyoruz - boylece bentleri_cikar'a chunk sinirlari HIC
    gorunmeyen, kesintisiz bir metin gidiyor ve bir bendin chunk
    sinirini asan kismi da dogru sekilde yakalaniyor.

    # Not: Bu fonksiyon dış modüller tarafından (örneğin 07_agent.py) yedek (fallback) mekanizması olarak çağrılmaktadır.
def ollama_ile_cevap_uret(prompt: str, model_override: str = None) -> str:
    """
    Prompt'u Ollama'nin yerel REST API'sine gonderir ve uretilen cevabi
    dondurur.
    """
    import requests
    
    kullanilacak_model = model_override if model_override else OLLAMA_MODEL

    try:
        yanit = requests.post(
            OLLAMA_API_URL,
            json={
                "model": kullanilacak_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 4096,         # LLM bağlam boyutu sınırlaması (RAM/Hız optimizasyonu)
                    "num_predict": 1024,     # Maksimum üretilecek token (Cevapların uzayıp sarkmasını önler)
                    "keep_alive": "15m"      # Hybrid agent düğümler arası geçişte modeli VRAM'de sıcak tutar
                }
            },
            timeout=400
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
            f"'{OLLAMA_MODEL}' modelinin indirilmis oldugundan emin olun "
            f"('ollama pull {OLLAMA_MODEL}')."
        )

    return yanit.json()["response"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanim: python 06_llm_answer.py \"<soru>\"")
        sys.exit(1)

    soru = sys.argv[1]

    try:
        koleksiyon = _retriever.koleksiyonu_ac()
    except RuntimeError as hata:
        print(f"\n\u274c HATA: {hata}")
        sys.exit(1)

    print("Embedding modeli yukleniyor...")
    embedding_modeli = _retriever.modeli_yukle()

    print(f"Soru embed ediliyor ve en alakali {TOP_K} chunk araniyor...")
    sonuc = _retriever.ara(embedding_modeli, koleksiyon, soru, top_k=TOP_K)

    if not sonuc["documents"][0]:
        print("\nHicbir alakali bilgi bulunamadi, cevap uretilemiyor.")
        sys.exit(0)

    sonuc = madde_parcalarini_genislet(koleksiyon, sonuc)

    en_alakali_madde_no = sonuc["metadatas"][0][0]["madde_no"]
    ilgili_kayitlar = sorted(
        [
            (doc, meta) for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0])
            if meta["madde_no"] == en_alakali_madde_no
        ],
        key=lambda x: x[1]["chunk_index"]
    )

    # Parçaları (chunk) örtüşmeleri temizleyerek birleştirip tek, kesintisiz bir metin oluşturuyoruz. Bu sayede chunk sınırlarına denk gelen tüm ifadeler eksiksiz olarak işlenebilmektedir.
    TURKCE_ALFABE = "abcçdefgğhıijklmnoöprsştuüvyz"

    def alfabe_sira_no(harf):
        try:
            return TURKCE_ALFABE.index(harf)
        except ValueError:
            return 999

    birlesik_metin = chunklari_ortusmeyi_temizleyerek_birlestir(ilgili_kayitlar)
    bentler = bentleri_cikar(birlesik_metin)

    if len(bentler) >= 2:
        print(f"Asama 1/2: {len(bentler)} bent regex ile koddan cikarildi (LLM'e sorulmadi, garantili).")
        print("Asama 2/2: Giris cumlesi uretiliyor (format kod tarafinda sabit)...")
        onsel_metin = onsel_tanim_metnini_cikar(birlesik_metin)
        try:
            cevap = kod_tabanli_nihai_cevap_olustur(soru, en_alakali_madde_no, bentler, onsel_metin)
        except RuntimeError as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)
    else:
        print("Asama 1/2: Eksiksiz liste olusturuluyor...")
        liste_prompti = eksiksiz_liste_prompti_olustur(soru, sonuc)
        try:
            eksiksiz_liste = ollama_ile_cevap_uret(liste_prompti)
        except RuntimeError as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)

        print("Asama 2/2: Akici hale getiriliyor...")
        akici_prompt = akici_hale_getir_prompti_olustur(soru, eksiksiz_liste)
        try:
            cevap = ollama_ile_cevap_uret(akici_prompt)
        except RuntimeError as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)

    cevap = cevabi_temizle(cevap)

    print("\n" + "=" * 60)
    print(f"SORU: {soru}")
    print("=" * 60)
    print(f"\nCEVAP:\n{cevap}")

    print("\n" + "-" * 60)
    print("Kullanilan kaynaklar:")
    for meta in sonuc["metadatas"][0]:
        print(f"  - {meta['madde_no']} (sayfa {meta['sayfa_no']})")

    print("\n🎉 Faz 1 - Klasik RAG tamamlandi!")