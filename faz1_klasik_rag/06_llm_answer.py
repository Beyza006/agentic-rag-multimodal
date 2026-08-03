# -*- coding: utf-8 -*-
"""
Task 1.6 - LLM ile Cevap Uretme
---------------------------------
Bu script, Faz 1'in son adimi: Task 1.5'teki retriever'i kullanarak
alakali chunk'lari bulur, bunlari bir LLM'e (Ollama uzerinden calisan
yerel bir model) "baglam" (context) olarak verir ve LLM'in bu baglama
dayanarak DOGAL, AKICI bir Turkce cevap uretmesini saglar.

Bu, RAG'in "Generation" (uretim) kismidir - Retrieval (Task 1.5) +
Generation (bu script) = RAG (Retrieval-Augmented Generation).

Kullanilan LLM: Ollama uzerinden calisan yerel bir model (varsayilan:
gemma2:9b). Model adi asagida tek bir degiskende (OLLAMA_MODEL) tutuluyor,
boylece ileride farkli bir model denemek istersen (orn. qwen2.5:7b) SADECE
o satiri degistirmen yeterli - kodun geri kalanina dokunmana gerek yok.

ON KOSUL: Ollama'nin bilgisayarinda kurulu ve calisiyor olmasi, ve
'ollama pull <model_adi>' ile modelin onceden indirilmis olmasi gerekir.

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

# Ollama'da kullanilacak model - degistirmek istersen SADECE bu satiri guncelle
OLLAMA_MODEL = "gemma2:9b"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

TOP_K = 5  # kac chunk getirilecek (Task 1.5'teki ile ayni mantik)


# --- 05_retriever.py'yi import ediyoruz (dosya adi rakamla basladigi icin
# importlib ile dosya yolundan yukluyoruz, tipki 02/03/04'te yaptigimiz gibi) ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
_retriever_yolu = os.path.join(_current_dir, "05_retriever.py")
_spec = importlib.util.spec_from_file_location("retriever_module", _retriever_yolu)
_retriever = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_retriever)


def eksiksiz_liste_prompti_olustur(soru: str, sonuc: dict) -> str:
    """
    ASAMA 1: Modele SADECE eksiksiz bir liste cikarmasini istiyoruz -
    format/akicilik onemli degil bu asamada, tek onceligi HICBIR SEYI
    ATLAMAMAK. Kucuk modeller, tek ve basit bir gorevi (sadece listele)
    karmasik cok-kurallli bir gorevden (hem eksiksiz hem akici hem
    aciklamali) çok daha guvenilir sekilde yapar.
    """
    baglam_parcalari = []
    for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0]):
        baglam_parcalari.append(f"[{meta['madde_no']}, sayfa {meta['sayfa_no']}]\n{doc}")
    baglam = "\n\n".join(baglam_parcalari)

    return f"""Aşağıdaki KVKK madde metinlerini oku. Kullanıcının sorusuyla
ilgili TÜM şartları/bentleri (varsa a, b, c, ç, d, e, f... hepsini)
eksiksiz bir şekilde madde madde listele. Hiçbirini atlama. Her şart için:
1) Şartın hangi maddeye ait olduğunu belirt (örn. "MADDE 5").
   ÖNEMLİ: Sadece kaynak metinde GERÇEKTEN a), b), c) gibi harfli bentler
   varsa harf ekle (örn. "MADDE 5/a"). Kaynak metinde böyle bir harf YOKSA,
   harf UYDURMA - sadece madde numarasını yaz (örn. "MADDE 19"), fıkra
   numarasını parantez içinde belirtebilirsin (örn. "MADDE 19 (3)").
2) Şartın kendisini yaz.
3) Günlük hayattan somut, kısa bir örnek ekle.
4) Eğer soru TEK bir basit gerçeğe (örn. bir kurumun nerede kurulduğu,
   kaç üyesi olduğu gibi) işaret ediyorsa, sadece o cümleyi değil, o
   kurumun/kavramın ne olduğuna dair YAKIN BAĞLAMI da (kaynak metindeki
   ilgili diğer cümleleri) ekle - cevap tek bir izole cümle gibi kalmasın.
Format önemli değil, sadece EKSİKSİZLİK, DOĞRU madde/bent referansı
(uydurma yok) ve her şart için örnek bulunması önemli.

--- KANUN MADDELERİ ---
{baglam}
--- --- ---

Soru: {soru}

Eksiksiz liste (madde referansı + örnek + gerekli bağlam dahil):"""


def akici_hale_getir_prompti_olustur(soru: str, eksiksiz_liste: str) -> str:
    """
    ASAMA 2: Asama 1'de elde ettigimiz (garantili eksiksiz) ham listeyi
    alip, SORUNUN TURUNE GORE iki farkli formattan birine sokuyoruz:

    - Soru birden fazla sart/bent gerektiriyorsa: kisa bir giris cumlesi +
      altta net, kisa maddeler halinde liste.
    - Soru tek, dogrudan bir cevap gerektiriyorsa (liste yoksa): normal,
      makul uzunlukta bir aciklama paragrafi - ne çok kisa ne gereksiz uzun.

    Bu tasarim, "her cevap ayni kaliba zorlanmasin, sorunun yapisina uygun
    olsun" prensibine dayanir.
    """
    return f"""Aşağıda bir soruya verilmiş, ham/eksiksiz bir bilgi listesi
var. Bu bilgiyi kullanarak NİHAİ cevabı şu KURALLARA göre oluştur:

ÖNCE KARAR VER: Aşağıdaki ham bilgide KAÇ TANE ayrı şart/madde/referans var?
- Eğer 2 VEYA DAHA FAZLA ayrı şart/madde varsa → DURUM 1'i uygula.
- Eğer SADECE 1 TANE şart/bilgi varsa → DURUM 2'yi uygula.

DURUM 1 (birden fazla şart):
  - Önce 1-2 cümlelik bir giriş yaz. Bu giriş SORUYU DOĞRUDAN CEVAPLAMALI
    VE birden fazla farklı/bağımsız şart olduğunu açıkça belirtmeli (örn.
    "Kişisel veriler, açık rıza olmadan aşağıdaki şartlardan herhangi
    birinin varlığı hâlinde işlenebilir:" gibi). Girişi TEK bir şarta
    (örn. sadece "kanunlarda öngörülmesi"ne) indirgeme - birden fazla ve
    çeşitli gerekçe olduğunu (kanun, sözleşme, hukuki yükümlülük vb.)
    hissettir. Kullanıcının sorusunu tekrar etme veya "bilgi edinmek
    istiyoruz" gibi dolaylı/anlamsız cümleler kurma.
  - YÖN UYARISI: Liste, giriş cümlesinden SONRA (aşağıda) gelecek. Bu
    yüzden giriş cümlesinde "yukarıda belirtilen" gibi ifadeler KULLANMA -
    bu yanlış olur çünkü liste henüz yazılmadı. Bunun yerine "aşağıdaki"
    kelimesini kullan.
  - Ardından, HİÇBİRİNİ ATLAMADAN, her şartı ayrı bir satırda listele.
    Format: "MADDE X (veya MADDE X/y): Şartın açıklaması. Örnek: ..."
  - HER ŞART İÇİN BİR ÖRNEK OLSUN: Eğer ham bilgide o şart için zaten bir
    örnek varsa onu KORU; yoksa günlük hayattan somut, kısa bir örnek SEN
    EKLE. Her satırda hem açıklama hem örnek bulunmalı.
  - HİÇBİR YILDIZ (*) İŞARETİ KULLANMA - ne başlıkta ne örnekte, ne
    hiçbir yerde. Yıldız, kalın yazı, markdown işareti YOK. Sadece düz metin.
  - TEKRAR ETME: Aynı ifadeyi/kelime grubunu bir cümle içinde iki kez
    yazma (örn. "...suretiyle X suretiyle..." gibi kekeleme yapma).
    Her açıklamayı bir kez, temiz şekilde yaz.
  - Madde referanslarını AYNEN kaynaktaki gibi koru, uydurma harf/bent ekleme.

DURUM 2 (tek bir şart/bilgi):
  - Soruyu DOĞRUDAN ve TAM cümlelerle cevapla - liste/madde satırı YAPMA,
    sadece 1-3 cümlelik akıcı bir paragraf yaz.
  - Kaynaktaki İLGİLİ EK BAĞLAMI da (bu madde/kurum/kavram ne hakkında)
    cevaba doğal şekilde dahil et ki cevap tek başına anlamlı ve tam olsun.
  - Madde referansını cümle içinde doğal şekilde geç (örn. "MADDE 19'a
    göre..."), ayrı bir satır/liste maddesi olarak YAZMA.
  - Cevabın SONUNA madde numarasını tekrar eden ayrı bir satır veya
    tırnak içinde alıntı EKLEME - madde referansı sadece cümle içinde,
    bir kez geçsin, tekrar etme.
  - HİÇBİR YILDIZ (*) İŞARETİ KULLANMA.

Soru: {soru}

Ham bilgi:
{eksiksiz_liste}

Nihai cevap (yıldız/markdown YOK, dogrudan cevapla basla, madde referanslari doğru):"""


def prompt_olustur(soru: str, sonuc: dict) -> str:
    """
    Retriever'dan gelen chunk'lari ve kullanicinin sorusunu, LLM'e
    gonderilecek tam prompt'a donusturur.

    Onemli tasarim karari: LLM'e "SADECE verilen maddelere dayan, kendi
    bilgini ekleme" diye acikca talimat veriyoruz. Bu, "grounded generation"
    (Faz 0'da ogrendigimiz kavram) icin kritik - aksi halde model, kanunla
    ilgisiz ama kulaga dogru gelen bilgiler "uydurabilir" (hallucination).
    """
    baglam_parcalari = []
    dokumanlar = sonuc["documents"][0]
    metadatalar = sonuc["metadatas"][0]

    for doc, meta in zip(dokumanlar, metadatalar):
        baglam_parcalari.append(
            f"[{meta['madde_no']}, sayfa {meta['sayfa_no']}]\n{doc}"
        )

    baglam = "\n\n".join(baglam_parcalari)

    prompt = f"""Sen, Kişisel Verilerin Korunması Kanunu (KVKK) konusunda bilgili bir asistansın.

Aşağıda KVKK'dan alınmış ilgili madde metinleri verilmiştir. SADECE bu
maddelerde yer alan bilgiye dayanarak, kullanıcının sorusunu Türkçe, açık
ve anlaşılır bir dille cevapla. Eğer verilen maddeler soruyu cevaplamaya
yeterli değilse, bunu açıkça belirt - bilgi uydurma.

ÖNEMLİ - EKSİKSİZLİK KURALI: Verilen madde metninde kaç tane şart/bent
(a, b, c, ç, d, e, f...) olduğunu ÖNCE KENDİ KENDİNE SAY. Cevabını
yazdıktan SONRA, yazdığın cevapta bu sayılan şartların HEPSİNİN geçip
geçmediğini kontrol et - eğer bir tanesini bile atladıysan, cevabını o
şartı da ekleyecek şekilde tamamla. Kısmi/özet bir açıklama vermek yerine
eksiksiz olmayı tercih et, bu kural FORMAT kuralından (asagida) daha
önceliklidir - gerekirse biraz daha uzun yaz ama hiçbir şartı atlama.

ÖNEMLİ - AÇIKLAMA KURALI: Her şartı SADECE olduğu gibi kopyalama; her
birine günlük hayattan somut bir örnek veya sade bir açıklama ekle.

ÖNEMLİ - FORMAT KURALI: Cevabını madde işaretleri (*, -, a) b) c) gibi
liste formatı) veya kalın/bold başlıklar KULLANMADAN, akıcı bir düz yazı
(paragraf) şeklinde yaz - sanki bir uzmana soru sorup sözlü, doğal bir
açıklama alıyormuşsun gibi. Şartlardan bahsederken bunları cümle akışı
içinde doğal şekilde geçir (örn. "Örneğin, kanunlarda açıkça öngörülmesi
halinde..." gibi), numaralı/madde işaretli bir liste sunma. Birden fazla
kısa paragraf kullanabilirsin ama her şart ayrı bir madde satırı olmasın.
UNUTMA: Format ne kadar akıcı olursa olsun, EKSİKSİZLİK KURALI'ndaki
tüm şartlar mutlaka cevabın içinde (cümle içinde geçecek şekilde) yer
almalı - akıcı yazmak, bazı şartları atlamak icin bir bahane degildir.

Cevabının sonunda hangi madde(ler)e dayandığını belirt.

--- İLGİLİ KANUN MADDELERİ ---
{baglam}
--- --- ---

Soru: {soru}

Cevap:"""

    return prompt


def cevabi_temizle(cevap: str) -> str:
    """
    Modelin bazen talimata ragmen ekledigi, gereksiz/tekrarci bir kapanis
    ifadesini (orn. "MADDE 19 (3): Kurumun merkezi Ankara'dadir." gibi,
    zaten cumle icinde soylenmis bilgiyi tekrar eden bir ek ifade)
    programatik olarak temizler.

    ONEMLI DUZELTME: Ilk versiyon metni "\\n" (gercek satir sonu) karakterine
    gore ayirip kontrol ediyordu, ama Ollama'nin urettigi metin bazen TEK
    bir paragraf halinde geliyor (satir sonu YOK), terminal ekraninda
    sadece GORSEL olarak sarilarak iki satirmis gibi GORUNUYOR. Bu yuzden
    satir bazli kontrol hicbir zaman eslesme bulamiyordu. Simdi bunun yerine
    metnin TAMAMI icinde regex ile "MADDE X (Y): ..." kalibini ariyoruz -
    bu, gercek satir sonu olup olmamasindan bagimsiz calisir.

    Mantik: Toplamda sadece 1 tane "MADDE <sayi>" gecen bir cevapta (DURUM 2,
    tekil cevap), eger bir cumle sonundan (". ") veya satir basindan hemen
    sonra "MADDE X (Y):" ile baslayan bir ek ifade varsa, bu neredeyse kesin
    gereksiz bir tekrardir - siliniyor. DURUM 1 (liste) cevaplarinda birden
    fazla "MADDE" gectigi icin (tum_madde_sayisi > 1) hicbir sey silinmiyor.
    """
    import re

    tum_madde_sayisi = len(re.findall(r"\bMADDE\s+\d+\b", cevap, re.IGNORECASE))

    if tum_madde_sayisi == 1:
        # Herhangi bir noktalama isaretinden (":", "-", "," vb.) bagimsiz
        # olarak, sadece "cumle/satir sonu + MADDE <sayi>" kalibini ariyoruz.
        # match.start() > 0 sarti, cevabin EN BASINDA meşru bir referans
        # varsa (orn. "MADDE 19'a gore...") ona DOKUNMAMAMIZI saglar - sadece
        # metnin ORTASINDA/SONUNDA, bir cumleyi bitirdikten SONRA tekrar
        # "MADDE" ile baslayan ek bir ifadeyi (gereksiz tekrari) hedefliyoruz.
        desen = re.compile(r"(\.\s+|\n\s*)(MADDE\s+\d+\b.*)", re.IGNORECASE | re.DOTALL)
        eslesme = desen.search(cevap)
        if eslesme and eslesme.start() > 0:
            cevap = cevap[:eslesme.start()].rstrip()

    # Model, "yildiz kullanma" talimatina ragmen zaman zaman '*' ekliyor
    # (orn. "*Örnek: ...*" gibi kalin yazi/vurgu niyetiyle). Buna modelin
    # kendisine guvenmek yerine, kod tarafinda KESIN olarak temizliyoruz -
    # yildizi silmek metnin anlamini degistirmez, sadece bicimini duzeltir.
    cevap = cevap.replace("*", "")

    # Model bazen (Asama 1'de kod-tabanli listede olmasa bile) Asama 2'de
    # kendiliğinden "haklarına sahiptir" gibi bir kapanis ifadesini tekrar
    # ekleyebiliyor - bunu nihai cevaptan da temizliyoruz.
    cevap = re.sub(
        r",?\s*haklarına sahiptir\.?",
        ".",
        cevap,
        flags=re.IGNORECASE
    )

    # Modelin bazen urettigi "kekeleme" tarzi tekrarlari (ayni ifadenin
    # ust uste iki kez yazilmasi, orn. "...suretiyle X suretiyle...")
    # genel bir regex ile tespit edip tek hale getiriyoruz.
    tekrar_deseni = re.compile(r"(\b.{15,100}?)\s+\1", re.IGNORECASE)
    onceki_hal = None
    while onceki_hal != cevap:
        onceki_hal = cevap
        cevap = tekrar_deseni.sub(r"\1", cevap, count=1)

    # Yıldız (*) temizligi ve nokta/bosluk fazlaliklarini duzelt
    cevap = re.sub(r"\.{2,}", ".", cevap)
    cevap = re.sub(r"\s+\.", ".", cevap)

    return cevap.strip()


def madde_parcalarini_genislet(koleksiyon, sonuc: dict) -> dict:
    """
    Task 1.2'de uzun maddeler (orn. MADDE 11, 9 hak icerdigi icin uzun)
    birden fazla alt-chunk'a bolunmustu. Top-k retrieval, bir maddenin
    SADECE bir alt-chunk'ini getirebilir, digerlerini (farkli maddelerin
    daha "yakin" cikmasi nedeniyle) getirmeyebilir - bu da eksik cevaba
    yol acar (orn. MADDE 11'in 9 hakkindan sadece 2'sinin gelmesi gibi).

    Bu fonksiyon, en alakali sonucun ait oldugu maddeyi tespit edip, o
    maddenin TUM alt-chunklarini ChromaDB'den ekstra olarak ceker ve
    sonuca ekler - boylece bolunmus bir madde bile TAM icerigiyle LLM'e
    ulasir.
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


def bentleri_cikar(metin: str) -> list:
    """
    Ham chunk metninde "a) ...", "b) ...", "ç) ..." gibi harfli bentleri
    REGEX ile (LLM'e sormadan) kod tarafinda cikarir. Bu, Task 1.2'de
    PDF'ten cikan metnin her bendi genelde ayri bir satirda tuttugu
    gercegine dayanir.

    Neden bu onemli: LLM'e "kac tane bent var, hepsini say" demek,
    9 gibi yuksek sayilarda GUVENILMEZ (MADDE 11'de 9 bent oldugunda
    model bazen sadece 2 tanesini yakalayabiliyor). Ama bentler PDF'te
    zaten net sekilde ayrilmis, bu yuzden REGEX ile %100 guvenilir
    sekilde cikarilabilir - LLM'e sadece bu GARANTILI listeyi
    zenginlestirmesi (ornek ekleme, akici hale getirme) birakiliyor.

    Donen deger: [(harf, icerik), ...] seklinde bir liste. Hicbir harfli
    bent bulunamazsa bos liste doner (bu, DURUM 2 - tekil cevap demektir).
    """
    satirlar = metin.split("\n")
    bent_deseni = re.compile(r"^([a-zçğıöşü])\)\s*(.*)$")
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
        elif mevcut_harf:
            mevcut_metin.append(satir_temiz)

    if mevcut_harf:
        bentler.append((mevcut_harf, " ".join(mevcut_metin).strip()))

    # Turkce kanun metinlerinde bentler genelde tek bir buyuk cumlenin
    # parcalaridir ve en sonunda hepsini kapatan ortak bir ifade gelir
    # (orn. "...haklarına sahiptir." veya "...hâlinde mümkündür.").
    # Bizim satir-bazli cikarma mantigimiz bu kapanis ifadesini yanlislikla
    # SON bendin icerigine ekliyor. Bilinen kapanis kaliplarini tespit
    # edip SADECE son bentten temizliyoruz.
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


def kod_tabanli_eksiksiz_liste_olustur(madde_no: str, bentler: list) -> str:
    """
    Regex ile GARANTILI olarak cikardigimiz bentleri, Asama 2'ye (akici
    hale getirme) girdi olarak verecegimiz duz metin formatina cevirir.
    """
    satirlar = []
    for harf, icerik in bentler:
        satirlar.append(f"{madde_no}/{harf}) {icerik}")
    return "\n".join(satirlar)


def giris_cumlesi_uret(soru: str, madde_no: str, bent_sayisi: int) -> str:
    """
    Sadece TEK bir giris cumlesi istiyoruz - liste yazmasini ISTEMIYORUZ,
    boylece modelin "liste formatini degistirme" riski bu adimda hic
    olmuyor (cunku zaten liste yazmasini istemiyoruz).
    """
    prompt = f"""Soru: {soru}

Bu soruya, {madde_no} kapsamında {bent_sayisi} farklı şart/hak ile cevap
verilecek. SADECE bu duruma uygun, 1 CÜMLELİK bir giriş yaz (örn.
"Bu haklar/şartlar aşağıdaki gibidir:" tarzı). Liste YAZMA, sadece giriş
cümlesini yaz, başka hiçbir şey ekleme.

ÇOK ÖNEMLİ: Sana verilmeyen hiçbir bilgiyi (kanun numarası, tarih, madde
dışında ek referans vb.) UYDURMA/EKLEME - sadece "{madde_no}" ifadesini
kullanabilirsin, başka hiçbir sayı veya isim EKLEME."""
    return ollama_ile_cevap_uret(prompt).strip()


def kod_tabanli_nihai_cevap_olustur(soru: str, madde_no: str, bentler: list) -> str:
    """
    Nihai cevabi TAMAMEN kod tarafinda, deterministik olarak birlestirir.
    LLM'den sadece giris cumlesini aliyoruz (dar, basit bir gorev); bent
    icerikleri dogrudan kaynak metinden (regex ile cikarilmis, garantili
    dogru) geliyor - ornek uretme adimi kaldirildi (basitlik ve hiz icin).
    """
    giris = giris_cumlesi_uret(soru, madde_no, len(bentler))

    satirlar = [giris, ""]
    for harf, icerik in bentler:
        satirlar.append(f"{madde_no}/{harf}) {icerik}")

    return "\n".join(satirlar)


def ollama_ile_cevap_uret(prompt: str) -> str:
    """
    Prompt'u Ollama'nin yerel REST API'sine gonderir ve uretilen cevabi
    dondurur. Ollama, kurulduktan sonra otomatik olarak arka planda
    http://localhost:11434 adresinde bir sunucu gibi calisir.
    """
    import requests

    try:
        yanit = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=400  # buyuk modeller CPU'da yavas olabilir, genis zaman taniyoruz
        )
        yanit.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama'ya baglanilamadi. Ollama'nin bilgisayarinizda calistigindan "
            "emin olun (kurulumdan sonra genelde otomatik baslar). "
            "Kontrol icin terminalde 'ollama list' calistirabilirsiniz."
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
        print("Ornek:    python 06_llm_answer.py \"Kişisel veriler ne zaman açık rıza olmadan işlenebilir?\"")
        sys.exit(1)

    soru = sys.argv[1]

    # --- Adim 1: Retrieval (Task 1.5'teki fonksiyonlari kullaniyoruz) ---
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

    # En alakali maddenin TUM alt-chunklarini garantiye almak icin genislet
    sonuc = madde_parcalarini_genislet(koleksiyon, sonuc)

    # --- Adim 2: Iki asamali generation ---
    # Once, en alakali sonucun ait oldugu maddede REGEX ile tespit
    # edilebilen harfli bentler (a, b, c...) var mi kontrol ediyoruz.
    # Varsa, "kac tane bent var" sorusunu LLM'e sormuyoruz (guvenilmez),
    # bunun yerine koddan GARANTILI olarak cikariyoruz.
    en_alakali_madde_no = sonuc["metadatas"][0][0]["madde_no"]
    ilgili_metinler = "\n".join(
        doc for doc, meta in zip(sonuc["documents"][0], sonuc["metadatas"][0])
        if meta["madde_no"] == en_alakali_madde_no
    )
    bentler = bentleri_cikar(ilgili_metinler)

    if len(bentler) >= 2:
        # DURUM 1 (liste): Formati TAMAMEN kod tarafinda garanti altina
        # aliyoruz. LLM'e sadece "giris cumlesi yaz" ve "ornekleri yaz"
        # gibi DAR, tek-isli gorevler veriyoruz - genel YAPI (siralama,
        # madde etiketleri, format) artik modelin inisiyatifine
        # birakilmiyor, bu da onceki calismalarda gordugumuz "model bazen
        # farkli bir format secebiliyor" tutarsizligini ortadan kaldirir.
        print(f"Asama 1/2: {len(bentler)} bent regex ile koddan cikarildi (LLM'e sorulmadi, garantili).")
        print("Asama 2/2: Giris cumlesi ve ornekler uretiliyor (format kod tarafinda sabit)...")
        try:
            cevap = kod_tabanli_nihai_cevap_olustur(soru, en_alakali_madde_no, bentler)
        except RuntimeError as hata:
            print(f"\n\u274c HATA: {hata}")
            sys.exit(1)
    else:
        # DURUM 2 (tekil bilgi): Harfli bent yoksa eski iki asamali
        # yontemle devam ediyoruz (bu akista format sorunu yasamamistik).
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

    # --- Sonucu goster ---
    print("\n" + "=" * 60)
    print(f"SORU: {soru}")
    print("=" * 60)
    print(f"\nCEVAP:\n{cevap}")

    print("\n" + "-" * 60)
    print("Kullanilan kaynaklar:")
    for meta in sonuc["metadatas"][0]:
        print(f"  - {meta['madde_no']} (sayfa {meta['sayfa_no']})")

    print("\n\U0001F389 Faz 1 - Klasik RAG tamamlandi!")
