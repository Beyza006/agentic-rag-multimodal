# Faz 0 — Araştırma Notları

## Task 0.1 — LLM Temelleri

- Transformer: Vaswani ve arkadaşları tarafından 2017 yılında "Attention Is All You Need" makalesiyle tanıtılan, girdi dizisini ardışık değil paralel olarak işleyen bir sinir ağı mimarisidir. RNN ve LSTM gibi tekrarlayan (recurrent) yapıların aksine, dizideki tüm elemanlar arasındaki ilişkiyi self-attention mekanizması aracılığıyla eşzamanlı olarak hesaplar. Bu sayede hem eğitim süresi kısalır hem de uzun mesafeli bağımlılıklar (long-range dependencies) daha etkili biçimde modellenir. Mimari, encoder ve decoder olmak üzere iki ana bloktan oluşur.

- Attention Mechanism: Bir dizideki her elemanın, dizinin diğer elemanlarıyla olan ilişkisini ağırlıklandırarak hesaplayan mekanizmadır. Query, Key ve Value vektörleri üzerinden hesaplanan benzerlik skorları (attention scores), softmax fonksiyonuyla normalize edilerek her elemanın bağlamsal temsiline katkı oranını belirler. Self-attention, bu hesaplamanın dizinin kendi elemanları arasında yapılmasıdır ve modelin bağlamsal (contextual) anlam çıkarımı yapmasını sağlar.

- Token / Context Window: Token, bir metnin model tarafından işlenebilecek en küçük birimidir; bir kelime, kelimenin bir parçası veya bir karakter olabilir (tokenizasyon algoritmasına bağlı olarak, örn. Byte Pair Encoding). Context window ise modelin tek bir ileri geçişte (forward pass) dikkate alabildiği maksimum token sayısıdır; modelin kısa süreli bağlamsal kapasitesini belirler ve bu sınırın aşılması durumunda önceki bilgi modelin işlem kapsamı dışında kalır.

- Prompt Engineering: Modelden istenen çıktının kalitesini ve doğruluğunu artırmak amacıyla girdi metninin (prompt) yapılandırılması sürecidir. Talimat netliği, örnek verme (few-shot prompting), rol tanımlama ve adım adım akıl yürütme talebi (chain-of-thought) gibi teknikler bu sürece dahildir; model ağırlıklarında herhangi bir değişiklik yapılmaz.

- Fine-tuning: Önceden büyük ve genel bir veri kümesi üzerinde eğitilmiş (pre-trained) bir modelin, belirli bir görev veya alana özgü daha küçük bir veri kümesiyle ek eğitime tabi tutularak model parametrelerinin güncellenmesi sürecidir. Amaç, modelin genel dil yeteneklerini korurken belirli bir alanda performansını artırmaktır.

- Inference: Eğitimi tamamlanmış bir modelin, öğrenilen parametreleri kullanarak yeni girdiler üzerinde çıktı (tahmin/üretim) ürettiği aşamadır. Eğitim (training) aşamasının aksine, inference sırasında model ağırlıklarında herhangi bir güncelleme yapılmaz; yalnızca ileri yönlü hesaplama (forward pass) gerçekleştirilir.

## Task 0.2 — Embedding

- Embedding nedir: Bir metinsel veya kategorik veriyi (kelime, cümle, doküman vb.) sabit boyutlu, sürekli değerli bir vektör uzayında temsil etme yöntemidir. Embedding modelleri, semantik olarak benzer girdilerin bu vektör uzayında birbirine yakın konumlanmasını sağlayacak şekilde eğitilir; bu sayede anlam, sayısal işlemlerle karşılaştırılabilir hale gelir.

- Semantic / Similarity Search: Sorgu ile doküman arasındaki ilişkiyi yüzeysel kelime eşleşmesi (lexical matching) yerine anlamsal benzerlik üzerinden kuran arama yöntemidir. Sorgu ve dokümanlar embedding vektörlerine dönüştürülür, ardından bu vektörler arasındaki benzerlik (genellikle cosine similarity ile) hesaplanarak en alakalı sonuçlar sıralanır.

- Cosine Similarity: İki vektör arasındaki açının kosinüsünü hesaplayarak benzerlik ölçen bir metriktir. Değer aralığı -1 ile 1 arasındadır; 1'e yakın değerler yüksek benzerliği, 0 ilişkisizliği, -1'e yakın değerler ise zıt anlamlılığı ifade eder. Vektörlerin büyüklüğünden (magnitude) bağımsız olarak yalnızca yönlerini karşılaştırdığı için, farklı uzunluktaki metinlerin embedding'lerini karşılaştırmada tercih edilir.

- Vector Search: Yüksek boyutlu bir vektör uzayında, verilen bir sorgu vektörüne en yakın (en benzer) vektörlerin bulunması işlemidir. Büyük ölçekli veri kümelerinde bu aramayı verimli hale getirmek için ANN (Approximate Nearest Neighbor) algoritmaları (örn. HNSW, IVF) kullanılır; bu algoritmalar tam (exact) arama yerine yaklaşık ama çok daha hızlı sonuçlar üretir.

## Task 0.3 — Vector Database Karşılaştırması

- ChromaDB: Python tabanlı, minimalist bir açık kaynak vector database'dir; sıfır konfigürasyonla bellek içi (in-memory) veya kalıcı (persistent) modda çalışabilir.

- FAISS (Facebook AI Similarity Search): C++ dilinde yazılmış, Python bağlamaları sunan, ultra hızlı vektör benzerlik araması için tasarlanmış bir kütüphanedir. Bağımsız bir veritabanı değildir; yalnızca arama motorudur, metadata yönetimi ve kalıcılık gibi özellikler ayrıca inşa edilmelidir.

- Milvus: Büyük ölçekli, dağıtık (distributed) mimariye sahip, production ortamları hedefleyen açık kaynak bir vector database'dir. Yatay ölçeklenebilirlik sağlar ancak kurulum/operasyonel yönetimi daha karmaşıktır.

- Weaviate: Bir bilgi grafiği (knowledge graph) ve modüler makine öğrenmesi modellerini entegre eden, bulut-native bir vector database'dir. GraphQL API, gerçek zamanlı sorgular ve multimodal (metin+görsel) veri desteği sunar.

| DB | Avantaj | Dezavantaj | Kullanım Alanı |
|---|---|---|---|
| ChromaDB | Basit kurulum, hızlı prototipleme | Sınırlı ölçeklenme, temel filtreleme | Küçük-orta ölçekli RAG, öğrenme projeleri |
| FAISS | Çok yüksek performans, milyonlarca/milyarlarca vektörde çalışabilir | Bağımsız DB değil, metadata/kalıcılık manuel | Performans kritik, local/backend sistemler |
| Milvus | Yüksek ölçeklenebilirlik, dağıtık deployment | Kurulum/yönetim karmaşıklığı | Büyük ölçekli production RAG, öneri sistemleri |
| Weaviate | Hibrit arama, knowledge graph, multimodal destek | Daha ağır kurulum/öğrenme eğrisi | Kurumsal arama, soru-cevap sistemleri |

**Proje Kapsamında Seçim:** Bu staj projesi kapsamında, kurulum kolaylığı, Python ekosistemiyle uyumluluğu, LangChain/LlamaIndex entegrasyon desteği ve yerel makinede bağımsız çalışabilme özellikleri göz önünde bulundurularak ChromaDB tercih edilmiştir. Veri hacminin ileride önemli ölçüde artması (milyonlarca doküman düzeyi) durumunda, Milvus gibi dağıtık mimariye sahip çözümlerin değerlendirilmesi planlanmaktadır.

## Task 0.4 — Multimodal & Benchmarking (Jetson AI Lab arşivi, kavramsal seviyede)

- Multimodal AI nedir: Birden fazla veri modalitesini (metin, görsel, ses, video, sensör verisi vb.) eş zamanlı olarak işleyebilen ve bu modaliteler arasında anlamsal ilişki kurabilen yapay zekâ sistemlerine verilen genel addır. Tekil modaliteli (unimodal) sistemlerin aksine, farklı veri türlerinden gelen bilgiyi ortak bir temsil (embedding) uzayında birleştirerek daha kapsamlı çıkarımlar yapılmasını sağlar.

- LLM ile VLM arasındaki fark: Large Language Model (LLM), yalnızca metinsel girdi üzerinde eğitilmiş ve yalnızca metinsel çıktı üretebilen bir mimaridir. Vision Language Model (VLM) ise görsel girdiyi (resim, video karesi) metinsel girdiyle birlikte işleyebilen, iki modaliteyi ortak bir gömme uzayında ilişkilendiren bir mimaridir. Bu sayede VLM, görsel soru-cevap (Visual Question Answering) ve görsel açıklama üretme (Image Captioning) gibi görevleri yerine getirebilir; LLM bu görevleri gerçekleştiremez çünkü eğitim verisi yalnızca metinden oluşur.

- VLM çalışma mekanizması: VLM, görüntüyü insan algısına benzer şekilde doğrudan "görmez"; bir görüntü, bilgisayar için piksel değerlerinden oluşan sayısal bir matristir. Model önce bu piksel bilgilerini bir görüntü kodlayıcısı (vision encoder) aracılığıyla işleyerek anlamlı bir vektör temsiline (embedding) dönüştürür. Bu görsel embedding, metinsel embedding ile aynı ortak anlam uzayında ilişkilendirilerek modelin görsel içerik hakkında dil tabanlı çıkarım yapması sağlanır.

- VLM temel görevleri: VLM'lerin başlıca uygulama alanları arasında resme açıklama üretme (Image Captioning), görsel içerik hakkında doğal dilde soru cevaplama (Visual Question Answering) ve OCR destekli belge anlama (bir belgedeki yazı, tablo ve grafiği birlikte yorumlama) yer alır.

- Vision Transformer (ViT) temel mantığı: ViT, orijinal Transformer mimarisinin görüntü işleme alanına uyarlanmış halidir. Görüntü, sabit boyutlu karelere (patch) bölünür; her patch düzleştirilip doğrusal bir projeksiyonla bir embedding vektörüne dönüştürülür. Bu patch embedding'leri, bir cümledeki kelime token'larına benzer şekilde ele alınarak self-attention katmanlarından geçirilir. Bu yaklaşım, evrişimli sinir ağlarının (CNN) yerel filtre tabanlı işleyişinin aksine, görüntünün farklı bölgeleri arasındaki global ilişkilerin doğrudan modellenmesine imkân tanır.

- Image Generation kavramı: Metinsel bir açıklamadan (prompt) yeni bir görsel içerik üretme sürecidir. Günümüzde en yaygın yaklaşım, gürültülü bir görüntüyü adım adım anlamlı bir görüntüye dönüştüren difüzyon modelleridir (diffusion models); bu modeller eğitim sırasında bir görüntüye kademeli olarak gürültü eklemeyi öğrenir, üretim sırasında ise bu süreci tersine çevirerek rastgele gürültüden görüntü sentezler.

- Audio Language Models: Ses verisini girdi ve/veya çıktı olarak işleyebilen model ailesidir. İki temel görev öne çıkar: Automatic Speech Recognition (ASR) konuşma sesini metne dönüştürür; Text-to-Speech (TTS) ise metni doğal insan sesine benzer bir ses çıktısına dönüştürür.

- GenAI Benchmarking kavramı: Üretken yapay zekâ modellerinin performansının, farklı model veya donanım konfigürasyonları arasında standart ölçütlerle karşılaştırılması sürecidir. Bu ölçütler genellikle modelin çıktı doğruluğundan (accuracy) ziyade çalışma zamanı performansına odaklanır; başlıca metrikler throughput (birim zamanda üretilen token sayısı) ve latency'dir (yanıtın üretilmeye başlama süresi). Özellikle donanım kaynaklarının sınırlı olduğu edge cihazlarda (örn. NVIDIA Jetson), bu metrikler modelin pratikte kullanılabilir olup olmadığını belirlemede kritik rol oynar.

- Jetson AI Lab arşivi notları (genel): NVIDIA Jetson AI Lab arşivi, farklı modalitelere (metin, görsel, ses) yönelik üretken yapay zekâ modellerinin kaynak kısıtlı edge cihazlar üzerinde nasıl çalıştırılabileceğine dair pratik rehberler sunmaktadır. Bu içerikler, belirli bir model seçiminden bağımsız olarak, multimodal bir sistemin bileşenlerinin (LLM, VLM, embedding, vector database, ASR/TTS) genel mimari içindeki rolünü kavramaya yöneliktir.

- Projede multimodal kullanım planı: Geliştirilecek Agentic RAG sisteminde kullanıcı yalnızca PDF/Word değil, görsel ve video da yükleyebilecektir. Agent, gelen sorunun niteliğini analiz ederek uygun aracı otomatik seçecektir: soru bir görsel/grafiğe ilişkinse Vision Tool (VLM), soru bir doküman içeriğine ilişkinse RAG mekanizması devreye girecektir.

- Video QA yaklaşımı: Video verisi, boyutu ve işlem yükü nedeniyle doğrudan modele gönderilmez; bunun yerine önce anlamlı karelere (frame) ayrıştırılır ve bu kareler VLM'e iletilir. Sorunun tek bir ana ilişkin olması durumunda (örn. "elinde ne var?") tek kare yeterli olabilirken, sorunun zaman içindeki bir olay dizisini kapsaması durumunda (örn. "kapıyı açmadan önce ne yaptı?") birden fazla karenin birlikte değerlendirilmesi gerekir; bu yaklaşım ileride Temporal Reasoning (zamansal akıl yürütme) olarak ele alınacaktır.

> Not: Bu aşamada model seçimi yapılmadı (Qwen, LLaVA vb.) — model seçimi, proje gereksinimleri netleştikten ve prototip geliştirme aşamasına geçildikten sonra farklı alternatiflerin denenmesiyle (deneme-yanılma) yapılacaktır.

## Faz 2 — Agentic RAG

### Task 2.1 — Agent Kavramı

- Agent (Yapay Zekâ Ajanı) nedir: Bir LLM'in plan yapmak, araç (tool) çağırmak ve durumu (state) takip etmek amacıyla bir döngü (loop) içinde çalıştırıldığı sistemdir. Sıradan bir sohbet botundan (chatbot) temel farkı, agent'ın yalnızca metin üretmekle kalmayıp eylemde bulunabilmesi ve bu eylemlerin sonucunu doğrulayabilmesidir. Tipik bir agent döngüsü şu adımları izler: girdiyi analiz etme, bir sonraki adımı planlama, gerekli aracı çağırma, sonucu gözlemleme ve gerekirse planı güncelleme.

- Workflow ile Agent Arasındaki Fark: Anthropic'in "Building Effective AI Agents" adlı makalesinde yapılan ayrıma göre, workflow'lar LLM'lerin önceden tanımlanmış, sabit kod yolları üzerinden yönlendirildiği sistemlerdir; agent'lar ise LLM'lerin kendi süreçlerini ve araç kullanımını çalışma zamanında (runtime) dinamik olarak yönettiği sistemlerdir. Workflow'larda adım sırası ve kullanılacak araçlar önceden bellidir; agent'larda ise bu kararlar her çalıştırmada girdiye göre değişebilir. Anthropic, karmaşıklığın gerekmediği durumlarda workflow'ların varsayılan/önerilen yaklaşım olduğunu, agent'ların ise daha az öngörülebilir, daha açık uçlu görevler için tercih edilmesi gerektiğini belirtmektedir.

- Planning (Planlama): Bir agent'ın, kullanıcı isteğini karşılamak için hangi adımların hangi sırayla izleneceğine karar verme sürecidir. Daha gelişmiş sistemlerde bu karar, görev bağlamına göre dinamik olarak inşa edilen bir karar ağacı şeklinde işler; yani agent, sabit bir kural kümesi yerine mevcut duruma göre bir sonraki adımı belirler.

- Tool Calling (Araç Çağırma): LLM'in, kullanıcı isteğini önceden tanımlanmış araç şemalarıyla (genellikle JSON Schema formatında) karşılaştırarak hangi aracı hangi parametrelerle çağıracağını yapılandırılmış bir çıktı (structured output) olarak üretmesi sürecidir. Önemli bir ayrıntı: LLM aracı doğrudan çalıştırmaz; yalnızca "şu araç, şu parametrelerle çağrılsın" şeklinde bir istek üretir, asıl çalıştırma işlemini agent'ı yöneten kod (bizim projemizde LangGraph) gerçekleştirir.

- Projemizle bağlantısı: Faz 1'de kurduğumuz sistem (PDF yükleme → chunking → embedding → arama → cevap üretme) teknik olarak bir workflow'dur; adımlar her seferinde aynı sabit sırayla işlemektedir. Faz 2'de kurulacak agent ise, kullanıcının sorusunun türüne göre (doküman sorusu mu, güncel bilgi mi gerektiriyor, görsel/video mu ilgilendiriyor) hangi aracın çağrılacağına çalışma zamanında karar verecektir.

- Kaynaklar: LangGraph (State/Node/Edge tabanlı agent workflow kütüphanesi, Task 2.3'te detaylandırılacak), OpenAI Agents SDK ve Anthropic Tool Use (farklı sağlayıcıların tool calling mekanizmalarına dair referans noktaları).

### Task 2.2 — Tool Calling & MCP

- Tool Calling (Araç Çağırma): Bir LLM'in, doğal dil ile ifade edilen bir kullanıcı isteğini karşılamak için önceden tanımlanmış bir aracın çağrılması gerektiğine karar verip, bu çağrıyı yapılandırılmış (structured) bir formatta (genellikle JSON) üretmesi mekanizmasıdır. Kritik bir ayrım: LLM aracı doğrudan çalıştırmaz; yalnızca "hangi araç, hangi parametrelerle çağrılsın" şeklinde bir istek üretir. Aracın fiilen çalıştırılması, agent'ı yöneten dış kod (bizim projemizde LangGraph) tarafından gerçekleştirilir. Bu ayrım önemlidir çünkü LLM'in üretebileceği potansiyel olarak hatalı veya güvenli olmayan bir çağrının, çalıştırılmadan önce kod tarafında doğrulanabilmesini mümkün kılar.

- Function Calling: Tool Calling ile büyük ölçüde eş anlamlı kullanılan, özellikle OpenAI'nin API dokümantasyonunda tercih edilen bir terimdir. LLM'e önceden bir fonksiyon şeması (isim, parametre adları ve tipleri) tanıtılır; model, kullanıcı isteğine uygun olduğunda bu şemaya uyan bir çağrı (fonksiyon adı + parametre değerleri) üretir. Bu çağrı, geliştirici tarafından yazılan gerçek fonksiyona iletilerek çalıştırılır.

- Structured Output (Yapılandırılmış Çıktı): Bir LLM'in çıktısını serbest biçimli doğal dil yerine, önceden tanımlanmış bir şemaya (örn. JSON Schema) uygun şekilde üretmesini sağlayan tekniktir. Bu yaklaşım, LLM çıktısının bir program tarafından güvenilir şekilde ayrıştırılabilmesini (parse edilebilmesini) sağlar; aksi hâlde serbest metin çıktılarından programatik karar üretmek güvenilmez olurdu. Projemizde Task 1.6 ve 2.3'te, LLM'e "yalnızca X veya Y yaz" şeklinde kısıtlı bir çıktı formatı dayatılması, yapılandırılmış çıktının basitleştirilmiş bir uygulamasıdır.

- Tool Registry (Araç Kayıt Defteri): Bir agent sisteminde kullanılabilir tüm araçların tanımlarının tutulduğu merkezi bir yapıdır. Araç sayısı arttıkça, tüm araç tanımlarının her seferinde LLM'e sunulması hem bağlam penceresini (context window) gereksiz yere doldurur hem de model performansını olumsuz etkileyebilir. Bu nedenle modern sistemlerde Tool Discovery (Araç Keşfi) adı verilen bir ön adım kullanılır: kullanıcının isteğiyle alakalı olabilecek araçlar (bir vektör arama veya MCP sunucusu aracılığıyla) önceden filtrelenir, yalnızca bu alt küme LLM'e sunulur.

- MCP (Model Context Protocol): Anthropic tarafından geliştirilen, yapay zekâ sistemlerinin farklı araçlara ve veri kaynaklarına (dosya sistemleri, API'ler, veritabanları vb.) standart, birlikte çalışabilir (interoperable) bir protokol üzerinden bağlanabilmesini sağlayan açık bir standarttır. MCP'den önce, her yapay zekâ uygulamasının her araca özel bir entegrasyon kodu yazması gerekiyordu; MCP, bu entegrasyonu tek bir ortak protokole indirgeyerek, bir istemcinin (client) MCP standardına uyan herhangi bir sunucuya (server) bağlanabilmesini mümkün kılar.

- Projemizle bağlantısı: `07_agent.py`'de uygulanan `karar_dugumu` fonksiyonu, tool calling mantığının basitleştirilmiş bir örneğidir — LLM'e yapılandırılmış bir çıktı ("İLGİLİ" / "ALAKASIZ") ürettirilir, bu çıktı koşullu kenar (conditional edge) tarafından okunarak hangi düğüme (aracın karşılığı) gidileceğine karar verilir. Proje şu an MCP kullanmamaktadır; yalnızca 2 "araç" (RAG düğümü ve doğrudan cevap düğümü) bulunduğu için bir Tool Registry veya MCP'ye ihtiyaç duyulmamaktadır. İleride (Faz 4'te Web Search gibi) araç sayısı arttığında bu yapıların değerlendirilmesi planlanmaktadır.
