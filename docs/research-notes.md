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
