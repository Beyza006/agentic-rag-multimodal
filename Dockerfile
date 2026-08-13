# Agentic RAG Multimodal - Docker Imaji
# --------------------------------------
# Bu imaj SADECE uygulamayi (Python + Gradio arayuzu + tum ajan mantigi)
# paketler. Ollama, HOST makinede (container DISINDA, senin bilgisayaninda)
# calismaya devam eder - modelleri tekrar indirmene gerek YOK.

FROM python:3.13-slim

# DUZELTME (gercek testte bulunan sorun): Python, container icinde
# calisirken cikti akislarini (stdout/stderr) VARSAYILAN olarak
# "buffered" (biriktirerek) yaziyor - yani print() ile yazdirilan
# satirlar, buffer dolana veya program bitene kadar "docker logs"
# komutunda GORUNMUYOR. Bu, uygulama aslinda calisirken bile "hicbir
# cikti yok" gibi gorunmesine yol aciyordu. Bu ortam degiskeni,
# Python'u "unbuffered" (anlik yazan) moda zorluyor - artik her print()
# satiri ANINDA "docker logs" ile gorunur oluyor.
ENV PYTHONUNBUFFERED=1

# Sistem bagimliliklari:
# - libgl1, libglib2.0-0: opencv-python (10_video.py, cv2) calismasi icin gerekli
# - libgomp1: paddleocr/paddlepaddle (11_ocr.py) calismasi icin gerekli
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ONCE sadece requirements.txt kopyalanip kurulur - boylece SADECE kod
# degistiginde (requirements.txt degismedigi surece) Docker'in katman
# onbellegi (layer cache) sayesinde pip install adimi TEKRAR calismaz,
# yeniden build alma suresi cok kisalir.
COPY requirements.txt .

# ONEMLI DUZELTME (gercek build sirasinda bulunan sorun): sentence-transformers,
# "torch" paketini bagimlilik olarak istiyor - ama pip varsayilan olarak
# torch'un TAM (GPU/CUDA destekli) surumunu kurmaya calisiyor, bu da
# torch'un kendisi (~527 MB) + onlarca NVIDIA/CUDA paketi (cublas, cudnn,
# cusolver, nvshmem, triton vb. - TOPLAMDA BIRKAC GB) indirmesine yol
# aciyordu. Bu container'in GPU erisimi YOK ve proje zaten CPU'da
# calisacak sekilde tasarlandi - bu yuzden torch'u ONCE, PyTorch'un
# KENDI CPU-ONLY adresinden kuruyoruz. Boylece asagidaki "pip install -r
# requirements.txt" adimi calistiginda, torch ZATEN kurulu (CPU surumu)
# bulunuyor ve GEREKSIZ CUDA paketlerini INDIRMIYOR.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Projenin tum kaynak kodunu kopyala.
# NOT: data/ klasoru (ChromaDB, dokumanlar, gorseller, videolar) BILEREK
# imaja DAHIL EDILMIYOR (bkz. .dockerignore) - bunun yerine docker-compose
# ile "volume" olarak disaridan baglaniyor, boylece:
#   1) Imaj boyutu kucuk kalir
#   2) Zaten olusturdugun ChromaDB'yi yeniden uretmene gerek kalmaz
#   3) Container silinip yeniden olusturulsa bile verilerin kaybolmaz
COPY . .

# Gradio, VARSAYILAN olarak sadece "127.0.0.1" (container'in kendi ici)
# uzerinde dinler - bu, container DISINDAN (yani senin tarayicindan)
# HICBIR SEKILDE erisilemez anlamina gelir, port yonlendirmesi (-p) bile
# ise yaramaz. Bu ortam degiskeni, Gradio'nun TUM aglardan ("0.0.0.0")
# dinlemesini saglar - 14_gradio_app.py'nin kendisine HICBIR DOKUNMA
# gerekmeden, Gradio bu degiskeni kendisi okur.
ENV GRADIO_SERVER_NAME=0.0.0.0

# Ollama, bu container'in DISINDA, HOST makinende calisiyor.
# "host.docker.internal", Docker Desktop'in (Windows/Mac) host makineye
# erismek icin sagladigi ozel adrestir. (Linux'ta docker-compose.yml
# icindeki "extra_hosts" ayari bunun calismasini saglar.)
ENV OLLAMA_API_URL=http://host.docker.internal:11434/api/generate

# Gradio'nun varsayilan portu
EXPOSE 7860

CMD ["python", "faz5_arayuz/14_gradio_app.py"]
