# faz7_deployment

Bu klasör, roadmap'teki Faz 7 (Deployment) asamasina ait notlari icerir.

## Docker ile calistirma

Projenin Docker ile paketlenmesine iliskin dosyalar, Docker'in kendi
kurali geregi (build baglaminin tum proje dosyalarina erisebilmesi icin)
**proje kokunde** bulunur, bu klasorde degil:

- `../Dockerfile` — uygulama imajinin tanimi
- `../docker-compose.yml` — tek komutla ayaga kaldirma yapilandirmasi
- `../.dockerignore` — imaja dahil edilmeyecek dosyalar

## Mimari karar

Ollama (yerel LLM/VLM servisi), Docker container'inin **DISINDA**, host
makinede (bu bilgisayarda) calismaya devam eder - modellerin container
icinde yeniden indirilmesine gerek yoktur. Container, Ollama'ya
`host.docker.internal` adresi uzerinden aglar.

## Calistirma

```
docker compose up --build
```

Sonra tarayicida `http://localhost:7860` adresine gidilir.

## Veri kalicligi

`data/` klasoru (ChromaDB, dokumanlar, gorseller, videolar), Docker
imajina GOMULMEZ - bunun yerine `docker-compose.yml` icinde bir "volume"
olarak host'taki `data/` klasorune baglanir. Boylece:
- Zaten olusturulmus ChromaDB dogrudan kullanilir, yeniden uretmeye gerek yoktur.
- Container silinip yeniden olusturulsa bile veriler kaybolmaz.
