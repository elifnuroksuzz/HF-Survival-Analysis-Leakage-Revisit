# Colab'da çalıştırma rehberi

Notebook'ların ikisi de Google Colab'da baştan sona çalışacak biçimde hazırlandı. Kodu
GitHub'dan kendileri indirir, gerekli paketleri kurar ve tüm çıktıları Google Drive'daki
`MyDrive/HF_Revisit_Project/` klasörüne yazar. Bağlantı koparsa çıktılar kaybolmaz.

## Notebook 01: geliştirme analizleri (isteğe bağlı, toplam 2–4 saat)

`results/` klasöründeki tablo ve şekiller bu notebook'un çıktılarıdır ve repoda zaten var.
Notebook 01'i yalnızca sonuçları kendiniz yeniden üretmek isterseniz çalıştırmanız gerekir.

1. https://colab.research.google.com → **File → Open notebook → GitHub**
2. `elifnuroksuzz/HF-Survival-Analysis-Leakage-Revisit` yazın, **Branch** olarak
   `claude/ecstatic-wright-61dz0b` seçin ve `notebooks/01_development_analyses.ipynb` dosyasını açın.
3. Hücreleri sırayla çalıştırın. İlk hücre Drive izni ister, onaylayın.
4. Analizler 6 hücreye bölünmüştür (1/6 … 6/6). Bağlantı koparsa önce kurulum hücresini,
   sonra yarıda kalan hücreyi yeniden çalıştırın. Bitmiş hücreleri tekrarlamanız gerekmez.

## Notebook 02: MIMIC-III dış doğrulama (gerekli, 30–60 dakika)

1. Colab'a, **PhysioNet'in BigQuery erişimi verdiği Google hesabıyla** girin. MIMIC-III
   sayfasında "Access to the GCP BigQuery was previously awarded to ..." satırında yazan
   hesap budur (ör. okul hesabınız). Açılan tüm giriş pencerelerinde de bu hesabı seçin.
2. Notebook 01'deki gibi GitHub sekmesinden `notebooks/02_external_validation_mimic.ipynb`
   dosyasını açın.
3. İlk hücredeki `GCP_PROJECT` değerini kontrol edin. Aynı hesapla
   https://console.cloud.google.com adresine girin; üstteki proje seçicide görünen
   **Project ID** buraya yazılmalı. Hiç proje yoksa **New project** ile oluşturun.
4. Hücreleri sırayla çalıştırın:
   - **Access check:** Birkaç saniye sürer. "MIMIC-III access OK" yazarsa devam edin.
     Hata verirse mesajda ne yapmanız gerektiği yazar.
   - **Step 1:** Modelleri eğitip kaydeder (yaklaşık 2 dk).
   - **Step 2:** BigQuery'den veriyi çeker ve dış doğrulamayı çalıştırır (10–40 dk).
     Çekilen veri Drive'a önbelleklenir; ikinci çalıştırmada sorgu tekrarlanmaz.
   - **Step 3:** EF kontrolü için 200 rapor seçer. Daha önce doldurduğunuz bir sayfa varsa
     onun üzerine yazmaz.
5. **"Text to send back"** hücresinin çıktısını kopyalayıp sohbete yapıştırın. Bu çıktıda
   yalnızca özet sayılar bulunur ve 10'dan küçük hücreler gizlenir.

## EF etiketleme (sonra yapılabilir)

`HF_Revisit_Project/data/mimic_cache/ef_annotation_sheet.csv` dosyasını Drive'dan açın.
Siz ve bir meslektaşınız birbirinizden bağımsız olarak her raporda yazan EF değerini
`ef_manual_a` ve `ef_manual_b` sütunlarına girin. Değer bir sayı (`55` ya da `55,5`)
olmalı; raporda EF yazmıyorsa `NA` yazın ya da boş bırakın. Dosyayı Excel (`;` ayraçlı)
veya Google Sheets ile kaydedebilirsiniz, iki biçim de okunur. Ardından notebook'un son
hücresini çalıştırın.

**Paylaşmayın:** `data/mimic_cache/` klasörü ve etiketleme sayfası hasta metni içerir
(PhysioNet sözleşmesi). Yalnızca `results/tables/mimic_*.csv` dosyaları paylaşılabilir.

## Sık görülen hatalar

| Mesaj | Çözüm |
|---|---|
| `Access Denied`, `User does not have permission` | Yanlış Google hesabı. **Runtime → Disconnect and delete runtime** yapın, yeniden çalıştırın ve doğru hesabı seçin. |
| `Project ... not found`, `bigquery.jobs.create` | `GCP_PROJECT` yanlış ya da başka bir hesaba ait. Project ID'yi console.cloud.google.com adresinden kopyalayın. |
| `Billing has not been enabled` | Projede BigQuery Sandbox'ı açın (ücretsiz) ya da faturalandırmayı etkinleştirin. |
| Başka bir hata | Kırmızı hata metninin son 20 satırını sohbete yapıştırın. |
