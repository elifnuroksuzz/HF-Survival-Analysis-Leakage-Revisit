# Re-evaluating Heart Failure Survival: From Data Leakage to Robust Clinical Prognostics

[](https://www.python.org/downloads/)
[](https://opensource.org/licenses/MIT)
[](https://colab.research.google.com/)

Bu çalışma, kalp yetmezliği (Heart Failure) sağkalım tahminleme çalışmalarındaki metodolojik boşlukları ele almaktadır. Proje, literatürde sıkça kullanılan Kaggle HF veri setindeki "time leakage" (zaman sızıntısı) sorununu kanıtlamakta ve geleneksel sınıflandırıcılar yerine **Survival Analysis (Sağkalım Analizi)** modellerinin gerekliliğini vurgulamaktadır.

## 🚀 Öne Çıkan Özellikler

  * **Leakage Proof:** `time` değişkeninin klinik bir tahmin edici değil, bir sonuç değişkeni olduğunun matematiksel ispatı.
  * **Methodological Correction:** Klasik ML yerine Cox-PH, Random Survival Forests (RSF) ve DeepSurv modelleriyle sağkalım analizi.
  * **External Validation:** Kaggle'da eğitilen modellerin **MIMIC-III** veri setinde (Cross-dataset) test edilmesi.
  * **XAI & Fairness:** SHAP ile model kararlarının açıklanması ve demografik gruplar arası adalet (fairness) analizi.

-----

## 📂 Proje Mimarisi

Proje, bağımsız çalışabilen ve sonuçları otomatik olarak `/results/` klasörüne raporlayan 5 modüler notebook üzerine kurulmuştur:

| Notebook | Açıklama | Anahtar Çıktı |
| :--- | :--- | :--- |
| **00\_Setup** | Veri ön işleme ve Kaggle/PhysioNet entegrasyonu. | Baseline Statistics |
| **01\_Replication** | Chicco & Jurman (2020) çalışmasının 10 model ile replikasyonu. | Benchmark Metrics (MCC/F1) |
| **02\_Leakage\_Proof** | "Follow-up time" özelliğindeki hedef sızıntısının deneysel kanıtı. | Bias Analysis Report |
| **03\_Survival** | Sağkalım modellerinin (Cox, RSF, DeepSurv) eğitimi. | C-Index & KM Curves |
| **04\_External\_Val** | MIMIC-III kohortu üzerinde dış doğrulama. | Generalization Gap |
| **05\_XAI\_Fairness** | SHAP Interpretability ve Subgroup Fairness denetimi. | Feature Importance / Odds |

-----

## 🛠️ Kurulum ve Bağımlılıklar

1.  Repoyu klonlayın:
    ```bash
    git clone https://github.com/kullanici-adi/HF-Survival-Revisit.git
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install scikit-survival pycox lifelines shap fairlearn
    ```
3.  **MIMIC-III Erişimi:** Notebook 04'ü çalıştırmak için PhysioNet üzerinden CITI sertifikasına ve MIMIC-III erişim iznine sahip olmanız gerekmektedir.

-----

## 📊 Önemli Bulgular (Özet)

  * **Sızıntı Etkisi:** `time` değişkeni dahil edildiğinde MCC skorları \~0.60 seviyesindeyken, bu değişken çıkarıldığında gerçek klinik performansın \~0.40 bandına gerilediği gözlemlenmiştir.
  * **Model Üstünlüğü:** Random Survival Forest (RSF), sansürlenmiş verilerle (right-censored) başa çıkmada klasik Random Forest modellerine göre daha yüksek kalibrasyon başarısı göstermiştir.
  * **XAI:** SHAP analizleri, **Ejection Fraction** ve **Serum Creatinine** seviyelerinin her iki veri setinde de en tutarlı prediktörler olduğunu doğrulamıştır.

-----

## 🧬 Veri Kaynakları

  * **Kaggle HF:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
  * **MIMIC-III:** [Medical Information Mart for Intensive Care III](https://physionet.org/content/mimiciii/1.4/)

-----

## 📝 Atıf ve Kaynakça

Bu çalışma, Davide Chicco ve Giuseppe Jurman'ın (2020) araştırmasını temel alarak geliştirilmiştir.

> *Chicco, D., Jurman, G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak 20, 16 (2020).*

-----

### 💡 Geliştirici Notu

Bu proje, Colab T4 GPU üzerinde optimize edilmiştir. Derin öğrenme tabanlı sağkalım modelleri (DeepSurv) için GPU hızlandırmasının açık olması önerilir.

-----


