# Makale Rehberi — SCI/SCIE İndeksli Dergi İçin

## 1. Makalenin tek cümlelik iddiası

> Kalp yetmezliği mortalite tahmininde sıkça kullanılan 299 hastalık veri setinde
> takip süresi (`time`) sonuçtan türetilmiş bir değişkendir; onu tahmin edici olarak
> kullanmak performansı yapay olarak şişirir. Doğru çerçeve sağkalım analizidir ve bu
> çerçevede modeller bağımsız bir ABD kohortunda (MIMIC-III) dış doğrulamaya tabi
> tutulmalıdır.

Bu bir **metodolojik yeniden değerlendirme + dış doğrulama** makalesidir. "Yeni model"
iddiası yoktur; gücü titizlikten gelir. Hakemler bu yüzden *her* sayıyı sorgulayacaktır.

## 1b. Araştırma soruları ve hipotezler

| RQ | Hipotez | Test eden analiz | Çıktı |
|---|---|---|---|
| RQ1 Sızıntı var mı, büyüklüğü ne? | **H1** `time` eklenince MCC belirgin artar. **H1a** Yalnız `time` ≈ tüm değişkenler + `time`. **H1b** `time` karıştırılınca artış kaybolur. | Deney B–D, düzeltilmiş t-testi | `leakage_BCD_*`, Şekil 1 |
| RQ1' Sızıntı yapısal mı? | **H1c** Hiçbir değişkenin etkisi olmayan simüle kohortlarda da `time` yüksek MCC üretir. | Simülasyon (200 kohort, sıfır ve gözlenen etki) | `simulation_*`, Ek Şekil |
| RQ2 Sızıntısız gerçek performans? | **H2** Baseline değişkenlerle sağkalım modelleri C ≈ 0.70 ve KM-null'dan daha iyi IBS. | Tekrarlı 5×10 CV | `survival_cv_summary`, Şekil 3 |
| RQ3 Sade model yeterli mi? | **H3** EF + SC Cox, 11 değişkenli Cox'tan anlamlı farklı değil; esnek modeller (RSF, GB-Cox) Cox'u anlamlı geçmez. | Eşleştirilmiş model karşılaştırması | `survival_paired_comparison` |
| RQ4 Klinik fayda ve kalibrasyon? | **H4** 90 günlük OOF risk iyi kalibre (O/E ≈ 1) ve makul eşiklerde "herkesi tedavi et"ten yüksek net fayda. | Sansür-uyumlu DCA ve KM kalibrasyonu | `dca_*`, `calibration_oof_*`, Şekil 6 |
| RQ5 Taşınabilirlik? | **H5** MIMIC-III'te ayırt edicilik kısmen korunur, kalibrasyon bozulur, yeniden kalibrasyon düzeltir. | Dış doğrulama | `mimic_*`, Şekil 5 |
| RQ6 Adalet? | **H6** Cinsiyet ve yaş alt gruplarında C-index farkı yok (CI'lar örtüşür). | OOF alt grup analizi | `table3_subgroup_cindex` |

Örneklem yeterliliği (Riley ve ark.) ayrıca raporlanır (`sample_size_riley`). Hızlı denemede n_min ≈ 351 > 299 çıktı; bu bir **sınırlılıktır** ve shrinkage beklentisiyle (≈0.87) birlikte dürüstçe yazılmalı.

Makaledeki **tüm sayılar** `results/RESULTS_SUMMARY.md` dosyasından alınmalı. Bu dosya tablolardan otomatik üretilir.

## 2. Hedef dergi önerileri

| Dergi | İndeks | Uygunluk |
|---|---|---|
| *BMC Medical Informatics and Decision Making* | SCIE | Chicco & Jurman (2020) burada yayımlandı; yorum/yeniden analiz için doğal yer. |
| *Computers in Biology and Medicine* | SCIE | Sızıntı + dış doğrulama ilgisini çeker. |
| *International Journal of Medical Informatics* | SCIE | Klinik bilişim, TRIPOD+AI uyumu bekler. |
| *JAMIA Open* / *Diagnostic and Prognostic Research* | ESCI / Scopus | Metodoloji odaklı, kısa "technical note" için. |

Gönderimden önce derginin güncel kapsamını ve indeks durumunu kontrol edin.

## 3. Raporlama standardı: TRIPOD+AI (Collins ve ark., BMJ 2024)

Gönderime doldurulmuş TRIPOD+AI kontrol listesi ekleyin. Kod tabanındaki karşılıkları:

| TRIPOD+AI maddesi | Nerede |
|---|---|
| Veri kaynağı, katılımcılar | Tablo 1 (`table1_cohort_by_outcome.csv`), MIMIC attrition (`mimic_primary_attrition.csv`) → akış diyagramı |
| Sonuç tanımı ve zamanlaması | Geliştirme: takip süresince ölüm (sansürlü); MIMIC: başvurudan 90 gün içinde ölüm |
| Tahmin ediciler ve ölçüm zamanı | Yalnız başlangıç (baseline) değişkenleri; MIMIC'te ilk 24 saat lab + indeks yatış echo |
| Örneklem büyüklüğü | 96 olay / 11 aday parametre ≈ 8.7 EPP → Riley ve ark. (2020) ölçütüyle sınırda; *limitations*'ta tartışın |
| Eksik veri | Geliştirme: yok. MIMIC: CPK %~40 eksik → geliştirme medyanı ile tek imputasyon + tam-vaka duyarlılık analizi |
| Analiz / iç doğrulama | Tekrarlı tabakalı 5-kat CV (10 tekrar), tüm ön işleme fold içinde |
| Performans ölçütleri | Harrell & Uno C, zaman-bağımlı AUC, Brier, IBS (KM null karşılaştırmalı); dış: kalibrasyon eğimi, CITL, ECE |
| Model güncelleme | Çapraz-uydurmalı yeniden kalibrasyon (intercept, lojistik, izotonik) |
| Adalet (fairness) | Cinsiyet ve yaş alt gruplarında C-index / AUC, duyarlılık, özgüllük farkları |
| Kod ve veri erişimi | Bu repo (Zenodo DOI alın); MIMIC kod-sorguları paylaşılır, veri paylaşılmaz |

## 4. Önerilen yapı (≈ 4000–5000 kelime)

1. **Introduction** — Ahmad (2017) veri seti ve Chicco & Jurman (2020) etkisi (yüksek atıf); veri
   setinin onlarca sonraki çalışmada `time` ile kullanılması; hedef sızıntısı kavramı
   (Kaufman ve ark. 2012; Kapoor & Narayanan 2023, *Patterns*).
2. **Methods** — 2.1 Kohortlar; 2.2 Sızıntı deneyleri A–E; 2.3 Sağkalım modelleri; 2.4 İç
   doğrulama; 2.5 MIMIC-III kohortu ve uyumlaştırma; 2.6 Dış doğrulama ve kalibrasyon;
   2.7 Açıklanabilirlik ve alt grup analizi; 2.8 İstatistik ve yazılım.
3. **Results** — Şekil 1 (sızıntı), Tablo 2 (Cox HR), Şekil 3 (C-index), Tablo 3 (alt gruplar),
   Şekil 5 (MIMIC kalibrasyon).
4. **Discussion** — Literatüre etkisi (ör. `time` kullanan çalışmaların sayısını sistematik
   bir taramayla belirlemek makaleyi çok güçlendirir), klinik yorum, sınırlılıklar.

## 5. Methods için İngilizce taslak paragraflar

**Target leakage.** *In the development cohort, the variable `time` records the number of days
from enrolment to death or censoring. It is therefore determined by the outcome and is unknown
at the time a prognosis would be made. We quantified the resulting leakage with five
experiments on identical stratified 70/30 splits (100 repetitions): (A) the distribution of
follow-up by vital status; (B) a classifier using `time` alone; (C) the 11 baseline variables
with versus without `time`; (D) a permutation control in which `time` was shuffled independently
within the training and test partitions; and (E) a leakage-free reformulation predicting death
within a fixed horizon (90 and 180 days) from baseline variables, excluding patients censored
before the horizon. Paired differences were tested with the corrected resampled t-test
(Nadeau & Bengio, 2003).*

**Survival models.** *We fitted a ridge-penalised Cox proportional-hazards model (11 baseline
variables and, separately, ejection fraction and serum creatinine only), a random survival
forest (500 trees, minimum leaf size 15) and a gradient-boosted Cox model. Right-skewed
laboratory values (creatinine phosphokinase, serum creatinine, platelets) were
log-transformed and all predictors standardised within each training fold. Performance was
estimated by 10 × repeated stratified 5-fold cross-validation using Harrell's and Uno's
concordance, cumulative/dynamic time-dependent AUC at 30, 60, 90 and 180 days, the Brier score
and the integrated Brier score (15–240 days) compared with a Kaplan–Meier null model. The
proportional-hazards assumption was assessed with scaled Schoenfeld residuals.*

**External validation.** *Models fitted on the full development cohort were applied without
re-estimation to adults with a first hospital admission coded ICD-9 428.x in MIMIC-III v1.4.
Predictors were the first laboratory measurement between 6 h before and 24 h after admission,
diagnoses coded during the index admission, and left-ventricular ejection fraction extracted
by a validated rule-based parser from echocardiography reports of the index stay (or, if
absent, the preceding year). Units were harmonised to the development cohort (platelets
K/µL × 1000; anaemia defined as haematocrit < 36 %). The primary endpoint was all-cause
mortality within 90 days of admission from the PATIENTS table (hospital and Social Security
death records). We report discrimination (Harrell's C, AUC at 90 days), calibration
(calibration-in-the-large, calibration slope, expected calibration error, calibration
plots), and cross-fitted recalibration.*

## 6. Gönderimden önce yapılması zorunlu olanlar (yapılacaklar listesi)

- [ ] `python scripts/run_external_validation.py` — yeni (düzeltilmiş) MIMIC sonuçlarını üretin;
      eski NB04/NB07 sayılarını (AUC 0.59–0.60) **kullanmayın**.
- [ ] **EF ayrıştırıcı doğrulaması:** rastgele 200 echo raporunu iki kişi elle etiketlesin;
      ayrıştırıcının duyarlılık/kesinlik ve ortalama mutlak hatasını Ek Tablo olarak verin.
- [ ] MIMIC `DOD` takip kapsamını (SSA ölüm kayıtları) belgelerden teyit edin; 90 gün birincil,
      180 gün duyarlılık analizi (`--horizon 180`).
- [ ] Chicco & Jurman (2020) ve Ahmad ve ark. (2017) değerlerini makalelerden
      `data/reference/chicco2020_reported.TEMPLATE.csv` şablonuna aktarın (iki kişi kontrol).
- [ ] Ahmad (2017) anemi tanımını (Hct < %36) ve birimlerini (CPK mcg/L, trombosit
      kiloplatelets/mL) orijinal makaleden teyit edin.
- [ ] DeepSurv'ü GPU'lu ortamda `--with-deepsurv` ile çalıştırın (bu ortamda PyTorch yoktu).
- [ ] Repoyu Zenodo'ya bağlayıp DOI alın; makalede "Code availability" bölümüne koyun.
- [ ] Etik beyan: MIMIC-III için PhysioNet DUA + CITI; geliştirme verisi kamuya açık (CC BY 4.0).

## 7. Beklenen hakem soruları ve cevap kaynağı

| Soru | Cevap |
|---|---|
| "Time'ı çıkarmak bilgi kaybı değil mi?" | Deney D (karıştırılmış time kazancı yok) + Deney B (yalnız time ≈ tam model) |
| "Neden sınıflandırma değil sağkalım?" | Sansürlü hastaların %68'i; Deney E (sabit ufuk) sansürlüleri dışlamak zorunda kalır |
| "n = 299 çok küçük" | Tekrarlı CV, fold dağılımları, EPP tartışması, dış doğrulama |
| "MIMIC farklı popülasyon" | Bu bir *transportability* testi; case-mix tablosu + yeniden kalibrasyon + yaş ≥ 40 duyarlılık analizi |
| "EF metinden çıkarıldı, ne kadar doğru?" | Elle etiketlenmiş örneklem doğrulaması (yukarıda) + yalnız sayısal EF duyarlılık analizi |
| "Hangi dış doğrulama endpoint'i?" | 90 günlük tüm-neden ölüm (zaman-olay); hastane içi ölüm ikincil |
