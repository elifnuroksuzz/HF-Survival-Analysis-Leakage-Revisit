# Kod İnceleme Raporu — Orijinal Notebook'lar (legacy/)

Bu belge, `legacy/` klasöründeki orijinal notebook'larda bulunan hataları, bunların
makaleye etkisini ve yeni kod tabanında (`src/hfrevisit/`, `scripts/`) nasıl
düzeltildiğini listeler. Önem derecesi: **K** = kritik (yayınlanırsa sonuç yanlış
olur), **Y** = yüksek (hakem reddi sebebi), **O** = orta (metodolojik zayıflık),
**D** = düşük (kod kalitesi).

## 1. Sonucu doğrudan değiştiren hatalar

| # | Notebook | Önem | Sorun | Etki | Düzeltme |
|---|---|---|---|---|---|
| 1 | NB03 Hücre 9 | **K** | `concordance_index_censored(e, t, -ph)` — scikit-survival *risk skoru* bekler (yüksek = erken olay). Kısmi hazardın negatifi verilince C-index ters döner. | Cox CV C-index **0.285** raporlanmış; doğrusu **0.715** (aynı fold'larla doğrulandı). Makalede "Cox rastgeleden kötü" denirdi. | `survival.evaluate_fold` her modelde `model.predict()` (risk) kullanır; `tests/test_core.py::test_cindex_orientation` ile korunur. |
| 2 | NB03 Hücre 11 | **K** | RSF CV döngüsü silinmiş (`# ... aynı kalıyor ...`). `df_rsf_cv`, `rsf`, `rsf_ci_mean` notebook'ta tanımlı değil; sonuçlar gizli kernel durumundan geliyor. | RSF C-index = 0.735 yeniden üretilemez. | `survival.cross_validate` tüm modelleri aynı tekrarlı 5×10 CV'de değerlendirir. |
| 3 | NB03 Hücre 15 | **K** | DeepSurv'de erken durdurma validasyonu `X_tr_s[:val_size]` — CSV `time`'a göre **sıralı** olduğundan validasyon seti sistematik olarak en kısa takipli (çoğu ölmüş) hastalardan oluşuyor. | Erken durdurma yanlı; DeepSurv sonucu güvenilmez. | Yükleyici veriyi sabit seed ile karıştırır; `DeepSurv` olay-tabakalı rastgele validasyon kullanır. |
| 4 | NB04 / NB07 | **K** | Trombosit birim uyumsuzluğu: Kaggle `263358` (/µL), MIMIC `229` (K/µL). Model MIMIC'teki tüm hastaları ağır trombositopenik görüyor. | MIMIC AUC=0.59 ve felaket kalibrasyon (ECE=0.41) büyük ölçüde bu hatadan. Tablo 1'de de "263358 vs 215" görünüyor. | `mimic/cohort.py`: MIMIC trombosit ×1000. Test: `test_build_cohort_harmonisation`. |
| 5 | NB04 / NB07 | **K** | EF, hastanın **ilk echo notundan** alınıyor ve `SUBJECT_ID` ile birleştiriliyor; not başka (hatta sonraki) bir yatıştan olabilir → dış kohortta **gelecekten bilgi sızıntısı**. Ayrıca metin 3000 karakterde kesiliyor (MIMIC'te EF genelde "Conclusions" kısmında, sonda). | EF hatalı/yanlı; makalenin ana tezi "leakage" olduğu için hakem için çok ağır bir tutarsızlık. | `queries.echo_sql`: yalnız indeks yatış veya önceki 1 yıl; tam metin; `extract_index_ef` indeks yatışı tercih eder. |
| 6 | NB04 / NB07 | **Y** | İlk yatış `MIN(HADM_ID)` ile seçiliyor. MIMIC-III'te HADM_ID kronolojik değildir. | Yanlış "indeks" yatış. | `ROW_NUMBER() ... ORDER BY ADMITTIME`. |
| 7 | NB04 / NB07 | **Y** | CPK için ITEMID 50911 (CK-MB, ng/mL) de alınıyor — farklı analit ve birim. Ayrıca sodyumda iki ITEMID'nin `MAX`'ı alınıyor (ilk ölçüm değil). | CPK karışık birimde. | Yalnız 50910; her laboratuvar için kronolojik ilk değer. |
| 8 | NB04 | **Y** | Kayıtlı model bulunamadı; **fallback** RF (farklı hiperparametre, `class_weight='balanced'`) kullanıldı. Raporlanan dış doğrulama, makalede tarif edilen model değil. | Sonuçlar modelle eşleşmiyor. | `run_development.py --stages export` modelleri kaydeder; dış doğrulama yalnızca bunları kullanır. |
| 9 | NB04 | **Y** | Uç nokta uyumsuzluğu: geliştirme kohortunda "takip süresince ölüm" (~130 gün), MIMIC'te hastane içi ölüm. | Dış doğrulama farklı bir hedefi ölçüyor. | Birincil uç nokta: başvurudan itibaren 90 günlük tüm-neden ölüm (`PATIENTS.DOD`), zaman-olay formatında; hastane içi ölüm ikincil. |
| 10 | NB05 | **Y** | "Test seti" bulunamayınca ham veri yeniden bölünmüş ve fallback model eğitilmiş: SHAP ve fairness **n = 60** hastada (bazı alt gruplarda n = 7). | Alt grup sonuçları istatistiksel olarak anlamsız. | Fairness ve açıklamalar 299 hastanın tümünün **out-of-fold** tahminlerinde yapılır; bootstrap GA ve "olay < 20 → keşifsel" işareti. |
| 11 | NB05 Hücre 9, 14 | **Y** | `CHICCO_BORDA_RANK` ve `ZAHID_REPORTED` değerleri elle yazılmış; kaynakla doğrulanmamış. Zahid ve ark. 2019 çalışması bildiğim kadarıyla *PLoS ONE*'da yayımlanmış bir Cox modeli çalışmasıdır, kodda ise "J Med Syst 43:298" ve AUC/duyarlılık değerleri yazıyor. SHAP ile Borda arasında ρ = 0.973 çıkması da bu tablonun SHAP'a bakılarak yazılmış olabileceğini düşündürüyor. | Doğrulanmamış literatür değeri yayımlamak ciddi bir bütünlük riski. | Kodda sabit literatür değeri yok. Yayınlanmış değerler `data/reference/chicco2020_reported.TEMPLATE.csv` şablonuna makaleden **elle aktarılmalı ve ikinci kişi kontrol etmeli**. |
| 12 | NB01 Hücre 16 | **Y** | `CHICCO_REFERENCE` (ör. RF MCC = 0.587) doğrulanmamış. Hatırladığım kadarıyla Chicco & Jurman Tablo 4'te RF MCC ≈ 0.38'dir; bu da sizin **S2** (time hariç) sonucunuza yakın, S1'e değil. Bu tahminim kaynaktan doğrulanmalı. | "Replikasyon başarısız (2/10)" yorumu yanlış senaryoyla yapılmış olabilir. | Aynı şablon; karşılaştırma senaryo bazında yapılır (`replication_vs_published.csv`). |

## 2. Hatalı ama sonucu daha az etkileyen sorunlar

| # | Notebook | Önem | Sorun | Düzeltme |
|---|---|---|---|---|
| 13 | NB02 Hücre 2 | O | `load_and_verify_data(path)` → path `verify` parametresine gidiyor; ardından `load_and_verify_data(...)` (Ellipsis) ve tüm sütunları BÜYÜK harfe çeviren yama. | Tek yükleyici `data.load_cohort`, şema doğrulaması. |
| 14 | NB03 | O | Brier ve zaman-bağımlı AUC hücreleri `no field of name time` hatasıyla **sessizce atlanmış** (`Surv.from_dataframe('DEATH_EVENT','TIME')` alan adları). Makalede vaat edilen kalibrasyon metrikleri hiç hesaplanmamış. | `data.to_structured` alan adları sabit (`event`,`time`); Brier(30/60/90/180), IBS, KM-null IBS hesaplanır. |
| 15 | NB03 | O | Cox L2 cezası (`penalizer=0.1`) ölçeklenmemiş değişkenlerde → ceza trombosit (10⁵ ölçeği) ile yaş arasında keyfi. | Çarpık laboratuvar değerleri log, tümü standardize, sonra ceza. HR tablosu klinik birimlerle (10 yaş, EF 5 puan, kreatinin iki katı) cezasız modelden. |
| 16 | NB03 | O | Permutation importance **eğitim verisinde** (in-sample). | Out-of-fold permutation importance (C-index düşüşü). |
| 17 | NB03 Hücre 20 | O | "Overall KM" eğrisi `DEATH_EVENT`'e göre tabakalanmış — sonuca göre KM çizmek anlamsız (yaşayan eğrisi hep 1). | Kaldırıldı; tertile KM + log-rank. |
| 18 | NB03 | O | 5 fold ortalamasının bootstrap'ı "95% CI" diye raporlanmış; NB02'de "ort ± 1.96 SD" GA diye adlandırılmış. | Tekrarlı CV'de fold dağılımının 2.5–97.5 yüzdelikleri "empirik aralık" olarak; model farkları için Nadeau–Bengio düzeltilmiş t-testi. |
| 19 | NB02 Deney B | O | `time` grubuna göre katmanlayıp grup içinde model eğitmek sızıntıyı kanıtlamaz (grup içi MCC düşüklüğü sınıf dengesizliğinden). | Daha güçlü deneyler: *yalnız time* modeli, *karıştırılmış time* kontrolü, sabit-ufuk yeniden çerçeveleme (Deney B–E). |
| 20 | NB04 | O | Hosmer–Lemeshow n ≈ 4000'de birincil kalibrasyon kanıtı olarak kullanılmış; ayrıca serbestlik derecesi her zaman `g-2`. | Kalibrasyon eğimi, CITL, ECE birincil; HL yalnız ek bilgi. |
| 21 | NB07 | O | Kalibrasyon tek 50/50 bölmede. | Çapraz-uydurmalı (5-fold) yeniden kalibrasyon: her hasta örneklem-dışı düzeltilmiş risk alır. |
| 22 | NB04 | O | MIMIC eksik CPK'sı **MIMIC medyanı** ile dolduruluyor (NB07'de Kaggle medyanı) — iki notebook tutarsız. | Birincil: geliştirme kohortu medyanı; duyarlılık: CPK tam-vaka. |
| 23 | NB04 | O | Anemi tanımı NB04'te Hct<36 **veya** Hb<12, NB07'de WHO cinsiyete özgü. | Geliştirme kohortunun tanımı (Ahmad 2017: Hct < %36) — makaleden teyit edin. |
| 24 | NB04 | O | Yaş `DATE_DIFF(..., YEAR)` (takvim yılı farkı) ve 89 üstü 89'a sıkıştırılmış. | Gün/365.2425; >89 → 90. |
| 25 | NB04 | D | Domain shift tablosunda Mortalite satırı `KeyError` ile hata vermiş; `p = 0.001  # sembolik` gibi uydurma p-değeri. | `table_one` ile gerçek testler. |
| 26 | NB01 | D | `use_label_encoder` (kaldırılmış XGBoost parametresi), `SVC(probability=True)` (iç CV ile yavaş ve AUC'yi değiştirmez), `copy.deepcopy` yerine `sklearn.clone`. | Düzeltildi. |
| 27 | Hepsi | D | `warnings.filterwarnings('ignore')` global; yollar Colab'a sabit; `Hücre 1 — Paket Kurulumu` başlıkları boş; Türkçe/İngilizce karışık çıktı. | `HF_PROJECT_ROOT` ortam değişkeni; uyarılar susturulmaz. |

## 3. Repo düzeyi

* **`06_cross_training_matrix` notebook'u bu projeye ait değil** (AI-üretimi görüntü tespiti,
  MMSP 2026 makalesi). Repodan kaldırıldı; git geçmişinde (`d35e10d`) duruyor. Başka bir
  makalenin hakem sürecindeki materyalini kamuya açık bir repoda tutmamanız önerilir.
* MIMIC-III türevi sonuçlar (hasta düzeyi tahminler, SHAP değerleri) PhysioNet DUA gereği
  paylaşılamaz; `.gitignore` bunları dışarıda bırakır.
* README'de "MIT License" yazıyordu ama `LICENSE` dosyası yoktu.

## 4. Yeniden üretilebilirlik

```bash
pip install -e ".[dev]"
python scripts/run_development.py --download   # tüm geliştirme analizleri
pytest                                          # 21 test
python scripts/run_external_validation.py --gcp-project <PROJE>   # MIMIC-III erişimi gerekir
```
