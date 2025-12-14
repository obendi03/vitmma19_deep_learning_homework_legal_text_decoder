## Deep Learning tantárgy (VITMMA19) — Legal Text Decoder

# Projektinformáció
Kiválasztott téma: Legal Text Decoder

Hallgató: Oláh Bendegúz István

Cél: +1 pont elérése: [Nem]

A projekt célja egy olyan természetes nyelvfeldolgozási (Natural Language Processing, NLP)
modell létrehozása, amely képes megjósolni, hogy egy adott Általános Szerződési Feltételek
(ÁSZF) és/vagy Általános Felhasználási Feltételek (ÁFF) szövegének egy bekezdése mennyire
könnyen vagy nehezen érthető egy átlagos felhasználó számára. A modell egy 1-től 5-ig
terjedő skálán adja meg az érthetőséget, segítve ezzel a jogi szövegek komplexitásának
felmérését.
# Megoldás rövid leírása
Moduláris, reprodukálható mélytanulási pipeline-t készítettem jogi szövegek osztályozására. Főként LSTM‑variánsokat (BiLSTM, attention), focal loss-t és HPO-t alkalmaztam.

## Adatelőkészítés (Data Preparation)

A pipeline alapértelmezett működése: a konténer indításakor a `src/00_download_data.py` script próbálja automatikusan letölteni és kicsomagolni a szükséges nyers adatkészleteket a `data/` mappába (konténer belső útvonala: `/app/data`). Amennyiben Dockerrel futtatod, csatold a helyi `data` mappát a konténerhez a `-v "%cd%\data:/app/data"` opcióval a `docker run` parancsban.

Ha az automatikus letöltés nem sikerül (például SharePoint hitelesítés vagy hálózati korlátozás miatt), kövesd a manuális eljárást:

- Töltsd le a ZIP fájlt a SharePoint megosztásról a böngészővel.
- Nevezd át a letöltött fájlt `downloaded.zip`-re.
- Helyezd a `downloaded.zip` fájlt a repository `data/` mappájába (konténerben: `/app/data`).


## Kísérleti beállítások

### Hardver és futtatási környezet
Az összes kísérlet GPU-n futott; a futtatási naplók szerint a használt GPU: **NVIDIA GeForce GTX 1050 Ti** (CUDA capability 6.1). A scriptek naplózzák a `Using device: cuda` sort és a GPU-adatokat (például: [lstm_version_7_data_augmentation log](model_developement/overfit_full_dataset/lstm_version_7_data_augmentation/log/run_2025-12-13_17-40-12.log#L178)). Ray Tune HPO esetén a naplók 1 logikai GPU használatát mutatják (példa: [lstm_version_8_hpo log](model_developement/overfit_full_dataset/lstm_version_8_hpo/log/run_2025-12-13_19-58-17.log)).

- **Baseline modell és kiértékelés:** a TF-IDF + Logistic Regression baseline és a kiértékelő pipeline a [src/03_1_evaluation_test_set.py](src/03_1_evaluation_test_set.py#L1-L30) fájlban található (accuracy, weighted F1, confusion matrix, classification report).

	Rövid konfigurációs összegzés (log alapján):

	- **TF-IDF:** `TfidfVectorizer(max_features=5000)` (fit a `train.csv`-en; teszten `transform`).
		- `max_features`: 5000
		- `ngram_range`: (1, 1) (alapértelmezett)
		- `analyzer`: 'word' (alapértelmezett)
		- `stop_words`: None (alapértelmezett)
		- `norm`: 'l2' (alapértelmezett)
		- `use_idf`: True (alapértelmezett)
	- **LogisticRegression:** `LogisticRegression(max_iter=1000)` (implicit: `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `class_weight=None`).
		- `max_iter`: 1000
		- `penalty`: 'l2' (alapértelmezett)
		- `C`: 1.0 (alapértelmezett)
		- `solver`: 'lbfgs' (alapértelmezett)
		- `class_weight`: None (alapértelmezett)
		- `multi_class`: 'auto' (alapértelmezett)
		- `tol`: 1e-4 (alapértelmezett)
	- **Kiértékelés:** `accuracy_score`, `f1_score(..., average='weighted')`, `confusion_matrix`, `classification_report`.

- **Train / Validation / Test split:** a split logikája és a mentés a `train.csv`/`val.csv`/`test.csv` fájlokba a [src/01_data_preprocessing.py](src/01_data_preprocessing.py#L123-L124) fájlban található (`train_test_split` hívások). A futtatásokban használt példa-méretek a naplókban szerepelnek (pl. Train=2434, Val=272, Test=307).

- **Consensus adatok:** a konszenzusos címkéket használtam a modell végeleges minősítésére a teszt adathalmazon elért legjobb konfiguráció mellett, mindezt úgy hogy a az egyes paragrafusokra adott értékelseket átlagoltam és ez az átlag volt a mit elvártam a modelltől, gogy jósoljon. A [src/01_data_preprocessing.py](src/01_data_preprocessing.py#L94-L119) fájlban kiszűröm azokat a bekezdéseket ; a konszenzusos adathalmaz csak a végső kiértékeléshez használatos (lásd: [src/03_2_evaluation_consensus.py](src/03_2_evaluation_consensus.py#L1-L30)).

- **Modellválasztás és HPO:** az alap tréning/val kiválasztás a [src/02_train.py](src/02_train.py#L1-L40) fájlban van megvalósítva (checkpointing, early stopping). A Ray Tune konfiguráció és keresőtér a [src/02_train_ray_tune.py](src/02_train_ray_tune.py#L162-L175) fájlban (`search_space`) található; a HPO futtatást a fájl `tune.run` hívása indítja.

- **Inference:** predikciós (inference) kód és eredmény-mentés a [src/04_inference.py](src/04_inference.py#L1-L40) fájlban található.




## Megjegyzések az HPO-hoz és a kiválasztáshoz
Az `lstm_version_8_hpo` és `lstm_version_8_best_param_hpo` könyvtárak hiperparaméter-keresési kísérleteket tartalmaznak. Az HPO befejezése után a legjobb trialok a `top1/`, `TOP2/`, `top3/` mappákban találhatók model checkpoint-ekkel és naplókkal. Ezek jó kiindulópontok a végső kiértékeléshez és abláci vizsgálatokhoz.

## **Modell developement fontosabb Eredmények (rendezett)**
Az alábbiakban rövid, rendezett eredmények találhatók a rendelkezésre álló naplók alapján. A linkek a `model_developement` mappában tárolt futtatási naplókra mutatnak.

- **Single batch (sanity overfit)** — [model_developement/single_batch/log/small_lstm_single_batch_overfit.log](model_developement/single_batch/log/small_lstm_single_batch_overfit.log)
	- Adat: 32 minta
	- Modell: MultiLayerLSTM — Összes paraméter: 17,389 (embedding: 14,976)
	- Tanulás: kb. 11. epoch körül elérte a 100% train pontosságot és az 104. epochig megtartotta; végső train veszteség ≈ 0.001 (overfit megfigyelhető)
	- Cél: a tanulási ciklus és gradiensek ellenőrzése; az overfit várt.

- **Half dataset (gyors overfit diagnosztika)** — [model_developement/overfit_half_dataset/log/small_lstm_overfit_half_dataset.log](model_developement/overfit_half_dataset/log/small_lstm_overfit_half_dataset.log)
	- Adat: fél adatkészlet (Train=973, Val=244)
	- Modell: MultiLayerLSTM — Összes paraméter: 239,021 (embedding: 236,608)
	- Tanulás: 11. epoch körül elérte a 99% train pontosságot; early-stop miatt leállt
	- Cél: gyors ellenőrzés, hogy a modell gyorsan tanul-e kisebb adaton; az eredmény az overfitting kockázatára utal.

- **Teljes adatkészlet (kiemelt futtatások)**
	- `lstm_version_1_first_try` — [model_developement/overfit_full_dataset/lstm_version_1_first_try/log/run_2025-12-13_14-54-11.log](model_developement/overfit_full_dataset/lstm_version_1_first_try/log/run_2025-12-13_14-54-11.log)
		- Adatok: Train=2434, Val=272, Test=307; vocab≈7072; embedding_dim=64
		- Modell: összes paraméter ≈455,085
		- Eredmény: Test Loss=1.4641, Test Acc=28.99%, Weighted F1=0.2195 (early stop 41. epoch után)
		- Increment: Inc#0 (Baseline)

	- `lstm_version_6_attention` — [model_developement/overfit_full_dataset/lstm_version_6_attention/log/attention_increment.log](model_developement/overfit_full_dataset/lstm_version_6_attention/log/attention_increment.log)
		- Adatok: Train=2434, Val=272, Test=307; Word2Vec vocab≈24k; embedding_dim=64
		- Modell: attention + LSTM — összes paraméter ≈1,782,342 (embedding ≈1,550,144)
		- Eredmény: Test Loss=1.3492, Test Acc=41.37%, Weighted F1=0.3542 (early stop 40. epoch után)
		- Increment: Inc#4 (Attention — improvement)

	- `lstm_version_8_hpo` (Ray Tune összegzés) — [model_developement/overfit_full_dataset/lstm_version_8_hpo/log/run_2025-12-13_19-58-17.log](model_developement/overfit_full_dataset/lstm_version_8_hpo/log/run_2025-12-13_19-58-17.log)
		- HPO: 20 trial (változó hidden_dim, lr, dropout, bidirectional, num_layers)
		- Legjobb megfigyelt val_acc ≈ 0.43382 (43.38%); több trial 41–43% tartományban
		- Példa erős konfiguráció: hidden_dim=64, lr≈6.05e-4, bidirectional=True, num_layers=3, fc_dropout≈0.55
		- Increment: Inc#6 (HPO — final search / improvement)

## **Részletes modell fejlesztés lépések**

Megjegyzés: a részletes Modell fejlesztés lépések verziók leírásai külön fájlban találhatók. Kérlek, másold át vagy szerkeszd a részleteket a következő fájlban: [model_developement/PER-LSTM_versions.md](model_developement/PER-LSTM_versions.md).




## Végleges modell és eredmények

Az alábbi információk a `log/run.log` fájlból lettek kinyerve (modell-összegzés, futtatási konfiguráció és végső kiértékelési metrikák). A pontos számok és a teljes log a fenti fájlban találhatók.

**Modell (végső checkpoint):**
- Architektúra: `MultiLayerLSTM` — Embedding(23838, 64, padding_idx=0) → LSTM(64 → 128, num_layers=2, dropout=0.3, batch_first=True) → Attention (Linear(128→1)) → Dropout(0.4) → FC(128→5).
- Paraméterek: összesen **1,757,830**; embedding-paraméterek: **1,525,632**.
**Modell (végső checkpoint):**
```
MultiLayerLSTM(
  (emb): Embedding(23840, 64, padding_idx=0)
  (lstm): LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.3)
  (attention): Linear(in_features=128, out_features=1, bias=True)
  (dropout): Dropout(p=0.4, inplace=False)
  (fc): Linear(in_features=128, out_features=5, bias=True)
)
```
- Total parameters      : 1,757,958
- Trainable parameters  : 1,757,958
- Frozen parameters     : 0
- Embedding parameters  : 1,525,760

**Tanítási / konfigurációs beállítások (a log alapján):**
- Adatméretek: Train=2434, Val=272, Test=307
- Early stopping: `EARLY_STOP_PATIENCE=20` (korai leállás 20 epóchnál)
- Word2Vec: vocab=23837, `embedding_dim=64`, mentve `W2V_PATH=/app/data/w2v.model`
- Fő hyperparaméterek: `BATCH_SIZE=32`, `LR=0.001`, `NUM_LSTM_LAYERS=2`, `SEQ_LEN=500`, `FC_DROPOUT=0.4`, `LSTM_DROPOUT=0.3`

### LSTM-specifikus konfiguráció (a `log/run.log` alapján)

Az alábbi lista a futás konfigurációs blokkja alapján tartalmaz minden, közvetlenül az LSTM architektúrához és tréninghez kapcsolódó beállítást:

- `MODEL_NAME`: LSTM
- `NUM_LSTM_LAYERS`: 2
- `LSTM_HIDDEN_DIM`: 128
- `LSTM_DROPOUT`: 0.3
- `FC_DROPOUT`: 0.4
- `NUM_CLASSES`: 5
- `SEQ_LEN`: 500
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001
- `WEIGHT_DECAY`: 0.0001
- `NUM_EPOCHS`: 1000
- `WORD2_VEC_VECTOR_SIZE`: 64
- `WORD2_VEC_MIN_COUNT`: 1
- `AUGMENTATION`: True (ha alkalmazva: `AUG_PROB`=0.15)


**Eredmények (log összegzés):**
**Eredmények (log összegzés):**
- Tesztkészlet (307 minta): LSTM — Test Loss **1.3459**, Test Acc **42.35%**, Weighted F1 **0.3754**.
- TF-IDF+LR baseline (teszt): Acc **36.81%**, F1 **0.3350**.
- Konszenzus (595 minta): LSTM — Loss **1.5002**, Acc **27.06%**, Weighted F1 **0.2181**; TF-IDF+LR baseline: Acc **34.79%**, F1 **0.3096**.

Megjegyzés: a konszenzusos címkék esetén a TF-IDF+LR baseline jobb eredményt ért el; részletes zavarási mátrixok és classification reportok a logban találhatók.


**Konfúziós mátrixok (vizuális beillesztés)**

Az előző futtatásból készült kép (ha létezik) be van helyezve az `evaluation_plots` mappában:

- Teszt konfúziós mátrix: [evaluation_plots/confusion_matrices_all.png](evaluation_plots/confusion_matrices_all.png)



# Naplózás
Az általam beállított naplózás a `src/utils.py` (vagy a kísérletekhez `model_developement/utils.py`) logger segédletét használja, amely az stdout-ra ír, így a Docker futtatás elfogja a logokat. A beadáshoz szükséges naplóelemeket a train scriptek biztosítják, például:

- Konfiguráció: kiírt hiperparaméterek (epochok, batch méret, tanulási ráta, optimizer, seed).
- Adatfeldolgozás: megerősítések, dataset méretek, előfeldolgozás sikeressége.
- Modell architektúra: modell összegzés, összes paraméter (trainable / nem trainable).
- Tanulási előrehaladás: epoch-onkénti metrikák (veszteség, pontosság vagy feladat-specifikus metrikák) és LR információk.
- Validáció: epoch-onkénti validációs metrikák és checkpointok.
- Végső kiértékelés: metrikák a tesztkészleten (pontosság, F1, zavarási mátrix stb.).

 Alapértelmezés szerint a gyökér `log/run.log` a repository szintű fő log fájl. Az egyes kísérletek saját `log/` mappáiba is írnak naplókat (pl. `model_developement/overfit_full_dataset/lstm_version_8_hpo/log/`).

### Log szekciók (minta egy pipeline futtatás naplójából)

Egy futtatási log általában a következő, jól elkülöníthető szekciókat tartalmazza — az alábbi lista megmutatja, mit kell keresned minden szekcióban:

- **Pipeline start / end:** indítás/lezárás időbélyeggel (pl. "=== Starting pipeline... ===", "=== Finished ... ===").

- **Script indítások:** `Running script: <script_name>` bejegyzések minden futtatott scripthez.

- **Adat letöltés / kicsomagolás:** letöltési státuszok, ZIP kicsomagolás, fájlérvényesítés és esetleges `ERROR loading ...` hibák.

- **Előfeldolgozás:** bemeneti DataFrame oszlopok/méretek (`Base DF columns`, `Consensus DF shape`), kiszűrt sorok, aggregációs statisztikák és mentett fájlok (`inference.csv`, `train/val/test split`).

- **Tréning / Word2Vec:** `==== FULL DATASET TRAIN/VAL START ====` blokk, eszköz-információ (`Using device`), Word2Vec vocab/epoch naplók és embedding mentések.

- **Konfiguráció & hyperparaméterek:** `==== CONFIG & HYPERPARAMETERS START ====` után kulcs=érték felsorolás (BATCH_SIZE, LR, SEQ_LEN, NUM_EPOCHS, W2V_PATH, stb.).

- **Model summary:** `==== MODEL SUMMARY START ====` – architektúra, teljes/ trainable/embedding paraméter-számok.

- **Epoch naplók & checkpointing:** `Epoch N/M | Train Loss: ... | Val Loss: ...` sorok, checkpoint mentés üzenetek, `Early stopping triggered` üzenet.

- **HPO / trial naplók:** Ray Tune esetén trial-specifikus naplók és az összegző exportok a HPO mappában.

- **Kiértékelés (test set):** `==== EVALUATION START ====` blokk, betöltött mintaszám, model/embedding betöltés, Test Loss/Acc/F1, confusion matrix és classification report; figyelmeztetések (pl. UndefinedMetricWarning).

- **Consensus evaluation:** külön blokk a konszenzusos adatokon végzett kiértékelés hasonló metrikákkal.

- **Inference:** `==== INFERENCE START ====`, betöltött inference minták száma, példa predikciók (printed rows), mentett `inference_results.csv`.

- **Hibák és figyelmeztetések:** explicit `ERROR` sorok és Python/library warningok (további vizsgálat szükséges esetén).

Ezeket a szekciókat a README-ben szereplő parancsokkal könnyen fájlba irányíthatod (`> log/run.log 2>&1`), ami megkönnyíti a hibakeresést és a futtatások reprodukálását.

## Docker Instructions (konténerizáció)
A repository tartalmaz egy `Dockerfile`-t a reprodukálható környezethez. Az alábbiakban a `Build` és `Run` parancsok külön blokkokban találhatók (a parancsok változatlanok):

### Build
Futtasd a projekt gyökérkönyvtárából:
```bat
docker build -t lstm_gpu_pipeline_full .
```

### Run
Windows `cmd.exe` példa (a parancs a helyi `log` és `data` mappákat csatolja a konténerhez és a kimenetet a `log/run.log`-ba menti):
```bat
docker run --gpus all --rm -v "%cd%\log:/app/log" -v "%cd%\data:/app/data" lstm_gpu_pipeline_full > log/run.log 2>&1
```

Megjegyzések:
- A fenti `docker run` parancs Windows `cmd.exe` szintaxisra van ( `%cd%` és backslash). PowerShell használata esetén cseréld `%cd%`-t `${PWD}`-re vagy `$(Get-Location)`-ra, és ügyelj az idézőjelekre.
- A `--gpus all` opció biztosítja, hogy a konténer GPU-hozzáférést kapjon; győződj meg róla, hogy a Docker Engine és az NVIDIA Container Toolkit telepítve van a hoszton.
- Futtasd először a `docker build` parancsot, majd a `docker run`-t.

## Fájlszerkezet (aktuális)
- `Dockerfile`: Leírja, hogyan épül és konfigurálódik a projekt konténeres futtatókörnyezete.
- `requirements.txt`: Tartalmazza a projekt futtatásához szükséges Python-csomagok és verziók listáját.
- `README.md`: Áttekintést, futtatási és beadási utasításokat, valamint a projekt struktúráját és eredményeit ismerteti.
- `evaluation_plots/`: Itt tároljuk a kiértékelési vizualizációkat (pl. confusion matrix, metrika-plotok).
- `log/`: A pipeline és kísérletek futtatási naplóit (pl. `run.log`) és kísérleti logokat gyűjti.
- `model_developement/`: Kísérleti futtatások, model-verziók, checkpointok és kapcsolódó naplók tárolóhelye.
- `notebooks/`: Jupyter jegyzetfüzetek az EDA-hoz, elemzésekhez és kísérletek dokumentálásához.
- `src/` — lásd részletes listát lent: a pipeline forrásai (adatletöltés, előfeldolgozás, tréning, HPO, kiértékelés, inference és segédmodulok).
- `data/` (nem verziózott; ide kerülnek a `train/val/test` fájlok): a nyers és feldolgozott bemeneti adatok helye, amelyet a pipeline használ.

### `src/` —  tartalom
- `00_download_data.py` — adatletöltés és kicsomagolás 
- `01_data_preprocessing.py` — annotációk betöltése, tisztítás, konszenzus aggregálás, átlagoás, /val/test CSV előállítás
- `02_train.py` — fő tréning script (modellépítés, training loop, checkpointing)
- `02_train_ray_tune.py` — Ray Tune HPO futtatások és konfiguráció
- `03_1_evaluation_test_set.py` — tesztkészlet kiértékelés (metrikák, riportok)
- `03_2_evaluation_consensus.py` — konszenzusos címkékkel végzett kiértékelés
- `04_inference.py` — inferencia és predikciók mentése 
- `config.py` — projekt-szintű konfigurációk és alapértelmezések
- `model.py` — modell/osztály definíciók (pl. LSTM/attention komponentek)
- `run.sh` — futtatási segédfájl (shell)
- `utils.py` — logger konfiguráció, tokenizálás, collate fn, metrikák és egyéb segédfüggvények


# Függőségek
Az összes Python függőség a `requirements.txt` fájlban található.