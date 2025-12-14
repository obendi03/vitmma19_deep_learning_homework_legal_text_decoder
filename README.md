## Deep Learning tantárgy (VITMMA19) — Projekt

**Projektinformáció**
- **Kiválasztott téma:** Jogszöveg dekódoló
- **Hallgató:** Oláh Bendegúz István
- **Aiming for +1 mark:** Nem

**Rövid megoldásleírás**
Moduláris, reprodukálható mélytanulási pipeline jogi szövegek érthetőségének előrejelzésére. Főbb komponensek: automatizált adatok letöltése/előfeldolgozása (`src/00_download_data.py`, `src/01_data_preprocessing.py`), baseline TF-IDF+LR, LSTM alapú modellek (BiLSTM, attention), HPO Ray Tune-nal, és evaluációs/kiértékelő scriptek (`src/03_1_evaluation_test_set.py`, `src/03_2_evaluation_consensus.py`).

**Kísérleti beállítások (összefoglaló)**
- Eszköz: GPU (naplókban `Using device: cuda` szerepel)
- Baseline: TF-IDF + LogisticRegression
- Fő modell: MultiLayerLSTM (+opcionális attention)
- Fő metrikák: accuracy, weighted F1, confusion matrix

**Adatelőkészítés**
- Automatikus: futtasd a `src/00_download_data.py`-t (ha elérhető források) és aztán `src/01_data_preprocessing.py`-t. Ezek előállítják a feldolgozott CSV fájlokat a `data/` mappába (konténerben: `/app/data`).
- Manuális: töltsd le a SharePoint ZIP-et, nevezd át `downloaded.zip`-re, csomagold ki és helyezd a `data/` mappába.
- A `src/01_data_preprocessing.py` fájl tartalmazza a train/val/test felosztás logikáját és a konszenzus kezelést.

**Naplózási követelmények**
Minden futtatásnál a log fájlnak (`log/run.log`) tartalmaznia kell:
- Konfiguráció: használt hiperparaméterek (pl. epoch, batch size, lr)
- Adatfeldolgozás: sikeres betöltés és előfeldolgozás visszaigazolása
- Modell architektúra: modell-összegzés, paraméterszám
- Tanulási előrehaladás: epoch-onkénti metrikák (loss/accuracy vagy feladatra jellemző metrikák)
- Validáció: epochonkénti validation metrikák
- Végső kiértékelés: teszt eredmények (accuracy, F1, confusion matrix)

Megjegyzés: a projekt `src/utils.py`-ja kezeli a logger konfigurációját; a logger stdout-ra ír, így a Docker futtatás esetén a kimenet a `log/run.log`-ba irányítható.

**Futtatási lépések (lokálisan / konténerben)**
1. Ellenőrizd, hogy a `data/` mappa tartalmazza a feldolgozott adatokat (pl. `train.csv`, `val.csv`, `test.csv`).
2. Lokálisan: pip install -r requirements.txt, majd `python src/02_train.py` (vagy használj Ray Tune futtatást a HPO-hoz).
3. Konténerben (példa Windows `cmd.exe`):
```bat
docker build -t dl-project .
docker run --gpus all --rm -v "%cd%\log:/app/log" -v "%cd%\data:/app/data" dl-project > log/run.log 2>&1
```

**Fájlstruktúra (gyors összegzés)**
- `Dockerfile` — konténerépítés
- `requirements.txt` — Python függőségek
- `src/` — pipeline forrásai: `00_download_data.py`, `01_data_preprocessing.py`, `02_train.py`, `02_train_ray_tune.py`, `03_1_evaluation_test_set.py`, `03_2_evaluation_consensus.py`, `04_inference.py`, `config.py`, `utils.py`, `model.py`
- `model_developement/` — kísérleti futtatások és naplók
- `evaluation_plots/` — vizuális kimenetek (pl. confusion matrix)
- `log/` — futtatási naplók (pl. `run.log`)

**Beadáshoz ellenőrzés (Submission checklist)**
- Project Information: kitöltve (téma, név): **OK**
- Solution Description: megtalálható a README-ben: **OK**
- Data Preparation: `src/01_data_preprocessing.py` és `src/00_download_data.py` megvannak: **OK**
- Dependencies: `requirements.txt` létezik: **OK**
- Configuration: `src/config.py` tartalmaz konfigurációs változókat (pl. `NUM_EPOCHS`): **OK**
- Logging: `src/utils.py` létezik, `log/run.log` jelenleg a repo-ban található: **OK**
- Dockerfile: létezik: **OK**
- Scripts: `src/02_train.py`, `src/03_1_evaluation_test_set.py`, `src/04_inference.py` megtalálhatók: **OK**

**Eltérések / Hiányosságok (Eltérések fejezet)**
- `data/` mappa: jelenleg nincs a repository gyökérében. A pipeline elvárja a `data/`-t a feldolgozott `train/val/test` fájlokkal. (Teendő: vagy futtasd `src/00_download_data.py`/`src/01_data_preprocessing.py` helyben, vagy töltsd fel a `data/` mappát és mount-oljad a konténerhez.)
- Docker image build/run státusz nincs automatikusan ellenőrizve itt — javasolt helyben futtatni `docker build` és `docker run` parancsokat, és ellenőrizni, hogy a konténer ténylegesen lefut-e teljes pipeline-nal.

Ha szeretnéd, elvégzem a következőket helyetted:
- Futtatok egy gyors ellenőrzést, hogy `docker build` lefut-e (ha engeded és rendelkezel Dockerrel a gépen).
- Létrehozom a hiányzó `data/` mappa helyőrző fájlokat, vagy generálok egy rövid példa `train.csv`-t a pipeline kipróbálásához.

---

## Deep Learning tantárgy (VITMMA19) — Projekt

# Projektinformáció
Kiválasztott téma: Jogszöveg dekódoló

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


Megjegyzés a manuális letöltésről: ha az automatikus letöltés nem működik vagy a link hitelesítést igényel, használd a következő, egyszerű manuális eljárást a *Legal Text Decoder* adatkészlethez:

- Nyisd meg a SharePoint linket (a projekthez tartozó `Legal Text Decoder` megosztást) a böngésződben: másold be a linket a címsorba és nyisd meg.
- Töltsd le a ZIP fájlt a böngészővel.
- Nevezd át a letöltött fájlt `downloaded.zip`-re.
- Helyezd a `downloaded.zip` fájlt a projekt `data/` mappájába (a konténerben ez a `/app/data`).



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

## **Per-LSTM verzió állapota (részletes fejezeteka modell fejlesztésről)**

Megjegyzés: a részletes per-LSTM verziók leírásai külön fájlban találhatók. Kérlek, másold át vagy szerkeszd a részleteket a következő fájlban: [model_developement/PER-LSTM_versions.md](model_developement/PER-LSTM_versions.md).

## Adatelőkészítés

A pipeline alapértelmezett működése: a konténer indításakor a `src/00_download_data.py` script próbálja automatikusan letölteni és kicsomagolni a szükséges nyers adatkészleteket a `data/` mappába (konténer belső útvonala: `/app/data`). Amennyiben Dockerrel futtatod, csatold a helyi `data` mappát a konténerhez a `-v "%cd%\data:/app/data"` opcióval a `docker run` parancsban.

Ha az automatikus letöltés nem sikerül (például SharePoint hitelesítés vagy hálózati korlátozás miatt), kövesd a manuális eljárást:

- Töltsd le a ZIP fájlt a SharePoint megosztásról a böngészővel.
- Nevezd át a letöltött fájlt `downloaded.zip`-re.
- Helyezd a `downloaded.zip` fájlt a repository `data/` mappájába (konténerben: `/app/data`).



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
A repository tartalmaz egy `Dockerfile`-t a reprodukálható környezethez. Használd az alábbi parancsokat (Windows példák a futtatáshoz):

Build (futtasd a projekt gyökérkönyvtárából):
```bat
docker build -t lstm_gpu_pipeline_full .
```

Run (Windows `cmd.exe` példa, a parancs a helyi `log` és `data` mappákat csatolja a konténerhez és a kimenetet a `log/run.log`-ba menti):
```bat
docker run --gpus all --rm -v "%cd%\log:/app/log" -v "%cd%\data:/app/data" lstm_gpu_pipeline_full > log/run.log 2>&1
```

Megjegyzések:
- A fenti `docker run` parancs Windows `cmd.exe` szintaxisra van ( `%cd%` és backslash). PowerShell használata esetén cseréld `%cd%`-t `${PWD}`-re vagy `$(Get-Location)`-ra, és ügyelj az idézőjelekre.
- A `--gpus all` opció biztosítja, hogy a konténer GPU-hozzáférést kapjon; győződj meg róla, hogy a Docker Engine és az NVIDIA Container Toolkit telepítve van a hoszton.
- Futtasd először a `docker build` parancsot, majd a `docker run`-t.

## Fájlszerkezet (összefoglaló)

- `Dockerfile`: a projekt konténerizálásához és futtatási környezet építéséhez használatos (része a reprodukálható pipeline-nak).
- `requirements.txt`: a Python függőségek listája, szükséges a környezet telepítéséhez (része a környezet előkészítésének).
- `README.md`: dokumentáció és használati utasítások — nem futtatható komponens, de fontos a reprodukálhatósághoz.
- `src/`: a pipeline magja — letöltés, adatelőkészítés, tanítás, kiértékelés és inference scriptjei (`00_download_data.py`, `01_data_preprocessing.py`, `02_train.py`, `03_evaluation.py`, `04_inference.py`, `config.py`, `utils.py`); ez a futtatható pipeline.
- `src/`: a pipeline magja — letöltés, adatelőkészítés, tanítás, kiértékelés és inference scriptjei (`00_download_data.py`, `01_data_preprocessing.py`, `02_train.py`, `03_evaluation.py`, `04_inference.py`, `config.py`, `utils.py`); ez a futtatható pipeline.

### `src/` tartalom (rövid felsorolás)

- `00_download_data.py`: letölti vagy előkészíti a nyers forrásfájlokat (ha szükséges) — Pipeline: opcionális elindító lépés.

- `01_data_preprocessing.py`: beolvassa az annotációkat, tisztít, aggregál (consensus), és menti a `train.csv`/`val.csv`/`test.csv` fájlokat — Pipeline: kötelező előfeldolgozás.
	- Fő lépések és kapcsolódó kódrészletek:
		- Annotációk betöltése és feldolgozása: [src/01_data_preprocessing.py#L9-L30] (függvények: `load_annotations`, `load_annotations_from_folder`).
		- Konszenzus kezelése, szűrés és export (`inference.csv`): [src/01_data_preprocessing.py#L94-L119].
		- Train/Val/Test split rögzített seed-del és CSV mentése: [src/01_data_preprocessing.py#L119] / [src/01_data_preprocessing.py#L123-L124].

- `02_train.py`: a fő tréning script; betölti a feldolgozott adatokat, felépíti a modellt, futtatja az epoch-okat és menti a checkpoint-okat — Pipeline: core (központi).
	- Fő lépések és kapcsolódó kódrészletek:
		- Tokenizálás és szekvencia előkészítés (pad/trim, `SEQ_LEN`, `tokens_to_indices`): [src/02_train.py#L47-L56].
		- Word2Vec embedding tanítás és betöltés (HPO esetén is hívva): [src/02_train.py#L47], [src/02_train_ray_tune.py#L51].
		- A train script bemutatja a `pd.read_csv`-t, `Dataset` / `DataLoader` használatot és batching-et: lásd a fájl elejét és a `model_developement/overfit_full_dataset/02_train.py` példáit.

- `03_evaluation.py` / `03_1_evaluation_test_set.py`: kiértékeli a mentett checkpoint-okat a teszt- és konszenzus-készleteken, generál metrikákat és riportokat — Pipeline: kötelező utófeldolgozás.

- `04_inference.py`: betölti a végső checkpoint-ot és predikciókat készít új (címkézetlen) adatokra — Pipeline: opcionális (deployment/inference lépés).

- `config.py`: projekt-szintű alapértelmezett konfigurációk (útvonalak, hyperparaméterek, seed) — Pipeline: szükséges referencia.

- `utils.py`: segédfüggvények (tokenizálás, `Dataset`/`DataLoader` collate, metrikák, logger) — Pipeline: támogató komponens.

- `logs/`: futtatási naplók gyűjtőhelye (helyi kimenet és HPO trial naplók) — Pipeline: kimenet/diagnosztika.

- `model_developement/`: modellek, segédfüggvények és kísérleti futtatások forrásai (iteratív fejlesztés és per-experiment scriptek) — a fejlesztési/eksperimentációs része a munkafolyamatnak.

- `data/`: a nyers és feldolgozott adatok tára (nem verziózott) — bemenete a pipeline-nak, de maga nem kód.

- `log/`: a futtatási naplók helye (`run.log` és kísérleti `log/` almappák) — a pipeline kimenete és diagnosztikai forrása.

# Függőségek
Az összes Python függőség a `requirements.txt` fájlban található.