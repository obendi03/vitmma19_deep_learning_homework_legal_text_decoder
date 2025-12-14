## **Per-LSTM verzió állapota (részletes fejezetek)**

### Hardver / futtatási környezet
A kísérletek GPU-n futottak; a naplók szerint a használt GPU: **NVIDIA GeForce GTX 1050 Ti** (CUDA capability 6.1). A futtatási naplók (`Using device: cuda`) és a GPU-részletek például itt találhatók: [lstm_version_6_attention log](overfit_full_dataset/lstm_version_6_attention/log/attention_increment.log#L170) és [lstm_version_7_data_augmentation log](overfit_full_dataset/lstm_version_7_data_augmentation/log/run_2025-12-13_17-40-12.log#L178).

### Formátum minden verziónál
1) Eredmény (rövid): futás fő metrikái és diagnózis (link a logra).
2) Módosított paraméterek az előző fázishoz képest (`paraméter: érték`) — hivatkozás `config.txt` és `model_description.txt` mappára.
3) Következő futtatásra javasolt módosítandó elemek (egymás alá felsorolva).

Az alábbi per‑verziós összefoglalók a fenti sablont követik.

### `lstm_version_1_first_try`
- Increment: Inc#0 (Baseline)
- Eredmény: Test Acc ≈ 28.99%, Weighted F1 ≈ 0.2195 — erős overfitting (log: model_developement/overfit_full_dataset/lstm_version_1_first_try/log/run_2025-12-13_14-54-11.log).
- Módosított paraméterek (vs korábbi baseline):
    - `BATCH_SIZE`: 32
    - `LEARNING_RATE`: 0.04
    - `LSTM_HIDDEN_DIM`: 8
    - `NUM_LSTM_LAYERS`: 1
    - `SEQ_LEN`: 10
    - `FC_DROPOUT`: 0.1
    - `LSTM_DROPOUT`: 0.1
    (lásd: model_developement/overfit_full_dataset/lstm_version_1_first_try/config.txt és `model_description.txt`)
- Következő futtatásra módosítandó elemek:
    - `LEARNING_RATE`: 0.001
    - `LSTM_HIDDEN_DIM`: 64–128
    - `NUM_LSTM_LAYERS`: 2
    - `SEQ_LEN`: 100–200
    - `FC_DROPOUT`: 0.3–0.4
    - `LSTM_DROPOUT`: 0.2–0.3
    - `EARLY_STOP_PATIENCE`: 40

### `lstm_version_2_better_parameters`
- Increment: Inc#1 (Improvement)
- Eredmény: Test Acc ≈ 41.37%, Weighted F1 ≈ 0.3775 — Inc#1 beállításokkal sikeres, baseline felülmúlva (log: model_developement/overfit_full_dataset/lstm_version_2_better_parameters/log/run_2025-12-12_15-04-08.log).
- Módosított paraméterek (vs `lstm_version_1_first_try`):
    - `BATCH_SIZE`: 16
    - `LEARNING_RATE`: 0.001
    - `LSTM_HIDDEN_DIM`: 128
    - `NUM_LSTM_LAYERS`: 2
    - `SEQ_LEN`: 200
    - `FC_DROPOUT`: 0.4
    - `LSTM_DROPOUT`: 0.3
    (lásd: model_developement/overfit_full_dataset/lstm_version_2_better_parameters/config.txt és `model_description.txt`)
- Következő futtatásra módosítandó elemek:
    - Próbálni enyhe `CLASS_WEIGHTS` alkalmazását vagy `FocalLoss(gamma=2)` tesztelését.
    - Monitor: per-class metrics és confusion matrix; ha összeomlik, visszavenni a súlyokat.

### `lstm_version_3` (class weights próbálkozás)
- Increment: Inc#2a (class-weights attempt — failed)
- Eredmény: predikciós kollapszus egyes futásoknál (automatikus súlyozás túl agresszív) — log: model_developement/overfit_full_dataset/lstm_version_3/log/run_2025-12-13_16-04-44.log.
- Módosított paraméterek (vs `lstm_version_2_better_parameters`):
    - `CLASS_WEIGHTS`: (auto-számolt, erősebb súlyok alkalmazva)
    (lásd: model_developement/overfit_full_dataset/lstm_version_3/config.txt)
- Következő futtatásra módosítandó elemek:
    - Használni "mild" súlyozást (clip/scale), vagy visszatérni Inc#1+FocalLoss alternatívára.

### `lstm_version_4_focal_loss`
- Increment: Inc#2b (Focal Loss — recovery / improvement)
- Eredmény: Focal Loss bevezetése stabilizálta a predikciókat; Test Acc ≈ 40.07%, Weighted F1 ≈ 0.3626 (log: model_developement/overfit_full_dataset/lstm_version_4_focal_loss/log/run_2025-12-13_16-38-06.log).
- Módosított paraméterek (vs `lstm_version_3`):
    - `LOSS`: FocalLoss(alpha=1,gamma=2)
    - Alap hiperparaméterek megtartva (`LEARNING_RATE`:0.001, `LSTM_HIDDEN_DIM`:128, droputs).
    (lásd: model_developement/overfit_full_dataset/lstm_version_4_focal_loss/config.txt)
- Következő futtatásra módosítandó elemek:
    - Finomhangolni `gamma` értékét; ha túl agresszív, csökkenteni.

### `lstm_version_5_bilstm`
- Increment: Inc#3 (BiLSTM attempt — failed)
- Eredmény: BiLSTM kipróbálása összeomláshoz vezetett (predict collapse) — log: model_developement/overfit_full_dataset/lstm_version_5_bilstm/log/run_2025-12-13_17-07-24.log.
- Módosított paraméterek (vs `lstm_version_4_focal_loss`):
    - `BIDIRECTIONAL`: True
    - `LSTM_HIDDEN_DIM`: 64–128 (kétirányúság miatt áttervezve)
    (lásd: model_developement/overfit_full_dataset/lstm_version_5_bilstm/config.txt)
- Következő futtatásra módosítandó elemek:
    - Revert a stabil Inc#1/Inc#2 konfigurációra és teszteljük BiLSTM-et fokozatosan (CrossEntropy először, majd FocalLoss óvatosan).

### `lstm_version_6_attention`
- Increment: Inc#4 (Attention — improvement)
- Eredmény: Attention hozzáadása stabil javulást hozott; Test Acc ≈ 41.37%, Weighted F1 ≈ 0.3542 (log: model_developement/overfit_full_dataset/lstm_version_6_attention/log/attention_increment.log).
- Módosított paraméterek (vs korábbi stabil konfigurációk):
    - `ATTENTION`: True (Linear(hidden_dim,1) attention)
    - `LSTM_HIDDEN_DIM`: 128
    (lásd: model_developement/overfit_full_dataset/lstm_version_6_attention/model_description.txt és `config.txt`)
- Következő futtatásra módosítandó elemek:
    - Add adat augmentáció a train készletre; HPO attention dropout/scale paraméterekre.

### `lstm_version_7_data_augmentation`
- Increment: Inc#5 (Data augmentation — improvement)
- Eredmény: Adat augmentáció javította a generalizációt; Test Acc ≈ 43.65%, Weighted F1 ≈ 0.3853 (log: model_developement/overfit_full_dataset/lstm_version_7_data_augmentation/log/run_2025-12-13_17-40-12.log).
- Módosított paraméterek (vs `lstm_version_6_attention`):
    - `DATA_AUGMENTATION`: True (train set only)
    - Egyéb hiperparaméterek megtartva (`LEARNING_RATE`:0.001, `LSTM_HIDDEN_DIM`:128)
    (lásd: model_developement/overfit_full_dataset/lstm_version_7_data_augmentation/config.txt)
- Következő futtatásra módosítandó elemek:
    - Finomhangolni augmentációs arányt és stratégiát; futtatni HPO-t a legjobb attention+augmentation beállításokon.

### `lstm_version_8_hpo` / `lstm_version_8_best_param_hpo`
- Increment: Inc#6 (HPO — top-trials / final search)
- Eredmény: Ray Tune HPO — top futások tesztpontossága ≈ 42–45%, legjobb ≈ 45.28% (log: model_developement/overfit_full_dataset/lstm_version_8_hpo/log/run_2025-12-13_19-58-17.log).
- Módosított paraméterek (összegzés a top trialokból):
    - `hidden_dim`: 64
    - `lr`: ≈6e-4
    - `bidirectional`: True (néhány trial)
    - `num_layers`: 3
    - `fc_dropout`: ≈0.55
    (lásd: model_developement/overfit_full_dataset/lstm_version_8_hpo/config.txt és top checkpoint mappák)
- Következő futtatásra módosítandó elemek:
    - Exportálni top checkpointokat, futtatni részletes eval-t (QWK, MAE, per-class metrics) és dönteni a végső modellről.

---
Az egyes verziók pontos konfigurációit és a modell felépítését megtalálod a verzió-specifikus mappákban (`model_developement/overfit_full_dataset/<verzió>/config.txt` és `model_description.txt`), valamint a futtatási naplókban (`log/` almappák).
