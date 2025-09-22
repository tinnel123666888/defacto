# DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning

This repository contains the official implementation of **DeFacto**, as described in our paper:
**DeFacto: Counterfactual Thinking with Images for Enforcing Evidence-Grounded and Faithful Reasoning**.

---

## 🛠️ Environment Setup

```bash
conda create -n defacto python=3.10
conda activate defacto
bash setup.sh
```

---

## 📂 Dataset Construction

```bash
# Convert JSON-format data into dataset
python DeFacto_train/dataset_maker.py
```

---

## 🚀 Training

```bash
cd DeFacto_train/src/scripts
bash 0917.sh
```

---

## 🔎 Batch Inference

```bash
# First start an inference service with vllm
# Then run the following code to perform inference on JSON data 
# and save it as a new JSON file (with multi-threading)
python DeFacto_train/inference_vllm.py
```

---

## 🖼️ Single Image Inference (with bbox visualization)

```bash
# First start a service by loading the model
python DeFacto_train/app-service.py

# Then run single image inference
python DeFacto_train/inference.py
```

---

## 📜 Key Code (Reward Module)

```bash
DeFacto_train/src/virft/src/open_r1/defacto_0917.py
```

