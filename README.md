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

Our dataset is available at https://huggingface.co/datasets/tinnel123/defacto_dataset

The dataset is organized into numbered subfolders (starting from `1`, `2`, `3`, …).
Each subfolder contains the following files:

* **original.(ext)** → the original input image
* **original\_smask.(ext)** → the image with task-relevant regions *masked out* (counterfactual supervision)
* **original\_rmask.(ext)** → the image with task-irrelevant regions *randomly masked*
* **boxes.txt** → all bounding boxes in the image
* **sboxes.txt** → boxes corresponding to task-relevant regions
* **outside\_boxes.txt** → boxes corresponding to task-irrelevant regions
* **random\_boxes.txt** → boxes of randomly masked regions
* **question.txt** → the question associated with this image
* **answer.txt** → the ground-truth answer

Example structure:

```
dataset/
├── 1/
│   ├── original.png
│   ├── original_smask.png
│   ├── original_rmask.png
│   ├── boxes.txt
│   ├── sboxes.txt
│   ├── outside_boxes.txt
│   ├── random_boxes.txt
│   ├── question.txt
│   └── answer.txt
├── 2/
│   ├── ...
```

To convert JSON-format data into the above dataset structure:

```bash
python DeFacto_train/dataset_maker.py
```

---

## 🚀 Training

```bash
cd DeFacto_train/src/scripts
bash 0917.sh
```

* **`bash 0917.sh`**: launches distributed training using `torchrun`.
  It sets environment variables (CUDA paths, checkpoint paths, dataset path, log path) and runs the training script `grpo_defacto.py` with DeepSpeed support.

---

## 🔎 Batch Inference

```bash
# Start an inference service with vllm first
# Then run the following code to perform inference on JSON data 
# and save it as a new JSON file (multi-threading supported)
python DeFacto_train/inference_vllm.py
```

---

## 🖼️ Single Image Inference (with bbox visualization)

```bash
# Start a service by loading the model
python defacto/app-service.py

# Then run single image inference
python defacto/inference.py
```

---

## 📜 Key Code (Reward Module)

```bash
DeFacto_train/src/virft/src/open_r1/grpo_defacto.py
```
