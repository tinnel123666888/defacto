

## 🛠️ Environment Setup

```bash
conda create -n grpo python=3.10
conda activate grpo
bash setup.sh
```

## Dataset Construction

```bash
Convert JSON-format data into dataset
python Grpo_train/dataset_maker.py
```

## Training

```bash
cd Grpo_train/src/scripts
bash 0917.sh
```

## Batch Inference

```bash
First start an inference service with vllm
Then run the following code to perform inference on JSON data 
and save it as a new JSON file (with multi-threading)
python Grpo_train/inference_vllm.py
```

## Single Image Inference (with bbox visualization)

```bash
First start a service by loading the model
python Grpo_train/app-service.py
Then run single image inference
python Grpo_train/inference.py
```

## Key Code (Reward)

```bash
Grpo_train/src/virft/src/open_r1/grpo_0917.py
```


