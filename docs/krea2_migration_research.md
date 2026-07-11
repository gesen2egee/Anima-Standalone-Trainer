# Krea 2 移植研究

## 結論

Krea 2 訓練可以移植到本 REPO，但不是複製四個入口腳本就能完成。外部 `musubi-tuner` 的 Krea 2 實作依賴一套較新的 dataset、cache 與 `NetworkTrainer` 介面；本 REPO 目前使用 kohya/sd-scripts 風格的 `train_network.py` 與 `library.train_util`。建議採「獨立 Krea 2 架構模組 + 現有訓練框架 adapter」方式，避免把整套 Musubi Tuner 基礎層直接覆蓋進來。

本次先完成共同環境基線：`transformers==4.57.6`，以及獨立 CUDA 12.8 requirements 的 `torch==2.7.1+cu128`、`torchvision==0.22.1+cu128`。Krea 2 程式碼尚未在本階段加入。

## 外部實作拆解

來源：`D:\SDXL\ai-toolkit\tasks\musubi-tuner`，主要檔案如下：

| 類別 | 外部檔案 | 作用 |
|---|---|---|
| 入口 | `krea2_cache_latents.py` | 使用 Qwen-Image VAE 快取 image latent |
| 入口 | `krea2_cache_text_encoder_outputs.py` | 快取 Qwen3-VL-4B 的 selected hidden states |
| 入口 | `krea2_train_network.py` | RAW DiT 的 flow-matching LoRA 訓練 |
| 入口 | `krea2_generate_image.py` | RAW／Turbo 推論與 LoRA 載入 |
| 模型 | `src/musubi_tuner/krea2/krea2_encoder.py` | Qwen3-VL-4B conditioner 與 ComfyUI 權重轉換 |
| 模型 | `src/musubi_tuner/krea2/krea2_mmdit.py` | Krea 2 single-stream MMDiT |
| 演算法 | `src/musubi_tuner/krea2/krea2_sampling.py` | patchify、resolution-aware timestep、Euler sampling |
| 載入 | `src/musubi_tuner/krea2/krea2_utils.py` | DiT／text encoder／LoRA／FP8 載入協調 |
| LoRA | `src/musubi_tuner/networks/lora_krea2.py` | Krea 2 的 Linear target 與權重介面 |

Krea 2 的建議流程是 RAW checkpoint 訓練、Turbo checkpoint 推論。訓練端需要 `--dit`、`--vae`、預先快取的文字輸出與 `networks.lora_krea2`；推論端還需要 Qwen3-VL-4B text encoder。

## 本 REPO 的可重用元件

- `library/qwen_image_autoencoder_kl.py` 已存在 Qwen-Image VAE 實作，可作為 Krea 2 的 VAE 基礎。
- `library/anima_utils.py` 與 `anima_train_network.py` 已有 Qwen 系列文字編碼／快取的部分經驗，但目前 Anima 使用的不是 Krea 2 所需的 Qwen3-VL-4B conditioner。
- `networks/lora.py`、`networks/lora_flux.py` 可提供通用 LoRA 基礎，但 Krea 2 仍需要獨立 target module、命名與 checkpoint adapter。
- `library/train_util.py` 已有 latent／text encoder cache 流程，但 Krea 2 的文字 cache 是「selected layers + varlen valid token」格式，不能直接假設現有固定長度 cache 可相容。

## 主要相容性差異與移植工作

### 1. Transformer 與文字編碼器

外部 Krea 2 使用 `Qwen3VLConfig`、`Qwen3VLForConditionalGeneration` 與 `Qwen2TokenizerFast`。因此 `transformers 4.57.6` 是必要基線；本 REPO 原本的 4.54.1 不應作為 Krea 2 執行環境。

### 2. 訓練框架介面

外部 `krea2_train_network.py` 繼承 Musubi 的 `hv_train_network.NetworkTrainer`，並使用 `ARCHITECTURE_KREA2`、新的 dataset blueprint、block swap 與 model-specific `call_dit`。本 REPO 的 `train_network.NetworkTrainer` 使用另一套 `library.train_util.DatasetGroup` 與模型載入流程，兩者不能直接互換。

### 3. 文字輸出快取

Krea 2 會保存 Qwen3-VL 的多層 hidden-state stack，並壓縮掉 padding token。這是訓練正確性的關鍵，尤其 prompt template 可能形成「有效 token、padding、有效 suffix」的 interior padding；移植時要保留 valid-token compaction，不可只截取 leading prefix。

### 4. DiT 與 LoRA

Krea 2 是 single-stream MMDiT，不是現有 SDXL、FLUX 或 Anima 的既有 U-Net／DiT 類別。外部預設對 DiT 的全部 Linear layers 建立 LoRA（包含 attention、MLP、text fusion 與 projection MLP），因此需要 Krea 2 專用 model class 與 LoRA target adapter。

### 5. 顯存與取樣

外部實作提供 scaled FP8、block swap、RAW／Turbo 權重切換與 Qwen3-VL／VAE 的 CPU shuttle。移植後若要在 24GB 顯卡執行，這些能力要保留；單純把 model 放入現有 `train_network.py` 而不移植 memory hooks，實務上很可能無法完成推論或 sample preview。

## 建議移植順序

1. **環境基線（本次）**：分離 `requirements-cu128.txt`，先安裝 CUDA 12.8 的 PyTorch，再安裝一般 requirements。
2. **純模型核心**：先移植 `krea2/`、`lora_krea2.py` 與必要的 safetensors／attention adapter，完成 CPU 端 config、state-dict key mapping 與純 tensor 單元測試。
3. **快取 adapter**：把 Qwen-Image VAE 接到現有 latent cache，另建 Krea 2 varlen text cache 格式與讀取器。
4. **訓練 adapter**：在現有 `train_network.NetworkTrainer` 增加 Krea 2 model hooks，接入 flow-matching timestep、patch/unpatch 與 Krea 2 LoRA；不要直接改寫既有 SD／Flux／Anima 路徑。
5. **推論與 UI**：先完成命令列 RAW／Turbo smoke test，再把模型選項接到 Training UI。

## 驗證門檻

- 不帶模型權重即可通過：Krea2 timestep、varlen text compaction、patch/unpatch shape、LoRA target enumeration。
- 使用小型本地資料集與 RAW checkpoint 完成一次 cache → one-step forward → LoRA save。
- 使用 Turbo checkpoint 完成固定 seed 推論，並確認 LoRA 套用後輸出不是 RAW/Turbo 權重混用。
- 在目標顯卡上驗證 bf16、scaled FP8 與 block swap 三種顯存路徑。

## 目前風險評估

可行性為「中高」，但工作量集中在訓練框架 adapter，而非安裝依賴。最主要風險是現有 cache metadata 與 Krea 2 varlen hidden-state cache 的資料結構差異，以及現有 LoRA 命名／載入器與 Krea 2 single-stream MMDiT 的 checkpoint key 不一致。
