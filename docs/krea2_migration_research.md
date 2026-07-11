# Krea 2 移植研究

## 結論

Krea 2 訓練可以移植到本 REPO，但不是複製四個入口腳本就能完成。外部 `musubi-tuner` 的 Krea 2 實作依賴一套較新的 dataset、cache 與 `NetworkTrainer` 介面；本 REPO 目前使用 kohya/sd-scripts 風格的 `train_network.py` 與 `library.train_util`。建議採「獨立 Krea 2 架構模組 + 現有訓練框架 adapter」方式，避免把整套 Musubi Tuner 基礎層直接覆蓋進來。

本次已完成共同環境基線與第一版共用管道：`transformers==4.57.6`，以及獨立 CUDA 12.8 requirements 的 `torch==2.7.1+cu128`、`torchvision==0.22.1+cu128`。

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

1. **環境基線（已完成）**：分離 `requirements-cu128.txt`，先安裝 CUDA 12.8 的 PyTorch，再安裝一般 requirements。
2. **純模型核心（已完成）**：移植 `library/krea2/`、`networks/lora_krea2.py` 與必要的 attention adapter。
3. **快取 adapter（已完成）**：Qwen-Image VAE 接入既有 latent cache，另建 Krea 2 varlen text cache 格式。
4. **訓練 adapter（已完成第一版）**：以 `Krea2NetworkTrainer` 接入 flow-matching timestep、patch/unpatch、FP8、block swap 與 Krea 2 LoRA。
5. **推論與 UI（已完成第一版）**：新增 `krea2_generate_image.py`，並接入 Training UI architecture registry 與 Krea 2 LoRA 選項。

## 驗證門檻

- 不帶模型權重即可通過：Krea2 timestep、varlen text compaction、patch/unpatch shape、LoRA target enumeration。
- 使用小型本地資料集與 RAW checkpoint 完成一次 cache → one-step forward → LoRA save。
- 使用 Turbo checkpoint 完成固定 seed 推論，並確認 LoRA 套用後輸出不是 RAW/Turbo 權重混用。
- 在目標顯卡上驗證 bf16、scaled FP8 與 block swap 三種顯存路徑。

## 目前風險評估

可行性為「中高」，但工作量集中在訓練框架 adapter，而非安裝依賴。最主要風險是現有 cache metadata 與 Krea 2 varlen hidden-state cache 的資料結構差異，以及現有 LoRA 命名／載入器與 Krea 2 single-stream MMDiT 的 checkpoint key 不一致。

## 已實作的共用訓練入口

目前已新增 `krea2_train_network.py`。它直接繼承本 REPO 的 `train_network.NetworkTrainer`，因此沿用原本的 dataset config、latent cache、text cache、optimizer、Accelerate、LoRA 儲存與 checkpoint 流程；Anima 的 `anima_train_network.py` 未被替換。

Krea 2 的一般訓練建議先建立 text cache；若啟用 caption augmentation，adapter 會改用 dynamic text encoding。一般快取訓練範例：

```powershell
accelerate launch --mixed_precision bf16 krea2_train_network.py `
  --dit path/to/raw.safetensors `
  --vae path/to/qwen_image_vae.safetensors `
  --text_encoder path/to/qwen3vl_4b_bf16.safetensors `
  --dataset_config path/to/dataset.toml `
  --cache_latents --cache_latents_to_disk `
  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk `
  --timestep_sampling krea2_shift `
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 `
  --optimizer_type AdamW8bit --learning_rate 1e-4 `
  --gradient_checkpointing --max_train_epochs 16 `
  --output_dir path/to/output --output_name krea2_lora
```

若顯存不足，可加上 `--fp8_scaled` 與 `--blocks_to_swap`。訓練中的 sample preview 仍保留為後續工作；目前可用 Training UI 的生成按鈕，或直接執行 `krea2_generate_image.py` 進行固定 seed 推論。Krea2 設定使用獨立的 `krea2_arguments`，不會改寫既有 Anima 的 `anima_arguments`。

### AIT 類似的低顯存設定

Training UI 的 Krea 2 選項現在包含：

- `Transformer DType = FP8 Scaled（Krea 2）`：會自動同時傳入 `fp8_base` 與 `fp8_scaled`。Krea 2 目前只實作 Scaled FP8，未提供會造成誤用的 plain FP8 選項。
- `Krea 2 訓練時即時編碼 Text Encoder（不使用 Cache）`：設定 `--krea2_dynamic_text_encoder`，直接在每個 batch 產生 Qwen3-VL conditioning，不需要建立 TE cache。
- `將 Text Encoder 保留在 CPU`：再加上 `--krea2_dynamic_text_encoder_cpu`。TE 權重會留在 CPU RAM，訓練時只把輸入／輸出傳輸到流程中，速度較慢但可降低 VRAM。

AIT 的 `layer_offloading_transformer_percent: 0.7` 可用 28 層 Krea 2 DiT 的 `--blocks_to_swap 20` 近似（約 71.4%）；12GB VRAM 建議從 batch size 1、gradient checkpointing、`--fp8_scaled --blocks_to_swap 20` 開始，再依實際 OOM 調高至 22–24。`blocks_to_swap` 不可與 `--cpu_offload_checkpointing` 同時使用。

CLI 範例：

```powershell
accelerate launch --mixed_precision bf16 krea2_train_network.py `
  --dit path/to/raw.safetensors `
  --vae path/to/qwen_image_vae.safetensors `
  --text_encoder path/to/qwen3vl_4b_bf16.safetensors `
  --krea2_dynamic_text_encoder --krea2_dynamic_text_encoder_cpu `
  --fp8_base --fp8_scaled --blocks_to_swap 20
```

## Anima 功能在 Krea2 的支援狀態

- `lora_anima`、`LoKR`、`CDKA`、`KRONA` 現在都能透過 Krea2 的 `SingleStreamDiT` Linear target 建立 adapter；`lora_anima` 會轉接到標準 Krea2 LoRA，其他三種使用共用 LyCORIS architecture adapter。
- `Model Guidance`、`CFG-Zero`、`CIOP`、`differential_guidance_scale` 與 Anima weighting 已接入 Krea2 flow-matching forward。
- `sigma`、`uniform`、`sigmoid`、`shift`、`autoshift`、`autoshift_wavelet`、`flux_shift`、`plora`、`krea2_shift` timestep sampling 均可使用；`autoshift*` 需要 alpha mask。
- Caption prefix/suffix、wildcard、caption dropout、caption shuffle、token warmup、caption tag dropout 與 FAD 會自動切換為 dynamic Qwen3-VL encoding，不使用固定 text cache；可加 `--krea2_dynamic_text_encoder_cpu` 降低顯存但會變慢。
- `masked_loss` 與 alpha-mask loss 可使用；Krea2 不支援 Anima 的 `train_inpainting` 輸入格式。

啟用 caption augmentation 時，請預期 Qwen3-VL 會與 DiT 同時佔用顯存；若顯存不足，使用 dynamic CPU text encoder。若要使用 CDKA／KRONA 推論合併 LoRA，請不要在生成命令啟用 `--fp8_scaled`，因為這兩種 adapter 需要先以完整權重合併。
