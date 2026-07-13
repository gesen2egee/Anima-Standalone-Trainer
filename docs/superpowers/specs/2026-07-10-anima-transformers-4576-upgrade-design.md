# Anima 新版 Python 訓練環境升級設計

## 目標

將 Anima 主環境的 Python 訓練依賴逐項比較 Anima 與 Musubi Tuner，採其中較新的版本，作為 Krea 2 整合的共同基線；本階段假設 API 可向後相容，仍以實際安裝與 smoke test 驗證。

## 範圍與非目標

- 比較兩 repo 的版本宣告後，依較新值更新根目錄 `requirements.txt` 中共同的 Python 訓練依賴。
- 先在新建的 CUDA 12.8 staging venv 套用變更，確認完整 Python 依賴解析與 Anima 訓練核心可匯入；通過後才切換既有 `venv`。
- 執行既有 UI 單元測試與 Python 訓練核心匯入 smoke test。
- Node UI 套件獨立評估，不會在沒有必要時與 Python 訓練相依混為一個升級單位。
- 不在這個變更內加入 Krea 2 訓練腳本或 UI。

## 比較來源與判定規則

版本比較只讀取下列工作區檔案，不查詢或採用網路上的最新發行版：

- Anima：`E:\workspace\Anima-Standalone-Trainer\requirements.txt`
- Musubi Tuner：`D:\SDXL\ai-toolkit\tasks\musubi-tuner\pyproject.toml`，目前本機 checkout

比較只涵蓋直接 Python 訓練依賴。名稱以 Python 套件正規化後比較（不分大小寫、`_` 與 `-` 等同）；extras 不是另一套件，例如 Anima 的 `diffusers[torch]` 保留其 extra。兩邊均有精確版本時取較大值；一邊是最低版本範圍時，採該最低版本並在 requirements 鎖成精確值；一邊未宣告時不新增或改動。Git URL、未固定版本、environment marker 及傳遞依賴不是兩 repo 可比較的版本，維持 Anima 原設定。

PyTorch 是環境基線而非「共同直接依賴」：Anima requirements 未列出它，但目前 Anima venv 是 CUDA 12.8 wheel；Musubi 為 CUDA 12.8 宣告 `torch>=2.7.1`、`torchvision>=0.22.1`。依使用者的完整升級要求，staging venv 將把 CUDA 12.8 基線鎖為 `torch==2.7.1+cu128` 與 `torchvision==0.22.1+cu128`。它們不寫進 `requirements.txt`，而放在獨立 `requirements-cu128.txt`，並由該檔專用 PyTorch CU128 index 安裝；一般 PyPI requirements 不受該 index 影響。

`requirements-cu128.txt` 的完整內容固定為：

```text
--index-url https://download.pytorch.org/whl/cu128
torch==2.7.1+cu128
torchvision==0.22.1+cu128
```

它必須獨立先安裝；後續安裝 `requirements.txt` 時不帶 PyTorch index，仍使用預設 PyPI。

| 正規化套件 | Anima 宣告 | Musubi 宣告 | 選定設定 | 判定 |
|---|---|---|---|---|
| accelerate | 1.6.0 | 1.6.0 | `accelerate==1.6.0` | 相同 |
| transformers | 4.54.1 | 4.57.6 | `transformers==4.57.6` | Musubi 較新 |
| diffusers | 0.32.1（含 `torch` extra） | 0.32.1 | `diffusers[torch]==0.32.1` | 相同，保留 Anima extra |
| opencv-python | 4.10.0.84 | 4.10.0.84 | `opencv-python==4.10.0.84` | 相同 |
| einops | 0.7.0 | 0.7.0 | `einops==0.7.0` | 相同 |
| safetensors | 0.4.5 | 0.4.5 | `safetensors==0.4.5` | 相同 |
| toml | 0.10.2 | 0.10.2 | `toml==0.10.2` | 相同 |
| voluptuous | 0.15.2 | 0.15.2 | `voluptuous==0.15.2` | 相同 |
| huggingface-hub | 0.34.3 | 0.34.3 | `huggingface-hub==0.34.3` | 相同 |
| ftfy | 6.3.1 | 6.3.1 | `ftfy==6.3.1` | 相同 |
| sentencepiece | 0.2.1 | 0.2.1 | `sentencepiece==0.2.1` | 相同 |

`av`、`pillow`、`tqdm`、`easydict` 只由 Musubi 宣告；Anima 的 ONNX、標註、optimizer、TensorBoard、VCS dependencies 與未釘版本只由 Anima 宣告。因此本階段一律不新增或變更它們。

## 設計決策

採直接升級主環境的兩 repo 版本比較方案。以 Anima `requirements.txt` 與 Musubi Tuner `pyproject.toml` 為唯一版本來源：共同依賴逐項取較新版本；只存在 Anima 的專用最佳化器、ONNX 與標註套件維持原設定；只存在 Musubi 的套件不會在未整合其功能前加入 Anima。Musubi Tuner 的 Krea 2 版本需求會成為共同依賴的較新值。

升級前以既有 venv 的同一支 Python 執行 `python -m pip freeze --all > tasks/upgrade/requirements-before-freeze.txt`，並以 UTF-8 記錄 Python、CUDA、torch、torchvision 與 NVIDIA driver 資訊。升級後輸出同格式的 `requirements-after-freeze.txt`；這些暫存紀錄位於已忽略的 `tasks/upgrade/`。升級只允許 Python 3.10–3.12，且必須驗證既有 NVIDIA driver 能執行 CU128 wheel。

升級以與現有 venv 相同的 base Python 建立 `venv-upgrade`。先以 `venv-upgrade\\Scripts\\python.exe -m pip install -r requirements-cu128.txt` 安裝 CU128 Torch，再以同一 Python 執行 `-m pip install -r requirements.txt`。接著用 staging Python 執行 `python -m pip check`、採集版本，確認 Anima 的 Qwen3 模型／訓練工具可匯入，最後執行 Python 與 UI 測試。若驗證失敗，刪除 `venv-upgrade`，既有 `venv` 完全不受影響。

通過 staging 驗證後，先停止 UI，並記錄現有 venv 的 `sys.base_prefix`。將既有 `venv` 重新命名為帶時間戳的 `venv-before-upgrade-*`，然後用已記錄的 base Python 在原本的 `venv` 路徑重新執行 `-m venv venv`，依相同順序安裝 `requirements-cu128.txt` 與 `requirements.txt`，並用 canonical `venv\\Scripts\\python.exe` 重跑全部驗證。不可將 `venv-upgrade` 直接重新命名，因為 Windows venv 的 console scripts 含建立時的絕對路徑。若 canonical venv 驗證失敗，刪除新 `venv` 後將備份資料夾改回 `venv`，即可完整回退。

## 驗收條件

1. `requirements.txt` 只依上方矩陣變更共同依賴，未改動單邊專用套件；新建 `requirements-cu128.txt` 只鎖 CUDA 12.8 的 Torch 基線。
2. staging venv 的 Python 為 3.10–3.12、CUDA wheel 為 CU128，且實際安裝版本符合矩陣與 `pip check` 無衝突。
3. `library.anima_models`、`library.anima_train_utils` 與 `anima_train_network` 能以 UTF-8 Python 環境匯入。
4. `node --test training-ui/lib/*.test.js` 及 `node --check training-ui/server.js`、`node --check training-ui/public/js/app.js` 通過。
5. 先以 `venv-upgrade\\Scripts\\python.exe -c "import torch, torchvision, transformers; import library.anima_models, library.anima_train_utils, anima_train_network; print(torch.__version__, torchvision.__version__, transformers.__version__)"` 驗證 staging 核心匯入與版本；切換後以 canonical `venv\\Scripts\\python.exe` 重跑同一命令。
6. 以 staging 與 canonical venv 分別執行 `node --test training-ui/lib/*.test.js`、`node --check training-ui/server.js`、`node --check training-ui/public/js/app.js` 與 `git diff --check`，均不得有錯誤；既有的 `library/anima_utils.py` 未提交修改不被納入本次提交。
