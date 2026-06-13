// === Anima Training UI — Client ===
const DEFAULT_NEGATIVE_PROMPT =
  "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, low quality, worst quality, blurry, bad anatomy, extra limbs, deformed, watermark, text, signature, bareness, artifacts, hands, copyrights name, jpeg_artifacts, scan_artifacts, bad hands, missing fingers, extra digit, fewer digits, artistic error, ye-pop, deviantart, logo, patreon logo";
let currentJob = null;
let ws = null;
let isDirty = false;
let lastSavedConfig = null;
let lastSavedDataset = null;
let lastSavedPrompts = [];
let lastSavedNegativePrompt = "";
let samplesPollTimer = null;
let isDraggingBg = false;
let bgPosPercent = { x: 50, y: 50 };
let currentSubsets = [];
let archRegistry = null; // Loaded from /api/architectures
// --- DOM Refs ---
const $ = (id) => document.getElementById(id);
const jobListEl = $("job-list");
const emptyState = $("empty-state");
const jobEditor = $("job-editor");
const jobTitle = $("job-title");
const consoleOutput = $("console-output");

// ==========================================
//  Localization
// ==========================================
const I18N_STORAGE_KEY = "ui_language";
const SUPPORTED_LANGUAGES = ["zh-TW", "zh-CN", "en"];
const originalTextNodes = new WeakMap();
const originalAttrs = new WeakMap();

const UI_TRANSLATIONS = {
  "🎯 Jobs": { "zh-TW": "🎯 工作", "zh-CN": "🎯 任务" },
  "+ New": { "zh-TW": "+ 新增", "zh-CN": "+ 新建" },
  "Language": { "zh-TW": "語言", "zh-CN": "语言" },
  "⚙️ Global Settings": { "zh-TW": "⚙️ 全域設定", "zh-CN": "⚙️ 全局设置" },
  "No Job Selected": { "zh-TW": "尚未選擇工作", "zh-CN": "尚未选择任务" },
  "Create a new training job or select one from the sidebar": {
    "zh-TW": "建立新的訓練工作，或從側邊欄選擇一個工作",
    "zh-CN": "创建新的训练任务，或从侧边栏选择一个任务",
  },
  "Job Name": { "zh-TW": "工作名稱", "zh-CN": "任务名称" },
  "💾 Save": { "zh-TW": "💾 儲存", "zh-CN": "💾 保存" },
  "Discard": { "zh-TW": "放棄變更", "zh-CN": "放弃更改" },
  "Clone": { "zh-TW": "複製", "zh-CN": "克隆" },
  "▶ Train": { "zh-TW": "▶ 訓練", "zh-CN": "▶ 训练" },
  "⏹ Stop": { "zh-TW": "⏹ 停止", "zh-CN": "⏹ 停止" },
  "Delete": { "zh-TW": "刪除", "zh-CN": "删除" },
  "Training": { "zh-TW": "訓練", "zh-CN": "训练" },
  "Dataset": { "zh-TW": "資料集", "zh-CN": "数据集" },
  "Network": { "zh-TW": "網路", "zh-CN": "网络" },
  "Multi-GPUs": { "zh-TW": "多 GPU", "zh-CN": "多 GPU" },
  "Prompts": { "zh-TW": "提示詞", "zh-CN": "提示词" },
  "Samples": { "zh-TW": "樣本", "zh-CN": "样本" },
  "Console": { "zh-TW": "主控台", "zh-CN": "控制台" },
  "📊 TensorBoard": { "zh-TW": "📊 TensorBoard", "zh-CN": "📊 TensorBoard" },
  "Optimization": { "zh-TW": "最佳化", "zh-CN": "优化" },
  "Learning Rate": { "zh-TW": "學習率", "zh-CN": "学习率" },
  "Text Encoder LR": { "zh-TW": "Text Encoder 學習率", "zh-CN": "Text Encoder 学习率" },
  "Optimizer": { "zh-TW": "優化器", "zh-CN": "优化器" },
  "LR Scheduler": { "zh-TW": "學習率排程", "zh-CN": "学习率调度" },
  "LR Warmup Steps": { "zh-TW": "學習率預熱步數", "zh-CN": "学习率预热步数" },
  "Weight Decay": { "zh-TW": "Weight Decay", "zh-CN": "Weight Decay" },
  "Seed": { "zh-TW": "Seed", "zh-CN": "Seed" },
  "Duration": { "zh-TW": "訓練長度", "zh-CN": "训练长度" },
  "Max Epochs": { "zh-TW": "最大訓練輪數", "zh-CN": "最大训练轮数" },
  "Save Every N Epochs": { "zh-TW": "每 N 輪儲存", "zh-CN": "每 N 轮保存" },
  "Max Steps": { "zh-TW": "最大步數", "zh-CN": "最大步数" },
  "Save Every N Steps": { "zh-TW": "每 N 步儲存", "zh-CN": "每 N 步保存" },
  "Checkpoint Management": { "zh-TW": "Checkpoint 管理", "zh-CN": "Checkpoint 管理" },
  "Save Last N Steps": { "zh-TW": "保留最近 N 步", "zh-CN": "保留最近 N 步" },
  "Save Last N Epochs": { "zh-TW": "保留最近 N 輪", "zh-CN": "保留最近 N 轮" },
  "Save Training State": { "zh-TW": "儲存訓練 State", "zh-CN": "保存训练 State" },
  "Save State at Train End": { "zh-TW": "訓練結束儲存 State", "zh-CN": "训练结束保存 State" },
  "Save Last N Step States": { "zh-TW": "保留最近 N Step States", "zh-CN": "保留最近 N Step States" },
  "Save Last N Epoch States": { "zh-TW": "保留最近 N Epoch States", "zh-CN": "保留最近 N Epoch States" },
  "blank = keep all": { "zh-TW": "空白 = 全部保留", "zh-CN": "空白 = 全部保留" },
  "blank = use checkpoint count": { "zh-TW": "空白 = 使用 checkpoint 保留數", "zh-CN": "空白 = 使用 checkpoint 保留数" },
  "Output Name": { "zh-TW": "輸出名稱", "zh-CN": "输出名称" },
  "Save Format": { "zh-TW": "儲存格式", "zh-CN": "保存格式" },
  "Performance": { "zh-TW": "效能", "zh-CN": "性能" },
  "Mixed Precision": { "zh-TW": "混合精度", "zh-CN": "混合精度" },
  "Transformer DType": { "zh-TW": "Transformer DType", "zh-CN": "Transformer DType" },
  "Save Precision": { "zh-TW": "儲存精度", "zh-CN": "保存精度" },
  "DataLoader Workers": { "zh-TW": "DataLoader Workers", "zh-CN": "DataLoader Workers" },
  "KNN Noise K": { "zh-TW": "KNN Noise K", "zh-CN": "KNN Noise K" },
  "CEP Noise": { "zh-TW": "CEP Noise", "zh-CN": "CEP Noise" },
  "Persistent DataLoader Workers": { "zh-TW": "持續保留 DataLoader Workers", "zh-CN": "持续保留 DataLoader Workers" },
  "Gradient Checkpointing": { "zh-TW": "Gradient Checkpointing", "zh-CN": "Gradient Checkpointing" },
  "Flash Attention": { "zh-TW": "Flash Attention", "zh-CN": "Flash Attention" },
  "Torch Compile": { "zh-TW": "Torch Compile", "zh-CN": "Torch Compile" },
  "Low RAM Optimization": { "zh-TW": "低 RAM 最佳化", "zh-CN": "低 RAM 优化" },
  "Blocks to Swap": { "zh-TW": "要 Swap 的 Blocks", "zh-CN": "要 Swap 的 Blocks" },
  "Activation Offload": { "zh-TW": "Activation Offload", "zh-CN": "Activation Offload" },
  "Caching": { "zh-TW": "快取", "zh-CN": "缓存" },
  "Cache Latents to Disk": { "zh-TW": "快取 Latents 到硬碟", "zh-CN": "缓存 Latents 到硬盘" },
  "VAE Batch Size": { "zh-TW": "VAE Batch Size", "zh-CN": "VAE Batch Size" },
  "VAE Chunk Size": { "zh-TW": "VAE Chunk Size", "zh-CN": "VAE Chunk Size" },
  "Cache Text Encoder Outputs to": { "zh-TW": "快取 Text Encoder Outputs 到", "zh-CN": "缓存 Text Encoder Outputs 到" },
  "Disk": { "zh-TW": "硬碟", "zh-CN": "硬盘" },
  "Disable VAE Cache": { "zh-TW": "停用 VAE Cache", "zh-CN": "禁用 VAE Cache" },
  "Timestep Sample Method": { "zh-TW": "Timestep 取樣方式", "zh-CN": "Timestep 采样方式" },
  "Flow Shift": { "zh-TW": "Flow Shift", "zh-CN": "Flow Shift" },
  "Weighting Scheme": { "zh-TW": "權重方案", "zh-CN": "权重方案" },
  "Sigmoid Scale": { "zh-TW": "Sigmoid Scale", "zh-CN": "Sigmoid Scale" },
  "Qwen3 Max Tokens": { "zh-TW": "Qwen3 最大 Tokens", "zh-CN": "Qwen3 最大 Tokens" },
  "T5 Max Tokens": { "zh-TW": "T5 最大 Tokens", "zh-CN": "T5 最大 Tokens" },
  "Global Dataset Settings": { "zh-TW": "資料集全域設定", "zh-CN": "数据集全局设置" },
  "Resolution(s)": { "zh-TW": "解析度", "zh-CN": "分辨率" },
  "Batch Size(s)": { "zh-TW": "Batch Size", "zh-CN": "Batch Size" },
  "Progressive Resolution Schedule": { "zh-TW": "漸進式解析度排程", "zh-CN": "渐进式分辨率调度" },
  "Gradient Accumulation": { "zh-TW": "Gradient Accumulation", "zh-CN": "Gradient Accumulation" },
  "Caption Extension": { "zh-TW": "Caption 副檔名", "zh-CN": "Caption 扩展名" },
  "Alpha Mask": { "zh-TW": "Alpha Mask", "zh-CN": "Alpha Mask" },
  "Enable Aspect Ratio Bucketing": { "zh-TW": "啟用長寬比分桶", "zh-CN": "启用宽高比分桶" },
  "Do Not Upscale": { "zh-TW": "不要放大圖片", "zh-CN": "不要放大图片" },
  "Images": { "zh-TW": "圖片", "zh-CN": "图片" },
  "Min Bucket Resolution": { "zh-TW": "最小 Bucket 解析度", "zh-CN": "最小 Bucket 分辨率" },
  "Max Bucket Resolution": { "zh-TW": "最大 Bucket 解析度", "zh-CN": "最大 Bucket 分辨率" },
  "Bucket Resolution Steps": { "zh-TW": "Bucket 解析度步進", "zh-CN": "Bucket 分辨率步进" },
  "Disable Bucket Shuffle": { "zh-TW": "停用 Bucket Shuffle", "zh-CN": "禁用 Bucket Shuffle" },
  "Dataset Folders": { "zh-TW": "資料集資料夾", "zh-CN": "数据集文件夹" },
  "+ Add Dataset": { "zh-TW": "+ 新增資料集", "zh-CN": "+ 新增数据集" },
  "Training Type": { "zh-TW": "訓練類型", "zh-CN": "训练类型" },
  "LoRA Configuration": { "zh-TW": "LoRA 設定", "zh-CN": "LoRA 设置" },
  "Network Module": { "zh-TW": "網路模組", "zh-CN": "网络模块" },
  "Network Dim (Rank)": { "zh-TW": "Network Dim (Rank)", "zh-CN": "Network Dim (Rank)" },
  "Network Alpha": { "zh-TW": "Network Alpha", "zh-CN": "Network Alpha" },
  "Train UNet Only": { "zh-TW": "只訓練 UNet", "zh-CN": "只训练 UNet" },
  "Network Dropout": { "zh-TW": "Network Dropout", "zh-CN": "Network Dropout" },
  "Network Args": { "zh-TW": "Network Args", "zh-CN": "Network Args" },
  "Network Weights (LoRA checkpoint)": { "zh-TW": "Network Weights (LoRA checkpoint)", "zh-CN": "Network Weights (LoRA checkpoint)" },
  "Full Finetune Configuration": { "zh-TW": "Full Finetune 設定", "zh-CN": "Full Finetune 设置" },
  "Freeze LLM Adapter": { "zh-TW": "凍結 LLM Adapter", "zh-CN": "冻结 LLM Adapter" },
  "Prompt List": { "zh-TW": "提示詞列表", "zh-CN": "提示词列表" },
  "+ Add Prompt": { "zh-TW": "+ 新增提示詞", "zh-CN": "+ 新增提示词" },
  "Generate Sample": { "zh-TW": "生成樣本", "zh-CN": "生成样本" },
  "Sampling": { "zh-TW": "取樣", "zh-CN": "采样" },
  "Sample Schedule": { "zh-TW": "樣本排程", "zh-CN": "样本调度" },
  "Sample at First": { "zh-TW": "開始前先產生樣本", "zh-CN": "开始前先生成样本" },
  "Enable Sampling During Training": { "zh-TW": "訓練中產生樣本", "zh-CN": "训练中生成样本" },
  "Sample Every N Steps": { "zh-TW": "每 N 步產生樣本", "zh-CN": "每 N 步生成样本" },
  "Refresh": { "zh-TW": "重新整理", "zh-CN": "刷新" },
  "TensorBoard": { "zh-TW": "TensorBoard", "zh-CN": "TensorBoard" },
  "Running on port": { "zh-TW": "執行中，port", "zh-CN": "运行中，端口" },
  "Not running": { "zh-TW": "未執行", "zh-CN": "未运行" },
  "Starting...": { "zh-TW": "啟動中...", "zh-CN": "启动中..." },
  "Launch": { "zh-TW": "啟動", "zh-CN": "启动" },
  "Open": { "zh-TW": "開啟", "zh-CN": "打开" },
  "Stop TensorBoard": { "zh-TW": "停止 TensorBoard", "zh-CN": "停止 TensorBoard" },
  "Job Maintenance": { "zh-TW": "工作維護", "zh-CN": "任务维护" },
  "📂 Open Job Folder": { "zh-TW": "📂 開啟工作資料夾", "zh-CN": "📂 打开任务文件夹" },
  "Logging": { "zh-TW": "記錄", "zh-CN": "日志" },
  "🗑️ Clear TensorBoard Logs": { "zh-TW": "🗑️ 清除 TensorBoard Logs", "zh-CN": "🗑️ 清除 TensorBoard Logs" },
  "Danger Zone": { "zh-TW": "危險區", "zh-CN": "危险区" },
  "⚠️ Reset Config to Defaults": { "zh-TW": "⚠️ 重設為預設值", "zh-CN": "⚠️ 重置为默认值" },
  "Create New Training Job": { "zh-TW": "建立新的訓練工作", "zh-CN": "创建新的训练任务" },
  "Job Name": { "zh-TW": "工作名稱", "zh-CN": "任务名称" },
  "Dataset 1 Image Folder": { "zh-TW": "資料集 1 圖片資料夾", "zh-CN": "数据集 1 图片文件夹" },
  "Cancel": { "zh-TW": "取消", "zh-CN": "取消" },
  "Create": { "zh-TW": "建立", "zh-CN": "创建" },
  "Clone Job": { "zh-TW": "複製工作", "zh-CN": "克隆任务" },
  "New Job Name": { "zh-TW": "新工作名稱", "zh-CN": "新任务名称" },
  "Confirm": { "zh-TW": "確認", "zh-CN": "确认" },
  "Are you sure?": { "zh-TW": "確定嗎？", "zh-CN": "确定吗？" },
  "Python Venv Path": { "zh-TW": "Python Venv 路徑", "zh-CN": "Python Venv 路径" },
  "Theme": { "zh-TW": "主題", "zh-CN": "主题" },
  "Background Image": { "zh-TW": "背景圖片", "zh-CN": "背景图片" },
  "Choose Image": { "zh-TW": "選擇圖片", "zh-CN": "选择图片" },
  "Remove": { "zh-TW": "移除", "zh-CN": "移除" },
  "Positioning (Drag to focal point)": { "zh-TW": "位置調整（拖曳焦點）", "zh-CN": "位置调整（拖拽焦点）" },
  "Dim Level (Overlay Opacity):": { "zh-TW": "暗化程度（遮罩透明度）：", "zh-CN": "暗化程度（遮罩透明度）：" },
  "Image Brightness:": { "zh-TW": "圖片亮度：", "zh-CN": "图片亮度：" },
  "Backdrop Blur:": { "zh-TW": "背景模糊：", "zh-CN": "背景模糊：" },
  "Text Shadow Intensity:": { "zh-TW": "文字陰影強度：", "zh-CN": "文字阴影强度：" },
  "Save Settings": { "zh-TW": "儲存設定", "zh-CN": "保存设置" },
  "LoRA Output Location": { "zh-TW": "LoRA 輸出位置", "zh-CN": "LoRA 输出位置" },
  "Application": { "zh-TW": "應用程式", "zh-CN": "应用程序" },
  "Anima Models": { "zh-TW": "Anima 模型", "zh-CN": "Anima 模型" },
  "Lumina Models": { "zh-TW": "Lumina 模型", "zh-CN": "Lumina 模型" },
  "DiT Model Path": { "zh-TW": "DiT 模型路徑", "zh-CN": "DiT 模型路径" },
  "Qwen3 Text Encoder Path": { "zh-TW": "Qwen3 Text Encoder 路徑", "zh-CN": "Qwen3 Text Encoder 路径" },
  "VAE Path": { "zh-TW": "VAE 路徑", "zh-CN": "VAE 路径" },
  "Image Directory": { "zh-TW": "圖片資料夾", "zh-CN": "图片文件夹" },
  "Num Repeats": { "zh-TW": "重複次數", "zh-CN": "重复次数" },
  "Keep Tokens": { "zh-TW": "保留 Tokens", "zh-CN": "保留 Tokens" },
  "Caption Prefix": { "zh-TW": "Caption 前綴", "zh-CN": "Caption 前缀" },
  "Caption Dropout Rate": { "zh-TW": "Caption Dropout Rate", "zh-CN": "Caption Dropout Rate" },
  "Tag Dropout Rate": { "zh-TW": "Tag Dropout Rate", "zh-CN": "Tag Dropout Rate" },
  "Dropout Every N Epochs": { "zh-TW": "每 N 輪 Dropout", "zh-CN": "每 N 轮 Dropout" },
  "Shuffle Captions": { "zh-TW": "打亂 Captions", "zh-CN": "打乱 Captions" },
  "Enable Wildcard": { "zh-TW": "啟用 Wildcard", "zh-CN": "启用 Wildcard" },
  "Flip Augmentations": { "zh-TW": "Flip Augmentations", "zh-CN": "Flip Augmentations" },
  "Regularization Dataset": { "zh-TW": "正則化資料集", "zh-CN": "正则化数据集" },
  "Enter prompt text...": { "zh-TW": "輸入提示詞...", "zh-CN": "输入提示词..." },
  "Base Model (No LoRA)": { "zh-TW": "基礎模型（無 LoRA）", "zh-CN": "基础模型（无 LoRA）" },
  "No jobs yet": { "zh-TW": "還沒有工作", "zh-CN": "还没有任务" },
  "Connecting...": { "zh-TW": "連線中...", "zh-CN": "连接中..." },
  "Idle": { "zh-TW": "閒置", "zh-CN": "空闲" },
  "Single GPU": { "zh-TW": "單 GPU", "zh-CN": "单 GPU" },
  "No NVIDIA GPUs detected (CPU only).": { "zh-TW": "未偵測到 NVIDIA GPU（僅 CPU）。", "zh-CN": "未检测到 NVIDIA GPU（仅 CPU）。" },
  "No NVIDIA GPUs detected.": { "zh-TW": "未偵測到 NVIDIA GPU。", "zh-CN": "未检测到 NVIDIA GPU。" },
  "Job saved": { "zh-TW": "工作已儲存", "zh-CN": "任务已保存" },
  "Changes discarded": { "zh-TW": "已放棄變更", "zh-CN": "已放弃更改" },
  "Global settings saved": { "zh-TW": "全域設定已儲存", "zh-CN": "全局设置已保存" },
  "Job created": { "zh-TW": "工作已建立", "zh-CN": "任务已创建" },
  "Job cloned": { "zh-TW": "工作已複製", "zh-CN": "任务已克隆" },
  "Job deleted": { "zh-TW": "工作已刪除", "zh-CN": "任务已删除" },
  "Training started": { "zh-TW": "訓練已開始", "zh-CN": "训练已开始" },
  "Training stopped": { "zh-TW": "訓練已停止", "zh-CN": "训练已停止" },
  "Generation started": { "zh-TW": "生成已開始", "zh-CN": "生成已开始" },
  "Add sample prompts first": { "zh-TW": "請先新增樣本提示詞", "zh-CN": "请先新增样本提示词" },
  "Unloading model...": { "zh-TW": "正在卸載模型...", "zh-CN": "正在卸载模型..." },
  "Model unloaded": { "zh-TW": "模型已卸載", "zh-CN": "模型已卸载" },
  "Checkpoints refreshed": { "zh-TW": "Checkpoints 已重新整理", "zh-CN": "Checkpoints 已刷新" },
  "TensorBoard launched": { "zh-TW": "TensorBoard 已啟動", "zh-CN": "TensorBoard 已启动" },
  "TensorBoard stopped": { "zh-TW": "TensorBoard 已停止", "zh-CN": "TensorBoard 已停止" },
  "Logs cleared": { "zh-TW": "Logs 已清除", "zh-CN": "Logs 已清除" },
  "Config reset to defaults": { "zh-TW": "設定已重設為預設值", "zh-CN": "设置已重置为默认值" },
  "Background updated!": { "zh-TW": "背景已更新！", "zh-CN": "背景已更新！" },
  "Background removed": { "zh-TW": "背景已移除", "zh-CN": "背景已移除" },
  "Please enter a directory path first": { "zh-TW": "請先輸入資料夾路徑", "zh-CN": "请先输入文件夹路径" },
};

const NETWORK_MODULE_PRESETS = {
  "networks.krona": { learningRate: "5e-4" },
  "networks.cdka": { learningRate: "1e-4" },
};

Object.assign(UI_TRANSLATIONS, {
  "Anima LoRA Training": { "zh-TW": "Anima LoRA 訓練", "zh-CN": "Anima LoRA 训练" },
  "Duration Unit": { "zh-TW": "訓練長度單位", "zh-CN": "训练时长单位" },
  "Epochs": { "zh-TW": "輪數", "zh-CN": "轮数" },
  "Steps": { "zh-TW": "步數", "zh-CN": "步数" },
  "Bucketing": { "zh-TW": "分桶", "zh-CN": "分桶" },
  "Do Not Upscale Images": { "zh-TW": "不要放大圖片", "zh-CN": "不要放大图片" },
  "Groups images by aspect ratio for efficient training.": {
    "zh-TW": "依照圖片長寬比分組，讓訓練更有效率。",
    "zh-CN": "按照图片宽高比分组，让训练更有效率。",
  },
  "Images smaller than the bucket resolution will not be upscaled, saving VRAM and disk space.": {
    "zh-TW": "小於 bucket 解析度的圖片不會被放大，可節省 VRAM 與硬碟空間。",
    "zh-CN": "小于 bucket 分辨率的图片不会被放大，可节省 VRAM 与硬盘空间。",
  },
  "Keeps DataLoader order within resolution buckets.": {
    "zh-TW": "保留每個解析度 bucket 內的 DataLoader 順序。",
    "zh-CN": "保留每个分辨率 bucket 内的 DataLoader 顺序。",
  },
  "Max resolution(s) for bucketing. Comma-separate for multi-resolution caching (e.g. 512, 1024).": {
    "zh-TW": "分桶用最大解析度。多解析度快取可用逗號分隔，例如 512, 1024。",
    "zh-CN": "分桶用最大分辨率。多分辨率缓存可用逗号分隔，例如 512, 1024。",
  },
  "Comma-separate to assign varying batch sizes per resolution.": {
    "zh-TW": "用逗號分隔，可替不同解析度指定不同 batch size。",
    "zh-CN": "用逗号分隔，可给不同分辨率指定不同 batch size。",
  },
  "Training Schedule": { "zh-TW": "訓練排程", "zh-CN": "训练调度" },
  "Checkpoint Management": { "zh-TW": "Checkpoint 管理", "zh-CN": "Checkpoint 管理" },
  "Save Last N Steps": { "zh-TW": "保留最近 N 步", "zh-CN": "保留最近 N 步" },
  "Save Last N Epochs": { "zh-TW": "保留最近 N 輪", "zh-CN": "保留最近 N 轮" },
  "Save Training State": { "zh-TW": "儲存訓練 State", "zh-CN": "保存训练 State" },
  "Save State at Train End": { "zh-TW": "訓練結束儲存 State", "zh-CN": "训练结束保存 State" },
  "Save Last N Step States": { "zh-TW": "保留最近 N Step States", "zh-CN": "保留最近 N Step States" },
  "Save Last N Epoch States": { "zh-TW": "保留最近 N Epoch States", "zh-CN": "保留最近 N Epoch States" },
  "Only applies when saving by steps. Leave blank to keep all checkpoints.": {
    "zh-TW": "只在依 steps 儲存時生效。留空代表保留全部 checkpoints。",
    "zh-CN": "只在按 steps 保存时生效。留空代表保留全部 checkpoints。",
  },
  "Only applies when saving by epochs. Leave blank to keep all checkpoints.": {
    "zh-TW": "只在依 epochs 儲存時生效。留空代表保留全部 checkpoints。",
    "zh-CN": "只在按 epochs 保存时生效。留空代表保留全部 checkpoints。",
  },
  "Also saves optimizer and scheduler state when a checkpoint is saved.": {
    "zh-TW": "儲存 checkpoint 時，同時儲存 optimizer 與 scheduler 狀態。",
    "zh-CN": "保存 checkpoint 时，同时保存 optimizer 与 scheduler 状态。",
  },
  "Saves a final state folder when training finishes.": {
    "zh-TW": "訓練完成時額外儲存最後的 state 資料夾。",
    "zh-CN": "训练完成时额外保存最后的 state 文件夹。",
  },
  "Overrides step checkpoint retention for state folders.": {
    "zh-TW": "覆蓋 step state 資料夾的保留數。",
    "zh-CN": "覆盖 step state 文件夹的保留数量。",
  },
  "Overrides epoch checkpoint retention for state folders.": {
    "zh-TW": "覆蓋 epoch state 資料夾的保留數。",
    "zh-CN": "覆盖 epoch state 文件夹的保留数量。",
  },
  "blank = keep all": { "zh-TW": "空白 = 全部保留", "zh-CN": "空白 = 全部保留" },
  "blank = use checkpoint count": {
    "zh-TW": "空白 = 使用 checkpoint 保留數",
    "zh-CN": "空白 = 使用 checkpoint 保留数",
  },
  "Restart Cycles": { "zh-TW": "重啟週期", "zh-CN": "重启周期" },
  "Number of times the learning rate restarts from max to min.": {
    "zh-TW": "學習率從最大值重新下降到最小值的次數。",
    "zh-CN": "学习率从最大值重新下降到最小值的次数。",
  },
  "Min LR Ratio": { "zh-TW": "最小 LR 比例", "zh-CN": "最小 LR 比例" },
  "Minimum LR as a fraction of the initial LR (e.g. 0.1 = decays to 10%).": {
    "zh-TW": "最小學習率佔初始學習率的比例，例如 0.1 代表降到 10%。",
    "zh-CN": "最小学习率占初始学习率的比例，例如 0.1 代表降到 10%。",
  },
  "Decouple Weight Decay": { "zh-TW": "Decouple Weight Decay", "zh-CN": "Decouple Weight Decay" },
  "Separates weight penalty from gradient scaling to prevent overfitting when using Weight Decay.": {
    "zh-TW": "將權重懲罰與梯度縮放分離，使用 Weight Decay 時可降低過擬合風險。",
    "zh-CN": "将权重惩罚与梯度缩放分离，使用 Weight Decay 时可降低过拟合风险。",
  },
  "Activation Checkpointing": { "zh-TW": "Activation Checkpointing", "zh-CN": "Activation Checkpointing" },
  "Cache Text Encoder Outputs to Disk": {
    "zh-TW": "快取 Text Encoder Outputs 到硬碟",
    "zh-CN": "缓存 Text Encoder Outputs 到硬盘",
  },
  "Pre-encode images as .safetensors files for faster training.": {
    "zh-TW": "預先把圖片編碼成 .safetensors，加快訓練讀取。",
    "zh-CN": "预先把图片编码成 .safetensors，加快训练读取。",
  },
  "Pre-encode captions. Required if text encoder is frozen.": {
    "zh-TW": "預先編碼 captions。凍結 text encoder 時需要啟用。",
    "zh-CN": "预先编码 captions。冻结 text encoder 时需要启用。",
  },
  "Can reduce VAE memory if chunking is still not enough.": {
    "zh-TW": "如果 VAE chunking 仍不夠，可再降低 VAE 記憶體使用。",
    "zh-CN": "如果 VAE chunking 仍不够，可进一步降低 VAE 内存使用。",
  },
  "0 disables chunked VAE decode. Lower values use less VRAM.": {
    "zh-TW": "0 代表停用 VAE 分塊 decode。數值越低越省 VRAM。",
    "zh-CN": "0 代表禁用 VAE 分块 decode。数值越低越省 VRAM。",
  },
  "Trades compute for VRAM savings. Recommended on.": {
    "zh-TW": "用額外計算換取 VRAM 節省，建議開啟。",
    "zh-CN": "用额外计算换取 VRAM 节省，建议开启。",
  },
  "Requires flash-attn package.": { "zh-TW": "需要 flash-attn 套件。", "zh-CN": "需要 flash-attn 包。" },
  "Uses dynamo inductor backend. Slower first step, faster training after.": {
    "zh-TW": "使用 dynamo inductor backend。第一步較慢，之後訓練可能更快。",
    "zh-CN": "使用 dynamo inductor backend。第一步较慢，之后训练可能更快。",
  },
  "Moves transformer blocks to RAM to save VRAM. 0=Off.": {
    "zh-TW": "把 transformer blocks 移到 RAM 以節省 VRAM。0 = 關閉。",
    "zh-CN": "把 transformer blocks 移到 RAM 以节省 VRAM。0 = 关闭。",
  },
  "Offloads activations to CPU during backward pass. Requires Gradient Checkpointing.": {
    "zh-TW": "backward 時把 activations 卸載到 CPU。需要 Gradient Checkpointing。",
    "zh-CN": "backward 时把 activations 卸载到 CPU。需要 Gradient Checkpointing。",
  },
  "Keeps workers alive between epochs (less startup lag).": {
    "zh-TW": "讓 workers 在 epochs 間保持啟動，減少重新啟動延遲。",
    "zh-CN": "让 workers 在 epochs 间保持启动，减少重新启动延迟。",
  },
  "Timestep Sample Method": { "zh-TW": "Timestep 取樣方式", "zh-CN": "Timestep 采样方式" },
  "Uniform (Kohya Default)": { "zh-TW": "Uniform（Kohya 預設）", "zh-CN": "Uniform（Kohya 默认）" },
  "Training Type": { "zh-TW": "訓練類型", "zh-CN": "训练类型" },
  "LoRA trains lightweight adapter weights. Full Finetune trains the entire DiT model.": {
    "zh-TW": "LoRA 只訓練輕量 adapter 權重；Full Finetune 會訓練整個 DiT 模型。",
    "zh-CN": "LoRA 只训练轻量 adapter 权重；Full Finetune 会训练整个 DiT 模型。",
  },
  "LoRA Configuration": { "zh-TW": "LoRA 設定", "zh-CN": "LoRA 设置" },
  "LoRA": { "zh-TW": "LoRA", "zh-CN": "LoRA" },
  "Full Finetune": { "zh-TW": "Full Finetune", "zh-CN": "Full Finetune" },
  "Full Finetune Options": { "zh-TW": "Full Finetune 選項", "zh-CN": "Full Finetune 选项" },
  "Freeze text encoder. Recommended for most LoRA training.": {
    "zh-TW": "凍結 text encoder。大多數 LoRA 訓練建議開啟。",
    "zh-CN": "冻结 text encoder。大多数 LoRA 训练建议开启。",
  },
  "Dropout rate (0-1). Randomly zeroes LoRA neurons to reduce overfitting. 0 = off.": {
    "zh-TW": "Dropout 比例（0-1）。隨機歸零 LoRA 神經元以降低過擬合；0 = 關閉。",
    "zh-CN": "Dropout 比例（0-1）。随机归零 LoRA 神经元以降低过拟合；0 = 关闭。",
  },
  "Space-separated key=value pairs passed to the network module.": {
    "zh-TW": "傳給 network module 的 key=value 參數，使用空格分隔。",
    "zh-CN": "传给 network module 的 key=value 参数，使用空格分隔。",
  },
  "Initialize from an existing LoRA. Often used for fine-tuning the lora. Usually not needed.": {
    "zh-TW": "從既有 LoRA 初始化。常用於接續微調，一般不需要。",
    "zh-CN": "从已有 LoRA 初始化。常用于接续微调，一般不需要。",
  },
  "Keep the LLM adapter weights frozen during training. Recommended — the adapter is pre-trained and retraining it risks degrading text understanding and causes DDP graph errors.": {
    "zh-TW": "訓練時保持 LLM adapter 權重凍結。建議開啟，因為 adapter 已預訓練，重訓可能降低文字理解並造成 DDP graph 錯誤。",
    "zh-CN": "训练时保持 LLM adapter 权重冻结。建议开启，因为 adapter 已预训练，重训可能降低文本理解并造成 DDP graph 错误。",
  },
  "Resume Training": { "zh-TW": "恢復訓練", "zh-CN": "恢复训练" },
  "Auto-resume from last saved state": { "zh-TW": "自動從最後儲存的 state 恢復", "zh-CN": "自动从最后保存的 state 恢复" },
  "Automatically resumes from the most recent saved state in the output folder.": {
    "zh-TW": "自動從輸出資料夾中最新的 state 恢復訓練。",
    "zh-CN": "自动从输出文件夹中最新的 state 恢复训练。",
  },
  "Resume State Folder": { "zh-TW": "恢復用 State 資料夾", "zh-CN": "恢复用 State 文件夹" },
  "Resume training state (optimizer, scheduler, step count). Leave blank when auto-resume is enabled.": {
    "zh-TW": "恢復訓練狀態（optimizer、scheduler、step 數）。啟用自動恢復時請留空。",
    "zh-CN": "恢复训练状态（optimizer、scheduler、step 数）。启用自动恢复时请留空。",
  },
  "In-Training Sampling": { "zh-TW": "訓練中產生樣本", "zh-CN": "训练中生成样本" },
  "Enable Sampling": { "zh-TW": "啟用樣本產生", "zh-CN": "启用样本生成" },
  "Sample at First": { "zh-TW": "開始前先產生樣本", "zh-CN": "开始前先生成样本" },
  "Generate sample images before training starts.": {
    "zh-TW": "訓練開始前先產生樣本圖。",
    "zh-CN": "训练开始前先生成样本图。",
  },
  "Sample Every N Epochs": { "zh-TW": "每 N 輪產生樣本", "zh-CN": "每 N 轮生成样本" },
  "Sample Every N Steps": { "zh-TW": "每 N 步產生樣本", "zh-CN": "每 N 步生成样本" },
  "Test Generation": { "zh-TW": "測試生成", "zh-CN": "测试生成" },
  "LoRA Strength": { "zh-TW": "LoRA 強度", "zh-CN": "LoRA 强度" },
  "GPU Selection": { "zh-TW": "GPU 選擇", "zh-CN": "GPU 选择" },
  "Multi-GPU Mode": { "zh-TW": "多 GPU 模式", "zh-CN": "多 GPU 模式" },
  "Keep Model Loaded": { "zh-TW": "保留模型載入", "zh-CN": "保持模型载入" },
  "Unload Model": { "zh-TW": "卸載模型", "zh-CN": "卸载模型" },
  "Generate": { "zh-TW": "生成", "zh-CN": "生成" },
  "Sample Prompts": { "zh-TW": "樣本提示詞", "zh-CN": "样本提示词" },
  "Negative Prompt": { "zh-TW": "負面提示詞", "zh-CN": "负面提示词" },
  "Apply to All": { "zh-TW": "套用到全部", "zh-CN": "应用到全部" },
  "No sample prompts yet. Click \"+ Add Prompt\" to add one.": {
    "zh-TW": "尚未新增樣本提示詞。點擊「+ 新增提示詞」來新增。",
    "zh-CN": "尚未新增样本提示词。点击「+ 新增提示词」来新增。",
  },
  "Generated Samples": { "zh-TW": "已生成樣本", "zh-CN": "已生成样本" },
  "No sample images yet. They will appear here during training.": {
    "zh-TW": "尚未產生樣本圖，訓練期間會顯示在這裡。",
    "zh-CN": "尚未生成样本图，训练期间会显示在这里。",
  },
  "Training Console": { "zh-TW": "訓練主控台", "zh-CN": "训练控制台" },
  "Waiting for training to start...": { "zh-TW": "等待訓練開始...", "zh-CN": "等待训练开始..." },
  "Clear": { "zh-TW": "清除", "zh-CN": "清除" },
  "Running on port": { "zh-TW": "執行中，port", "zh-CN": "运行中，port" },
  "Click \"Launch\" to start TensorBoard and view training metrics.": {
    "zh-TW": "點擊「啟動」開啟 TensorBoard 並查看訓練指標。",
    "zh-CN": "点击「启动」开启 TensorBoard 并查看训练指标。",
  },
  "Job Maintenance": { "zh-TW": "工作維護", "zh-CN": "任务维护" },
  "Open the job's directory in file explorer.": {
    "zh-TW": "在檔案總管中開啟此工作的資料夾。",
    "zh-CN": "在文件资源管理器中打开此任务的文件夹。",
  },
  "Logs directory:": { "zh-TW": "Logs 資料夾：", "zh-CN": "Logs 文件夹：" },
  "Delete all TensorBoard event files for this job.": {
    "zh-TW": "刪除此工作的所有 TensorBoard event 檔案。",
    "zh-CN": "删除此任务的所有 TensorBoard event 文件。",
  },
  "Revert all settings to template defaults.": {
    "zh-TW": "將所有設定還原為模板預設值。",
    "zh-CN": "将所有设置还原为模板默认值。",
  },
  "Hardware Allocation": { "zh-TW": "硬體分配", "zh-CN": "硬件分配" },
  "Multi-GPU Optimization": { "zh-TW": "多 GPU 最佳化", "zh-CN": "多 GPU 优化" },
  "Parallelism Mode": { "zh-TW": "平行模式", "zh-CN": "并行模式" },
  "DDP Options": { "zh-TW": "DDP 選項", "zh-CN": "DDP 选项" },
  "DeepSpeed Options": { "zh-TW": "DeepSpeed 選項", "zh-CN": "DeepSpeed 选项" },
  "TP/SP Options": { "zh-TW": "TP/SP 選項", "zh-CN": "TP/SP 选项" },
  "Use CUDA Direct Backend": { "zh-TW": "使用 CUDA Direct Backend", "zh-CN": "使用 CUDA Direct Backend" },
  "Windows-only custom backend replacing NCCL for native multi-GPU. Auto-detected in TP/SP mode. Incompatible with Torch Compile.": {
    "zh-TW": "Windows 專用自訂 backend，用於 native 多 GPU 時取代 NCCL。TP/SP 模式會自動偵測，且不相容 Torch Compile。",
    "zh-CN": "Windows 专用自定义 backend，用于 native 多 GPU 时替代 NCCL。TP/SP 模式会自动检测，且不兼容 Torch Compile。",
  },
  "Gradient as Bucket View": { "zh-TW": "Gradient as Bucket View", "zh-CN": "Gradient as Bucket View" },
  "Static Graph": { "zh-TW": "Static Graph", "zh-CN": "Static Graph" },
  "Reduces gradient memory overhead by eliminating a copy per step. Recommended for DDP training.": {
    "zh-TW": "每 step 少一次複製，降低 gradient 記憶體開銷。DDP 訓練建議開啟。",
    "zh-CN": "每 step 少一次复制，降低 gradient 内存开销。DDP 训练建议开启。",
  },
  "Allows DDP to overlap communication and computation more aggressively. Recommended when model structure does not change between steps.": {
    "zh-TW": "讓 DDP 更積極重疊通訊與計算。模型結構每 step 不變時建議開啟。",
    "zh-CN": "让 DDP 更积极重叠通信与计算。模型结构每 step 不变时建议开启。",
  },
  "Sharding Strategy": { "zh-TW": "Sharding 策略", "zh-CN": "Sharding 策略" },
  "CPU Offloading": { "zh-TW": "CPU Offloading", "zh-CN": "CPU Offloading" },
  "Enable Resharding After Forward": { "zh-TW": "啟用 Forward 後 Reshard", "zh-CN": "启用 Forward 后 Reshard" },
  "Enable FSDP Activation Checkpointing": {
    "zh-TW": "啟用 FSDP Activation Checkpointing",
    "zh-CN": "启用 FSDP Activation Checkpointing",
  },
  "CPU RAM Efficient Loading": { "zh-TW": "CPU RAM 省記憶體載入", "zh-CN": "CPU RAM 省内存加载" },
  "Backward Prefetch": { "zh-TW": "Backward Prefetch", "zh-CN": "Backward Prefetch" },
  "Forward Prefetch": { "zh-TW": "Forward Prefetch", "zh-CN": "Forward Prefetch" },
  "Use Original Parameters": { "zh-TW": "使用原始 Parameters", "zh-CN": "使用原始 Parameters" },
  "Limit All-Gathers": { "zh-TW": "限制 All-Gathers", "zh-CN": "限制 All-Gathers" },
  "Auto Wrap Policy": { "zh-TW": "Auto Wrap Policy", "zh-CN": "Auto Wrap Policy" },
  "Transformer Layer to Wrap": { "zh-TW": "要 Wrap 的 Transformer Layer", "zh-CN": "要 Wrap 的 Transformer Layer" },
  "Parameters Threshold": { "zh-TW": "Parameters 門檻", "zh-CN": "Parameters 阈值" },
  "Diagnostics": { "zh-TW": "診斷", "zh-CN": "诊断" },
  "Enable Step Profiling": { "zh-TW": "啟用 Step Profiling", "zh-CN": "启用 Step Profiling" },
  "Track Microbatches": { "zh-TW": "追蹤 Microbatches", "zh-CN": "追踪 Microbatches" },
  "Prints per-step timing breakdown (forward, backward, communication, optimizer, Python overhead) to the training log.": {
    "zh-TW": "在訓練 log 輸出每個 step 的耗時拆解：forward、backward、通訊、optimizer、Python overhead。",
    "zh-CN": "在训练 log 输出每个 step 的耗时拆解：forward、backward、通信、optimizer、Python overhead。",
  },
  "Also print per-microbatch fwd/bwd times within each step (only available with step profiling enabled).": {
    "zh-TW": "同時輸出每個 step 內各 microbatch 的 fwd/bwd 時間，只在 step profiling 開啟時可用。",
    "zh-CN": "同时输出每个 step 内各 microbatch 的 fwd/bwd 时间，只在 step profiling 开启时可用。",
  },
  "Path to the Python virtual environment with training dependencies.": {
    "zh-TW": "包含訓練依賴套件的 Python venv 路徑。",
    "zh-CN": "包含训练依赖包的 Python venv 路径。",
  },
  "Drag the focal point in the preview above. Auto-scales to fit.": {
    "zh-TW": "拖曳上方預覽中的焦點位置，會自動縮放適配。",
    "zh-CN": "拖拽上方预览中的焦点位置，会自动缩放适配。",
  },
  "Apply these settings to all prompts below": {
    "zh-TW": "將這些設定套用到下方所有提示詞",
    "zh-CN": "将这些设置应用到下方所有提示词",
  },
  "Enter negative prompt tags...": { "zh-TW": "輸入負面提示詞 tags...", "zh-CN": "输入负面提示词 tags..." },
  "Refresh Checkpoints": { "zh-TW": "重新整理 Checkpoints", "zh-CN": "刷新 Checkpoints" },
  "Free VRAM by unloading the generation model": {
    "zh-TW": "卸載生成模型以釋放 VRAM",
    "zh-CN": "卸载生成模型以释放 VRAM",
  },
  "Open in new tab": { "zh-TW": "在新分頁開啟", "zh-CN": "在新标签页打开" },
  "e.g. 512, 1024, 2048": { "zh-TW": "例如 512, 1024, 2048", "zh-CN": "例如 512, 1024, 2048" },
  "e.g. 1, 2, 4": { "zh-TW": "例如 1, 2, 4", "zh-CN": "例如 1, 2, 4" },
  "e.g. conv_dim=4 conv_alpha=4": {
    "zh-TW": "例如 conv_dim=4 conv_alpha=4",
    "zh-CN": "例如 conv_dim=4 conv_alpha=4",
  },
  "e.g. Block": { "zh-TW": "例如 Block", "zh-CN": "例如 Block" },
  "e.g. Aemeath_copy": { "zh-TW": "例如 Aemeath_copy", "zh-CN": "例如 Aemeath_copy" },
  "my_job_v1": { "zh-TW": "my_job_v1", "zh-CN": "my_job_v1" },
  "e.g. Aemeath_v2": { "zh-TW": "例如 Aemeath_v2", "zh-CN": "例如 Aemeath_v2" },
  "D:\\datasets\\my_images": { "zh-TW": "D:\\datasets\\my_images", "zh-CN": "D:\\datasets\\my_images" },
});

Object.assign(UI_TRANSLATIONS, {
  "Backend": { "zh-TW": "Backend", "zh-CN": "Backend" },
  "Better Multi-GPU support.": { "zh-TW": "較好的多 GPU 支援。", "zh-CN": "较好的多 GPU 支持。" },
  "Loading GPUs...": { "zh-TW": "正在載入 GPUs...", "zh-CN": "正在加载 GPUs..." },
  "Requires flash-attn.": { "zh-TW": "需要 flash-attn。", "zh-CN": "需要 flash-attn。" },
  "Scale": { "zh-TW": "Scale", "zh-CN": "Scale" },
  "W": { "zh-TW": "寬", "zh-CN": "宽" },
  "H": { "zh-TW": "高", "zh-CN": "高" },
  "Safetensors": { "zh-TW": "Safetensors", "zh-CN": "Safetensors" },
  "Checkpoint": { "zh-TW": "Checkpoint", "zh-CN": "Checkpoint" },
  "GitHub Dark": { "zh-TW": "GitHub 深色", "zh-CN": "GitHub 深色" },
  "GitHub Light": { "zh-TW": "GitHub 淺色", "zh-CN": "GitHub 浅色" },
  "Midnight Blue": { "zh-TW": "午夜藍", "zh-CN": "午夜蓝" },
  "Cherry Pink": { "zh-TW": "櫻桃粉", "zh-CN": "樱桃粉" },
  "Positioning (Drag to focal point)": { "zh-TW": "位置調整（拖曳焦點）", "zh-CN": "位置调整（拖拽焦点）" },
  "Dim Level (Overlay Opacity):": { "zh-TW": "暗化程度（遮罩透明度）：", "zh-CN": "暗化程度（遮罩透明度）：" },
  "Image Brightness:": { "zh-TW": "圖片亮度：", "zh-CN": "图片亮度：" },
  "Backdrop Blur:": { "zh-TW": "背景模糊：", "zh-CN": "背景模糊：" },
  "Text Shadow Intensity:": { "zh-TW": "文字陰影強度：", "zh-CN": "文字阴影强度：" },
  "📁 Choose Image": { "zh-TW": "📁 選擇圖片", "zh-CN": "📁 选择图片" },
  "🔄 Refresh": { "zh-TW": "🔄 重新整理", "zh-CN": "🔄 刷新" },
  "🗑️ Delete Selected": { "zh-TW": "🗑️ 刪除選取", "zh-CN": "🗑️ 删除选中" },
  "🚀 Launch": { "zh-TW": "🚀 啟動", "zh-CN": "🚀 启动" },
  "↗ Open": { "zh-TW": "↗ 開啟", "zh-CN": "↗ 打开" },
  "For more information, please refer to the": {
    "zh-TW": "更多資訊請參考",
    "zh-CN": "更多信息请参考",
  },
  "official PyTorch docs": { "zh-TW": "PyTorch 官方文件", "zh-CN": "PyTorch 官方文档" },
  "Select a strategy to see details.": { "zh-TW": "選擇策略後會顯示詳細說明。", "zh-CN": "选择策略后会显示详细说明。" },
  "DDP replicates the full model on each GPU and syncs gradients. FSDP1/FSDP2 shard parameters to reduce VRAM (FSDP2 is the newer, simpler API).": {
    "zh-TW": "DDP 會在每張 GPU 複製完整模型並同步 gradients。FSDP1/FSDP2 會切分參數以降低 VRAM 使用量（FSDP2 是較新的簡化 API）。",
    "zh-CN": "DDP 会在每张 GPU 复制完整模型并同步 gradients。FSDP1/FSDP2 会切分参数以降低 VRAM 使用量（FSDP2 是较新的简化 API）。",
  },
  "DDP — Data Parallel (standard)": { "zh-TW": "DDP — Data Parallel（標準）", "zh-CN": "DDP — Data Parallel（标准）" },
  "FSDP1 — Fully Sharded Data Parallel v1": { "zh-TW": "FSDP1 — Fully Sharded Data Parallel v1", "zh-CN": "FSDP1 — Fully Sharded Data Parallel v1" },
  "FSDP2 — Fully Sharded Data Parallel v2 (Linux only)": {
    "zh-TW": "FSDP2 — Fully Sharded Data Parallel v2（僅 Linux）",
    "zh-CN": "FSDP2 — Fully Sharded Data Parallel v2（仅 Linux）",
  },
  "DeepSpeed — ZeRO Optimizer Sharding": {
    "zh-TW": "DeepSpeed — ZeRO Optimizer Sharding",
    "zh-CN": "DeepSpeed — ZeRO Optimizer Sharding",
  },
  "TP/SP — Tensor + Sequence Parallel": {
    "zh-TW": "TP/SP — Tensor + Sequence Parallel",
    "zh-CN": "TP/SP — Tensor + Sequence Parallel",
  },
  "Parallel CFG (Speed) - run pos/neg on separate GPUs": {
    "zh-TW": "Parallel CFG（速度）- 正/負提示詞分別跑在不同 GPU",
    "zh-CN": "Parallel CFG（速度）- 正/负提示词分别跑在不同 GPU",
  },
  "Sharding (VRAM) - split model across GPUs": {
    "zh-TW": "Sharding（省 VRAM）- 將模型切分到多張 GPU",
    "zh-CN": "Sharding（省 VRAM）- 将模型切分到多张 GPU",
  },
  "Disable Fused QKV": { "zh-TW": "停用 Fused QKV", "zh-CN": "禁用 Fused QKV" },
  "Debug option. Leaves attention q/k/v projections unfused instead of using the packed QKV/KV TP path.": {
    "zh-TW": "除錯選項。保留 attention q/k/v projections 不融合，不使用 packed QKV/KV TP 路徑。",
    "zh-CN": "调试选项。保留 attention q/k/v projections 不融合，不使用 packed QKV/KV TP 路径。",
  },
  "TP Degree": { "zh-TW": "TP Degree", "zh-CN": "TP Degree" },
  "Number of GPUs for tensor parallelism. Must equal the number of selected GPUs above.": {
    "zh-TW": "Tensor parallelism 使用的 GPU 數量，必須等於上方選取的 GPU 數。",
    "zh-CN": "Tensor parallelism 使用的 GPU 数量，必须等于上方选择的 GPU 数。",
  },
  "Sequence Parallel (SP)": { "zh-TW": "Sequence Parallel（SP）", "zh-CN": "Sequence Parallel（SP）" },
  "Always enabled for this mode. Spatial tokens are split across GPUs alongside weight sharding.": {
    "zh-TW": "此模式會固定啟用。Spatial tokens 會和權重切分一起分散到多張 GPU。",
    "zh-CN": "此模式会固定启用。Spatial tokens 会和权重切分一起分散到多张 GPU。",
  },
  "TP/SP notes:": { "zh-TW": "TP/SP 注意事項：", "zh-CN": "TP/SP 注意事项：" },
  "All standard training configs (LR, batch size, optimizer, network dim, etc.) work. Sample generation during training is disabled. Backend is selected above. Torch Compile is incompatible.": {
    "zh-TW": "標準訓練設定（LR、batch size、optimizer、network dim 等）都可用。訓練中 sample 會停用。Backend 由上方選擇。Torch Compile 不相容。",
    "zh-CN": "标准训练设置（LR、batch size、optimizer、network dim 等）都可用。训练中 sample 会禁用。Backend 由上方选择。Torch Compile 不兼容。",
  },
  "Use Gloo on native Windows. Use NCCL from WSL/Linux. Auto lets the TP/SP script choose.": {
    "zh-TW": "native Windows 使用 Gloo；WSL/Linux 使用 NCCL。Auto 會讓 TP/SP script 自行選擇。",
    "zh-CN": "native Windows 使用 Gloo；WSL/Linux 使用 NCCL。Auto 会让 TP/SP script 自行选择。",
  },
  "DeepSpeed note:": { "zh-TW": "DeepSpeed 注意事項：", "zh-CN": "DeepSpeed 注意事项：" },
  "This mode uses Accelerate DeepSpeed launch options plus existing training arguments.": {
    "zh-TW": "此模式會使用 Accelerate DeepSpeed 啟動選項，並搭配現有訓練參數。",
    "zh-CN": "此模式会使用 Accelerate DeepSpeed 启动选项，并搭配现有训练参数。",
  },
  "ZeRO Stage": { "zh-TW": "ZeRO Stage", "zh-CN": "ZeRO Stage" },
  "Higher stage saves more VRAM but can increase communication overhead.": {
    "zh-TW": "Stage 越高越省 VRAM，但可能增加通訊成本。",
    "zh-CN": "Stage 越高越省 VRAM，但可能增加通信成本。",
  },
  "Optimizer Offload Device": { "zh-TW": "Optimizer Offload 裝置", "zh-CN": "Optimizer Offload 设备" },
  "Optimizer NVMe Path": { "zh-TW": "Optimizer NVMe 路徑", "zh-CN": "Optimizer NVMe 路径" },
  "Parameter Offload Device": { "zh-TW": "Parameter Offload 裝置", "zh-CN": "Parameter Offload 设备" },
  "Parameter NVMe Path": { "zh-TW": "Parameter NVMe 路徑", "zh-CN": "Parameter NVMe 路径" },
  "Enable ZeRO-3 Init": { "zh-TW": "啟用 ZeRO-3 Init", "zh-CN": "启用 ZeRO-3 Init" },
  "Save 16-bit Model with ZeRO-3": { "zh-TW": "使用 ZeRO-3 儲存 16-bit 模型", "zh-CN": "使用 ZeRO-3 保存 16-bit 模型" },
  "FP16 Master Weights and Gradients": {
    "zh-TW": "FP16 Master Weights and Gradients",
    "zh-CN": "FP16 Master Weights and Gradients",
  },
  "Primarily useful for ZeRO-Offload configurations that support this mode.": {
    "zh-TW": "主要用於支援此模式的 ZeRO-Offload 設定。",
    "zh-CN": "主要用于支持此模式的 ZeRO-Offload 设置。",
  },
  "Moves parameters to Main System RAM when not in use. Slows down training but drastically reduces VRAM requirements.": {
    "zh-TW": "未使用時把 parameters 移到系統 RAM。訓練會變慢，但可大幅降低 VRAM 需求。",
    "zh-CN": "未使用时把 parameters 移到系统 RAM。训练会变慢，但可大幅降低 VRAM 需求。",
  },
  "Frees gathered parameters after the forward pass. Highly recommended for maximizing VRAM savings.": {
    "zh-TW": "forward 後釋放已 gather 的 parameters。想最大化節省 VRAM 時強烈建議開啟。",
    "zh-CN": "forward 后释放已 gather 的 parameters。想最大化节省 VRAM 时强烈建议开启。",
  },
  "Uses FSDP's native activation checkpointing. Can be used alongside standard gradient checkpointing for maximum VRAM savings.": {
    "zh-TW": "使用 FSDP 原生 activation checkpointing，可和標準 gradient checkpointing 搭配以最大化節省 VRAM。",
    "zh-CN": "使用 FSDP 原生 activation checkpointing，可和标准 gradient checkpointing 搭配以最大化节省 VRAM。",
  },
  "Only rank 0 loads the model from disk; other ranks receive weights via broadcast. Reduces peak system RAM by ~50% during startup. Automatically enables Sync Module States (required by accelerate).": {
    "zh-TW": "只有 rank 0 從硬碟載入模型，其他 ranks 透過 broadcast 接收權重。啟動時可降低約 50% 系統 RAM 峰值，並自動啟用 accelerate 需要的 Sync Module States。",
    "zh-CN": "只有 rank 0 从硬盘加载模型，其他 ranks 通过 broadcast 接收权重。启动时可降低约 50% 系统 RAM 峰值，并自动启用 accelerate 需要的 Sync Module States。",
  },
  "Overlaps parameter all-gather with gradient computation in the backward pass.": {
    "zh-TW": "在 backward pass 中重疊 parameter all-gather 與 gradient 計算。",
    "zh-CN": "在 backward pass 中重叠 parameter all-gather 与 gradient 计算。",
  },
  "BACKWARD_PRE": { "zh-TW": "BACKWARD_PRE", "zh-CN": "BACKWARD_PRE" },
  "BACKWARD_POST": { "zh-TW": "BACKWARD_POST", "zh-CN": "BACKWARD_POST" },
  "BACKWARD_PRE (Recommended)": { "zh-TW": "BACKWARD_PRE（建議）", "zh-CN": "BACKWARD_PRE（建议）" },
  "None (No Prefetch)": { "zh-TW": "None（不 Prefetch）", "zh-CN": "None（不 Prefetch）" },
  "Pre-fetches the next layer's parameters during the forward pass to overlap communication with computation. Safe for DiT/UNet (static graphs). Speeds up forward pass at a small VRAM cost.": {
    "zh-TW": "forward pass 時預先抓取下一層 parameters，讓通訊與計算重疊。對 DiT/UNet（static graphs）安全，會用少量 VRAM 換取 forward 加速。",
    "zh-CN": "forward pass 时预先抓取下一层 parameters，让通信与计算重叠。对 DiT/UNet（static graphs）安全，会用少量 VRAM 换取 forward 加速。",
  },
  "Required for LoRA training. Keeps original parameter references so FSDP handles mixed frozen/trainable parameters (frozen UNet + LoRA hooks) correctly. Disable only if you know the entire model has uniform requires_grad.": {
    "zh-TW": "LoRA 訓練需要。保留原始 parameter references，讓 FSDP 正確處理凍結/可訓練混合參數（凍結 UNet + LoRA hooks）。只有確定整個模型 requires_grad 一致時才關閉。",
    "zh-CN": "LoRA 训练需要。保留原始 parameter references，让 FSDP 正确处理冻结/可训练混合参数（冻结 UNet + LoRA hooks）。只有确定整个模型 requires_grad 一致时才关闭。",
  },
  "Prevents too many simultaneous all-gather ops from piling up, reducing CUDA malloc retries and potential OOM spikes. Recommended on.": {
    "zh-TW": "避免太多 all-gather 同時堆積，降低 CUDA malloc 重試與潛在 OOM 尖峰。建議開啟。",
    "zh-CN": "避免太多 all-gather 同时堆积，降低 CUDA malloc 重试与潜在 OOM 峰值。建议开启。",
  },
  "Determines how FSDP clusters parameters into sharded units. TRANSFORMER_BASED_WRAP is recommended for large models.": {
    "zh-TW": "決定 FSDP 如何把 parameters 分組成 sharded units。大型模型建議使用 TRANSFORMER_BASED_WRAP。",
    "zh-CN": "决定 FSDP 如何把 parameters 分组成 sharded units。大型模型建议使用 TRANSFORMER_BASED_WRAP。",
  },
  "Determines how FSDP2 groups parameters into sharded units. TRANSFORMER_BASED_WRAP is recommended for large DiT/UNet models: each transformer block is all-gathered and reduce-scattered independently, enabling communication/computation overlap.": {
    "zh-TW": "決定 FSDP2 如何把 parameters 分組成 sharded units。大型 DiT/UNet 建議 TRANSFORMER_BASED_WRAP：每個 transformer block 獨立 all-gather/reduce-scatter，可重疊通訊與計算。",
    "zh-CN": "决定 FSDP2 如何把 parameters 分组成 sharded units。大型 DiT/UNet 建议 TRANSFORMER_BASED_WRAP：每个 transformer block 独立 all-gather/reduce-scatter，可重叠通信与计算。",
  },
  "Exact class name of the transformer block to wrap per-layer. Required for TRANSFORMER_BASED_WRAP.": {
    "zh-TW": "要逐層 wrap 的 transformer block 精確 class 名稱。TRANSFORMER_BASED_WRAP 必填。",
    "zh-CN": "要逐层 wrap 的 transformer block 精确 class 名称。TRANSFORMER_BASED_WRAP 必填。",
  },
  "The exact class name of the transformer block. Required for TRANSFORMER_BASED_WRAP.": {
    "zh-TW": "transformer block 的精確 class 名稱。TRANSFORMER_BASED_WRAP 必填。",
    "zh-CN": "transformer block 的精确 class 名称。TRANSFORMER_BASED_WRAP 必填。",
  },
  "Only shards modules with at least this many parameters. Default is 100M (1e8).": {
    "zh-TW": "只有參數量達到此門檻的 modules 才會 shard。預設 100M（1e8）。",
    "zh-CN": "只有参数量达到此阈值的 modules 才会 shard。默认 100M（1e8）。",
  },
  "FSDP2 vs FSDP1:": { "zh-TW": "FSDP2 與 FSDP1：", "zh-CN": "FSDP2 与 FSDP1：" },
  "FSDP2 uses": { "zh-TW": "FSDP2 使用", "zh-CN": "FSDP2 使用" },
  "This is not available on Windows.": { "zh-TW": "Windows 無法使用此功能。", "zh-CN": "Windows 无法使用此功能。" },
  "(PyTorch ≥ 2.4). Parameters are always exposed as originals (no FlatParameter), making it compatible with LoRA. Backward prefetch, forward prefetch, and limit-all-gathers are handled automatically. Sharding strategy is replaced by the single Reshard After Forward toggle.": {
    "zh-TW": "（PyTorch ≥ 2.4）。Parameters 會一直以原始形式暴露（沒有 FlatParameter），因此相容 LoRA。Backward prefetch、forward prefetch、limit-all-gathers 會自動處理。Sharding strategy 會改由單一 Reshard After Forward 開關控制。",
    "zh-CN": "（PyTorch ≥ 2.4）。Parameters 会一直以原始形式暴露（没有 FlatParameter），因此兼容 LoRA。Backward prefetch、forward prefetch、limit-all-gathers 会自动处理。Sharding strategy 会改由单一 Reshard After Forward 开关控制。",
  },
  "When enabled (default), frees unsharded parameters after each forward pass and re-all-gathers them in backward. Maximizes VRAM savings. Disable to keep parameters unsharded between forward and backward — saves one backward all-gather at the cost of higher VRAM.": {
    "zh-TW": "啟用時（預設）會在每次 forward 後釋放 unsharded parameters，並在 backward 重新 all-gather。這最省 VRAM。關閉後會在 forward/backward 間保留 unsharded parameters，可少一次 backward all-gather，但會使用更多 VRAM。",
    "zh-CN": "启用时（默认）会在每次 forward 后释放 unsharded parameters，并在 backward 重新 all-gather。这最省 VRAM。关闭后会在 forward/backward 间保留 unsharded parameters，可少一次 backward all-gather，但会使用更多 VRAM。",
  },
  "Frees intermediate activations during forward and recomputes them in backward. Reduces VRAM at the cost of extra compute. Combine with Reshard After Forward for maximum VRAM savings.": {
    "zh-TW": "forward 時釋放中間 activations，並在 backward 重新計算。用額外計算換取 VRAM 節省；搭配 Reshard After Forward 最省。",
    "zh-CN": "forward 时释放中间 activations，并在 backward 重新计算。用额外计算换取 VRAM 节省；搭配 Reshard After Forward 最省。",
  },
  "Only rank 0 loads the model checkpoint; other ranks receive weights via broadcast. Reduces peak system RAM by ~50% during startup.": {
    "zh-TW": "只有 rank 0 載入模型 checkpoint，其他 ranks 透過 broadcast 接收權重。啟動時可降低約 50% 系統 RAM 峰值。",
    "zh-CN": "只有 rank 0 加载模型 checkpoint，其他 ranks 通过 broadcast 接收权重。启动时可降低约 50% 系统 RAM 峰值。",
  },
  "Moves sharded parameters and gradients to CPU when not in use. Drastically reduces VRAM at the cost of H2D/D2H copy overhead per step.": {
    "zh-TW": "未使用時把 sharded parameters 與 gradients 移到 CPU。可大幅降低 VRAM，但每 step 會增加 H2D/D2H copy 成本。",
    "zh-CN": "未使用时把 sharded parameters 与 gradients 移到 CPU。可大幅降低 VRAM，但每 step 会增加 H2D/D2H copy 成本。",
  },
  "Loads model to VRAM to save system RAM.": {
    "zh-TW": "將模型載入 VRAM 以節省系統 RAM。",
    "zh-CN": "将模型加载到 VRAM 以节省系统 RAM。",
  },
  "Train at each resolution sequentially (low → high) instead of mixing them. Requires at least 2 resolutions above. Works with both epochs and max steps.": {
    "zh-TW": "依序訓練各解析度（低 → 高），而不是混合訓練。上方至少需要 2 個解析度，epochs 與 max steps 都可用。",
    "zh-CN": "依序训练各分辨率（低 → 高），而不是混合训练。上方至少需要 2 个分辨率，epochs 与 max steps 都可用。",
  },
  "Each fraction is the portion of total steps for that resolution. Must sum to 1.0.": {
    "zh-TW": "每個比例代表該解析度佔總 steps 的比例，總和必須為 1.0。",
    "zh-CN": "每个比例代表该分辨率占总 steps 的比例，总和必须为 1.0。",
  },
  "Use image alpha channel as loss mask. Images without alpha train normally.": {
    "zh-TW": "使用圖片 alpha channel 作為 loss mask。沒有 alpha 的圖片會正常訓練。",
    "zh-CN": "使用图片 alpha channel 作为 loss mask。没有 alpha 的图片会正常训练。",
  },
  "Higher = more capacity, more VRAM.": {
    "zh-TW": "數值越高容量越大，也會使用更多 VRAM。",
    "zh-CN": "数值越高容量越大，也会使用更多 VRAM。",
  },
  "Scaling factor. Usually same as dim.": {
    "zh-TW": "縮放係數，通常與 dim 相同。",
    "zh-CN": "缩放系数，通常与 dim 相同。",
  },
  "Gradient tensors become views into the all-reduce communication buffer instead of copies. Reduces peak memory and eliminates one copy per step. Safe to always enable for DDP.": {
    "zh-TW": "Gradient tensor 會成為 all-reduce 通訊 buffer 的 view，而不是複製。可降低記憶體峰值並省掉每 step 一次 copy。DDP 可放心開啟。",
    "zh-CN": "Gradient tensor 会成为 all-reduce 通信 buffer 的 view，而不是复制。可降低内存峰值并省掉每 step 一次 copy。DDP 可放心开启。",
  },
  "Tells DDP the computation graph is identical every step (same layers, same order). Enables more aggressive comm/compute overlap. Safe for standard UNet/DiT architectures.": {
    "zh-TW": "告訴 DDP 每個 step 的計算圖都相同（相同 layers、相同順序），可更積極重疊通訊與計算。標準 UNet/DiT 架構可安全使用。",
    "zh-CN": "告诉 DDP 每个 step 的计算图都相同（相同 layers、相同顺序），可更积极重叠通信与计算。标准 UNet/DiT 架构可安全使用。",
  },
  "Keep the model loaded in VRAM for faster subsequent generations": {
    "zh-TW": "將模型保留在 VRAM 中，讓後續生成更快",
    "zh-CN": "将模型保留在 VRAM 中，让后续生成更快",
  },
  "TP/SP mode always runs tensor parallel and sequence parallel together.": {
    "zh-TW": "TP/SP 模式固定一起執行 tensor parallel 與 sequence parallel。",
    "zh-CN": "TP/SP 模式固定一起执行 tensor parallel 与 sequence parallel。",
  },
  "Throttles in-flight all-gather operations so they don't all queue up at once. Prevents excessive CUDA memory allocation retries caused by too many unsharded parameter buffers being live simultaneously. Recommended to leave enabled.": {
    "zh-TW": "限制進行中的 all-gather，避免一次排隊太多。可避免太多 unsharded parameter buffers 同時存在造成 CUDA 記憶體配置重試。建議保持開啟。",
    "zh-CN": "限制进行中的 all-gather，避免一次排队太多。可避免太多 unsharded parameter buffers 同时存在造成 CUDA 内存分配重试。建议保持开启。",
  },
  "While executing the forward pass through layer N, FSDP simultaneously all-gathers layer N+1's parameters. Reduces forward pass time by overlapping communication with computation. Only safe with static graphs (same layer execution order every step) â€” DiT/UNet architectures qualify.": {
    "zh-TW": "執行第 N 層 forward 時，FSDP 會同時 all-gather 第 N+1 層 parameters。透過重疊通訊與計算縮短 forward 時間。只適用 static graphs（每 step 層順序相同），DiT/UNet 架構符合。",
    "zh-CN": "执行第 N 层 forward 时，FSDP 会同时 all-gather 第 N+1 层 parameters。通过重叠通信与计算缩短 forward 时间。只适用 static graphs（每 step 层顺序相同），DiT/UNet 架构符合。",
  },
  "Overlaps all-gather communication with gradient computation during the backward pass. BACKWARD_PRE fetches the next layer's parameters while computing the current layer's gradients â€” best throughput but uses slightly more peak VRAM. BACKWARD_POST fetches after finishing gradients â€” less overlap but safer. None is sequential and slowest.": {
    "zh-TW": "在 backward pass 中重疊 all-gather 通訊與 gradient 計算。BACKWARD_PRE 會在計算當前層 gradient 時抓下一層 parameters，吞吐最好但 VRAM 峰值略高。BACKWARD_POST 在 gradient 完成後再抓，重疊較少但較保守。None 則循序執行最慢。",
    "zh-CN": "在 backward pass 中重叠 all-gather 通信与 gradient 计算。BACKWARD_PRE 会在计算当前层 gradient 时抓下一层 parameters，吞吐最好但 VRAM 峰值略高。BACKWARD_POST 在 gradient 完成后再抓，重叠较少但较保守。None 则顺序执行最慢。",
  },
  "Keeps references to the original model parameters instead of FSDP's internal flattened FlatParameter. Required when the same FSDP unit contains both frozen and trainable parameters (e.g. frozen UNet with LoRA hooks). Without this, the optimizer and gradient hooks can silently break on mixed requires_grad tensors.": {
    "zh-TW": "保留原始模型 parameters 的 references，而不是使用 FSDP 內部 flatten 後的 FlatParameter。當同一個 FSDP unit 同時包含凍結與可訓練參數（例如 frozen UNet + LoRA hooks）時必須啟用。否則 optimizer 與 gradient hooks 可能在 mixed requires_grad tensors 下悄悄失效。",
    "zh-CN": "保留原始模型 parameters 的 references，而不是使用 FSDP 内部 flatten 后的 FlatParameter。当同一个 FSDP unit 同时包含冻结与可训练参数（例如 frozen UNet + LoRA hooks）时必须启用。否则 optimizer 与 gradient hooks 可能在 mixed requires_grad tensors 下悄悄失效。",
  },
  "gives the best throughput by fetching the next layer while computing current gradients. Costs a small VRAM peak.": {
    "zh-TW": "會在計算目前 gradients 時抓取下一層，因此吞吐最好，但 VRAM 峰值會稍微增加。",
    "zh-CN": "会在计算当前 gradients 时抓取下一层，因此吞吐最好，但 VRAM 峰值会稍微增加。",
  },
});

function getInitialLanguage() {
  const saved = localStorage.getItem(I18N_STORAGE_KEY);
  if (SUPPORTED_LANGUAGES.includes(saved)) return saved;
  return "zh-TW";
}

let currentLanguage = getInitialLanguage();

function translatePhrase(text) {
  if (currentLanguage === "en") return text;
  return UI_TRANSLATIONS[text]?.[currentLanguage] || text;
}

function isRichTextFragment(node) {
  const parent = node.parentElement;
  if (!parent) return false;
  if (!parent.querySelector("a, strong, em, code")) return false;
  const fullText = parent.textContent.replace(/\s+/g, " ").trim();
  return fullText && fullText !== node.nodeValue.trim() && !UI_TRANSLATIONS[fullText];
}

function applyI18nTextNode(node) {
  const raw = node.nodeValue;
  if (!raw || !raw.trim()) return;
  if (isRichTextFragment(node)) return;
  if (!originalTextNodes.has(node)) originalTextNodes.set(node, raw);
  const original = originalTextNodes.get(node);
  const leading = original.match(/^\s*/)?.[0] || "";
  const trailing = original.match(/\s*$/)?.[0] || "";
  const core = original.trim();
  node.nodeValue = leading + translatePhrase(core) + trailing;
}

function applyI18nAttributes(el) {
  ["placeholder", "title"].forEach((attr) => {
    if (!el.hasAttribute(attr)) return;
    let attrMap = originalAttrs.get(el);
    if (!attrMap) {
      attrMap = {};
      originalAttrs.set(el, attrMap);
    }
    if (!attrMap[attr]) attrMap[attr] = el.getAttribute(attr);
    el.setAttribute(attr, translatePhrase(attrMap[attr]));
  });
}

function applyI18n(root = document.body) {
  document.documentElement.lang = currentLanguage;
  const switcher = $("ui-language");
  if (switcher) switcher.value = currentLanguage;

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      if (["SCRIPT", "STYLE", "TEXTAREA"].includes(parent.tagName)) return NodeFilter.FILTER_REJECT;
      return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(applyI18nTextNode);
  root.querySelectorAll?.("[placeholder], [title]").forEach(applyI18nAttributes);
}

function setLanguage(lang) {
  currentLanguage = SUPPORTED_LANGUAGES.includes(lang) ? lang : "zh-TW";
  localStorage.setItem(I18N_STORAGE_KEY, currentLanguage);
  applyI18n();
  updateGPUActivity();
  updateGenGPULabel();
  if (currentJob) checkTensorBoard();
}

function refreshI18n() {
  if (currentLanguage !== "en") applyI18n();
}
function getNetworkModuleLearningRate(moduleName) {
  return NETWORK_MODULE_PRESETS[moduleName]?.learningRate || "5e-4";
}
// ==========================================
//  API
// ==========================================
// Deletion API
async function deleteSamples(paths) {
  if (!currentJob) return;
  try {
    for (const fullPath of paths) {
      // Path is /api/jobs/:name/samples/samples/filename.png
      const parts = fullPath.split("/samples/");
      const relPath = parts[parts.length - 1];
      await fetch(`/api/jobs/${currentJob}/samples/${relPath}`, {
        method: "DELETE",
      });
    }
    // Remove from local state
    paths.forEach((p) => sampleState.selectedPaths.delete(p));
    // Refresh UI
    loadSamples();
  } catch (err) {
    console.error("Delete failed", err);
    showToast("Error deleting samples", "danger");
  }
}
async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const err = await res.json();
      msg = err.error || msg;
    } catch (_) { }
    throw new Error(msg);
  }
  return res.json();
}
// ==========================================
//  WebSocket
// ==========================================
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}`);
  ws.onopen = () => {
    if (currentJob) {
      ws.send(JSON.stringify({ type: "subscribe", job: currentJob }));
    }
  };
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      // hw_stats is global (not job-specific)
      if (msg.type === "hw_stats") {
        updateHwMonitor(msg.data);
        return;
      }
      if (msg.job !== currentJob) return;
      if (msg.type === "log") {
        appendConsole(msg.data);
      } else if (msg.type === "status") {
        if (msg.data === "generating") return; // Ignore generation status for Training button
        updateRunningState(msg.data === "running");
      }
    } catch (e) { }
  };
  ws.onclose = () => {
    setTimeout(connectWS, 3000);
  };
}
function subscribeToJob(jobName) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "subscribe", job: jobName }));
  }
}
// ==========================================
//  Hardware Monitor
// ==========================================
function formatHwBytes(bytes) {
  if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(1) + " GB";
  if (bytes >= 1048576) return (bytes / 1048576).toFixed(0) + " MB";
  return bytes + " B";
}
function getTempClass(temp) {
  if (temp >= 80) return "hw-temp-hot";
  if (temp >= 65) return "hw-temp-warm";
  return "hw-temp-cool";
}
function updateHwMonitor(stats) {
  const container = document.getElementById("hw-stats-container");
  if (!container) return;
  const cpuPct = Math.max(0, Math.min(100, stats.cpu || 0));
  const ramPct = stats.ram
    ? Math.round((stats.ram.used / stats.ram.total) * 100)
    : 0;
  const ramUsed = stats.ram ? formatHwBytes(stats.ram.used) : "?";
  const ramTotal = stats.ram ? formatHwBytes(stats.ram.total) : "?";
  let html = "";
  const cpuTempHtml =
    stats.cpuTemp != null
      ? `<span class="hw-temp ${getTempClass(stats.cpuTemp)}">${stats.cpuTemp}°C</span>`
      : "";
  // System section (CPU + RAM)
  html += `<div class="hw-section">
        <div class="hw-section-header">
            <span class="hw-section-title">System</span>
            ${cpuTempHtml}
        </div>
        <div class="hw-row">
            <span class="hw-metric-label">CPU</span>
            <div class="hw-bar-wrap"><div class="hw-bar" style="width:${cpuPct}%"></div></div>
            <span class="hw-metric-value">${cpuPct}%</span>
        </div>
        <div class="hw-row">
            <span class="hw-metric-label">RAM</span>
            <div class="hw-bar-wrap"><div class="hw-bar hw-bar-ram" style="width:${ramPct}%"></div></div>
            <span class="hw-metric-value">${ramPct}% &nbsp;${ramUsed} / ${ramTotal}</span>
        </div>
    </div>`;
  // GPU sections
  if (stats.gpus && stats.gpus.length > 0) {
    stats.gpus.forEach((gpu) => {
      const gpuPct = Math.max(0, Math.min(100, gpu.util || 0));
      const vramPct =
        gpu.memTotal > 0 ? Math.round((gpu.memUsed / gpu.memTotal) * 100) : 0;
      const vramUsed = (gpu.memUsed / 1024).toFixed(1);
      const vramTotal = (gpu.memTotal / 1024).toFixed(1);
      const tempClass = getTempClass(gpu.temp);
      const activeClass = gpu.activity ? " hw-active" : "";
      const activityBadge = gpu.activity
        ? `<span class="hw-activity-badge">${gpu.activity}</span>`
        : "";
      const powerPct = gpu.powerLimit > 0 ? Math.round((gpu.powerDraw / gpu.powerLimit) * 100) : 0;
      const powerLabel = gpu.powerLimit > 0 ? `${gpu.powerDraw}W / ${gpu.powerLimit}W` : `${gpu.powerDraw}W`;
      html += `<div class="hw-section hw-section-gpu${activeClass}">
                <div class="hw-section-header">
                    <span class="hw-section-title">GPU ${gpu.index}</span>
                    <span class="hw-temp ${tempClass}">${gpu.temp}°C</span>
                    ${activityBadge}
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">Core</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-gpu" style="width:${gpuPct}%"></div></div>
                    <span class="hw-metric-value">${gpuPct}%</span>
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">VRAM</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-vram" style="width:${vramPct}%"></div></div>
                    <span class="hw-metric-value">${vramPct}% &nbsp;${vramUsed} / ${vramTotal} GB</span>
                </div>
                <div class="hw-row">
                    <span class="hw-metric-label">Power</span>
                    <div class="hw-bar-wrap"><div class="hw-bar hw-bar-power" style="width:${powerPct}%"></div></div>
                    <span class="hw-metric-value">${powerLabel}</span>
                </div>
            </div>`;
    });
  }
  container.innerHTML = html;
  // Compact bar (collapsed view)
  const compact = document.getElementById("hw-compact-bar");
  if (compact) {
    let ch = `<div class="hw-compact-item">
            <span class="hw-compact-label">CPU</span>
            <span>${cpuPct}%${stats.cpuTemp != null ? ` · <span class="hw-temp ${getTempClass(stats.cpuTemp)}">${stats.cpuTemp}°C</span>` : ""}</span>
        </div>
        <div class="hw-compact-item">
            <span class="hw-compact-label">RAM</span>
            <span>${ramUsed} / ${ramTotal}</span>
        </div>`;
    if (stats.gpus && stats.gpus.length > 0) {
      stats.gpus.forEach((gpu) => {
        const gpuPct = Math.max(0, Math.min(100, gpu.util || 0));
        const vramUsed = (gpu.memUsed / 1024).toFixed(1);
        const vramTotal = (gpu.memTotal / 1024).toFixed(1);
        const tempClass = getTempClass(gpu.temp);
        const badge = gpu.activity
          ? ` <span class="hw-activity-badge">${gpu.activity}</span>`
          : "";
        ch += `<div class="hw-compact-sep"></div>
                <div class="hw-compact-item">
                    <span class="hw-compact-label">GPU ${gpu.index}</span>
                    <span>${gpuPct}% · ${vramUsed}/${vramTotal}GB · ${gpu.powerDraw}W · <span class="hw-temp ${tempClass}">${gpu.temp}°C</span>${badge}</span>
                </div>`;
      });
    }
    compact.innerHTML = ch;
  }
}
// Keep tab-content padding in sync with monitor height + handle collapse
(function initHwMonitorResize() {
  const monitor = document.getElementById("hw-monitor");
  const toggleBtn = document.getElementById("hw-toggle");
  if (!monitor) return;
  function syncToggleArrow(isCollapsed) {
    if (toggleBtn)
      toggleBtn.textContent = isCollapsed ? "▲  Hardware Monitor" : "▼";
  }
  // Restore collapsed state
  const collapsed = localStorage.getItem("hw_monitor_collapsed") === "true";
  if (collapsed) monitor.classList.add("hw-collapsed");
  syncToggleArrow(collapsed);
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const isNowCollapsed = monitor.classList.toggle("hw-collapsed");
      localStorage.setItem("hw_monitor_collapsed", isNowCollapsed);
      syncToggleArrow(isNowCollapsed);
    });
  }
  if (!window.ResizeObserver) return;
  new ResizeObserver(() => {
    document.documentElement.style.setProperty(
      "--hw-bar-height",
      monitor.offsetHeight + 6 + "px",
    );
  }).observe(monitor);
})();
// ==========================================
//  Job List
// ==========================================
async function loadJobs() {
  const jobs = await api("/api/jobs");
  jobListEl.innerHTML = "";
  if (jobs.length === 0) {
    jobListEl.innerHTML =
      '<div style="padding:20px;text-align:center;color:var(--text-muted)">No jobs yet</div>';
    refreshI18n();
    return jobs;
  }
  jobs.forEach((job) => {
    const el = document.createElement("div");
    el.className = `job-item${job.name === currentJob ? " active" : ""}${job.running ? " running" : ""}`;
    el.innerHTML = `
            <div class="status-dot"></div>
            <span class="job-name">${job.name}</span>
        `;
    el.addEventListener("click", () => selectJob(job.name));
    jobListEl.appendChild(el);
  });
  refreshI18n();
  return jobs;
}
function getExistingJobNames() {
  return new Set(
    Array.from(document.querySelectorAll(".job-name"))
      .map((el) => el.textContent.trim())
      .filter(Boolean),
  );
}
function nextVersionedName(baseName, startVersion = 1) {
  const existing = getExistingJobNames();
  const cleanBase = String(baseName || "my_job")
    .trim()
    .replace(/_v\d+$/i, "")
    .replace(/\s+v\d+$/i, "")
    || "my_job";
  let version = startVersion;
  let candidate = `${cleanBase}_v${version}`;
  while (existing.has(candidate)) {
    version += 1;
    candidate = `${cleanBase}_v${version}`;
  }
  return candidate;
}
function nextCloneName(name) {
  const raw = String(name || "my_job").trim() || "my_job";
  const match = raw.match(/^(.*?)(?:_v|\s+v)(\d+)$/i);
  if (match) return nextVersionedName(match[1], Number(match[2]) + 1);
  return nextVersionedName(raw, 2);
}
function clearCurrentJobSelection() {
  currentJob = null;
  isDirty = false;
  localStorage.removeItem("lastJob");
  if (samplesPollTimer) {
    clearInterval(samplesPollTimer);
    samplesPollTimer = null;
  }
  $("btn-save").classList.add("hidden");
  $("btn-discard").classList.add("hidden");
  jobEditor.classList.add("hidden");
  emptyState.classList.remove("hidden");
}
async function selectJob(name) {
  if (currentJob) savePromptTransientSettings();
  if (isDirty && !confirm("Unsaved changes. Switch anyway?")) return;
  isDirty = false;
  currentJob = name;
  if (samplesPollTimer) {
    clearInterval(samplesPollTimer);
    samplesPollTimer = null;
  }
  localStorage.setItem("lastJob", name);
  jobTitle.textContent = name;
  emptyState.classList.add("hidden");
  jobEditor.classList.remove("hidden");
  try {
    // Load available GPUs
    await loadGPUs();
    await loadGenGPUs();
    // Load job data
    const data = await api(`/api/jobs/${name}`);
    populateConfig(data.config);
    populateDataset(data.dataset);
    // Load prompts
    await loadPrompts();
    // Check run status
    const status = await api(`/api/jobs/${name}/train/status`);
    updateRunningState(status.running);
    // Set default negative prompt if no saved value exists for this job
    const savedTransient = localStorage.getItem(`prompt_transient_${name}`);
    if (!savedTransient || !JSON.parse(savedTransient).negative_prompt) {
      $("global-negative-prompt").value = DEFAULT_NEGATIVE_PROMPT;
    }
    // Save initial state for dirty checking
    lastSavedConfig = JSON.parse(JSON.stringify(gatherConfig()));
    lastSavedDataset = JSON.parse(JSON.stringify(gatherDataset()));
    lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
    lastSavedNegativePrompt = $("global-negative-prompt").value;
    // Subscribe WS
    subscribeToJob(name);
    // Reset console
    consoleOutput.textContent = "Waiting for training to start...";
    // Reset save button
    $("btn-save").classList.add("hidden");
    $("btn-discard").classList.add("hidden");
    // Refresh job list highlight
    loadJobs();
    // Load samples
    loadSamples();
    loadCheckpoints();
    loadPromptTransientSettings();
  } catch (err) {
    console.error(`Failed to load job "${name}":`, err);
    showToast(`Failed to load job: ${err.message}`, "danger");
    clearCurrentJobSelection();
  }
}
function updateRunningState(running) {
  $("btn-run").classList.toggle("hidden", running);
  $("btn-stop").classList.toggle("hidden", !running);
  // Update sidebar dot
  document.querySelectorAll(".job-item").forEach((el) => {
    const name = el.querySelector(".job-name").textContent;
    if (name === currentJob) {
      el.classList.toggle("running", running);
    }
  });
}
// ==========================================
//  Config UI Mapping
// ==========================================
function populateConfig(config) {
  const t = config.training_arguments || {};
  const n = config.network_arguments || {};
  const a = config.anima_arguments || {};
  const networkModule = n.network_module || "networks.krona";
  // Training
  $("cfg-learning-rate").value = t.learning_rate || getNetworkModuleLearningRate(networkModule);
  $("cfg-text-encoder-lr").value = t.text_encoder_lr ?? "0";
  $("cfg-optimizer").value = t.optimizer_type || "library.came.CAME";
  $("cfg-lr-scheduler").value = t.lr_scheduler || "constant_with_warmup";
  $("cfg-lr-warmup").value = t.lr_warmup_steps ?? 0.1;
  $("cfg-lr-scheduler-cycles").value = t.lr_scheduler_num_cycles ?? 1;
  $("cfg-lr-min-ratio").value = t.lr_scheduler_min_lr_ratio ?? 0;
  $("cfg-seed").value = t.seed ?? 42;
  // Extract weight decay
  let wdValue = "";
  let decoupleValue = true;
  if (t.optimizer_args && Array.isArray(t.optimizer_args)) {
    const wdArg = t.optimizer_args.find((arg) =>
      String(arg).startsWith("weight_decay="),
    );
    if (wdArg) {
      wdValue = wdArg.split("=")[1];
    }
    const decoupleArg = t.optimizer_args.find((arg) =>
      String(arg).startsWith("decouple="),
    );
    if (decoupleArg) {
      decoupleValue = decoupleArg.split("=")[1].toLowerCase() === "true";
    }
  }
  $("cfg-weight-decay").value = wdValue;
  $("cfg-decouple").checked = decoupleValue;
  // Update conditional visibility
  updateOptimizerOptions();
  updateLrSchedulerOptions();
  const maxSteps = t.max_train_steps;
  const isSteps = maxSteps && maxSteps > 0;
  document.querySelector(
    `input[name="duration-unit"][value="${isSteps ? "steps" : "epochs"}"]`,
  ).checked = true;
  updateDurationUnit();
  $("cfg-max-epochs").value = t.max_train_epochs ?? 20;
  $("cfg-save-every").value = t.save_every_n_epochs ?? 1;
  $("cfg-max-steps").value = t.max_train_steps ?? 3000;
  $("cfg-save-every-steps").value = t.save_every_n_steps ?? 200;
  $("cfg-save-last-steps").value = t.save_last_n_steps ?? "";
  $("cfg-save-last-epochs").value = t.save_last_n_epochs ?? "";
  $("cfg-save-state").checked = t.save_state ?? false;
  $("cfg-save-state-end").checked = t.save_state_on_train_end ?? false;
  $("cfg-save-last-steps-state").value = t.save_last_n_steps_state ?? "";
  $("cfg-save-last-epochs-state").value = t.save_last_n_epochs_state ?? "";
  $("cfg-output-name").value = t.output_name || "my_anima_lora";
  $("cfg-save-format").value = t.save_model_as || "safetensors";
  $("cfg-save-precision").value = t.save_precision || "bf16";
  $("cfg-mixed-precision").value = t.mixed_precision || "bf16";
  $("cfg-transformer-dtype").value = t.full_bf16 ? "bfloat16" : t.full_fp16 ? "float16" : "float32";
  $("cfg-workers").value = t.max_data_loader_n_workers ?? 2;
  $("cfg-grad-acc").value = t.gradient_accumulation_steps ?? 1;
  $("cfg-gradient-checkpointing").checked = t.gradient_checkpointing ?? true;
  $("cfg-flash-attn").checked = (t.flash_attn ?? false) || t.attn_mode === "flash";
  $("cfg-torch-compile").checked = t.torch_compile ?? false;
  $("cfg-lowram").checked = t.lowram ?? false;
  $("cfg-blocks-to-swap").value = t.blocks_to_swap ?? 0;
  $("cfg-knn-noise-k").value = t.knn_noise_k ?? 2;
  $("cfg-cep-noise").value = t.cep_noise ?? 0.05;
  // Activation offload mode
  if (t.unsloth_offload_checkpointing) {
    $("cfg-activation-offload").value = "unsloth";
  } else if (t.cpu_offload_checkpointing) {
    $("cfg-activation-offload").value = "cpu";
  } else {
    $("cfg-activation-offload").value = "none";
  }
  updateActivationOffloadUI();
  $("cfg-persistent-workers").checked =
    t.persistent_data_loader_workers ?? true;
  $("cfg-cache-latents").checked = t.cache_latents_to_disk ?? true;
  $("cfg-vae-batch").value = t.vae_batch_size ?? 1;
  $("cfg-vae-chunk-size").value = t.vae_chunk_size ?? 64;
  $("cfg-vae-disable-cache").checked = t.vae_disable_cache ?? false;
  $("cfg-cache-te").checked = t.cache_text_encoder_outputs_to_disk ?? true;
  $("cfg-disable-bucket-shuffle").checked = t.disable_bucket_shuffle ?? false;
  // Progressive resolution schedule
  if (t.resolution_schedule) {
    $("cfg-progressive-reso").checked = true;
    $("progressive-reso-panel").classList.remove("hidden");
    // Parse schedule string and populate fraction inputs after rendering phases
    window._pendingProgressiveSchedule = t.resolution_schedule;
  } else {
    $("cfg-progressive-reso").checked = false;
    $("progressive-reso-panel").classList.add("hidden");
    window._pendingProgressiveSchedule = null;
  }
  $("cfg-use-cuda-direct").checked = t.use_cuda_direct ?? false;
  $("cfg-ddp-gradient-as-bucket-view").checked =
    t.ddp_gradient_as_bucket_view ?? false;
  $("cfg-ddp-static-graph").checked = t.ddp_static_graph ?? false;
  // Multi-GPU mode selector (ddp/fsdp/fsdp2/deepspeed/tp_sp) — backward compat: infer from use_fsdp
  const restoredMode =
    t.multigpu_mode || (t.deepspeed ? "deepspeed" : t.use_fsdp ? "fsdp" : "ddp");
  $("cfg-multigpu-mode").value = restoredMode;
  applyMultiGpuMode(restoredMode);
  // TP/SP options
  if (t.tp_degree) $("cfg-tp-degree").value = t.tp_degree;
  if ($("cfg-tp-backend")) $("cfg-tp-backend").value = t.tp_backend || "auto";
  $("cfg-sequence-parallel").checked = true;
  if ($("cfg-no-fuse-qkv")) $("cfg-no-fuse-qkv").checked = t.no_fuse_qkv ?? false;
  $("cfg-fsdp-sharding-strategy").value = t.fsdp_sharding_strategy || "1";
  $("cfg-fsdp-offload-params").checked = t.fsdp_offload_params ?? false;
  $("cfg-fsdp-reshard-after-forward").checked =
    t.fsdp_reshard_after_forward ?? false;
  $("cfg-fsdp-activation-checkpointing").checked =
    t.fsdp_activation_checkpointing ?? false;
  $("cfg-fsdp-cpu-ram-efficient-loading").checked =
    t.fsdp_cpu_ram_efficient_loading ?? false;
  $("cfg-fsdp-backward-prefetch").value = t.fsdp_backward_prefetch || "";
  $("cfg-fsdp-forward-prefetch").checked = t.fsdp_forward_prefetch ?? false;
  $("cfg-fsdp-use-orig-params").checked = t.fsdp_use_orig_params ?? true;
  $("cfg-fsdp-limit-all-gathers").checked = t.fsdp_limit_all_gathers ?? true;
  $("cfg-fsdp-auto-wrap-policy").value = t.fsdp_auto_wrap_policy || "NO_WRAP";
  $("cfg-fsdp-min-num-params").value = t.fsdp_min_num_params || "100000000";
  $("cfg-fsdp-layer-to-wrap").value =
    t.fsdp_transformer_layer_cls_to_wrap || "";
  // fsdp-settings is always visible when group-fsdp is shown (mode dropdown controls visibility)
  $("fsdp-layer-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp-auto-wrap-policy").value !== "TRANSFORMER_BASED_WRAP",
  );
  $("fsdp-size-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp-auto-wrap-policy").value !== "SIZE_BASED_WRAP",
  );
  // FSDP2 restore
  $("cfg-fsdp2-reshard-after-forward").checked = t.fsdp2_reshard_after_forward ?? true;
  $("cfg-fsdp2-offload-params").checked = t.fsdp2_offload_params ?? false;
  $("cfg-fsdp2-activation-checkpointing").checked = t.fsdp2_activation_checkpointing ?? false;
  $("cfg-fsdp2-cpu-ram-efficient-loading").checked = t.fsdp2_cpu_ram_efficient_loading ?? false;
  $("cfg-fsdp2-auto-wrap-policy").value = t.fsdp2_auto_wrap_policy || "NO_WRAP";
  $("cfg-fsdp2-min-num-params").value = t.fsdp2_min_num_params || "100000000";
  $("cfg-fsdp2-layer-to-wrap").value = t.fsdp2_transformer_layer_cls_to_wrap || "";
  $("fsdp2-layer-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp2-auto-wrap-policy").value !== "TRANSFORMER_BASED_WRAP",
  );
  $("fsdp2-size-wrap-group").classList.toggle(
    "hidden",
    $("cfg-fsdp2-auto-wrap-policy").value !== "SIZE_BASED_WRAP",
  );
  // DeepSpeed restore
  $("cfg-ds-zero-stage").value = String(t.zero_stage ?? 2);
  $("cfg-ds-offload-optimizer-device").value =
    t.offload_optimizer_device || "none";
  $("cfg-ds-offload-optimizer-nvme-path").value =
    t.offload_optimizer_nvme_path || "";
  $("cfg-ds-offload-param-device").value = t.offload_param_device || "none";
  $("cfg-ds-offload-param-nvme-path").value =
    t.offload_param_nvme_path || "";
  $("cfg-ds-zero3-init-flag").checked = t.zero3_init_flag ?? false;
  $("cfg-ds-zero3-save-16bit-model").checked =
    t.zero3_save_16bit_model ?? false;
  $("cfg-ds-fp16-master-weights-and-gradients").checked =
    t.fp16_master_weights_and_gradients ?? false;
  updateDeepspeedOffloadUI();
  $("cfg-step-profile").checked = t.step_profile ?? false;
  $("cfg-profile-microbatch").checked = t.profile_microbatch ?? false;
  $("cfg-profile-microbatch-group").style.display = (t.step_profile ?? false) ? "" : "none";
  // Check GPU boxes based on config
  const savedIds = (config.gpu_ids || "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s);
  const savedGpuId = savedIds[0] || null;
  Array.from(document.querySelectorAll('input[name="gpu-select"]')).forEach((cb, index) => {
    cb.checked = savedGpuId ? cb.value === savedGpuId : index === 0;
    // Also update card class if parent exists
    const card = cb.closest(".gpu-card");
    if (card) card.classList.toggle("selected", cb.checked);
  });
  updateMultiGPUUI();
  const sampleInterval = t.sample_every_n_epochs;
  const sampleIntervalSteps = t.sample_every_n_steps;
  const sampleArgs = config.sample_arguments || {};
  const enableSampling =
    (sampleInterval !== null && sampleInterval > 0) ||
    (sampleIntervalSteps !== null && sampleIntervalSteps > 0);
  $("cfg-enable-sampling").checked = enableSampling;
  $("cfg-sample-every").value = sampleInterval || 1;
  $("cfg-sample-every-steps").value = sampleIntervalSteps || 100;
  $("cfg-sample-at-first").checked = t.sample_at_first ?? sampleArgs.sample_at_first ?? false;
  $("group-sample-every").classList.toggle("hidden", !enableSampling);
  // Anima
  $("cfg-timestep-method").value =
    a.timestep_sampling ||
    (a.timestep_sample_method === "logit_normal" ? "sigmoid" : a.timestep_sample_method) ||
    "sigmoid";
  $("cfg-flow-shift").value = a.discrete_flow_shift ?? 1.0;
  $("cfg-weighting-scheme").value = a.weighting_scheme || "uniform";
  $("cfg-sigmoid-scale").value = a.sigmoid_scale ?? 1.0;
  $("cfg-qwen3-max-token-length").value = a.qwen3_max_token_length ?? 512;
  $("cfg-t5-max-token-length").value = a.t5_max_token_length ?? 512;
  // Network / Training type
  const trainingType = n.network_module ? "lora" : "full_finetune";
  $("cfg-training-type").value = trainingType;
  updateTrainingTypeUI(trainingType);
  $("cfg-network-module").value = networkModule;
  $("cfg-network-dim").value = n.network_dim ?? 16;
  $("cfg-network-alpha").value = n.network_alpha ?? 16;
  updateNetworkModuleUI();
  $("cfg-unet-only").checked = n.network_train_unet_only ?? true;
  $("cfg-network-weights").value = n.network_weights || "";
  $("cfg-freeze-llm-adapter").checked = t.freeze_llm_adapter ?? true;
  $("cfg-auto-resume").checked = n.auto_resume_last_state ?? false;
  $("cfg-resume").value = n.resume || "";
  $("cfg-resume").disabled = $("cfg-auto-resume").checked;
  $("cfg-network-dropout").value = n.network_dropout ?? 0;
  $("cfg-network-args").value = (n.network_args || []).join(" ");
}
function populateDataset(dataset) {
  const g = dataset.general || {};
  let dArray = [];
  if (Array.isArray(dataset.datasets)) {
    dArray = dataset.datasets;
  } else if (dataset.datasets) {
    dArray = [dataset.datasets];
  }
  if (dArray.length === 0) dArray = [{}];

  const resArray = dArray.map(d => Array.isArray(d.resolution) ? d.resolution[0] : (d.resolution || 1024));
  const batchArray = dArray.map(d => d.batch_size ?? 1);

  $("cfg-resolution").value = resArray.join(", ");
  $("cfg-batch-size").value = batchArray.join(", ");

  const d = dArray[0];
  $("cfg-caption-ext").value = d.caption_extension || ".txt";
  $("cfg-enable-bucket").checked = g.enable_bucket ?? true;
  $("cfg-bucket-no-upscale").checked = g.bucket_no_upscale ?? true;
  $("cfg-min-bucket").value = g.min_bucket_reso ?? 512;
  $("cfg-max-bucket").value = g.max_bucket_reso ?? 2048;
  $("cfg-bucket-steps").value = g.bucket_reso_steps ?? 64;
  // Load Subsets into memory
  let subsetsRaw = d.subsets || [];
  if (!Array.isArray(subsetsRaw)) subsetsRaw = [subsetsRaw];
  // Convert to our internal state format
  currentSubsets = subsetsRaw.map((s) => ({
    image_dir: s.image_dir || "",
    num_repeats: s.num_repeats ?? 1,
    keep_tokens: s.keep_tokens ?? 5,
    flip_aug: s.flip_aug ?? false,
    caption_prefix: s.caption_prefix || "",
    caption_dropout_rate: s.caption_dropout_rate ?? 0.0,
    caption_tag_dropout_rate: s.caption_tag_dropout_rate ?? 0.3,
    caption_dropout_every_n_epochs: s.caption_dropout_every_n_epochs ?? 0,
    shuffle_caption: s.shuffle_caption ?? true,
    enable_wildcard: s.enable_wildcard ?? true,
    is_reg: s.is_reg ?? false,
  }));
  // Edge case: if empty, force at least 1
  if (currentSubsets.length === 0) {
    addSubset(false);
  }
  // Alpha mask: check if any subset has it enabled
  $("cfg-alpha-mask").checked = subsetsRaw.length === 0 || subsetsRaw.some((s) => s.alpha_mask === true);
  renderSubsets();
  // Rebuild progressive phase rows now that resolution field is populated
  if ($("cfg-progressive-reso").checked) renderProgressivePhases();
}
function updateOptimizerOptions() {
  const optimizer = $("cfg-optimizer").value;
  const isProdigy =
    optimizer.includes("Prodigy") || optimizer.includes("DAdapt");
  $("group-decouple").classList.toggle("hidden", !isProdigy);
}
function updateLrSchedulerOptions() {
  const scheduler = $("cfg-lr-scheduler").value;
  $("group-lr-scheduler-cycles").classList.toggle(
    "hidden",
    scheduler !== "cosine_with_restarts",
  );
  $("group-lr-min-ratio").classList.toggle(
    "hidden",
    scheduler !== "cosine_with_min_lr",
  );
}
function updateActivationOffloadUI() {
  const offload = $("cfg-activation-offload").value;
  const blocksInput = $("cfg-blocks-to-swap");
  const isOffload = offload !== "none";
  blocksInput.disabled = isOffload;
  if (isOffload) {
    blocksInput.value = 0;
  }
  // Auto-enable gradient checkpointing when offload is selected
  if (isOffload) {
    $("cfg-gradient-checkpointing").checked = true;
  }
}
// Helpers for safe parsing
function safeInt(val, fallback = 0) {
  if (val === "" || val === null || val === undefined) return fallback;
  const p = parseInt(val);
  return isNaN(p) ? fallback : p;
}
function safeFloat(val, fallback = 0.0) {
  if (val === "" || val === null || val === undefined) return fallback;
  const p = parseFloat(val);
  return isNaN(p) ? fallback : p;
}
function optionalPositiveInt(id) {
  const raw = $(id).value;
  if (raw === "" || raw === null || raw === undefined) return undefined;
  const parsed = safeInt(raw);
  return parsed > 0 ? parsed : undefined;
}
function gatherConfig() {
  const unit = document.querySelector(
    'input[name="duration-unit"]:checked',
  ).value;
  const isEpochs = unit === "epochs";
  const enableSampling = $("cfg-enable-sampling").checked;
  const isMultiGpu = false;
  const multiGpuMode = $("cfg-multigpu-mode").value;
  const optimizerArgs = [];
  const wdValue = $("cfg-weight-decay").value;
  if (wdValue !== "") {
    optimizerArgs.push(`weight_decay=${wdValue}`);
  }
  if (!$("group-decouple").classList.contains("hidden")) {
    const isDecoupled = $("cfg-decouple").checked;
    optimizerArgs.push(`decouple=${isDecoupled ? "True" : "False"}`);
  }
  const config = {
    training_arguments: {
      output_name: $("cfg-output-name").value,
      save_model_as: $("cfg-save-format").value,
      max_train_epochs: isEpochs
        ? safeInt($("cfg-max-epochs").value)
        : undefined,
      save_every_n_epochs: isEpochs
        ? safeInt($("cfg-save-every").value)
        : undefined,
      save_last_n_epochs: optionalPositiveInt("cfg-save-last-epochs"),
      save_last_n_epochs_state: optionalPositiveInt("cfg-save-last-epochs-state"),
      sample_every_n_epochs:
        isEpochs && enableSampling
          ? safeInt($("cfg-sample-every").value)
          : undefined,
      max_train_steps: !isEpochs
        ? safeInt($("cfg-max-steps").value)
        : undefined,
      save_every_n_steps: !isEpochs
        ? safeInt($("cfg-save-every-steps").value)
        : undefined,
      save_last_n_steps: optionalPositiveInt("cfg-save-last-steps"),
      save_last_n_steps_state: optionalPositiveInt("cfg-save-last-steps-state"),
      sample_at_first: $("cfg-sample-at-first").checked ? true : undefined,
      sample_every_n_steps:
        !isEpochs && enableSampling
          ? safeInt($("cfg-sample-every-steps").value)
          : undefined,
      save_state: $("cfg-save-state").checked ? true : undefined,
      save_state_on_train_end: $("cfg-save-state-end").checked ? true : undefined,
      log_with: "tensorboard",
      learning_rate: safeFloat($("cfg-learning-rate").value),
      text_encoder_lr: safeFloat($("cfg-text-encoder-lr").value),
      optimizer_type: $("cfg-optimizer").value,
      optimizer_args: optimizerArgs.length > 0 ? optimizerArgs : undefined,
      lr_scheduler: $("cfg-lr-scheduler").value,
      lr_scheduler_num_cycles:
        $("cfg-lr-scheduler").value === "cosine_with_restarts"
          ? safeInt($("cfg-lr-scheduler-cycles").value)
          : undefined,
      lr_scheduler_min_lr_ratio:
        $("cfg-lr-scheduler").value === "cosine_with_min_lr"
          ? safeFloat($("cfg-lr-min-ratio").value)
          : undefined,
      lr_warmup_steps: safeFloat($("cfg-lr-warmup").value),
      // Hardware
      mixed_precision: $("cfg-mixed-precision").value,
      save_precision: $("cfg-save-precision").value || undefined,
      ...($("cfg-transformer-dtype").value === "bfloat16" ? { full_bf16: true } : {}),
      ...($("cfg-transformer-dtype").value === "float16" ? { full_fp16: true } : {}),
      max_data_loader_n_workers: safeInt($("cfg-workers").value),
      gradient_accumulation_steps: safeInt($("cfg-grad-acc").value),
      max_grad_norm: 1.0,
      train_batch_size: safeInt(($("cfg-batch-size").value || "1").split(",")[0].trim(), 1),
      knn_noise_k: safeInt($("cfg-knn-noise-k").value),
      cep_noise: safeFloat($("cfg-cep-noise").value),
      gradient_checkpointing: $("cfg-gradient-checkpointing").checked,
      flash_attn: $("cfg-flash-attn").checked,
      torch_compile: $("cfg-torch-compile").checked,
      lowram: $("cfg-lowram").checked,
      blocks_to_swap: safeInt($("cfg-blocks-to-swap").value),
      ...($("cfg-activation-offload").value === "cpu" && {
        cpu_offload_checkpointing: true,
      }),
      ...($("cfg-activation-offload").value === "unsloth" && {
        unsloth_offload_checkpointing: true,
      }),
      persistent_data_loader_workers: $("cfg-persistent-workers").checked,
      seed: safeInt($("cfg-seed").value),
      cache_latents: $("cfg-cache-latents").checked,
      cache_latents_to_disk: $("cfg-cache-latents").checked,
      vae_batch_size: safeInt($("cfg-vae-batch").value),
      vae_chunk_size: safeInt($("cfg-vae-chunk-size").value),
      vae_disable_cache: $("cfg-vae-disable-cache").checked,
      cache_text_encoder_outputs_to_disk: $("cfg-cache-te").checked,
      ...($("cfg-disable-bucket-shuffle").checked && {
        disable_bucket_shuffle: true,
      }),
      multigpu_mode: isMultiGpu ? multiGpuMode : "ddp",
      ...(multiGpuMode === "tp_sp" && isMultiGpu
          ? {
              tp_degree: safeInt($("cfg-tp-degree").value) || 2,
              tp_backend: $("cfg-tp-backend")?.value || "auto",
              sequence_parallel: true,
              ...($("cfg-no-fuse-qkv")?.checked ? { no_fuse_qkv: true } : {}),
            }
          : {}),
      ...(multiGpuMode === "deepspeed" && isMultiGpu
        ? {
            deepspeed: true,
            zero_stage: safeInt($("cfg-ds-zero-stage").value, 2),
            offload_optimizer_device:
              $("cfg-ds-offload-optimizer-device").value !== "none"
                ? $("cfg-ds-offload-optimizer-device").value
                : undefined,
            offload_optimizer_nvme_path:
              $("cfg-ds-offload-optimizer-device").value === "nvme"
                ? $("cfg-ds-offload-optimizer-nvme-path").value.trim() ||
                  undefined
                : undefined,
            offload_param_device:
              $("cfg-ds-offload-param-device").value !== "none"
                ? $("cfg-ds-offload-param-device").value
                : undefined,
            offload_param_nvme_path:
              $("cfg-ds-offload-param-device").value === "nvme"
                ? $("cfg-ds-offload-param-nvme-path").value.trim() || undefined
                : undefined,
            zero3_init_flag: $("cfg-ds-zero3-init-flag").checked,
            zero3_save_16bit_model: $("cfg-ds-zero3-save-16bit-model").checked,
            fp16_master_weights_and_gradients: $(
              "cfg-ds-fp16-master-weights-and-gradients",
            ).checked,
          }
        : { deepspeed: false }),
      use_cuda_direct: isMultiGpu ? $("cfg-use-cuda-direct").checked : false,
      ddp_gradient_as_bucket_view: isMultiGpu
        ? $("cfg-ddp-gradient-as-bucket-view").checked
        : false,
      ddp_static_graph: isMultiGpu ? $("cfg-ddp-static-graph").checked : false,
      // FSDP Configs
      use_fsdp: isMultiGpu
        ? multiGpuMode === "fsdp" || multiGpuMode === "fsdp2"
        : false,
      fsdp_sharding_strategy: $("cfg-fsdp-sharding-strategy").value,
      fsdp_offload_params: $("cfg-fsdp-offload-params").checked,
      fsdp_reshard_after_forward: $("cfg-fsdp-reshard-after-forward").checked,
      fsdp_activation_checkpointing: $("cfg-fsdp-activation-checkpointing")
        .checked,
      fsdp_cpu_ram_efficient_loading: $("cfg-fsdp-cpu-ram-efficient-loading")
        .checked,
      fsdp_backward_prefetch: $("cfg-fsdp-backward-prefetch").value,
      fsdp_forward_prefetch: $("cfg-fsdp-forward-prefetch").checked,
      fsdp_use_orig_params: $("cfg-fsdp-use-orig-params").checked,
      fsdp_limit_all_gathers: $("cfg-fsdp-limit-all-gathers").checked,
      fsdp_auto_wrap_policy: $("cfg-fsdp-auto-wrap-policy").value,
      fsdp_min_num_params: safeInt($("cfg-fsdp-min-num-params").value),
      fsdp_transformer_layer_cls_to_wrap: $(
        "cfg-fsdp-layer-to-wrap",
      ).value.trim(),
      // FSDP2 Configs
      fsdp2_reshard_after_forward: $("cfg-fsdp2-reshard-after-forward").checked,
      fsdp2_offload_params: $("cfg-fsdp2-offload-params").checked,
      fsdp2_activation_checkpointing: $("cfg-fsdp2-activation-checkpointing").checked,
      fsdp2_cpu_ram_efficient_loading: $("cfg-fsdp2-cpu-ram-efficient-loading").checked,
      fsdp2_auto_wrap_policy: $("cfg-fsdp2-auto-wrap-policy").value,
      fsdp2_min_num_params: safeInt($("cfg-fsdp2-min-num-params").value),
      fsdp2_transformer_layer_cls_to_wrap: $("cfg-fsdp2-layer-to-wrap").value.trim(),
      // FFT options
      ...($("cfg-training-type").value === "full_finetune" && $("cfg-freeze-llm-adapter").checked
        ? { freeze_llm_adapter: true }
        : {}),
      // Diagnostics
      step_profile: $("cfg-step-profile").checked,
      profile_microbatch: $("cfg-profile-microbatch").checked,
      // Progressive resolution schedule
      ...(() => {
        if (!$("cfg-progressive-reso").checked) return {};
        const resList = ($("cfg-resolution").value || "1024").split(",").map(r => parseInt(r.trim())).filter(Boolean);
        const inputs = document.querySelectorAll(".prog-reso-frac");
        if (inputs.length === 0 || resList.length < 2) return {};
        const parts = resList.map((r, i) => {
          const frac = parseFloat(inputs[i]?.value || 0);
          return `${r}:${frac.toFixed(2)}`;
        });
        return { resolution_schedule: parts.join(",") };
      })(),
    },
    network_arguments: $("cfg-training-type").value === "full_finetune"
      ? {
          auto_resume_last_state: $("cfg-auto-resume").checked,
          ...($("cfg-resume").value && !$("cfg-auto-resume").checked && { resume: $("cfg-resume").value }),
        }
      : {
          network_module: $("cfg-network-module").value,
          network_dim: safeInt($("cfg-network-dim").value),
          network_alpha: safeInt($("cfg-network-alpha").value),
          network_train_unet_only: $("cfg-unet-only").checked,
          ...(safeFloat($("cfg-network-dropout").value) > 0 && {
            network_dropout: safeFloat($("cfg-network-dropout").value),
          }),
          ...($("cfg-network-args").value.trim() && {
            network_args: $("cfg-network-args").value.trim().split(/\s+/),
          }),
          ...($("cfg-network-weights").value && {
            network_weights: $("cfg-network-weights").value,
          }),
          auto_resume_last_state: $("cfg-auto-resume").checked,
          ...($("cfg-resume").value && !$("cfg-auto-resume").checked && { resume: $("cfg-resume").value }),
    },
    anima_arguments: {
      timestep_sampling: $("cfg-timestep-method").value,
      discrete_flow_shift: safeFloat($("cfg-flow-shift").value),
      sigmoid_scale: safeFloat($("cfg-sigmoid-scale").value),
      qwen3_max_token_length: safeInt($("cfg-qwen3-max-token-length").value),
      t5_max_token_length: safeInt($("cfg-t5-max-token-length").value),
      weighting_scheme: $("cfg-weighting-scheme").value,
    },
    gpu_ids:
      Array.from(document.querySelectorAll('input[name="gpu-select"]:checked'))[0]?.value || "",
  };
  return config;
}
function gatherDataset() {
  const res = safeInt($("cfg-resolution").value);
  return {
    general: {
      enable_bucket: $("cfg-enable-bucket").checked,
      bucket_no_upscale: $("cfg-bucket-no-upscale").checked,
      min_bucket_reso: safeInt($("cfg-min-bucket").value),
      max_bucket_reso: safeInt($("cfg-max-bucket").value),
      bucket_reso_steps: safeInt($("cfg-bucket-steps").value),
    },
    datasets: (() => {
      const resStr = $("cfg-resolution").value || "1024";
      const batchStr = $("cfg-batch-size").value || "1";

      const resList = resStr.split(",").map(r => safeInt(r.trim()));
      const batchListRaw = batchStr.split(",").map(b => safeInt(b.trim()));

      return resList.map((r, i) => {
        const b = batchListRaw[i] !== undefined ? batchListRaw[i] : batchListRaw[batchListRaw.length - 1];
        return {
          resolution: [r, r],
          batch_size: b,
          caption_extension: $("cfg-caption-ext").value,
          subsets: currentSubsets.map((s) => {
            const subset = {
              image_dir: s.image_dir,
              num_repeats: safeInt(s.num_repeats),
              keep_tokens: safeInt(s.keep_tokens),
              flip_aug: s.flip_aug,
              caption_prefix: s.caption_prefix,
              caption_dropout_rate: safeFloat(s.caption_dropout_rate),
              caption_tag_dropout_rate: safeFloat(s.caption_tag_dropout_rate),
              caption_dropout_every_n_epochs: safeInt(
                s.caption_dropout_every_n_epochs,
              ),
              shuffle_caption: s.shuffle_caption,
              enable_wildcard: s.enable_wildcard,
            };
            if (s.is_reg) subset.is_reg = true;
            if ($("cfg-alpha-mask").checked) subset.alpha_mask = true;
            return subset;
          }),
        };
      });
    })(),
  };
}
// ==========================================
//  Dataset Subsets
// ==========================================
function addSubset(shouldRender = true) {
  currentSubsets.push({
    image_dir: "",
    num_repeats: 1,
    keep_tokens: 5,
    flip_aug: false,
    caption_prefix: "",
    caption_dropout_rate: 0.0,
    caption_tag_dropout_rate: 0.3,
    caption_dropout_every_n_epochs: 0,
    shuffle_caption: true,
    enable_wildcard: true,
    is_reg: false,
    collapsed: false,
  });
  if (shouldRender) {
    renderSubsets();
    checkDirty();
  }
}
function deleteSubset(idx) {
  if (currentSubsets.length <= 1) return; // Prevent deleting the last one
  currentSubsets.splice(idx, 1);
  renderSubsets();
  checkDirty();
}
function renderSubsets() {
  const container = $("dataset-subsets-list");
  if (!container) return;
  container.innerHTML = "";
  currentSubsets.forEach((subset, idx) => {
    const isLastOne = currentSubsets.length === 1;
    const isCollapsed = !!subset.collapsed;
    const card = document.createElement("div");
    card.className = "prompt-card-edit";
    card.style.flexDirection = "column";
    card.style.alignItems = "stretch";
    card.style.padding = isCollapsed ? "8px 15px" : "15px";
    const dirName = subset.image_dir
      ? subset.image_dir.split(/[\\/]/).pop()
      : "Empty Path";
    card.innerHTML = `
            <div class="prompt-card-header" style="justify-content: space-between; align-items: center; border-bottom: ${isCollapsed ? "none" : "1px solid var(--border)"}; padding-bottom: ${isCollapsed ? "0" : "8px"}; margin-bottom: ${isCollapsed ? "0" : "12px"};">
                <div style="display: flex; align-items: center; gap: 10px; cursor: pointer; flex: 1;" class="subset-toggle">
                    <span style="font-size: 0.8rem; transition: transform 0.2s; transform: rotate(${isCollapsed ? "-90deg" : "0deg"})">▼</span>
                    <label style="font-weight: 600; cursor: pointer;">Dataset ${idx + 1} ${subset.is_reg ? '<span style="font-size: 0.7rem; color: var(--text-muted); background: var(--border); padding: 1px 6px; border-radius: 4px; margin-left: 6px;">REG</span>' : ""}<span style="font-weight: normal; font-size: 0.8rem; color: var(--text-muted); margin-left: 10px;">${isCollapsed ? "(" + dirName + ")" : ""}</span></label>
                </div>
                <button class="btn btn-ghost btn-sm btn-delete-subset" title="Delete Dataset" 
                    ${isLastOne ? "disabled" : ""} 
                    style="color: var(--danger, #ff4d4d); transition: transform 0.1s; ${isLastOne ? "opacity:0.3; cursor:not-allowed;" : ""}">🗑️</button>
            </div>
            <div class="subset-body" style="display: ${isCollapsed ? "none" : "block"}">
                <div class="form-group">
                    <label style="font-size: 0.8rem;">Image Directory</label>
                    <div style="display: flex; gap: 8px;">
                        <input type="text" class="sub-image-dir" value="${escapeHtml(subset.image_dir)}" placeholder="C:\\path\\to\\images" style="flex: 1;">
                        <button class="btn btn-secondary btn-open-dir" title="Open folder">📂</button>
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Num Repeats</label>
                        <input type="number" class="sub-num-repeats" value="${subset.num_repeats}" min="1">
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Keep Tokens</label>
                        <input type="number" class="sub-keep-tokens" value="${subset.keep_tokens}" min="0">
                    </div>
                </div>
                <div class="form-group" style="margin-top: 10px;">
                    <label style="font-size: 0.8rem;">Caption Prefix</label>
                    <input type="text" class="sub-caption-prefix" value="${escapeHtml(subset.caption_prefix)}" placeholder="e.g. A photo of,">
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Caption Dropout Rate</label>
                        <input type="number" class="sub-caption-dropout" value="${subset.caption_dropout_rate}" step="0.01" min="0" max="1">
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Tag Dropout Rate</label>
                        <input type="number" class="sub-tag-dropout" value="${subset.caption_tag_dropout_rate}" step="0.01" min="0" max="1">
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;">Dropout Every N Epochs</label>
                        <input type="number" class="sub-dropout-every-n" value="${subset.caption_dropout_every_n_epochs}" min="0">
                        <small style="display:block; font-size: 0.7rem; color: var(--text-muted);">0 = disabled</small>
                    </div>
                    <div class="form-group">
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-shuffle-caption" ${subset.shuffle_caption ? "checked" : ""}> Shuffle Captions</label>
                    </div>
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-enable-wildcard" ${subset.enable_wildcard ? "checked" : ""}> Enable Wildcard</label>
                    </div>
                </div>
                <div class="form-row" style="margin-top: 10px;">
                    <div class="form-group">
                        <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-flip-aug" ${subset.flip_aug ? "checked" : ""}> Flip Augmentations</label>
                    </div>
                    <div class="form-group">
                    </div>
                </div>
                <div class="form-group" style="margin-top: 10px;">
                    <label style="font-size: 0.8rem;"><input type="checkbox" class="sub-is-reg" ${subset.is_reg ? "checked" : ""}> Regularization Dataset</label>
                    <small style="display:block; font-size: 0.7rem; color: var(--text-muted);">Images in this folder are used as regularization (class images) to prevent overfitting.</small>
                </div>
            </div>
        `;
    // Toggle collapse
    card.querySelector(".subset-toggle").addEventListener("click", () => {
      subset.collapsed = !subset.collapsed;
      renderSubsets();
    });
    // Update memory immediately on input
    if (!isCollapsed) {
      const updateSubset = () => {
        subset.image_dir = card.querySelector(".sub-image-dir").value;
        subset.num_repeats = safeInt(
          card.querySelector(".sub-num-repeats").value,
        );
        subset.keep_tokens = safeInt(
          card.querySelector(".sub-keep-tokens").value,
        );
        subset.caption_prefix = card.querySelector(".sub-caption-prefix").value;
        subset.caption_dropout_rate = safeFloat(
          card.querySelector(".sub-caption-dropout").value,
        );
        subset.caption_tag_dropout_rate = safeFloat(
          card.querySelector(".sub-tag-dropout").value,
        );
        subset.caption_dropout_every_n_epochs = safeInt(
          card.querySelector(".sub-dropout-every-n").value,
        );
        subset.shuffle_caption = card.querySelector(
          ".sub-shuffle-caption",
        ).checked;
        subset.enable_wildcard = card.querySelector(
          ".sub-enable-wildcard",
        ).checked;
        subset.flip_aug = card.querySelector(".sub-flip-aug").checked;
        subset.is_reg = card.querySelector(".sub-is-reg").checked;
        checkDirty();
      };
      card.querySelectorAll("input").forEach((input) => {
        input.addEventListener("input", updateSubset);
        if (input.type === "checkbox") {
          input.addEventListener("change", updateSubset);
        }
      });
      card
        .querySelector(".btn-open-dir")
        .addEventListener("click", async () => {
          const dir = subset.image_dir.trim();
          if (!dir) {
            showToast("Please enter a directory path first");
            return;
          }
          const result = await api("/api/system/open-folder", {
            method: "POST",
            body: { path: dir },
          });
          if (result.error) {
            showToast("Error: " + result.error);
          }
        });
    }
    if (!isLastOne) {
      card
        .querySelector(".btn-delete-subset")
        .addEventListener("click", (e) => {
          e.stopPropagation(); // Don't trigger toggle
          deleteSubset(idx);
        });
    }
    container.appendChild(card);
  });
  refreshI18n();
}
// ==========================================
//  Save
// ==========================================
async function saveJob() {
  if (!currentJob) return;
  const config = gatherConfig();
  const dataset = gatherDataset();
  // Prevent duplicate directories
  const subPaths = dataset.datasets[0].subsets
    .map((s) => s.image_dir.trim().toLowerCase())
    .filter((p) => p !== "");
  const uniquePaths = new Set(subPaths);
  if (uniquePaths.size !== subPaths.length) {
    showToast(
      "Error: Duplicate Image Directories detected. Each subset must have a unique path.",
    );
    return;
  }
  // Save Config & Dataset
  await api(`/api/jobs/${currentJob}`, {
    method: "PUT",
    body: { config, dataset },
  });
  // Save Prompts
  await savePrompts();
  // Update last saved state
  lastSavedConfig = JSON.parse(JSON.stringify(config));
  lastSavedDataset = JSON.parse(JSON.stringify(dataset));
  lastSavedPrompts = JSON.parse(JSON.stringify(currentPrompts));
  lastSavedNegativePrompt = $("global-negative-prompt").value;
  checkDirty();
  showToast("Job saved");
}
function checkDirty() {
  if (!currentJob) return;
  const currentConfig = gatherConfig();
  const currentDataset = gatherDataset();
  // Deep compare
  const configChanged =
    JSON.stringify(currentConfig) !== JSON.stringify(lastSavedConfig);
  const datasetChanged =
    JSON.stringify(currentDataset) !== JSON.stringify(lastSavedDataset);
  const promptsChanged =
    JSON.stringify(currentPrompts) !== JSON.stringify(lastSavedPrompts);
  const negPromptChanged =
    ($("global-negative-prompt").value || "") !==
    (lastSavedNegativePrompt || "");
  isDirty =
    configChanged || datasetChanged || promptsChanged || negPromptChanged;
  if (isDirty) {
    $("btn-save").classList.remove("hidden");
    $("btn-discard").classList.remove("hidden");
  } else {
    $("btn-save").classList.add("hidden");
    $("btn-discard").classList.add("hidden");
  }
}
function discardChanges() {
  if (!currentJob || !isDirty) return;
  showConfirm(
    "Discard Changes",
    "Discard all unsaved changes and revert to last saved state?",
    () => {
      populateConfig(lastSavedConfig);
      populateDataset(lastSavedDataset);
      currentPrompts = JSON.parse(JSON.stringify(lastSavedPrompts));
      renderPrompts();
      isDirty = false;
      $("btn-save").classList.add("hidden");
      $("btn-discard").classList.add("hidden");
      showToast("Changes discarded");
    },
  );
}
// Show/hide microbatch option depending on step profile checkbox
$("cfg-step-profile").addEventListener("change", (e) => {
  $("cfg-profile-microbatch-group").style.display = e.target.checked ? "" : "none";
  if (!e.target.checked) $("cfg-profile-microbatch").checked = false;
});

// Show/hide LoRA-specific fields based on training type
function updateTrainingTypeUI(type) {
  const isLora = type === "lora";
  $("lora-config-section").classList.toggle("hidden", !isLora);
  $("fft-config-section").classList.toggle("hidden", isLora);
}
function updateNetworkModuleUI() {
  const moduleName = $("cfg-network-module").value;
  const hidesRankAlpha = moduleName === "networks.cdka" || moduleName === "networks.krona";
  $("network-rank-alpha-row").classList.toggle("hidden", hidesRankAlpha);
}
function applyNetworkModulePreset(moduleName) {
  $("cfg-learning-rate").value = getNetworkModuleLearningRate(moduleName);
}
$("cfg-training-type").addEventListener("change", (e) => {
  updateTrainingTypeUI(e.target.value);
  checkDirty();
});
$("cfg-network-module").addEventListener("change", () => {
  updateNetworkModuleUI();
  applyNetworkModulePreset($("cfg-network-module").value);
  checkDirty();
});

// Disable manual resume path when auto-resume is enabled
$("cfg-auto-resume").addEventListener("change", (e) => {
  $("cfg-resume").disabled = e.target.checked;
  if (e.target.checked) $("cfg-resume").value = "";
});

// Mark dirty on any input change
document.addEventListener("input", (e) => {
  if (e.target.closest(".tab-content") && e.target.closest(".tab-pane")) {
    checkDirty();
  }
});
// ==========================================
//  Prompts
// ==========================================
let currentPrompts = []; // Array of objects { text, w, h, s, l, d }
async function loadPrompts() {
  if (!currentJob) return;
  const data = await api(`/api/jobs/${currentJob}/prompts`);
  // Parse strings into objects
  currentPrompts = (data.prompts || []).map((line) => parsePromptLine(line));
  renderPrompts();
}
function parsePromptLine(line) {
  // Defaults
  const p = { text: "", w: 832, h: 1216, s: 20, l: 7.5, d: 1, skip: false };
  // Check if skipped
  if (line.trim().startsWith("#")) {
    p.skip = true;
    line = line.trim().substring(1).trim();
  }
  // Extract params
  const paramRegex = /\s+--([whdsl])\s+(\S+)/g;
  let match;
  while ((match = paramRegex.exec(line)) !== null) {
    const val = match[2];
    if (match[1] === "w") p.w = parseInt(val);
    if (match[1] === "h") p.h = parseInt(val);
    if (match[1] === "s") p.s = parseInt(val);
    if (match[1] === "d") p.d = parseInt(val);
    if (match[1] === "l") p.l = parseFloat(val);
  }
  // Extract text (strip out specific params and the negative prompt string)
  p.text = line
    .replace(/\s+--n\s+.*$/i, "") // Remove global negative prompt and everything after it
    .replace(/\s+--[whdsl]\s+\S+/gi, "") // Remove regular parameter flags
    .trim();
  return p;
}
function serializePrompt(p) {
  // Reconstruct line, ensuring no newlines break the backend parsing parser
  const safeText = p.text.replace(/[\r\n]+/g, " ").trim();
  let line = `${safeText} --w ${p.w} --h ${p.h} --s ${p.s} --d ${p.d} --l ${p.l}`;
  // Append global negative prompt without newlines
  const neg = $("global-negative-prompt")
    .value.replace(/[\r\n]+/g, " ")
    .trim();
  if (neg) {
    line += ` --n ${neg}`;
  }
  return p.skip ? `# ${line}` : line;
}
async function savePrompts() {
  // Filter out prompts that have no text before saving
  const validPrompts = currentPrompts.filter(
    (p) => p.text && p.text.trim().length > 0,
  );
  const lines = validPrompts.map(serializePrompt);
  await api(`/api/jobs/${currentJob}/prompts`, {
    method: "PUT",
    body: { prompts: lines },
  });
}
function renderPrompts() {
  const list = $("prompts-list");
  const empty = $("prompts-empty");
  if (currentPrompts.length === 0) {
    list.classList.add("hidden");
    empty.classList.remove("hidden");
    refreshI18n();
    return;
  }
  empty.classList.add("hidden");
  list.classList.remove("hidden");
  list.innerHTML = "";
  currentPrompts.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = `prompt-card-edit${p.skip ? " skipped" : ""}`;
    card.innerHTML = `
            <div class="prompt-card-header">
                <label class="skip-label">
                    <input type="checkbox" class="p-skip" ${p.skip ? "checked" : ""}> Skip
                </label>
            </div>
            <textarea class="p-text" rows="2" placeholder="Enter prompt text...">${escapeHtml(p.text)}</textarea>
            <div class="prompt-card-row">
                <div class="compact-input">
                    <label>W</label>
                    <input type="number" class="p-w" value="${p.w}" step="64">
                </div>
                <div class="compact-input">
                    <label>H</label>
                    <input type="number" class="p-h" value="${p.h}" step="64">
                </div>
                <div class="compact-input">
                    <label>Steps</label>
                    <input type="number" class="p-s" value="${p.s}">
                </div>
                <div class="compact-input">
                    <label>Scale</label>
                    <input type="number" class="p-l" value="${p.l}" step="0.5">
                </div>
                <div class="compact-input">
                    <label>Seed</label>
                    <input type="number" class="p-d" value="${p.d}">
                </div>
                <button class="btn btn-ghost btn-sm btn-delete-prompt" title="Delete">🗑️</button>
            </div>
        `;
    // Bind events
    const updateState = () => {
      p.skip = card.querySelector(".p-skip").checked;
      p.text = card.querySelector(".p-text").value;
      p.w = parseInt(card.querySelector(".p-w").value);
      p.h = parseInt(card.querySelector(".p-h").value);
      p.s = parseInt(card.querySelector(".p-s").value);
      p.l = parseFloat(card.querySelector(".p-l").value);
      p.d = parseInt(card.querySelector(".p-d").value);
      card.classList.toggle("skipped", p.skip);
      checkDirty();
    };
    const tx = card.querySelector(".p-text");
    const autoResize = () => {
      tx.style.height = "auto";
      tx.style.height = tx.scrollHeight + 2 + "px";
    };
    tx.addEventListener("input", autoResize);
    // Initial resize
    setTimeout(autoResize, 1);
    card.querySelectorAll("input, textarea").forEach((el) => {
      el.addEventListener("input", updateState);
    });
    card
      .querySelector(".btn-delete-prompt")
      .addEventListener("click", () => deletePrompt(idx));
    list.appendChild(card);
  });
  refreshI18n();
}
function deletePrompt(idx) {
  currentPrompts.splice(idx, 1);
  renderPrompts();
  checkDirty();
}
function addPrompt() {
  // Get defaults from global bar
  const w = parseInt($("global-w").value) || 832;
  const h = parseInt($("global-h").value) || 1216;
  const s = parseInt($("global-s").value) || 28;
  const l = parseFloat($("global-l").value) || 3.5;
  let d = parseInt($("global-d").value);
  // If global seed is 0 or empty, randomize for the new prompt
  if (!d || d === 0) {
    d = Math.floor(Math.random() * 99999) + 1;
  }
  currentPrompts.push({ text: "", w, h, s, l, d, skip: false });
  renderPrompts();
  checkDirty();
}
function applyGlobalSettings() {
  const w = parseInt($("global-w").value);
  const h = parseInt($("global-h").value);
  const s = parseInt($("global-s").value);
  const l = parseFloat($("global-l").value);
  const d = parseInt($("global-d").value);
  currentPrompts.forEach((p) => {
    if (w) p.w = w;
    if (h) p.h = h;
    if (s) p.s = s;
    if (l) p.l = l;
    // Seed handling: 0 = random for each prompt, non-zero = apply same seed to all
    if (d === 0) {
      p.d = Math.floor(Math.random() * 99999) + 1; // Random seed 1-99999
    } else if (d) {
      p.d = d;
    }
  });
  renderPrompts();
  renderPrompts();
  checkDirty();
  showToast(
    d === 0
      ? "Random seeds applied to all prompts"
      : "Global settings applied to all prompts",
  );
}
// === Prompt Tab Persistence ===
function savePromptTransientSettings() {
  if (!currentJob) return;
  const settings = {
    lora_mul: $("gen-lora-mul").value,
    keep_loaded: $("chk-keep-loaded").checked,
    flash_attn: $("gen-flash-attn").checked,
    sage_attn: $("gen-sage-attn").checked,
    global_w: $("global-w").value,
    global_h: $("global-h").value,
    global_s: $("global-s").value,
    global_l: $("global-l").value,
    global_d: $("global-d").value,
    selected_lora: $("gen-lora-select").value,
    negative_prompt: $("global-negative-prompt").value,
    gen_gpu_ids: getSelectedGenGPUs(),
    gen_multi_gpu_mode: $("gen-multi-gpu-mode").value,
  };
  localStorage.setItem(
    `prompt_transient_${currentJob}`,
    JSON.stringify(settings),
  );
}
function loadPromptTransientSettings() {
  if (!currentJob) return;
  const data = localStorage.getItem(`prompt_transient_${currentJob}`);
  if (!data) return;
  try {
    const settings = JSON.parse(data);
    if (settings.lora_mul !== undefined)
      $("gen-lora-mul").value = settings.lora_mul;
    if (settings.keep_loaded !== undefined)
      $("chk-keep-loaded").checked = settings.keep_loaded;
    if (settings.flash_attn !== undefined)
      $("gen-flash-attn").checked = settings.flash_attn;
    if (settings.sage_attn !== undefined)
      $("gen-sage-attn").checked = settings.sage_attn;
    if (settings.global_w !== undefined)
      $("global-w").value = settings.global_w;
    if (settings.global_h !== undefined)
      $("global-h").value = settings.global_h;
    if (settings.global_s !== undefined)
      $("global-s").value = settings.global_s;
    if (settings.global_l !== undefined)
      $("global-l").value = settings.global_l;
    if (settings.global_d !== undefined)
      $("global-d").value = settings.global_d;
    if (settings.negative_prompt !== undefined)
      $("global-negative-prompt").value = settings.negative_prompt;
    // Restore gen GPU selection
    if (settings.gen_gpu_ids !== undefined) {
      restoreGenGPUSelection(settings.gen_gpu_ids);
    }
    if (settings.gen_multi_gpu_mode !== undefined) {
      $("gen-multi-gpu-mode").value = settings.gen_multi_gpu_mode;
    }
    // selected_lora is handled in loadCheckpoints
  } catch (e) { }
}
// ==========================================
//  Console
// ==========================================
function appendConsole(text) {
  if (!text) return;
  if (consoleOutput.textContent.startsWith("Waiting")) {
    consoleOutput.textContent = "";
  }
  const wasNearBottom =
    consoleOutput.scrollHeight -
    consoleOutput.scrollTop -
    consoleOutput.clientHeight <
    100;
  // Standard Terminal logic: \r overwrites the CURRENT line.
  // We split current content and process the last line surgically.
  let fullText = consoleOutput.textContent + text;
  if (fullText.includes("\r")) {
    const lines = fullText.split("\n");
    const lastLine = lines[lines.length - 1];
    if (lastLine.includes("\r")) {
      const parts = lastLine.split("\r");
      let processedLine = "";
      for (let i = 0; i < parts.length; i++) {
        // If there is content after this \r, it overwrites what was before it.
        // If this is the very last part of the string, we keep it.
        if (i === parts.length - 1) {
          processedLine += parts[i];
        } else if (parts[i + 1].length > 0) {
          processedLine = ""; // Overwrite triggered by following content
        } else {
          processedLine = parts[i]; // Keep until something actually follows the \r
        }
      }
      lines[lines.length - 1] = processedLine;
      fullText = lines.join("\n");
    }
  }
  consoleOutput.textContent = fullText;
  if (wasNearBottom) {
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
  }
}
// ==========================================
//  Samples
// ==========================================
async function loadCheckpoints() {
  if (!currentJob) return;
  const jobAtStart = currentJob;
  const files = await api(`/api/jobs/${currentJob}/checkpoints`);
  if (currentJob !== jobAtStart) return; // job changed while fetching
  const select = $("gen-lora-select");
  // Save current selection
  const currentVal = select.value;
  select.innerHTML = '<option value="">Base Model (No LoRA)</option>';
  files.forEach((f) => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = `${f.name} (${new Date(f.mtime).toLocaleString()})`;
    select.appendChild(opt);
  });
  // Restore selection if exists
  const data = localStorage.getItem(`prompt_transient_${currentJob}`);
  let savedLora = null;
  if (data) {
    try {
      savedLora = JSON.parse(data).selected_lora;
    } catch (e) { }
  }
  const valToRestore = currentVal || savedLora;
  if (
    valToRestore &&
    Array.from(select.options).some((o) => o.value === valToRestore)
  ) {
    select.value = valToRestore;
  }
}
// Sample State
let sampleState = {
  selectedPaths: new Set(),
  lastSelectedPath: null,
  groups: {}, // enum -> [images]
  allImages: [], // flat list for index lookup
  isExplicitMultiSelect: false,
};
async function loadSamples(isUpdate = false) {
  if (!currentJob) return;
  const images = await api(`/api/jobs/${currentJob}/samples`);
  const container = $("samples-grid");
  const empty = $("samples-empty");
  if (!images || images.length === 0) {
    if (!isUpdate) {
      container.classList.add("hidden");
      empty.classList.remove("hidden");
    }
    return;
  }
  empty.classList.add("hidden");
  container.classList.remove("hidden");
  // Load manual order from localStorage
  const savedOrder = loadManualOrder();
  const orderMap = new Map();
  if (savedOrder) {
    savedOrder.forEach((item, index) => {
      orderMap.set(item.path, { group: item.group, index: index });
    });
  }
  // 1. Group Images
  const groups = {};
  images.forEach((img) => {
    let groupKey;
    if (orderMap.has(img.path)) {
      groupKey = orderMap.get(img.path).group;
    } else {
      const match = img.name.match(/_(\d{2,})_\d{14}/);
      groupKey = match ? match[1] : "default";
    }
    if (!groups[groupKey]) groups[groupKey] = [];
    groups[groupKey].push(img);
  });
  sampleState.groups = groups;
  sampleState.allImages = images;
  // 2. Render Groups
  container.innerHTML = "";
  const sortedGroupKeys = Object.keys(groups).sort((a, b) => {
    if (a === "default") return 1;
    if (b === "default") return -1;
    return parseInt(a) - parseInt(b);
  });
  sortedGroupKeys.forEach((key) => {
    const groupDiv = document.createElement("div");
    groupDiv.className = "sample-group";
    groupDiv.dataset.group = key;
    const header = document.createElement("div");
    header.className = "group-header";
    header.textContent =
      key === "default" ? "Uncategorized" : `Prompt ${parseInt(key) + 1}`;
    const gridDiv = document.createElement("div");
    gridDiv.className = "group-grid";
    gridDiv.addEventListener("dragover", handleDragOver);
    gridDiv.addEventListener("drop", handleDrop);
    // Sort images in group.
    // If they have a saved index, use it. Otherwise, use mtime (newest first).
    groups[key].sort((a, b) => {
      const orderA = orderMap.get(a.path);
      const orderB = orderMap.get(b.path);
      if (orderA && orderB) return orderA.index - orderB.index;
      if (orderA) return 1; // Saved items come after new items?
      if (orderB) return -1;
      return b.mtime - a.mtime; // Default newest first for items without saved order
    });
    groups[key].forEach((img) => {
      createSampleCard(img, gridDiv);
    });
    groupDiv.appendChild(header);
    groupDiv.appendChild(gridDiv);
    container.appendChild(groupDiv);
  });
  if (!window._samplesInitialized) {
    initSampleInteractions();
    window._samplesInitialized = true;
  }
  updateSelectionVisuals();
}
function saveManualOrder() {
  if (!currentJob) return;
  const order = [];
  document.querySelectorAll(".sample-group").forEach((group) => {
    const groupKey = group.dataset.group;
    group.querySelectorAll(".sample-card").forEach((card) => {
      order.push({
        path: card.dataset.path,
        group: groupKey,
      });
    });
  });
  localStorage.setItem(`sample_order_${currentJob}`, JSON.stringify(order));
}
function loadManualOrder() {
  if (!currentJob) return null;
  const data = localStorage.getItem(`sample_order_${currentJob}`);
  try {
    return data ? JSON.parse(data) : null;
  } catch (e) {
    return null;
  }
}
function createSampleCard(img, container) {
  const card = document.createElement("div");
  card.className = "sample-card";
  card.draggable = true;
  card.dataset.path = img.path;
  card.dataset.name = img.name;
  card.dataset.mtime = img.mtime; // for sorting reference
  // Check selection state
  if (sampleState.selectedPaths.has(img.path)) {
    card.classList.add("selected");
  }
  card.innerHTML = `
        <img src="${img.path}" alt="${escapeHtml(img.name)}" loading="lazy" draggable="false">
        <div class="sample-name">${escapeHtml(img.name)}</div>
        <button class="btn-delete-card" title="Delete Image">🗑</button>
    `;
  // Delete Card Logic
  card.querySelector(".btn-delete-card").addEventListener("click", (e) => {
    e.stopPropagation(); // Don't trigger selection/lightbox
    showConfirm("Delete Image", `Delete "${img.name}"?`, () => {
      deleteSamples([img.path]);
    });
  });
  // Click Selection Logic (Selection + Open Lightbox)
  card.addEventListener("click", (e) => handleSampleClick(e, img, card));
  // Drag Events
  card.addEventListener("dragstart", handleDragStart);
  card.addEventListener("dragover", handleDragOver);
  card.addEventListener("drop", handleDrop);
  card.addEventListener("dragenter", (e) => e.preventDefault());
  container.appendChild(card);
}
// ==========================================
//  Sample Interactions
// ==========================================
// ==========================================
//  Box Selection (Rubber Band)
// ==========================================
let boxSelection = {
  isSelecting: false,
  startX: 0,
  startY: 0,
  element: null,
};
function initSampleInteractions() {
  // Keyboard Navigation
  document.addEventListener("keydown", handleGlobalKeydown);
  // Batch Delete Button
  const btnDelete = $("btn-delete-selected");
  if (btnDelete) {
    btnDelete.addEventListener("click", () => {
      const count = sampleState.selectedPaths.size;
      if (count > 0) {
        showConfirm(
          "Delete Images",
          `Delete ${count} selected image(s)?`,
          () => {
            deleteSamples(Array.from(sampleState.selectedPaths));
          },
        );
      }
    });
  }
  // Box Selection Listeners (on container)
  const container = $("samples-grid"); // This might be hidden initially?
  // We can attach to document or a wrapper.
  // Attaching to 'samples-grid' is safest if it exists.
  if (container) {
    container.addEventListener("mousedown", handleBoxStart);
  }
  document.addEventListener("mousemove", handleBoxMove);
  document.addEventListener("mouseup", handleBoxEnd);
}
function handleBoxStart(e) {
  if (e.target.closest(".sample-card")) return;
  if (e.button !== 0) return;
  boxSelection.isSelecting = true;
  sampleState.isExplicitMultiSelect = true;
  boxSelection.startX = e.pageX;
  boxSelection.startY = e.pageY;
  // Create selection box element
  if (!boxSelection.element) {
    const el = document.createElement("div");
    el.className = "selection-box";
    document.body.appendChild(el);
    boxSelection.element = el;
  }
  const el = boxSelection.element;
  el.style.left = e.pageX + "px";
  el.style.top = e.pageY + "px";
  el.style.width = "0px";
  el.style.height = "0px";
  el.style.display = "block";
  if (!e.ctrlKey && !e.shiftKey) {
    clearSelection();
  }
}
function handleBoxMove(e) {
  if (!boxSelection.isSelecting) return;
  e.preventDefault(); // Stop text selection
  const currentX = e.pageX;
  const currentY = e.pageY;
  const minX = Math.min(boxSelection.startX, currentX);
  const maxX = Math.max(boxSelection.startX, currentX);
  const minY = Math.min(boxSelection.startY, currentY);
  const maxY = Math.max(boxSelection.startY, currentY);
  const el = boxSelection.element;
  el.style.left = minX + "px";
  el.style.top = minY + "px";
  el.style.width = maxX - minX + "px";
  el.style.height = maxY - minY + "px";
  // Update selection in real-time
  updateBoxSelection(minX, minY, maxX, maxY, e.ctrlKey);
}
function handleBoxEnd(e) {
  if (!boxSelection.isSelecting) return;
  boxSelection.isSelecting = false;
  if (boxSelection.element) {
    boxSelection.element.style.display = "none";
  }
}
function updateBoxSelection(x1, y1, x2, y2, isCtrl) {
  const cards = document.querySelectorAll(".sample-card");
  cards.forEach((card) => {
    const rect = card.getBoundingClientRect();
    // Get card coordinates relative to page (since box uses pageX/Y)
    const cardX1 = rect.left + window.scrollX;
    const cardY1 = rect.top + window.scrollY;
    const cardX2 = cardX1 + rect.width;
    const cardY2 = cardY1 + rect.height;
    // Check intersection
    const isOverlapping = !(
      cardX1 > x2 ||
      cardX2 < x1 ||
      cardY1 > y2 ||
      cardY2 < y1
    );
    if (isOverlapping) {
      sampleState.selectedPaths.add(card.dataset.path);
    } else if (!isCtrl) {
      // If not holding Ctrl, box selection is "set" logic, but real-time clearing
      // of things outside box is tricky if we started with a selection.
      // Simplified: Box Selection ADDS to selection during drag.
      // If we want "Select ONLY these", we cleared at start.
      // Scaling back: Standard behavior is Additive if Box touches.
      // To be strict:
      // If we cleared at start, then sampleState contains only what is currently overlapping.
      // But we need to NOT delete things we just added in this drag session if we shrink box.
      // This requires "initialSelection" state. Too complex for raw JS in one function.
      // CURRENT LOGIC: additive only during move.
      // If user shrinks box, items stay selected. (Minor UX quirk but acceptable).
    }
  });
  updateSelectionVisuals();
}
function handleSampleClick(e, img, card) {
  // Lightbox triggers on double click or specific action?
  // User request: "arrow keys to move... even if user is open a specific image"
  // Standard UI: Click = Select, Double Click = Open?
  // Or Click = Open?
  // Plan: Click = Select. Double Click = Lightbox.
  // If modifier keys are used, strictly selection.
  // BUT user said "open a specific image", implying lightbox.
  // Let's implement: Click selects. Double click opens.
  // Also, if you just click and no modifiers, maybe open?
  // "select multiple images then they can drag" -> implies single click might select.
  // Hybrid approach:
  // Simple Click: Selects (and clears others)
  // Ctrl+Click: Toggles
  // Shift+Click: Range
  // Double Click: Open Lightbox
  if (e.ctrlKey || e.metaKey) {
    sampleState.isExplicitMultiSelect = true;
    toggleSelection(img.path);
  } else if (e.shiftKey) {
    sampleState.isExplicitMultiSelect = true;
    selectRange(img.path);
  } else {
    // Simple click: Select and Open Lightbox
    sampleState.isExplicitMultiSelect = false;
    selectSingle(img.path);
    openLightbox(img.path, img.name);
  }
  sampleState.lastSelectedPath = img.path;
}
// Better to attach dblclick to card in createSampleCard, adding it here implies logic change
// Lets add logic in createSampleCard wrapper
// (Modified createSampleCard above needs dblclick listener)
function selectSingle(path) {
  sampleState.selectedPaths.clear();
  sampleState.selectedPaths.add(path);
  updateSelectionVisuals();
}
function toggleSelection(path) {
  if (sampleState.selectedPaths.has(path)) {
    sampleState.selectedPaths.delete(path);
  } else {
    sampleState.selectedPaths.add(path);
  }
  updateSelectionVisuals();
}
function selectRange(targetPath) {
  if (!sampleState.lastSelectedPath) {
    selectSingle(targetPath);
    return;
  }
  // Find indices in the flattened visual list
  // To do this right, we need the current visual DOM order
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  const startIdx = allCards.findIndex(
    (c) => c.dataset.path === sampleState.lastSelectedPath,
  );
  const endIdx = allCards.findIndex((c) => c.dataset.path === targetPath);
  if (startIdx === -1 || endIdx === -1) return;
  const [min, max] = [Math.min(startIdx, endIdx), Math.max(startIdx, endIdx)];
  // Add range
  // If ctrl not held, clear others? Standard behavior is usually yes for Shift-click
  // But lets keep it additive for now or clear?
  // Windows Explorer: Shift-click clears previous selection (except anchor)
  // Let's clear for simplicity.
  sampleState.selectedPaths.clear();
  for (let i = min; i <= max; i++) {
    sampleState.selectedPaths.add(allCards[i].dataset.path);
  }
  updateSelectionVisuals();
}
function updateSelectionVisuals() {
  const isMultiRoot =
    sampleState.selectedPaths.size > 1 || sampleState.isExplicitMultiSelect;
  const count = sampleState.selectedPaths.size;
  // Batch delete button
  const btnDelete = $("btn-delete-selected");
  if (btnDelete) {
    if (count > 0) {
      btnDelete.classList.remove("hidden");
      btnDelete.textContent = `🗑️ Delete (${count})`;
    } else {
      btnDelete.classList.add("hidden");
    }
  }
  // Toggle multi-select mode on all grids
  document.querySelectorAll(".group-grid").forEach((grid) => {
    grid.classList.toggle("multi-select-mode", isMultiRoot);
  });
  document.querySelectorAll(".sample-card").forEach((card) => {
    if (sampleState.selectedPaths.has(card.dataset.path)) {
      card.classList.add("selected");
    } else {
      card.classList.remove("selected");
    }
  });
}
function clearSelection() {
  sampleState.selectedPaths.clear();
  updateSelectionVisuals();
}
// Drag and Drop Logic
function handleDragStart(e) {
  const path = e.target.closest(".sample-card").dataset.path;
  // If dragging an unselected item, select it first
  if (!sampleState.selectedPaths.has(path)) {
    selectSingle(path);
  }
  e.dataTransfer.setData(
    "text/plain",
    JSON.stringify(Array.from(sampleState.selectedPaths)),
  );
  e.dataTransfer.effectAllowed = "move";
  e.target.closest(".sample-card").classList.add("dragging");
}
function handleDragOver(e) {
  e.preventDefault();
  e.dataTransfer.dropEffect = "move";
  // Remove existing drag indicators
  document
    .querySelectorAll(".drag-over-left, .drag-over-right, .drag-over-grid")
    .forEach((el) => {
      el.classList.remove(
        "drag-over-left",
        "drag-over-right",
        "drag-over-grid",
      );
    });
  const targetCard = e.target.closest(".sample-card");
  const targetGrid = e.target.closest(".group-grid");
  if (targetCard) {
    const rect = targetCard.getBoundingClientRect();
    const relX = e.clientX - rect.left;
    if (relX < rect.width / 2) {
      targetCard.classList.add("drag-over-left");
    } else {
      targetCard.classList.add("drag-over-right");
    }
  } else if (targetGrid) {
    // Visual feedback for dropping into the grid background
    targetGrid.classList.add("drag-over-grid");
  }
}
function handleDrop(e) {
  e.preventDefault();
  document
    .querySelectorAll(".drag-over-left, .drag-over-right, .drag-over-grid")
    .forEach((el) => {
      el.classList.remove(
        "drag-over-left",
        "drag-over-right",
        "drag-over-grid",
      );
    });
  const targetCard = e.target.closest(".sample-card");
  const targetGrid = e.target.closest(".group-grid");
  if (!targetGrid) return;
  try {
    const paths = JSON.parse(e.dataTransfer.getData("text/plain"));
    const allCards = Array.from(document.querySelectorAll(".sample-card"));
    const cardsToMove = allCards.filter((c) => paths.includes(c.dataset.path));
    if (targetCard) {
      const rect = targetCard.getBoundingClientRect();
      const relX = e.clientX - rect.left;
      const insertBefore = relX < rect.width / 2;
      cardsToMove.forEach((card) => {
        if (insertBefore) {
          targetGrid.insertBefore(card, targetCard);
        } else {
          targetGrid.insertBefore(card, targetCard.nextSibling);
        }
      });
    } else {
      // Drop in grid background -> Append to end
      cardsToMove.forEach((card) => {
        targetGrid.appendChild(card);
      });
    }
    cardsToMove.forEach((card) => card.classList.remove("dragging"));
    // Save the new state permanently
    saveManualOrder();
  } catch (err) {
    console.error("Drop error", err);
  }
}
// Keyboard Navigation & Lightbox
function handleGlobalKeydown(e) {
  // Lightbox navigation
  const lightbox = document.querySelector(".lightbox");
  if (lightbox) {
    const currentSrc = lightbox.querySelector("img").getAttribute("src");
    handleLightboxNavigation(e, currentSrc, lightbox);
    return;
  }
  // Grid navigation (Arrow Keys)
  // Only if focus is not in an input
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key.startsWith("Arrow")) {
    e.preventDefault();
    navigateGrid(e.key, e.ctrlKey);
  }
  if (e.key === "Enter") {
    const selected = Array.from(sampleState.selectedPaths);
    if (selected.length === 1) {
      const card = document.querySelector(
        `.sample-card[data-path="${selected[0]}"]`,
      );
      openLightbox(selected[0], card ? card.dataset.name : "");
    }
  }
}
// Navigation Helper
function calculateNextIndex(allCards, currentIdx, direction) {
  if (currentIdx === -1) return 0;
  let nextIdx = currentIdx;
  if (direction === "ArrowRight")
    nextIdx = Math.min(currentIdx + 1, allCards.length - 1);
  if (direction === "ArrowLeft") nextIdx = Math.max(currentIdx - 1, 0);
  if (direction === "ArrowUp" || direction === "ArrowDown") {
    const currentRect = allCards[currentIdx].getBoundingClientRect();
    const currentCenter = currentRect.left + currentRect.width / 2;
    const currentY = currentRect.top + currentRect.height / 2;
    let bestDist = Infinity;
    let bestCandidate = -1;
    allCards.forEach((c, i) => {
      if (i === currentIdx) return;
      const r = c.getBoundingClientRect();
      const y = r.top + r.height / 2;
      const x = r.left + r.width / 2;
      // Metric: Minimize vertical dist first, then horizontal.
      const distV = Math.abs(y - currentY);
      const distH = Math.abs(x - currentCenter);
      const score = distV * 2 + distH;
      if (
        (direction === "ArrowUp" && y < currentRect.top) ||
        (direction === "ArrowDown" && y > currentRect.bottom)
      ) {
        if (score < bestDist) {
          bestDist = score;
          bestCandidate = i;
        }
      }
    });
    if (bestCandidate !== -1) nextIdx = bestCandidate;
  }
  return nextIdx;
}
function navigateGrid(direction, isCtrl) {
  // Find current focus (last selected)
  // If no selection, select first
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  if (allCards.length === 0) return;
  let idx = -1;
  if (sampleState.lastSelectedPath) {
    idx = allCards.findIndex(
      (c) => c.dataset.path === sampleState.lastSelectedPath,
    );
  }
  if (idx === -1) {
    selectSingle(allCards[0].dataset.path);
    allCards[0].scrollIntoView({ block: "center" });
    sampleState.lastSelectedPath = allCards[0].dataset.path;
    return;
  }
  const nextIdx = calculateNextIndex(allCards, idx, direction);
  if (nextIdx !== idx) {
    const path = allCards[nextIdx].dataset.path;
    if (!isCtrl) {
      selectSingle(path);
    } else {
      selectSingle(path);
    }
    allCards[nextIdx].scrollIntoView({ block: "nearest" });
    sampleState.lastSelectedPath = path; // Update visual focus anchor
  }
}
function openLightbox(src, name) {
  // Remove existing
  const existing = document.querySelector(".lightbox");
  if (existing) existing.remove();
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.innerHTML = `
        <div class="lightbox-title">${name || ""}</div>
        <img src="${src}">
        <div class="lightbox-metadata hidden"></div>
        <div class="lightbox-nav">
            Use Arrow Keys to navigate | ESC to close
        </div>
    `;
  // Click background to close
  lb.addEventListener("click", (e) => {
    if (e.target === lb) lb.remove();
  });
  document.body.appendChild(lb);
  loadLightboxMetadata(src, lb);
  // Auto-fade navigation hint after 3s
  setTimeout(() => {
    const nav = lb.querySelector(".lightbox-nav");
    if (nav) nav.style.opacity = "0";
  }, 3000);
}
function handleLightboxNavigation(e, currentSrc, lightbox) {
  if (e.key === "Escape") {
    lightbox.remove();
    return;
  }
  if (!e.key.startsWith("Arrow")) return;
  e.preventDefault();
  // Find current index
  const allCards = Array.from(document.querySelectorAll(".sample-card"));
  const idx = allCards.findIndex((c) => c.dataset.path === currentSrc);
  if (idx === -1) return;
  const nextIdx = calculateNextIndex(allCards, idx, e.key);
  if (nextIdx !== idx) {
    const nextCard = allCards[nextIdx];
    const nextPath = nextCard.dataset.path;
    const nextName = nextCard.dataset.name;
    lightbox.querySelector("img").src = nextPath;
    const titleEl = lightbox.querySelector(".lightbox-title");
    if (titleEl) titleEl.textContent = nextName || "";
    // Update metadata
    loadLightboxMetadata(nextPath, lightbox);
    // Also update selection in background
    selectSingle(nextPath);
    nextCard.scrollIntoView({ block: "nearest" });
  }
}
async function loadLightboxMetadata(path, lightbox) {
  const metaEl = lightbox.querySelector(".lightbox-metadata");
  if (!metaEl) return;
  // Convert path from /api/jobs/NAME/samples/... to /api/jobs/NAME/metadata/...
  const metaUrl = path.replace("/samples/", "/metadata/");
  try {
    const res = await fetch(metaUrl);
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.parameters) {
      metaEl.textContent = data.parameters;
      metaEl.classList.remove("hidden");
    } else {
      metaEl.classList.add("hidden");
    }
  } catch (e) {
    metaEl.classList.add("hidden");
  }
}
// Add double click listener helper
function addDoubleClick(element, callback) {
  let lastClick = 0;
  element.addEventListener("click", (e) => {
    const now = new Date().getTime();
    if (now - lastClick < 300) {
      callback(e);
    }
    lastClick = now;
  });
}
// ==========================================
//  TensorBoard
// ==========================================
let tbUrl = null;
async function checkTensorBoard() {
  if (!currentJob) return;
  const status = await api(`/api/jobs/${currentJob}/tensorboard/status`);
  updateTbState(status.running, status.url);
}
function updateTbState(running, url) {
  $("btn-tb-launch").classList.toggle("hidden", running);
  $("btn-tb-stop").classList.toggle("hidden", !running);
  $("btn-tb-open").classList.toggle("hidden", !running);
  $("tb-status").textContent = running
    ? `${translatePhrase("Running on port")} ${new URL(url).port}`
    : translatePhrase("Not running");
  $("tb-status").style.color = running ? "var(--success)" : "var(--text-muted)";
  if (running && url) {
    tbUrl = url;
    $("tb-placeholder").classList.add("hidden");
    $("tb-iframe").classList.remove("hidden");
    // Only set src if it changed
    if ($("tb-iframe").src !== url) {
      $("tb-iframe").src = url;
    }
  } else {
    tbUrl = null;
    $("tb-placeholder").classList.remove("hidden");
    $("tb-iframe").classList.add("hidden");
    $("tb-iframe").src = "";
  }
}
async function launchTensorBoard() {
  if (!currentJob) return;
  $("btn-tb-launch").disabled = true;
  $("btn-tb-launch").textContent = translatePhrase("Starting...");
  const result = await api(`/api/jobs/${currentJob}/tensorboard`, {
    method: "POST",
  });
  if (result.error) {
    alert(result.error);
    $("btn-tb-launch").disabled = false;
    $("btn-tb-launch").textContent = `\uD83D\uDE80 ${translatePhrase("Launch")}`;
    return;
  }
  // Give TensorBoard a moment to start
  setTimeout(() => {
    updateTbState(true, result.url);
    $("btn-tb-launch").disabled = false;
    $("btn-tb-launch").textContent = `\uD83D\uDE80 ${translatePhrase("Launch")}`;
    showToast("TensorBoard launched");
  }, 2000);
}
async function stopTensorBoard() {
  if (!currentJob) return;
  await api(`/api/jobs/${currentJob}/tensorboard/stop`, { method: "POST" });
  updateTbState(false, null);
  showToast("TensorBoard stopped");
}
// ==========================================
//  Global Settings
// ==========================================
function applyTheme(theme) {
  const t = theme || "github-dark";
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("ui_theme", t);
}
// Build dynamic global settings tabs from architecture registry
function buildGlobalSettingsTabs(registry) {
  const nav = $("global-tabs-nav");
  const content = $("global-tabs-content");
  // Clear old dynamic tabs (keep the static Application tab)
  nav.innerHTML = "";
  content.querySelectorAll(".gtab-pane-dynamic").forEach((el) => el.remove());
  content.querySelectorAll(".gtab-pane").forEach((pane) => {
    pane.classList.remove("active");
    pane.classList.add("hidden");
  });
  const archs = registry.architectures;
  let isFirst = true;
  for (const [archId, arch] of Object.entries(archs)) {
    // Tab button
    const btn = document.createElement("button");
    btn.className = "tab" + (isFirst ? " active" : "");
    btn.dataset.gtab = archId;
    btn.textContent = arch.display_name + " Models";
    nav.appendChild(btn);
    // Tab pane
    const pane = document.createElement("div");
    pane.id = `gtab-${archId}`;
    pane.className =
      "gtab-pane gtab-pane-dynamic" + (isFirst ? " active" : " hidden");
    for (const [configKey, pathDef] of Object.entries(arch.global_paths)) {
      const group = document.createElement("div");
      group.className = "form-group";
      group.innerHTML = `
                <label>${pathDef.label}</label>
                <input type="text" id="cfg-global-${configKey}" placeholder="${pathDef.placeholder}">
            `;
      pane.appendChild(group);
    }
    if (isFirst) {
      const outputGroup = document.createElement("div");
      outputGroup.className = "form-group";
      outputGroup.style.marginTop = "15px";
      outputGroup.innerHTML = `
                <label>LoRA Output Location</label>
                <input type="text" id="cfg-global-jobs-dir" placeholder="E:\\Anima-LoRA-Outputs">
                <small>Folder where UI jobs, configs, LoRA outputs, and logs are stored.</small>
            `;
      pane.appendChild(outputGroup);
    }
    // All-in-One sync button
    if (arch.all_in_one && arch.all_in_one_source_key) {
      const syncGroup = document.createElement("div");
      syncGroup.style.marginTop = "8px";
      syncGroup.innerHTML = `
                <button class="btn btn-secondary btn-sm" id="btn-sync-${archId}">\uD83D\uDD04 Use as All-in-One Checkpoint</button>
                <small style="display: block; margin-top: 4px;">Copies the first path to all other fields for this architecture.</small>
            `;
      pane.appendChild(syncGroup);
    }
    // Insert before the static Application tab
    content.insertBefore(pane, $("gtab-app"));
    isFirst = false;
  }
  // Application tab button (always last)
  const appBtn = document.createElement("button");
  appBtn.className = "tab";
  appBtn.dataset.gtab = "app";
  appBtn.textContent = "Application";
  nav.appendChild(appBtn);
  // Bind tab switching
  nav.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      nav.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      content
        .querySelectorAll(".gtab-pane, .gtab-pane-dynamic")
        .forEach((p) => {
          p.classList.remove("active");
          p.classList.add("hidden");
        });
      tab.classList.add("active");
      const pane = $(`gtab-${tab.dataset.gtab}`);
      if (pane) {
        pane.classList.remove("hidden");
        pane.classList.add("active");
      }
    });
  });
  // Bind all-in-one sync buttons
  for (const [archId, arch] of Object.entries(archs)) {
    if (arch.all_in_one && arch.all_in_one_source_key) {
      const syncBtn = $(`btn-sync-${archId}`);
      if (syncBtn) {
        syncBtn.addEventListener("click", () => {
          const sourceInput = $(`cfg-global-${arch.all_in_one_source_key}`);
          if (sourceInput && sourceInput.value) {
            for (const configKey of Object.keys(arch.global_paths)) {
              $(`cfg-global-${configKey}`).value = sourceInput.value;
            }
            showToast(`${arch.display_name} paths synced!`);
          }
        });
      }
    }
  }
  refreshI18n();
}
function isDefaultGlobalPathValue(value) {
  return !String(value || "").trim();
}
function shouldPromptForGlobalSettings(config) {
  if (!archRegistry) return false;
  const firstArch = Object.values(archRegistry.architectures)[0];
  if (!firstArch) return false;
  const firstPagePathsAreDefault = Object.keys(firstArch.global_paths).every((configKey) =>
    isDefaultGlobalPathValue(config.model_paths?.[configKey]),
  );
  return firstPagePathsAreDefault && isDefaultGlobalPathValue(config.jobs_dir);
}
async function loadGlobalSettings() {
  // Fetch registry if not cached
  if (!archRegistry) {
    archRegistry = await api("/api/architectures");
    buildGlobalSettingsTabs(archRegistry);
  }
  const config = await api("/api/global-config");
  // Populate path inputs dynamically from registry
  for (const [archId, arch] of Object.entries(archRegistry.architectures)) {
    for (const configKey of Object.keys(arch.global_paths)) {
      const input = $(`cfg-global-${configKey}`);
      if (input) input.value = config.model_paths?.[configKey] || "";
    }
  }
  $("cfg-global-venv").value = config.venv_path || "";
  const jobsDirInput = $("cfg-global-jobs-dir");
  if (jobsDirInput) jobsDirInput.value = config.jobs_dir || "";
  // Theme
  const theme = config.ui?.theme || "github-dark";
  $("cfg-theme").value = theme;
  applyTheme(theme);
  // Background settings
  const pos = config.ui?.background_position || "50% 50%";
  const dim = config.ui?.dim_level ?? 70;
  const brightness = config.ui?.brightness_level ?? 100;
  const blur = config.ui?.blur_level ?? 10;
  const textShadow = config.ui?.text_shadow_size ?? 0;
  $("cfg-bg-dim").value = dim;
  $("val-bg-dim").textContent = dim + "%";
  $("cfg-bg-brightness").value = brightness;
  $("val-bg-brightness").textContent = brightness + "%";
  $("cfg-bg-blur").value = blur;
  $("val-bg-blur").textContent = blur + "px";
  $("cfg-text-shadow").value = textShadow;
  $("val-text-shadow").textContent = textShadow + "px";
  if (config.ui?.background) {
    applyBackground(
      config.ui.background,
      pos,
      dim,
      brightness,
      blur,
      textShadow,
    );
    $("bg-visual-controls").classList.remove("hidden");
  } else {
    $("bg-pos-group").classList.add("hidden");
    $("bg-visual-controls").classList.add("hidden");
  }
}
async function saveGlobalSettings() {
  // Read existing config first to preserve bg settings
  const existingConfig = await api("/api/global-config");
  const previousJobsDir = (existingConfig.jobs_dir || "").trim();
  const jobsDirInput = $("cfg-global-jobs-dir");
  const nextJobsDir = (jobsDirInput?.value || "").trim();
  const jobsDirChanged = previousJobsDir !== nextJobsDir;
  if (
    jobsDirChanged &&
    currentJob &&
    isDirty &&
    !confirm("Current job has unsaved changes. Change LoRA Output Location and discard those editor changes?")
  ) {
    return;
  }
  // Build model_paths dynamically from registry
  const model_paths = {};
  if (archRegistry) {
    for (const [archId, arch] of Object.entries(archRegistry.architectures)) {
      for (const configKey of Object.keys(arch.global_paths)) {
        const input = $(`cfg-global-${configKey}`);
        if (input) model_paths[configKey] = input.value;
      }
    }
  }
  const config = {
    model_paths,
    venv_path: $("cfg-global-venv").value,
    jobs_dir: jobsDirInput?.value || "",
    ui: {
      ...(existingConfig.ui || {}),
      theme: $("cfg-theme").value,
      background_position: `${bgPosPercent.x.toFixed(1)}% ${bgPosPercent.y.toFixed(1)}%`,
      dim_level: parseInt($("cfg-bg-dim").value),
      brightness_level: parseInt($("cfg-bg-brightness").value),
      blur_level: parseInt($("cfg-bg-blur").value),
      text_shadow_size: parseInt($("cfg-text-shadow").value),
    },
  };
  // Apply theme immediately
  applyTheme(config.ui.theme);
  // Live update background if one exists
  if (existingConfig?.ui?.background) {
    applyBackground(
      existingConfig.ui.background,
      config.ui.background_position,
      config.ui.dim_level,
      config.ui.brightness_level,
      config.ui.blur_level,
      config.ui.text_shadow_size,
    );
  }
  await api("/api/global-config", { method: "PUT", body: config });
  if (jobsDirChanged) {
    const previousJob = currentJob;
    isDirty = false;
    const jobs = await loadJobs();
    if (previousJob && jobs.some((job) => job.name === previousJob)) {
      await selectJob(previousJob);
    } else {
      clearCurrentJobSelection();
    }
  }
  closeModal("modal-global-settings");
  showToast("Global settings saved");
}
// === Background Image Functions ===
function applyBackground(
  url,
  position = "50% 50%",
  dim = 70,
  brightness = 100,
  blur = 10,
  textShadow = 0,
) {
  const appContainer = document.querySelector(".app");
  const preview = $("bg-drag-preview");
  const handle = $("bg-drag-handle");
  // Cache for early load
  localStorage.setItem(
    "ui_background",
    JSON.stringify({ url, position, dim, brightness, blur, textShadow }),
  );
  // Remove the early-load style once we have the real container
  const earlyStyle = document.getElementById("early-bg");
  if (earlyStyle) earlyStyle.remove();
  if (url && url !== "none" && url !== "") {
    const root = document.documentElement;
    appContainer.style.backgroundImage = `url('${url}')`;
    appContainer.style.backgroundPosition = position;
    appContainer.classList.add("has-bg");
    root.style.setProperty("--bg-dim", dim / 100);
    root.style.setProperty("--bg-brightness", brightness / 100);
    root.style.setProperty("--bg-blur", blur + "px");
    root.style.setProperty("--text-shadow-size", textShadow + "px");
    preview.style.backgroundImage = `url('${url}')`;
    preview.style.backgroundPosition = position;
    const parts = position.split(" ");
    if (parts.length === 2) {
      bgPosPercent.x = parseFloat(parts[0]);
      bgPosPercent.y = parseFloat(parts[1]);
      handle.style.left = bgPosPercent.x + "%";
      handle.style.top = bgPosPercent.y + "%";
    }
    $("btn-remove-bg").classList.remove("hidden");
    $("bg-pos-group").classList.remove("hidden");
    $("bg-visual-controls").classList.remove("hidden");
  } else {
    appContainer.style.backgroundImage = "none";
    appContainer.classList.remove("has-bg");
    $("btn-remove-bg").classList.add("hidden");
    $("bg-pos-group").classList.add("hidden");
    $("bg-visual-controls").classList.add("hidden");
  }
}
// Drag logic
function updateBgPosFromMouse(e) {
  const container = $("bg-drag-container");
  const rect = container.getBoundingClientRect();
  let x = ((e.clientX - rect.left) / rect.width) * 100;
  let y = ((e.clientY - rect.top) / rect.height) * 100;
  x = Math.max(0, Math.min(100, x));
  y = Math.max(0, Math.min(100, y));
  bgPosPercent = { x, y };
  const posStr = `${x.toFixed(1)}% ${y.toFixed(1)}%`;
  $("bg-drag-handle").style.left = x + "%";
  $("bg-drag-handle").style.top = y + "%";
  $("bg-drag-preview").style.backgroundPosition = posStr;
  document.querySelector(".app").style.backgroundPosition = posStr;
}
$("bg-drag-container").onmousedown = (e) => {
  isDraggingBg = true;
  updateBgPosFromMouse(e);
};
window.addEventListener("mousemove", (e) => {
  if (isDraggingBg) updateBgPosFromMouse(e);
});
window.addEventListener("mouseup", () => {
  isDraggingBg = false;
});
// Upload handler
$("cfg-bg-upload").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (event) => {
    const base64 = event.target.result;
    const res = await api("/api/global/background", {
      method: "POST",
      body: { image: base64 },
    });
    if (res.success) {
      const config = await api("/api/global-config");
      const pos = `${bgPosPercent.x}% ${bgPosPercent.y}%`;
      const dim = parseInt($("cfg-bg-dim").value);
      const brightness = parseInt($("cfg-bg-brightness").value);
      const blur = parseInt($("cfg-bg-blur").value);
      const textShadow = parseInt($("cfg-text-shadow").value);
      applyBackground(res.url, pos, dim, brightness, blur, textShadow);
      // Save to global config
      config.ui = config.ui || {};
      config.ui.background = res.url;
      config.ui.background_position = pos;
      config.ui.dim_level = dim;
      config.ui.brightness_level = brightness;
      config.ui.blur_level = blur;
      config.ui.text_shadow_size = textShadow;
      await api("/api/global-config", { method: "PUT", body: config });
      showToast("Background updated!");
    }
  };
  reader.readAsDataURL(file);
};
// Remove handler
$("btn-remove-bg").onclick = async () => {
  await api("/api/global/background", { method: "DELETE" });
  applyBackground(null);
  const config = await api("/api/global-config");
  if (config.ui) delete config.ui.background;
  await api("/api/global-config", { method: "PUT", body: config });
  showToast("Background removed");
};
// Slider live previews
$("cfg-bg-dim").oninput = (e) => {
  $("val-bg-dim").textContent = e.target.value + "%";
  document.documentElement.style.setProperty("--bg-dim", e.target.value / 100);
};
$("cfg-bg-brightness").oninput = (e) => {
  $("val-bg-brightness").textContent = e.target.value + "%";
  document.documentElement.style.setProperty(
    "--bg-brightness",
    e.target.value / 100,
  );
};
$("cfg-bg-blur").oninput = (e) => {
  $("val-bg-blur").textContent = e.target.value + "px";
  document.documentElement.style.setProperty(
    "--bg-blur",
    e.target.value + "px",
  );
};
$("cfg-text-shadow").oninput = (e) => {
  $("val-text-shadow").textContent = e.target.value + "px";
  document.documentElement.style.setProperty(
    "--text-shadow-size",
    e.target.value + "px",
  );
};
// Theme change handler (live preview)
$("cfg-theme").onchange = (e) => {
  applyTheme(e.target.value);
};
// ==========================================
//  Modals & Helpers
// ==========================================
function openModal(id) {
  $(id).classList.remove("hidden");
}
function closeModal(id) {
  $(id).classList.add("hidden");
}
function showConfirm(title, message, onConfirm) {
  $("confirm-title").textContent = translatePhrase(title);
  $("confirm-message").textContent = translatePhrase(message);
  const actions = $("confirm-actions");
  actions.innerHTML = "";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "btn btn-ghost";
  cancelBtn.textContent = translatePhrase("Cancel");
  cancelBtn.onclick = () => closeModal("modal-confirm");
  const confirmBtn = document.createElement("button");
  confirmBtn.className = "btn btn-danger";
  confirmBtn.textContent = translatePhrase("Confirm");
  confirmBtn.onclick = () => {
    closeModal("modal-confirm");
    onConfirm();
  };
  actions.appendChild(cancelBtn);
  actions.appendChild(confirmBtn);
  openModal("modal-confirm");
}
function showToast(msg) {
  const toast = document.createElement("div");
  toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; z-index: 300;
        padding: 12px 20px; border-radius: 8px;
        background: var(--bg-tertiary); border: 1px solid var(--border);
        color: var(--text-primary); font-size: 0.9rem;
        box-shadow: var(--shadow); animation: fadeIn 0.2s;
    `;
  toast.textContent = translatePhrase(msg);
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 2000);
}
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
// ==========================================
//  Tabs
// ==========================================
document.querySelectorAll(".tab").forEach((tab) => {
  // don't attach this listener if it's a global tab
  if (tab.closest("#global-tabs-nav")) return;
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".tab:not(#global-tabs-nav .tab)")
      .forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach((p) => {
      p.classList.remove("active");
      p.classList.add("hidden");
    });
    tab.classList.add("active");
    const pane = $(`tab-${tab.dataset.tab}`);
    pane.classList.remove("hidden");
    pane.classList.add("active");
    localStorage.setItem("lastTab", tab.dataset.tab);
    // Stop polling if switching away from samples (or just reset it)
    if (samplesPollTimer) {
      clearInterval(samplesPollTimer);
      samplesPollTimer = null;
    }
    // Auto-refresh data on tab switch
    if (tab.dataset.tab === "samples") {
      loadSamples();
      samplesPollTimer = setInterval(() => loadSamples(true), 3000);
    }
    if (tab.dataset.tab === "prompts") loadPrompts();
    if (tab.dataset.tab === "tensorboard") checkTensorBoard();
  });
});
// Global tab switching and sync buttons are now handled
// dynamically inside buildGlobalSettingsTabs()
// ==========================================
//  Event Listeners
// ==========================================
$("cfg-enable-sampling").addEventListener("change", (e) => {
  $("group-sample-every").classList.toggle("hidden", !e.target.checked);
});
document.querySelectorAll('input[name="duration-unit"]').forEach((el) => {
  el.addEventListener("change", updateDurationUnit);
});
function updateDurationUnit() {
  const unit = document.querySelector(
    'input[name="duration-unit"]:checked',
  ).value;
  const isEpochs = unit === "epochs";
  $("schedule-epochs").classList.toggle("hidden", !isEpochs);
  $("schedule-steps").classList.toggle("hidden", isEpochs);
  $("container-sample-every-epochs").classList.toggle("hidden", !isEpochs);
  $("container-sample-every-steps").classList.toggle("hidden", isEpochs);
}
// Multiple Datasets
const btnAddDataset = $("btn-add-dataset");
if (btnAddDataset) {
  btnAddDataset.addEventListener("click", () => addSubset(true));
}
// New Job
$("btn-new-job").addEventListener("click", () => {
  const defaultName = nextVersionedName("my_job", 1);
  $("new-job-name").value = defaultName;
  $("new-job-output-name").value = defaultName;
  $("new-job-output-name").dataset.autoValue = defaultName;
  $("new-job-network-module").value = "networks.krona";
  $("new-job-image-dir").value = "";
  $("new-job-max-steps").value = "3000";
  openModal("modal-new-job");
  $("new-job-name").focus();
  $("new-job-name").select();
});
$("new-job-name").addEventListener("input", () => {
  const outputInput = $("new-job-output-name");
  if (outputInput.value === outputInput.dataset.autoValue) {
    outputInput.value = $("new-job-name").value.trim();
    outputInput.dataset.autoValue = outputInput.value;
  }
});
$("btn-create-job").addEventListener("click", async () => {
  const name = $("new-job-name").value.trim();
  if (!name) return;
  const result = await api("/api/jobs", {
    method: "POST",
    body: {
      name,
      output_name: $("new-job-output-name").value.trim(),
      network_module: $("new-job-network-module").value,
      image_dir: $("new-job-image-dir").value.trim(),
      max_train_steps: safeInt($("new-job-max-steps").value, 3000),
    },
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  closeModal("modal-new-job");
  await loadJobs();
  selectJob(result.name);
  showToast("Job created");
});
// Load GPUs from server
async function loadGPUs() {
  const container = $("cfg-gpu-selection");
  try {
    const gpus = await api("/api/system/gpus");
    container.innerHTML = "";
    if (gpus.length === 0) {
      container.innerHTML =
        "<small>No NVIDIA GPUs detected (CPU only).</small>";
      refreshI18n();
      return;
    }
    gpus.forEach((gpu, i) => {
      const card = document.createElement("div");
      card.className = "gpu-card" + (i === 0 ? " selected" : "");
      card.dataset.index = gpu.index;
      card.id = `gpu-card-${gpu.index}`;
      card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <div class="gpu-status">
                    <div class="status-dot"></div>
                    <span class="gpu-status-text">Idle</span>
                </div>
                <input type="checkbox" name="gpu-select" value="${gpu.index}" ${i === 0 ? "checked" : ""} id="gpu-${gpu.index}">
            `;
      const selectOnlyThisGpu = () => {
        document.querySelectorAll('input[name="gpu-select"]').forEach((other) => {
          other.checked = false;
          const otherCard = other.closest(".gpu-card");
          if (otherCard) otherCard.classList.remove("selected");
        });
        const checkbox = card.querySelector("input");
        checkbox.checked = true;
        card.classList.add("selected");
      };
      card.addEventListener("click", (e) => {
        selectOnlyThisGpu();
        updateMultiGPUUI();
        checkDirty();
      });
      container.appendChild(card);
    });
    updateGPUActivity();
    updateMultiGPUUI();
    refreshI18n();
  } catch (err) {
    console.error("Failed to load GPUs:", err);
    container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
    refreshI18n();
  }
}
// Show the correct mode panel, hide the others.
// Panels use only the "hidden" class for visibility — no disabled-section.
function applyMultiGpuMode(mode) {
  const ddpGroup   = $("group-ddp-opts");
  const fsdpGroup  = $("group-fsdp");
  const fsdp2Group = $("group-fsdp2");
  const dsGroup    = $("group-deepspeed");
  const tpGroup    = $("group-tp-sp");
  if (!ddpGroup || !fsdpGroup || !tpGroup) return;

  ddpGroup.classList.toggle("hidden",   mode !== "ddp");
  fsdpGroup.classList.toggle("hidden",  mode !== "fsdp");
  if (fsdp2Group) fsdp2Group.classList.toggle("hidden", mode !== "fsdp2");
  if (dsGroup) dsGroup.classList.toggle("hidden", mode !== "deepspeed");
  tpGroup.classList.toggle("hidden",    mode !== "tp_sp");

  // Keep hidden checkbox in sync so reconcileFSDPConflicts still works
  const fsdpToggle = $("cfg-use-fsdp");
  if (fsdpToggle) fsdpToggle.checked = (mode === "fsdp" || mode === "fsdp2");

  updateCudaDirectForTpSp();
  updateDeepspeedOffloadUI();
}

function updateCudaDirectForTpSp() {
  const cudaGroup  = $("group-cuda-direct");
  const cudaToggle = $("cfg-use-cuda-direct");
  if (!cudaGroup || !cudaToggle) return;

  const mode   = $("cfg-multigpu-mode")?.value;
  const lockCudaDirect = mode === "tp_sp" || mode === "deepspeed";

  cudaGroup.classList.toggle("disabled-section", lockCudaDirect);
  cudaToggle.disabled = lockCudaDirect;
  if (lockCudaDirect) cudaToggle.checked = false;
}

function updateDeepspeedOffloadUI() {
  const optDevice = $("cfg-ds-offload-optimizer-device");
  const paramDevice = $("cfg-ds-offload-param-device");
  const optNvmeGroup = $("ds-offload-opt-nvme-group");
  const paramNvmeGroup = $("ds-offload-param-nvme-group");
  if (!optDevice || !paramDevice || !optNvmeGroup || !paramNvmeGroup) return;

  optNvmeGroup.classList.toggle("hidden", optDevice.value !== "nvme");
  paramNvmeGroup.classList.toggle("hidden", paramDevice.value !== "nvme");
}

function updateMultiGPUUI() {
  const modeGroup   = $("group-multigpu-mode");
  const cudaGroup   = $("group-cuda-direct");
  const cudaToggle  = $("cfg-use-cuda-direct");
  if (!cudaGroup || !cudaToggle) return;

  const count = document.querySelectorAll('input[name="gpu-select"]:checked').length;

  if (count > 1) {
    if (modeGroup) modeGroup.classList.remove("disabled-section");
    cudaGroup.classList.remove("disabled-section");
    // Keep tp_degree in sync with actual GPU count — the server uses GPU count
    // directly, so this just keeps the display honest.
    const tpDegreeInput = $("cfg-tp-degree");
    if (tpDegreeInput) tpDegreeInput.value = count;
    const tpBackend = $("cfg-tp-backend");
    if (tpBackend && !tpBackend.value) tpBackend.value = "auto";
    const spToggle = $("cfg-sequence-parallel");
    if (spToggle) spToggle.checked = true;
    applyMultiGpuMode($("cfg-multigpu-mode")?.value || "ddp");
  } else {
    // Single GPU — disable mode selector and cuda-direct, hide all panels
    if (modeGroup) modeGroup.classList.add("disabled-section");
    cudaGroup.classList.add("disabled-section");
    cudaToggle.checked = false;
    ["group-ddp-opts", "group-fsdp", "group-fsdp2", "group-deepspeed", "group-tp-sp"].forEach(id => {
      const el = $(id);
      if (el) el.classList.add("hidden");
    });
    const fsdpToggle = $("cfg-use-fsdp");
    if (fsdpToggle) fsdpToggle.checked = false;
  }
}
// Global state for restoring manual offloading values
let lastBlocksValue = "0";
let lastActivationValue = "none";
function reconcileFSDPConflicts() {
  const fsdpToggle = $("cfg-use-fsdp");
  const cudaDirectToggle = $("cfg-use-cuda-direct");
  const blocksInput = $("cfg-blocks-to-swap");
  const activationSelect = $("cfg-activation-offload");
  const blocksGroup = $("group-blocks-to-swap");
  const activationGroup = $("group-activation-offload");
  const strategySelect = $("cfg-fsdp-sharding-strategy");
  if (!fsdpToggle) return;
  if (fsdpToggle.checked) {
    // SAVE CURRENT VALUES before zeroing them (but only if they aren't already 0/none due to a previous toggle)
    if (blocksInput.value !== "0") lastBlocksValue = blocksInput.value;
    if (activationSelect.value !== "none")
      lastActivationValue = activationSelect.value;
    // FORCE TO 0 / NONE
    blocksInput.value = "0";
    activationSelect.value = "none";
    // DISABLE GROUPS
    if (blocksGroup) blocksGroup.classList.add("disabled-section");
    if (activationGroup) activationGroup.classList.add("disabled-section");
  } else {
    // RESTORE PREVIOUS VALUES
    blocksInput.value = lastBlocksValue;
    activationSelect.value = lastActivationValue;
    // ENABLE GROUPS
    if (blocksGroup) blocksGroup.classList.remove("disabled-section");
    if (activationGroup) activationGroup.classList.remove("disabled-section");
  }
  // SHARDING STRATEGY FILTERING (CUDA Direct / Windows compatibility)
  if (strategySelect) {
    const h1 = strategySelect.querySelector('option[value="4"]');
    const h2 = strategySelect.querySelector('option[value="5"]');
    if (cudaDirectToggle && cudaDirectToggle.checked) {
      if (h1) h1.disabled = true;
      if (h2) h2.disabled = true;
      // If current selection was a hybrid one, reset to FULL_SHARD
      if (strategySelect.value === "4" || strategySelect.value === "5") {
        strategySelect.value = "1";
      }
    } else {
      if (h1) h1.disabled = false;
      if (h2) h2.disabled = false;
    }
    // Force update of info box
    if (window.updateFSDPInfoBox) window.updateFSDPInfoBox();
  }
  // TORCH COMPILE vs CUDA DIRECT (mutually exclusive)
  const torchCompileToggle = $("cfg-torch-compile");
  const torchCompileGroup = $("group-torch-compile");
  if (cudaDirectToggle && torchCompileToggle && torchCompileGroup) {
    if (cudaDirectToggle.checked) {
      // Save state before disabling
      if (torchCompileToggle.checked) window._lastTorchCompile = true;
      torchCompileToggle.checked = false;
      torchCompileGroup.classList.add("disabled-section");
    } else {
      torchCompileGroup.classList.remove("disabled-section");
      // Restore previous state if it was saved
      if (window._lastTorchCompile) {
        torchCompileToggle.checked = true;
        window._lastTorchCompile = false;
      }
    }
  }
}
// Global initialization for extra elements
document.addEventListener("DOMContentLoaded", () => {
  const fsdpToggle = $("cfg-use-fsdp");
  const cudaDirectToggle = $("cfg-use-cuda-direct");
  const blocksInput = $("cfg-blocks-to-swap");
  const activationSelect = $("cfg-activation-offload");
  // EXPOSE for usage in other scripts or reactive functions
  window.updateFSDPInfoBox = () => {
    const strategySelect = $("cfg-fsdp-sharding-strategy");
    const fsdpInfo = $("cfg-fsdp-strategy-info");
    if (!strategySelect || !fsdpInfo) return;
    const strategyMap = {
      1: "<strong>FULL_SHARD</strong>: Shards optimizer states, gradients and parameters across all GPUs. Best for maximum VRAM savings.",
      2: "<strong>SHARD_GRAD_OP</strong>: Shards optimizer states and gradients (equivalent to ZeRO-2). Faster than FULL_SHARD but uses more VRAM.",
      3: "<strong>NO_SHARD</strong>: <strong>: Same as DDP</strong> Not recommended for real training",
      4: "<strong>HYBRID_SHARD</strong>: Shards optimizer states, gradients and parameters within each node while each node has a full copy. Use for multi-node setups.",
      5: "<strong>HYBRID_SHARD_ZERO2</strong>: Shards optimizer states and gradients within each node while each node has a full copy.",
    };
    fsdpInfo.innerHTML =
      strategyMap[strategySelect.value] || "Select a strategy to see details.";
  };
  // Multi-GPU mode selector
  const modeSelect = $("cfg-multigpu-mode");
  if (modeSelect) {
    modeSelect.addEventListener("change", (e) => {
      applyMultiGpuMode(e.target.value);
      reconcileFSDPConflicts();
    });
  }
  const dsOptDevice = $("cfg-ds-offload-optimizer-device");
  const dsParamDevice = $("cfg-ds-offload-param-device");
  if (dsOptDevice) dsOptDevice.addEventListener("change", updateDeepspeedOffloadUI);
  if (dsParamDevice) dsParamDevice.addEventListener("change", updateDeepspeedOffloadUI);
  // fsdpToggle is a hidden input
  if (cudaDirectToggle) {
    cudaDirectToggle.addEventListener("change", reconcileFSDPConflicts);
    // Auto Wrap Policy visibility toggle
    $("cfg-fsdp-auto-wrap-policy").addEventListener("change", (e) => {
      $("fsdp-layer-wrap-group").classList.toggle(
        "hidden",
        e.target.value !== "TRANSFORMER_BASED_WRAP",
      );
      $("fsdp-size-wrap-group").classList.toggle(
        "hidden",
        e.target.value !== "SIZE_BASED_WRAP",
      );
    });
    const fsdp2WrapPolicy = $("cfg-fsdp2-auto-wrap-policy");
    if (fsdp2WrapPolicy) {
      fsdp2WrapPolicy.addEventListener("change", (e) => {
        $("fsdp2-layer-wrap-group").classList.toggle(
          "hidden",
          e.target.value !== "TRANSFORMER_BASED_WRAP",
        );
        $("fsdp2-size-wrap-group").classList.toggle(
          "hidden",
          e.target.value !== "SIZE_BASED_WRAP",
        );
      });
    }
  }
  // Manual value tracking: Update "last known" value when user changes it MANUALLY
  if (blocksInput) {
    blocksInput.addEventListener("change", () => {
      if (!fsdpToggle || !fsdpToggle.checked)
        lastBlocksValue = blocksInput.value;
    });
  }
  if (activationSelect) {
    activationSelect.addEventListener("change", () => {
      if (!fsdpToggle || !fsdpToggle.checked)
        lastActivationValue = activationSelect.value;
    });
  }
  const strategySelect = $("cfg-fsdp-sharding-strategy");
  if (strategySelect) {
    strategySelect.addEventListener("change", window.updateFSDPInfoBox);
    window.updateFSDPInfoBox(); // Initial call
  }
  updateDeepspeedOffloadUI();
  // Initial reconcile call
  reconcileFSDPConflicts();
});
// Load GPUs for generation (separate from training GPU selection)
async function loadGenGPUs() {
  const container = $("gen-gpu-selection");
  if (!container) return;
  try {
    const gpus = await api("/api/system/gpus");
    container.innerHTML = "";
    if (gpus.length === 0) {
      container.innerHTML = "<small>No NVIDIA GPUs detected.</small>";
      refreshI18n();
      return;
    }
    gpus.forEach((gpu, i) => {
      const card = document.createElement("div");
      card.className = "gpu-card" + (i === 0 ? " selected" : "");
      card.dataset.index = gpu.index;
      card.id = `gen-gpu-card-${gpu.index}`;
      card.innerHTML = `
                <div class="gpu-index">GPU ${gpu.index}</div>
                <div class="gpu-name" title="${gpu.name}">${gpu.name}</div>
                <div class="gpu-mem">${gpu.memory}</div>
                <input type="checkbox" name="gen-gpu-select" value="${gpu.index}" ${i === 0 ? "checked" : ""} id="gen-gpu-${gpu.index}">
            `;
      const cb = card.querySelector("input[type=checkbox]");
      card.addEventListener("click", (e) => {
        document.querySelectorAll('input[name="gen-gpu-select"]').forEach((other) => {
          other.checked = false;
          const otherCard = other.closest(".gpu-card");
          if (otherCard) otherCard.classList.remove("selected");
        });
        cb.checked = true;
        card.classList.add("selected");
        updateGenGPULabel();
      });
      container.appendChild(card);
    });
    updateGenGPULabel();
    refreshI18n();
  } catch (err) {
    console.error("Failed to load gen GPUs:", err);
    container.innerHTML = `<small style="color:red">Error: ${err.message}</small>`;
    refreshI18n();
  }
}
function getSelectedGenGPUs() {
  const checked = document.querySelectorAll(
    'input[name="gen-gpu-select"]:checked',
  );
  return Array.from(checked)[0]?.value || "";
}
function restoreGenGPUSelection(gpuIds) {
  if (!gpuIds) return;
  const id = gpuIds.split(",").map((s) => s.trim()).filter(Boolean)[0];
  document.querySelectorAll('input[name="gen-gpu-select"]').forEach((cb) => {
    cb.checked = cb.value === id;
    const card = cb.closest(".gpu-card");
    if (card) card.classList.toggle("selected", cb.checked);
  });
  updateGenGPULabel();
}
function updateGenGPULabel() {
  const label = $("gen-gpu-mode-label");
  const optionsDiv = $("gen-multi-gpu-options");
  if (!label) return;
  const selected = document.querySelectorAll(
    'input[name="gen-gpu-select"]:checked',
  );
  label.textContent = selected.length === 1 ? `— ${translatePhrase("Single GPU")}` : "";
  label.style.color = "";
  if (optionsDiv) optionsDiv.style.display = "none";
}
async function updateGPUActivity() {
  try {
    const res = await fetch("/api/gpu/activity");
    if (!res.ok) return;
    const activity = await res.json(); // { "0": "training", "1": "sampling" }
    document.querySelectorAll(".gpu-card").forEach((card) => {
      const index = card.dataset.index;
      const status = activity[index] || "idle";
      const textEl = card.querySelector(".gpu-status-text");
      card.classList.remove("active-training", "active-sampling");
      if (status === "training") {
        card.classList.add("active-training");
        textEl.textContent = translatePhrase("Training");
      } else if (status === "sampling") {
        card.classList.add("active-sampling");
        textEl.textContent = translatePhrase("Sampling");
      } else {
        textEl.textContent = translatePhrase("Idle");
      }
    });
  } catch (err) {
    // Silently fail polling
  }
}
$("btn-cancel-job").addEventListener("click", () =>
  closeModal("modal-new-job"),
);
// Enter key in new job name
$("new-job-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("btn-create-job").click();
});
// Save
$("btn-save").addEventListener("click", saveJob);
// Keyboard shortcut: Ctrl+S
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "s") {
    e.preventDefault();
    if (currentJob && isDirty) saveJob();
  }
});
// Clone
$("btn-clone").addEventListener("click", () => {
  if (!currentJob) return;
  // Open Modal
  $("clone-job-name").value = nextCloneName(currentJob);
  openModal("modal-clone-job");
  $("clone-job-name").focus();
  $("clone-job-name").select();
});
// Confirm Clone
$("btn-confirm-clone").addEventListener("click", async () => {
  const newName = $("clone-job-name").value.trim();
  if (!newName) return;
  const result = await api(`/api/jobs/${currentJob}/clone`, {
    method: "POST",
    body: { newName: newName },
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  closeModal("modal-clone-job");
  await loadJobs();
  selectJob(result.name);
  showToast("Job cloned");
});
// Cancel Clone
$("btn-cancel-clone").addEventListener("click", () =>
  closeModal("modal-clone-job"),
);
// Enter key in clone job name
$("clone-job-name").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("btn-confirm-clone").click();
});
// Delete
$("btn-delete").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Delete Job",
    `Delete "${currentJob}" and all its files? This cannot be undone.`,
    async () => {
      const deletedJob = currentJob;
      await api(`/api/jobs/${deletedJob}`, { method: "DELETE" });
      // Clean up all localStorage keys for the deleted job
      localStorage.removeItem(`prompt_transient_${deletedJob}`);
      localStorage.removeItem(`sample_order_${deletedJob}`);
      clearCurrentJobSelection();
      await loadJobs();
      showToast("Job deleted");
    },
  );
});
// Train
$("btn-run").addEventListener("click", async () => {
  if (!currentJob) return;
  let warningMsg = "";
  // Check sampling Logic
  if (($("cfg-enable-sampling").checked || $("cfg-sample-at-first").checked) && currentPrompts.length === 0) {
    warningMsg =
      "Sampling is enabled but no prompts are defined.\n\nContinue training without generating samples...\n\n";
  }
  // Auto-save first
  if (isDirty) await saveJob();
  const result = await api(`/api/jobs/${currentJob}/train/start`, {
    method: "POST",
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  updateRunningState(true);
  consoleOutput.textContent = "";
  if (warningMsg) appendConsole(warningMsg);
  // Auto-switch to console tab
  document.querySelector('[data-tab="console"]').click();
  showToast("Training started");
});
// Generate
$("btn-gen-sample").addEventListener("click", async () => {
  if (!currentJob) return;
  savePromptTransientSettings();
  if (isDirty) await saveJob();
  if (currentPrompts.length === 0) {
    showToast("Add sample prompts first");
    return;
  }
  const payload = {};
  const loraPath = $("gen-lora-select").value;
  if (loraPath) {
    payload.network_weights = loraPath;
    payload.network_mul = parseFloat($("gen-lora-mul").value) || 1.0;
  }
  // Add Anima generation params
  payload.flow_shift = parseFloat($("cfg-flow-shift").value) || 3.0;
  payload.flash_attn = $("gen-flash-attn").checked;
  payload.sage_attn = $("gen-sage-attn").checked;
  payload.gen_gpu_ids = getSelectedGenGPUs();
  payload.gen_multi_gpu_mode = $("gen-multi-gpu-mode").value;
  const result = await api(`/api/jobs/${currentJob}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: Object.assign(payload, {
      keep_loaded: $("chk-keep-loaded").checked,
    }),
  });
  if (result.error) {
    alert(result.error);
    return;
  }
  appendConsole(
    `Starting generation...\n${loraPath ? `Using LoRA: ${loraPath} (x${payload.network_mul})` : "(Using base model)"}\nFlow Shift: ${payload.flow_shift}\n\n`,
  );
  showToast("Generation started");
});
// Unload Model
$("btn-unload-model").addEventListener("click", async () => {
  if (!currentJob) return;
  showToast("Unloading model...");
  const result = await api(`/api/jobs/${currentJob}/unload`, {
    method: "POST",
  });
  if (result.success) {
    showToast(result.message || "Model unloaded");
  } else {
    alert(result.error);
  }
});
$("btn-refresh-checkpoints").addEventListener("click", () => {
  loadCheckpoints();
  showToast("Checkpoints refreshed");
});
// Stop
$("btn-stop").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Stop Training",
    `Stop training for "${currentJob}"?`,
    async () => {
      await api(`/api/jobs/${currentJob}/train/stop`, { method: "POST" });
      updateRunningState(false);
      showToast("Training stopped");
    },
  );
});
// Console clear
$("btn-clear-console").addEventListener("click", () => {
  consoleOutput.textContent = "Waiting for training to start...";
});
// Samples refresh
$("btn-refresh-samples").addEventListener("click", loadSamples);
// TensorBoard
$("btn-tb-launch").addEventListener("click", launchTensorBoard);
$("btn-tb-stop").addEventListener("click", () => {
  showConfirm(
    "Stop TensorBoard",
    "Stop the TensorBoard server for this job?",
    stopTensorBoard,
  );
});
$("btn-tb-open").addEventListener("click", () => {
  if (tbUrl) window.open(tbUrl, "_blank");
});
// Global Settings
$("ui-language").addEventListener("change", (e) => {
  setLanguage(e.target.value);
});
$("btn-global-settings").addEventListener("click", () => {
  loadGlobalSettings();
  openModal("modal-global-settings");
});
$("btn-close-global").addEventListener("click", () =>
  closeModal("modal-global-settings"),
);
$("btn-save-global").addEventListener("click", saveGlobalSettings);
// Prompts
$("btn-add-prompt").addEventListener("click", addPrompt);
$("btn-apply-global").addEventListener("click", applyGlobalSettings);
// Persistence for Prompt Tab settings
[
  "gen-lora-select",
  "gen-lora-mul",
  "chk-keep-loaded",
  "gen-flash-attn",
  "gen-multi-gpu-mode",
  "global-w",
  "global-h",
  "global-s",
  "global-l",
  "global-d",
].forEach((id) => {
  $(id).addEventListener("change", savePromptTransientSettings);
  if ($(id).tagName === "INPUT") {
    $(id).addEventListener("input", savePromptTransientSettings);
  }
});
// Job Settings
$("btn-open-folder").addEventListener("click", async () => {
  if (!currentJob) return;
  await api(`/api/jobs/${currentJob}/open-folder`, { method: "POST" });
});
$("btn-clear-logs").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Clear Logs",
    "Delete all TensorBoard logs for this job?",
    async () => {
      await api(`/api/jobs/${currentJob}/clear-logs`, { method: "POST" });
      showToast("Logs cleared");
    },
  );
});
$("btn-reset-config").addEventListener("click", () => {
  if (!currentJob) return;
  showConfirm(
    "Reset Config",
    "Reset all settings to template defaults?",
    async () => {
      await api(`/api/jobs/${currentJob}/reset-config`, { method: "POST" });
      selectJob(currentJob);
      showToast("Config reset to defaults");
    },
  );
});
// Close modals on backdrop click
document.querySelectorAll(".modal").forEach((modal) => {
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.add("hidden");
    }
  });
});
// ==========================================
//  Progressive Resolution Schedule
// ==========================================
function renderProgressivePhases() {
  const resList = ($("cfg-resolution").value || "")
    .split(",").map(r => parseInt(r.trim())).filter(r => r > 0);
  const container = $("progressive-reso-phases");
  if (!container) return;

  // Preserve existing fraction values by index before clearing
  const existing = Array.from(container.querySelectorAll(".prog-reso-frac"))
    .map(el => parseFloat(el.value) || 0);

  container.innerHTML = "";

  if (resList.length < 2) {
    container.innerHTML = '<small>Enter at least 2 resolutions above to configure phases.</small>';
    updateProgressiveSum();
    return;
  }

  const defaultFrac = +(1 / resList.length).toFixed(2);

  // All phases go in a single form-row so they appear side-by-side
  const row = document.createElement("div");
  row.className = "form-row";

  resList.forEach((r, i) => {
    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("label");
    label.textContent = `${r}px`;

    const input = document.createElement("input");
    input.type = "number";
    input.className = "prog-reso-frac";
    input.min = "0.01";
    input.max = "0.99";
    input.step = "0.01";
    input.value = (existing[i] !== undefined && existing[i] > 0)
      ? existing[i].toFixed(2) : defaultFrac.toFixed(2);

    const hint = document.createElement("small");
    hint.textContent = `${Math.round(parseFloat(input.value) * 100)}% of steps`;

    input.addEventListener("input", () => {
      hint.textContent = `${Math.round(parseFloat(input.value) * 100)}% of steps`;
      updateProgressiveSum();
    });

    group.appendChild(label);
    group.appendChild(input);
    group.appendChild(hint);
    row.appendChild(group);
  });

  container.appendChild(row);

  // Restore fractions when loading from a saved config
  if (window._pendingProgressiveSchedule) {
    const fracs = window._pendingProgressiveSchedule
      .split(",").map(p => parseFloat(p.split(":")[1]) || 0);
    container.querySelectorAll(".prog-reso-frac").forEach((inp, i) => {
      if (fracs[i] !== undefined) {
        inp.value = fracs[i].toFixed(2);
        inp.dispatchEvent(new Event("input"));
      }
    });
    window._pendingProgressiveSchedule = null;
  }

  updateProgressiveSum();
}

function updateProgressiveSum() {
  const inputs = document.querySelectorAll(".prog-reso-frac");
  const sum = Array.from(inputs).reduce((acc, el) => acc + (parseFloat(el.value) || 0), 0);
  const hint = $("progressive-reso-sum-hint");
  if (!hint) return;
  const ok = Math.abs(sum - 1.0) < 0.015;
  const sumStr = `Sum: ${sum.toFixed(2)}`;
  // Show sum inline in the hint text with colour
  hint.innerHTML = `Each fraction is the portion of total steps for that resolution. Must sum to 1.0. &nbsp;<span style="font-weight:600;color:${ok ? "var(--success,#4caf50)" : "var(--error,#f44336)"}">${sumStr}</span>`;
}

// Toggle panel visibility and re-render phases
document.addEventListener("change", (e) => {
  if (e.target.id === "cfg-progressive-reso") {
    const panel = $("progressive-reso-panel");
    if (e.target.checked) {
      panel.classList.remove("hidden");
      renderProgressivePhases();
    } else {
      panel.classList.add("hidden");
    }
  }
});

// Re-render phases when the resolution list changes
document.addEventListener("input", (e) => {
  if (e.target.id === "cfg-resolution" && $("cfg-progressive-reso")?.checked) {
    renderProgressivePhases();
  }
});

// ==========================================
//  Init
// ==========================================
async function init() {
  // 1. FAST LOAD: Apply cached visual settings immediately (Flicker prevention)
  const cachedTheme = localStorage.getItem("ui_theme");
  if (cachedTheme) applyTheme(cachedTheme);
  const cachedBg = localStorage.getItem("ui_background");
  if (cachedBg) {
    try {
      const bg = JSON.parse(cachedBg);
      applyBackground(
        bg.url,
        bg.position,
        bg.dim,
        bg.brightness,
        bg.blur,
        bg.textShadow,
      );
    } catch (e) { }
  }
  // 2. Normal Init
  connectWS();
  applyI18n();
  await loadJobs();
  // Start status polling
  setInterval(updateGPUActivity, 3000);
  // Watch for config changes
  document.addEventListener("input", (e) => {
    if (e.target.id && e.target.id.startsWith("cfg-")) {
      checkDirty();
    }
  });
  document.addEventListener("change", (e) => {
    if (e.target.id && e.target.id.startsWith("cfg-")) {
      checkDirty();
    }
  });
  // Optimizer custom bindings
  $("cfg-optimizer").addEventListener("change", updateOptimizerOptions);
  $("cfg-lr-scheduler").addEventListener("change", updateLrSchedulerOptions);
  // Activation offload <-> blocks to swap mutual exclusivity
  $("cfg-activation-offload").addEventListener(
    "change",
    updateActivationOffloadUI,
  );
  // Discard Button
  $("btn-discard").addEventListener("click", discardChanges);
  // Mutual exclusivity for Flash/Sage Attention
  const enforceMutualAttention = (flashId, sageId) => {
    const flash = $(flashId);
    const sage = $(sageId);
    if (!flash || !sage) return;
    flash.addEventListener("change", () => {
      if (flash.checked) sage.checked = false;
      if (flashId.startsWith("gen-")) savePromptTransientSettings();
    });
    sage.addEventListener("change", () => {
      if (sage.checked) flash.checked = false;
      if (flashId.startsWith("gen-")) savePromptTransientSettings();
    });
  };
  enforceMutualAttention("gen-flash-attn", "gen-sage-attn");
  // Restore Job
  const lastJob = localStorage.getItem("lastJob");
  if (lastJob) {
    const jobExists = Array.from(
      document.querySelectorAll(".job-item .job-name"),
    ).some((el) => el.textContent === lastJob);
    if (jobExists) {
      await selectJob(lastJob);
    } else {
      localStorage.removeItem("lastJob");
    }
  }
  // Restore Tab
  const lastTab = localStorage.getItem("lastTab");
  if (lastTab && currentJob) {
    const tabEl = document.querySelector(`.tab[data-tab="${lastTab}"]`);
    if (tabEl) tabEl.click();
  }
  // 3. Sync Settings: Load from server and refresh cache
  if (!archRegistry) {
    archRegistry = await api("/api/architectures");
    buildGlobalSettingsTabs(archRegistry);
  }
  const globalConfig = await api("/api/global-config");
  if (globalConfig?.ui?.theme) {
    applyTheme(globalConfig.ui.theme);
  }
  // Apply saved background
  if (globalConfig?.ui?.background) {
    applyBackground(
      globalConfig.ui.background,
      globalConfig.ui.background_position || "50% 50%",
      globalConfig.ui.dim_level ?? 70,
      globalConfig.ui.brightness_level ?? 100,
      globalConfig.ui.blur_level ?? 10,
      globalConfig.ui.text_shadow_size ?? 0,
    );
  } else {
    applyBackground("none");
  }
  if (shouldPromptForGlobalSettings(globalConfig)) {
    await loadGlobalSettings();
    openModal("modal-global-settings");
  }
}
init();
window.addEventListener("beforeunload", () => savePromptTransientSettings());
