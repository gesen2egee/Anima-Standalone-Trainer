const WINDOWS_ACCESS_VIOLATION_CODES = new Set([
  3221225477,
  -1073741819,
  3221226505,
  -1073740791,
]);

function hasCompletionEvidence(logText) {
  const text = String(logText || "");
  
  // 1. 確認是否有明確的模型已儲存的日誌標記 (Anima, Flux, SDXL 等皆適用)
  if (/model saved\./i.test(text)) return true;
  if (/save trained model as/i.test(text)) return true;
  
  // 2. 檢查進度條與 checkpoint 儲存
  const has100Percent = /100%\|/.test(text);
  const hasSavingCheckpoint = /saving checkpoint/i.test(text);
  if (has100Percent && hasSavingCheckpoint) return true;
  
  // 3. 原本的嚴格匹配條件 (做為 fallback)
  return /steps:\s*100%\|/.test(text)
    && /saving checkpoint:[\s\S]*\.safetensors/i.test(text);
}

function isSuccessfulTrainingExit({ code, stoppedByRequest, logText }) {
  if (stoppedByRequest) return false;
  if (code === 0) return true;
  
  // 如果是 Windows 下常見的 Python/PyTorch 資源釋放崩潰代碼，若有完成證據則視為成功
  if (WINDOWS_ACCESS_VIOLATION_CODES.has(Number(code))) {
    return hasCompletionEvidence(logText);
  }
  
  // 其他非 0 狀態碼（例如 powershell 封裝過的 code），若有明確的完成證據亦視為成功
  return hasCompletionEvidence(logText);
}

module.exports = {
  hasCompletionEvidence,
  isSuccessfulTrainingExit,
};

