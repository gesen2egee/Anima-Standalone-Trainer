const WINDOWS_ACCESS_VIOLATION_CODES = new Set([
  3221225477,
  -1073741819,
]);

function hasCompletionEvidence(logText) {
  const text = String(logText || "");
  return /steps:\s*100%\|/.test(text)
    && /saving checkpoint:[\s\S]*\.safetensors/i.test(text);
}

function isSuccessfulTrainingExit({ code, stoppedByRequest, logText }) {
  if (stoppedByRequest) return false;
  if (code === 0) return true;
  if (WINDOWS_ACCESS_VIOLATION_CODES.has(Number(code))) {
    return hasCompletionEvidence(logText);
  }
  return false;
}

module.exports = {
  hasCompletionEvidence,
  isSuccessfulTrainingExit,
};
