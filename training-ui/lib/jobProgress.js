const fs = require("fs");
const path = require("path");

const CHECKPOINT_EXTENSIONS = [".safetensors", ".ckpt", ".pt"];

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function parseProgressName(name, outputName) {
  const base = String(outputName || "").trim();
  if (!base) return null;

  const escaped = escapeRegExp(base);
  const checkpointSuffix = `(?:${CHECKPOINT_EXTENSIONS.map((ext) => escapeRegExp(ext)).join("|")})`;
  const stepMatch = name.match(new RegExp(`^${escaped}-step(\\d+)(?:-state|${checkpointSuffix})$`));
  if (stepMatch) return { unit: "steps", value: Number.parseInt(stepMatch[1], 10) };

  const epochMatch = name.match(new RegExp(`^${escaped}-(\\d+)(?:-state|${checkpointSuffix})$`));
  if (epochMatch) return { unit: "epochs", value: Number.parseInt(epochMatch[1], 10) };

  return null;
}

function newestProgressEntry(outputDir, outputName, unit) {
  if (!fs.existsSync(outputDir)) return null;

  try {
    return fs.readdirSync(outputDir)
      .map((name) => {
        const parsed = parseProgressName(name, outputName);
        if (!parsed || parsed.unit !== unit) return null;

        const fullPath = path.join(outputDir, name);
        try {
          const stat = fs.statSync(fullPath);
          if (!stat.isFile() && !stat.isDirectory()) return null;
          return { ...parsed, name, mtime: stat.mtimeMs };
        } catch (_) {
          return null;
        }
      })
      .filter(Boolean)
      .sort((a, b) => b.mtime - a.mtime)[0] || null;
  } catch (_) {
    return null;
  }
}

function findLatestProgressSave(outputDir, outputName, unit) {
  const latest = newestProgressEntry(outputDir, outputName, unit);
  if (!latest) return null;
  return {
    unit: latest.unit,
    value: latest.value,
    name: latest.name,
  };
}

function positiveInt(value) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function emptyProgress(unit, target) {
  return {
    unit,
    current: 0,
    target,
    percent: 0,
    label: "",
    saveName: "",
  };
}

function calculateJobProgress(outputDir, options = {}) {
  const maxTrainSteps = positiveInt(options.maxTrainSteps);
  const maxTrainEpochs = positiveInt(options.maxTrainEpochs);
  const unit = maxTrainSteps > 0 ? "steps" : "epochs";
  const target = maxTrainSteps > 0 ? maxTrainSteps : maxTrainEpochs;

  if (target <= 0) return emptyProgress(unit, 0);

  const latest = findLatestProgressSave(outputDir, options.outputName, unit);
  if (!latest) return emptyProgress(unit, target);

  const percent = Math.min(100, Math.max(0, Math.round((latest.value / target) * 100)));
  const labelUnit = unit === "steps" ? "step" : "epoch";
  return {
    unit,
    current: latest.value,
    target,
    percent,
    label: `${labelUnit} ${latest.value} / ${target}`,
    saveName: latest.name,
  };
}

module.exports = {
  calculateJobProgress,
  findLatestProgressSave,
  parseProgressName,
};
