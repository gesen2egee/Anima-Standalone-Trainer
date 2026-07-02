const fs = require("fs");
const path = require("path");

function matchesOutputName(name, outputName, suffix) {
  const base = String(outputName || "").trim();
  if (!base) return name.endsWith(suffix);
  return name === `${base}${suffix}` || (name.startsWith(`${base}-`) && name.endsWith(suffix));
}

function newestExisting(outputDir, predicate) {
  if (!fs.existsSync(outputDir)) return null;
  try {
    return fs.readdirSync(outputDir)
      .map((name) => {
        const fullPath = path.join(outputDir, name);
        try {
          const stat = fs.statSync(fullPath);
          return predicate(name, stat) ? { path: fullPath, mtime: stat.mtimeMs } : null;
        } catch (_) {
          return null;
        }
      })
      .filter(Boolean)
      .sort((a, b) => b.mtime - a.mtime)[0]?.path || null;
  } catch (_) {
    return null;
  }
}

function findLatestStateDir(outputDir, outputName) {
  return newestExisting(outputDir, (name, stat) => (
    stat.isDirectory() && matchesOutputName(name, outputName, "-state")
  ));
}

const CHECKPOINT_EXTENSIONS = [".safetensors", ".ckpt", ".pt"];

function findLatestCheckpointFile(outputDir, outputName) {
  return newestExisting(outputDir, (name, stat) => (
    stat.isFile() && CHECKPOINT_EXTENSIONS.some((suffix) => matchesOutputName(name, outputName, suffix))
  ));
}

function findAutoResumeSource(outputDir, outputName, { allowCheckpoint = false } = {}) {
  const statePath = findLatestStateDir(outputDir, outputName);
  if (statePath) return { type: "state", path: statePath };

  if (allowCheckpoint) {
    const checkpointPath = findLatestCheckpointFile(outputDir, outputName);
    if (checkpointPath) return { type: "checkpoint", path: checkpointPath };
  }

  return null;
}

module.exports = {
  findAutoResumeSource,
  findLatestCheckpointFile,
  findLatestStateDir,
};
