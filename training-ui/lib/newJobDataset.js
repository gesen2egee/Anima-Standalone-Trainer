const fs = require('fs');
const path = require('path');

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']);

function stripQuotes(value) {
  if (typeof value !== 'string') return value;
  return value.replace(/^['"]+|['"]+$/g, '');
}

function normalizeCaptionPrefix(value) {
  const triggerWords = String(value || '').trim();
  if (!triggerWords) return '';
  return /,\s*$/.test(triggerWords) ? triggerWords : `${triggerWords},`;
}

function collectImageFiles(dir) {
  const files = [];
  if (!dir || !fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectImageFiles(fullPath));
    } else if (IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(fullPath);
    }
  }
  return files;
}

function countDirectImages(dir) {
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return 0;
  return fs.readdirSync(dir, { withFileTypes: true }).filter((entry) => (
    entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
  )).length;
}

function listFirstLevelDirs(nativeDir) {
  if (!nativeDir || !fs.existsSync(nativeDir) || !fs.statSync(nativeDir).isDirectory()) return [];
  return fs.readdirSync(nativeDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function parseBatchFolderName(folderName) {
  const match = String(folderName || '').match(/^(\d+)_(.+)$/);
  if (!match) return null;
  const repeats = Number.parseInt(match[1], 10);
  const triggerWords = match[2].trim();
  if (!Number.isFinite(repeats) || repeats < 1 || !triggerWords) return null;
  return { repeats, triggerWords };
}

function joinDisplayPath(parent, child) {
  const base = String(parent || '').replace(/[\\\/]+$/, '');
  if (/^[A-Za-z]:[\\\/]/.test(base) || base.startsWith('\\\\')) {
    return path.win32.join(base, child);
  }
  return path.join(base, child);
}

function withBaseSubset(baseSubset, values) {
  return {
    ...(baseSubset || {}),
    num_repeats: 1,
    ...(values || {}),
  };
}

function buildBatchSubsets({ imageDir, nativeImageDir, autoBalanceRepeats, baseSubset }) {
  const subsets = [];
  for (const folderName of listFirstLevelDirs(nativeImageDir)) {
    const parsed = parseBatchFolderName(folderName);
    if (!parsed) continue;

    const nativeChildPath = path.join(nativeImageDir, folderName);
    const imageCount = collectImageFiles(nativeChildPath).length;
    if (imageCount < 1) continue;

    subsets.push(withBaseSubset(baseSubset, {
      image_dir: joinDisplayPath(imageDir, folderName),
      num_repeats: autoBalanceRepeats
        ? Math.max(1, Math.ceil(100 / imageCount))
        : parsed.repeats,
      caption_prefix: normalizeCaptionPrefix(parsed.triggerWords),
    }));
  }
  return subsets;
}

function resolveSingleChildFallback(imageDir, nativeImageDir) {
  if (countDirectImages(nativeImageDir) > 0) return imageDir;

  const childDirs = listFirstLevelDirs(nativeImageDir);
  if (childDirs.length !== 1) return imageDir;

  const onlyChild = childDirs[0];
  const nativeChildPath = path.join(nativeImageDir, onlyChild);
  if (collectImageFiles(nativeChildPath).length < 1) return imageDir;
  return joinDisplayPath(imageDir, onlyChild);
}

function buildNewJobSubsets({
  imageDir = '',
  triggerCaptionPrefix = '',
  batchImport = false,
  autoBalanceRepeats = false,
  baseSubset = {},
  toNativePath = (value) => value,
} = {}) {
  const cleanImageDir = stripQuotes(String(imageDir || '').trim());
  const nativeImageDir = toNativePath(cleanImageDir);
  const captionPrefix = normalizeCaptionPrefix(triggerCaptionPrefix);

  if (cleanImageDir && batchImport) {
    const batchSubsets = buildBatchSubsets({
      imageDir: cleanImageDir,
      nativeImageDir,
      autoBalanceRepeats,
      baseSubset,
    });
    if (batchSubsets.length > 0) return batchSubsets;
  }

  if (!cleanImageDir && !captionPrefix) return [];

  const resolvedImageDir = cleanImageDir
    ? resolveSingleChildFallback(cleanImageDir, nativeImageDir)
    : '';
  const subset = withBaseSubset(baseSubset, {});
  if (resolvedImageDir) subset.image_dir = resolvedImageDir;
  if (captionPrefix) subset.caption_prefix = captionPrefix;
  return [subset];
}

module.exports = {
  buildNewJobSubsets,
  collectImageFiles,
  parseBatchFolderName,
};
