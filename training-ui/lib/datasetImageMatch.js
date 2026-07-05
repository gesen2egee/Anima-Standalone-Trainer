const fs = require('fs');
const path = require('path');

const SD_SCRIPT_IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp']);

function stripQuotes(value) {
  if (typeof value !== 'string') return value;
  return value.replace(/^['"]+|['"]+$/g, '');
}

function normalizeCaptionExtension(value) {
  const ext = String(value || '.txt').trim() || '.txt';
  return ext.startsWith('.') ? ext : `.${ext}`;
}

function normalizeCaptionPrefix(value) {
  const triggerWords = String(value || '').trim();
  if (!triggerWords) return '';
  return /,\s*$/.test(triggerWords) ? triggerWords : `${triggerWords},`;
}

function isSdScriptsImageName(name) {
  return SD_SCRIPT_IMAGE_EXTENSIONS.has(path.extname(String(name || '')).toLowerCase());
}

function sortPaths(paths) {
  return paths.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function listSdScriptsImages(dir) {
  if (!dir || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return [];
  const files = fs.readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && isSdScriptsImageName(entry.name))
    .map((entry) => path.join(dir, entry.name));
  return sortPaths(files);
}

async function listSdScriptsImagesAsync(dir) {
  try {
    const stat = await fs.promises.stat(dir);
    if (!stat.isDirectory()) return [];
    const entries = await fs.promises.readdir(dir, { withFileTypes: true });
    return sortPaths(entries
      .filter((entry) => entry.isFile() && isSdScriptsImageName(entry.name))
      .map((entry) => path.join(dir, entry.name)));
  } catch (_) {
    return [];
  }
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

function buildFolderMatch({ imageDir, nativeImageDir, repeats = 1, triggerWords = '' }) {
  const imagePaths = listSdScriptsImages(nativeImageDir);
  return {
    imageDir,
    nativeImageDir,
    imagePaths,
    imageCount: imagePaths.length,
    repeats,
    triggerWords,
    captionPrefix: normalizeCaptionPrefix(triggerWords),
  };
}

function resolveDatasetImageFolders({
  imageDir = '',
  nativeImageDir,
  batchImport = false,
  autoBalanceRepeats = false,
  toNativePath = (value) => value,
} = {}) {
  const cleanImageDir = stripQuotes(String(imageDir || '').trim());
  const rootNativeDir = nativeImageDir || toNativePath(cleanImageDir);
  const folders = [];

  if (cleanImageDir && batchImport) {
    for (const folderName of listFirstLevelDirs(rootNativeDir)) {
      const parsed = parseBatchFolderName(folderName);
      if (!parsed) continue;

      const childNativeDir = path.join(rootNativeDir, folderName);
      const match = buildFolderMatch({
        imageDir: joinDisplayPath(cleanImageDir, folderName),
        nativeImageDir: childNativeDir,
        repeats: parsed.repeats,
        triggerWords: parsed.triggerWords,
      });
      if (match.imageCount < 1) continue;
      if (autoBalanceRepeats) {
        match.repeats = Math.max(1, Math.ceil(100 / match.imageCount));
      }
      folders.push(match);
    }
    return {
      mode: 'batch',
      folders,
      matchedFolderCount: folders.length,
      totalImages: folders.reduce((sum, folder) => sum + folder.imageCount, 0),
    };
  }

  if (cleanImageDir) {
    const currentMatch = buildFolderMatch({
      imageDir: cleanImageDir,
      nativeImageDir: rootNativeDir,
    });
    if (currentMatch.imageCount > 0) {
      folders.push(currentMatch);
    } else {
      const childDirs = listFirstLevelDirs(rootNativeDir);
      if (childDirs.length === 1) {
        const onlyChild = childDirs[0];
        const childMatch = buildFolderMatch({
          imageDir: joinDisplayPath(cleanImageDir, onlyChild),
          nativeImageDir: path.join(rootNativeDir, onlyChild),
        });
        if (childMatch.imageCount > 0) folders.push(childMatch);
      }
    }
  }

  return {
    mode: 'single',
    folders,
    matchedFolderCount: folders.length,
    totalImages: folders.reduce((sum, folder) => sum + folder.imageCount, 0),
  };
}

async function inspectDatasetImageFolders({
  imageDir = '',
  captionExtension = '.txt',
  batchImport = false,
  autoBalanceRepeats = false,
  toNativePath = (value) => value,
} = {}) {
  const captionExt = normalizeCaptionExtension(captionExtension);
  const nativeImageDir = toNativePath(stripQuotes(String(imageDir || '').trim()));
  const exists = !!nativeImageDir && fs.existsSync(nativeImageDir);
  const isDirectory = exists && fs.statSync(nativeImageDir).isDirectory();
  const resolved = isDirectory
    ? resolveDatasetImageFolders({
      imageDir,
      nativeImageDir,
      batchImport,
      autoBalanceRepeats,
      toNativePath,
    })
    : { mode: batchImport ? 'batch' : 'single', folders: [], matchedFolderCount: 0, totalImages: 0 };

  let missingCaption = 0;
  let emptyCaption = 0;
  for (const folder of resolved.folders) {
    const imagePaths = await listSdScriptsImagesAsync(folder.nativeImageDir);
    folder.imagePaths = imagePaths;
    folder.imageCount = imagePaths.length;
    for (const imagePath of imagePaths) {
      const captionPath = path.join(
        path.dirname(imagePath),
        `${path.basename(imagePath, path.extname(imagePath))}${captionExt}`,
      );
      try {
        const content = await fs.promises.readFile(captionPath, 'utf8');
        if (!content.trim()) emptyCaption += 1;
      } catch (_) {
        missingCaption += 1;
      }
    }
  }

  const imageCount = resolved.folders.reduce((sum, folder) => sum + folder.imageCount, 0);
  return {
    path: imageDir || '',
    native_path: nativeImageDir || '',
    exists,
    is_directory: isDirectory,
    has_images: imageCount > 0,
    image_count: imageCount,
    matched_folder_count: resolved.folders.length,
    matched_folders: resolved.folders.map((folder) => ({
      image_dir: folder.imageDir,
      native_image_dir: folder.nativeImageDir,
      image_count: folder.imageCount,
      repeats: folder.repeats,
      trigger_words: folder.triggerWords,
    })),
    missing_caption: missingCaption,
    empty_caption: emptyCaption,
    caption_extension: captionExt,
    mode: resolved.mode,
  };
}

module.exports = {
  SD_SCRIPT_IMAGE_EXTENSIONS,
  inspectDatasetImageFolders,
  isSdScriptsImageName,
  joinDisplayPath,
  listSdScriptsImages,
  listSdScriptsImagesAsync,
  normalizeCaptionPrefix,
  normalizeCaptionExtension,
  parseBatchFolderName,
  resolveDatasetImageFolders,
  stripQuotes,
};
