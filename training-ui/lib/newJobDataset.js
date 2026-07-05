const {
  normalizeCaptionPrefix,
  resolveDatasetImageFolders,
  stripQuotes,
} = require('./datasetImageMatch');

function withBaseSubset(baseSubset, values) {
  return {
    ...(baseSubset || {}),
    num_repeats: 1,
    ...(values || {}),
  };
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
  const resolved = resolveDatasetImageFolders({
    imageDir: cleanImageDir,
    nativeImageDir,
    batchImport,
    autoBalanceRepeats,
    toNativePath,
  });

  if (cleanImageDir && batchImport && resolved.folders.length > 0) {
    return resolved.folders.map((folder) => withBaseSubset(baseSubset, {
      image_dir: folder.imageDir,
      num_repeats: folder.repeats,
      caption_prefix: normalizeCaptionPrefix(folder.triggerWords),
    }));
  }

  if (!cleanImageDir && !captionPrefix) return [];

  const subset = withBaseSubset(baseSubset, {});
  if (resolved.folders[0]?.imageDir) subset.image_dir = resolved.folders[0].imageDir;
  else if (cleanImageDir) subset.image_dir = cleanImageDir;
  if (captionPrefix) subset.caption_prefix = captionPrefix;
  return [subset];
}

module.exports = {
  buildNewJobSubsets,
  resolveDatasetImageFolders,
};
