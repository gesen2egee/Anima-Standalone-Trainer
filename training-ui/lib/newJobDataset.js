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
    return resolved.folders.map((folder) => {
      const hasTrigger = !!folder.triggerWords;
      return withBaseSubset(baseSubset, {
        image_dir: folder.imageDir,
        num_repeats: folder.repeats,
        caption_prefix: normalizeCaptionPrefix(folder.triggerWords),
        keep_tokens: hasTrigger ? 1 : 0,
        caption_dropout_rate: 0.1,
        caption_tag_dropout_rate: 0,
        enable_fad: true,
        fad_curriculum: true,
        fad_timestep: false,
        folder_shift: folder.folder_shift || 'global',
      });
    });
  }

  if (!cleanImageDir && !captionPrefix) return [];

  const subset = withBaseSubset(baseSubset, {});
  if (resolved.folders[0]?.imageDir) subset.image_dir = resolved.folders[0].imageDir;
  else if (cleanImageDir) subset.image_dir = cleanImageDir;
  
  if (captionPrefix) {
    subset.caption_prefix = captionPrefix;
    subset.keep_tokens = 1;
  } else {
    subset.keep_tokens = 0;
  }
  subset.caption_dropout_rate = 0.1;
  subset.caption_tag_dropout_rate = 0;
  subset.enable_fad = true;
  subset.fad_curriculum = true;
  subset.fad_timestep = false;
  subset.folder_shift = resolved.folders[0]?.folder_shift || 'global';
  
  return [subset];
}

module.exports = {
  buildNewJobSubsets,
  resolveDatasetImageFolders,
};
