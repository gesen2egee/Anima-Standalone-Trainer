function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function cloneValue(value) {
  if (Array.isArray(value)) return value.map(cloneValue);
  if (isPlainObject(value)) {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneValue(item)]));
  }
  return value;
}

function sameResolution(left, right) {
  const normalize = (value) => Array.isArray(value) ? value : [value];
  return JSON.stringify(normalize(left?.resolution)) === JSON.stringify(normalize(right?.resolution));
}

function mergeArray(existing, incoming, path) {
  const currentKey = path[path.length - 1];
  if (currentKey === "datasets") {
    const existingItems = Array.isArray(existing) ? existing : existing ? [existing] : [];
    const used = new Set();
    return incoming.map((item, index) => {
      let sourceIndex = existingItems.findIndex((candidate, candidateIndex) => (
        !used.has(candidateIndex) && sameResolution(candidate, item)
      ));
      if (sourceIndex < 0 && index < existingItems.length && !used.has(index)) sourceIndex = index;
      if (sourceIndex >= 0) used.add(sourceIndex);
      return mergeValue(sourceIndex >= 0 ? existingItems[sourceIndex] : undefined, item, [...path, index]);
    });
  }

  if (currentKey === "subsets") {
    const existingItems = Array.isArray(existing) ? existing : existing ? [existing] : [];
    const used = new Set();
    return incoming.map((item, index) => {
      const imageDir = String(item?.image_dir || "").trim().toLowerCase();
      let sourceIndex = imageDir
        ? existingItems.findIndex((candidate, candidateIndex) => (
            !used.has(candidateIndex)
            && String(candidate?.image_dir || "").trim().toLowerCase() === imageDir
          ))
        : -1;
      if (sourceIndex < 0 && index < existingItems.length && !used.has(index)) sourceIndex = index;
      if (sourceIndex >= 0) used.add(sourceIndex);
      return mergeValue(sourceIndex >= 0 ? existingItems[sourceIndex] : undefined, item, [...path, index]);
    });
  }

  return cloneValue(incoming);
}

function mergeValue(existing, incoming, path = []) {
  if (Array.isArray(incoming)) return mergeArray(existing, incoming, path);
  if (!isPlainObject(incoming)) return cloneValue(incoming);

  const result = isPlainObject(existing) ? cloneValue(existing) : {};
  for (const [key, value] of Object.entries(incoming)) {
    result[key] = mergeValue(isPlainObject(existing) ? existing[key] : undefined, value, [...path, key]);
  }
  return result;
}

function mergeDatasetConfigPreservingUnknown(existing, incoming) {
  return mergeValue(existing || {}, incoming || {}, []);
}

module.exports = { mergeDatasetConfigPreservingUnknown };
