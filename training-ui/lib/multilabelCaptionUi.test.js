const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("dataset card exposes multilabel caption tagging controls below caption prefix", () => {
  const appJs = read("public/js/app.js");
  const prefixIndex = appJs.indexOf("sub-caption-prefix");
  assert.ok(prefixIndex >= 0, "caption prefix control should exist");

  const taggingIndex = appJs.indexOf("sub-tagger-caption-ext");
  assert.ok(taggingIndex > prefixIndex, "tagging controls should follow caption prefix");

  [
    "sub-tagger-enable-char",
    "sub-tagger-enable-rating",
    "sub-tagger-enable-general",
    "btn-run-subset-tagger",
  ].forEach((needle) => assert.match(appJs, new RegExp(needle)));

  assert.match(appJs, /打標輸出成 Caption/);
  assert.match(appJs, /CHAR/);
  assert.match(appJs, /RATING/);
  assert.match(appJs, /GENERAL/);
});

test("dataset tagger button calls multilabel caption API", () => {
  const appJs = read("public/js/app.js");
  assert.match(appJs, /runSubsetTagger/);
  assert.match(appJs, /\/api\/jobs\/\$\{encodeURIComponent\(currentJob\)\}\/tag-captions/);
  assert.match(appJs, /caption_extension/);
  assert.match(appJs, /include_char/);
  assert.match(appJs, /include_rating/);
  assert.match(appJs, /include_general/);
  assert.match(appJs, /caption_extension:\s*captionExtension/);
  assert.match(appJs, /include_char:\s*includeChar/);
  assert.match(appJs, /include_rating:\s*includeRating/);
  assert.match(appJs, /include_general:\s*includeGeneral/);
});

test("dataset tagger refreshes folder caption status after completion", () => {
  const appJs = read("public/js/app.js");
  const start = appJs.indexOf("async function runSubsetTagger");
  const end = appJs.indexOf("\n// ==========================================\n//  Save", start);
  const block = appJs.slice(start, end);

  assert.match(block, /await updateFolderInspectionStatus\(/);
  assert.match(block, /imageDir,\s*captionExtension,\s*card\.querySelector\("\.sub-image-dir-status"\)/);
});
