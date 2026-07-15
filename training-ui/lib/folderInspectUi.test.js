const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("new job and dataset image folders expose picker buttons and caption status", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");

  assert.match(html, /id="btn-new-job-select-image-dir"/);
  assert.match(html, /id="new-job-image-dir-status"/);
  assert.match(appJs, /class="[^"]*btn-select-sub-image-dir/);
  assert.match(appJs, /class="[^"]*sub-image-dir-status/);
  assert.match(appJs, /function formatFolderInspection\(summary\)/);
  assert.match(appJs, /function inspectImageFolder\(imageDir, captionExtension/);
  assert.match(appJs, /\/api\/system\/select-folder/);
  assert.match(appJs, /\/api\/system\/inspect-image-folder/);
});

test("folder selection updates image dir fields and immediately checks captions", () => {
  const appJs = read("public/js/app.js");
  const newJobSelectBlock = appJs.match(/async function selectNewJobImageDir\(\) \{[\s\S]*?\n\}/)?.[0] || "";
  const subsetSelectBlock = appJs.match(/async function selectSubsetImageDir\(card, subset\) \{[\s\S]*?\n\}/)?.[0] || "";

  assert.match(appJs, /async function selectFolderPath\([^)]*\)/);
  assert.match(appJs, /LAST_IMAGE_FOLDER_KEY/);
  assert.match(appJs, /localStorage\.getItem\(LAST_IMAGE_FOLDER_KEY\)/);
  assert.match(appJs, /initial_path:\s*lastFolder/);
  assert.match(appJs, /localStorage\.setItem\(LAST_IMAGE_FOLDER_KEY,\s*selectedPath\)/);
  assert.match(appJs, /showToast\("沒有選到資料夾，請再試一次。", "warning"\)/);
  assert.match(appJs, /btn-new-job-select-image-dir/);
  assert.match(appJs, /selectFolderPath\(\$\("new-job-image-dir"\)\.value\.trim\(\)\)/);
  assert.match(appJs, /selectFolderPath\(subset\.image_dir \|\| ""\)/);
  assert.match(appJs, /updateNewJobImageDirStatus/);
  assert.match(appJs, /selectSubsetImageDir/);
  assert.match(appJs, /updateSubsetImageDirStatus/);
  assert.match(newJobSelectBlock, /await updateNewJobImageDirStatus\(\)/);
  assert.doesNotMatch(subsetSelectBlock, /await updateSubsetImageDirStatus\(card, subset\)/);
  assert.match(appJs, /missing_caption/);
  assert.match(appJs, /empty_caption/);
  assert.match(appJs, /image_count/);
  assert.match(appJs, /matched_folder_count/);
  assert.match(appJs, /batch_import/);
});
