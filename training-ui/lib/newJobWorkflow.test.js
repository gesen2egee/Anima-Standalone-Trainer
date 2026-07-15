const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");
const { buildNewJobSamplePrompts } = require("./newJobSamples");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function routeBlock(source, route) {
  const start = source.indexOf(`app.post('${route}'`);
  assert.notStrictEqual(start, -1, `${route} route should exist`);
  const nextRoute = source.indexOf("\napp.", start + 1);
  return source.slice(start, nextRoute === -1 ? source.length : nextRoute);
}

test("new job modal exposes trigger words and auto tag action", () => {
  const html = read("public/index.html");
  const networkIndex = html.indexOf('id="new-job-network-module"');
  const triggerIndex = html.indexOf('id="new-job-trigger-words"');
  const imageDirIndex = html.indexOf('id="new-job-image-dir"');
  const architectureIndex = html.indexOf('id="new-job-model-architecture"');

  assert.match(html, /id="new-job-trigger-words"/);
  assert.ok(triggerIndex >= 0 && triggerIndex < networkIndex, "trigger words should be above network module");
  assert.ok(architectureIndex >= 0 && architectureIndex < imageDirIndex, "model architecture should be first");
  assert.ok(imageDirIndex >= 0 && imageDirIndex < triggerIndex, "image folder should be above trigger words");
  assert.match(html, /id="btn-create-job-auto-tag"/);
  assert.match(html, /Create \+ Auto Tag/);
});

test("new job can generate two trigger-word sample prompts", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs");

  assert.match(html, /id="new-job-generate-samples"(?! checked)/);
  assert.match(appJs, /\$\("new-job-generate-samples"\)\.checked = false/);
  assert.match(appJs, /generate_samples:\s*\$\("new-job-generate-samples"\)\.checked/);
  assert.match(block, /buildNewJobSamplePrompts\(trigger_words\)/);
  assert.match(block, /delete config\.training_arguments\.sample_at_first/);
  assert.match(block, /delete config\.training_arguments\.sample_every_n_steps/);
});

test("generated samples contain trigger words and two distinct seeds", () => {
  const seeds = [10528, 31583];
  const prompts = buildNewJobSamplePrompts("vegapunk york", () => seeds.shift()).split("\n");

  assert.strictEqual(prompts.length, 2);
  assert.match(prompts[0], /^vegapunk york --w 832 --h 1216 --s 28 --d 10528 --l 3\.5 --n /);
  assert.match(prompts[1], /^vegapunk york --w 832 --h 1216 --s 28 --d 31583 --l 3\.5 --n /);
  assert.match(buildNewJobSamplePrompts("  ", () => 42), /^1girl --w 832 --h 1216 --s 28 --d 42 --l 3\.5 --n /);
});

test("new job trigger words and versioned name follow the image folder", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /function deriveTriggerWordsFromImagePath\(imageDir\)/);
  assert.match(appJs, /function nextArchitectureJobName\(baseName, architecture\)/);
  assert.match(appJs, /return `\$\{cleanBase\}\$\{versionPart\} \$\{architectureJobSuffix\(architecture\)\}`/);
  assert.match(appJs, /applyNewJobAutoNaming\(summary\)/);
  assert.match(appJs, /\|\| triggerInput\?\.value\.trim\(\)/);
  assert.match(appJs, /setAutoInputValue\(triggerInput, triggerWords\)/);
  assert.match(appJs, /setAutoInputValue\(\$\("new-job-name"\), jobName\)/);
  assert.match(appJs, /\$\("new-job-trigger-words"\)\.addEventListener\("input"/);
});

test("new job creation sends trigger words and can auto tag with defaults", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /async function createJobFromModal\(\{ autoTag = false \} = \{\}\)/);
  assert.match(appJs, /trigger_words:\s*\$\("new-job-trigger-words"\)\.value\.trim\(\)/);
  assert.match(appJs, /auto_output_name:\s*\$\("new-job-output-name"\)\.dataset\.autoValue/);
  assert.match(appJs, /\$\("btn-create-job-auto-tag"\)\.addEventListener\("click", \(\) => createJobFromModal\(\{ autoTag: true \}\)\)/);
  assert.match(appJs, /async function runNewJobDefaultTagger\(jobName\)/);
  assert.match(appJs, /\/api\/jobs\/\$\{encodeURIComponent\(jobName\)\}\/tag-captions/);
  assert.match(appJs, /caption_extension:\s*"\.txt"/);
  assert.match(appJs, /include_char:\s*true/);
  assert.match(appJs, /include_rating:\s*true/);
  assert.match(appJs, /include_general:\s*true/);
  assert.match(appJs, /updateNewJobImageDirStatus\(\)/);
  assert.match(appJs, /closeModal\("modal-new-job"\)/);
  assert.doesNotMatch(appJs.match(/async function runNewJobDefaultTagger\(jobName\)[\s\S]*?\n\}/)?.[0] || "", /image_dir:/);
});

test("create and auto tag keeps modal open until tagger progress finishes", () => {
  const appJs = read("public/js/app.js");
  const createBlock = appJs.match(/async function createJobFromModal\(\{ autoTag = false \} = \{\}\) \{[\s\S]*?\n\}/);
  const modalClickBlock = appJs.match(/document\.querySelectorAll\("\.modal"\)[\s\S]*?\n\}\);/);

  assert.ok(createBlock, "create job helper should exist");
  assert.ok(modalClickBlock, "modal backdrop handler should exist");
  assert.match(createBlock[0], /if \(autoTag\)[\s\S]*await runNewJobDefaultTagger\(result\.name\)/);
  assert.match(createBlock[0], /if \(autoTag\)[\s\S]*await updateNewJobImageDirStatus\(\)/);
  assert.match(createBlock[0], /if \(autoTag\)[\s\S]*closeModal\("modal-new-job"\)/);
  assert.doesNotMatch(createBlock[0], /closeModal\("modal-new-job"\);\s*await loadJobs\(\);\s*await selectJob/);
  assert.match(modalClickBlock[0], /modal\.id === "modal-new-job"/);
  assert.match(modalClickBlock[0], /return/);
});

test("new job modal can batch import first-level folders and auto balance repeats", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs");

  assert.match(html, /id="new-job-batch-import"/);
  assert.match(html, /id="new-job-auto-balance"/);
  assert.match(appJs, /\$\("new-job-batch-import"\)\.checked = false/);
  assert.match(appJs, /\$\("new-job-auto-balance"\)\.checked = false/);
  assert.match(appJs, /batch_import:\s*\$\("new-job-batch-import"\)\.checked/);
  assert.match(appJs, /auto_balance_repeats:\s*\$\("new-job-auto-balance"\)\.checked/);
  assert.match(serverJs, /buildNewJobSubsets/);
  assert.match(block, /batch_import/);
  assert.match(block, /auto_balance_repeats/);
});

test("server writes trigger words into first subset caption prefix", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs");

  assert.match(serverJs, /function normalizeCaptionPrefixFromTriggerWords\(value\)/);
  assert.match(block, /trigger_words/);
  assert.match(block, /normalizeCaptionPrefixFromTriggerWords\(trigger_words\)/);
  assert.match(block, /triggerCaptionPrefix/);
  assert.match(block, /buildNewJobSubsets/);
});
