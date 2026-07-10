const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("advanced tab exposes mask controls outside dataset tab", () => {
  const html = read("public/index.html");
  const advancedIndex = html.indexOf('id="tab-advanced"');
  const datasetIndex = html.indexOf('id="tab-dataset"');
  const alphaIndex = html.indexOf('id="cfg-alpha-mask"');
  const automaskIndex = html.indexOf('id="cfg-automask"');
  const maskedLossIndex = html.indexOf('id="cfg-masked-loss-random"');

  assert.ok(advancedIndex >= 0, "missing advanced tab");
  assert.ok(datasetIndex > advancedIndex, "dataset tab should follow advanced tab");
  assert.ok(alphaIndex >= 0, "missing alpha mask control");
  assert.ok(alphaIndex > advancedIndex && alphaIndex < datasetIndex, "alpha mask should live in advanced tab");
  assert.ok(automaskIndex > alphaIndex, "automask should appear after alpha mask");
  assert.ok(maskedLossIndex > automaskIndex && maskedLossIndex < datasetIndex, "random mask strength should live with mask controls");

  [
    "cfg-automask",
    "cfg-automask-alpha",
    "cfg-automask-shrink",
    "cfg-automask-blur",
    "cfg-automask-model",
  ].forEach((id) => assert.match(html, new RegExp(`id="${id}"`)));
});

test("UI loads and saves automask training arguments", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /\$\("cfg-automask"\)\.checked\s*=\s*t\.automask\s*\?\?\s*false/);
  assert.match(appJs, /\$\("cfg-automask-alpha"\)\.value\s*=\s*t\.automask_alpha\s*\?\?\s*128/);
  assert.match(appJs, /automask:\s*\$\("cfg-automask"\)\.checked/);
  assert.match(appJs, /automask_alpha:\s*safeInt\(\$\("cfg-automask-alpha"\)\.value,\s*128\)/);
  assert.match(appJs, /automask_model:\s*\$\("cfg-automask-model"\)\.value\.trim\(\)\s*\|\|\s*"base-nightly"/);
  assert.match(appJs, /if\s*\(\$\("cfg-automask"\)\.checked\)\s*\{\s*subset\.alpha_mask\s*=\s*true/);
});

test("autoshift exposes automask settings without checking automask loss", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");

  assert.match(html, /<option value="autoshift">Autoshift \(Mask Ratio\)<\/option>/);
  assert.match(appJs, /const autoshift = \$\("cfg-timestep-method"\)\.value === "autoshift"/);
  assert.match(appJs, /!\$\("cfg-automask"\)\.checked && !autoshift/);
  assert.doesNotMatch(appJs, /autoshift[\s\S]{0,100}cfg-automask"\)\.checked\s*=\s*true/);
});

test("alpha mask UI preserves per-subset values unless the global toggle is changed", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /alpha_mask:\s*s\.alpha_mask === true/);
  assert.match(appJs, /let alphaMaskTouched\s*=\s*false/);
  assert.match(appJs, /\$\("cfg-alpha-mask"\)\.indeterminate\s*=\s*alphaMaskStates\.some\(\(value\) => value\) && !alphaMaskStates\.every\(\(value\) => value\)/);
  assert.match(appJs, /\$\("cfg-alpha-mask"\)\.addEventListener\("change",\s*\(\) => \{\s*alphaMaskTouched = true/);
  assert.match(appJs, /if\s*\(\$\("cfg-automask"\)\.checked\)\s*\{\s*subset\.alpha_mask = true/);
  assert.match(appJs, /else if\s*\(alphaMaskTouched\)/);
  assert.match(appJs, /else if\s*\(s\.alpha_mask\)\s*\{\s*subset\.alpha_mask\s*=\s*true/);
});
