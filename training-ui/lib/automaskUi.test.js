const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("dataset tab exposes automask controls near alpha mask", () => {
  const html = read("public/index.html");
  const alphaIndex = html.indexOf('id="cfg-alpha-mask"');
  const automaskIndex = html.indexOf('id="cfg-automask"');

  assert.ok(alphaIndex >= 0, "missing alpha mask control");
  assert.ok(automaskIndex > alphaIndex, "automask should appear after alpha mask");

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
  assert.match(appJs, /if\s*\(\$\("cfg-alpha-mask"\)\.checked\s*\|\|\s*\$\("cfg-automask"\)\.checked\)\s*subset\.alpha_mask\s*=\s*true/);
});
