const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "public", "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(root, "public", "js", "app.js"), "utf8");

test("DOP controls expose class and flex-token preservation options", () => {
  assert.match(html, /id="cfg-diff-output-preservation"/);
  assert.match(html, /id="cfg-diff-output-preservation-multiplier"/);
  assert.match(html, /id="cfg-diff-output-preservation-class"/);
  assert.match(html, /留空使用過濾後的 flex tokens/);
});

test("DOP persists settings and disables text encoder cache", () => {
  assert.match(appJs, /diff_output_preservation:\s*\$\("cfg-diff-output-preservation"\)\.checked/);
  assert.match(appJs, /diff_output_preservation_class:\s*\$\("cfg-diff-output-preservation-class"\)\.value\.trim\(\)/);
  assert.match(appJs, /\$\("cfg-cache-te"\)\.checked = false/);
});
