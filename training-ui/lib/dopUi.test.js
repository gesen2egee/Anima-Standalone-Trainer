const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "public", "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(root, "public", "js", "app.js"), "utf8");

test("DOP controls expose class-only preservation options", () => {
  assert.match(html, /id="cfg-diff-output-preservation"/);
  assert.match(html, /id="cfg-diff-output-preservation-multiplier"/);
  assert.doesNotMatch(html, /cfg-diff-output-preservation-class/);
  assert.match(html, /DOP 只使用各 Dataset 的 Class Token/);
  assert.match(appJs, /class="sub-class-tokens"/);
  assert.match(appJs, /class_tokens:\s*\(s\.class_tokens \?\? ""\)\.trim\(\)/);
});

test("DOP persists settings and disables text encoder cache", () => {
  assert.match(appJs, /diff_output_preservation:\s*\$\("cfg-diff-output-preservation"\)\.checked/);
  assert.match(appJs, /\$\("cfg-cache-te"\)\.checked = false/);
});
