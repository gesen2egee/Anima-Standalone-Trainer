const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("Differential Guidance is not exposed as a misleading UI setting", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  assert.doesNotMatch(html, /cfg-differential-guidance-scale/);
  assert.doesNotMatch(appJs, /cfg-differential-guidance-scale/);
});

test("default training template relies on the neutral parser default", () => {
  const template = read("templates/config_template.toml");
  assert.doesNotMatch(template, /differential_guidance_scale/);
});
