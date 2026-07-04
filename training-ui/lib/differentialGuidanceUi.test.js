const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("UI loads and saves Differential Guidance scale", () => {
  const appJs = read("public/js/app.js");
  assert.match(appJs, /cfg-differential-guidance-scale/);
  assert.match(appJs, /differential_guidance_scale/);
});

test("default training template contains Differential Guidance default", () => {
  const template = read("templates/config_template.toml");
  assert.match(template, /differential_guidance_scale = 1/);
});
