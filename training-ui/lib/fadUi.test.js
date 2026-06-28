const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("UI loads and saves dataset-level FAD config values", () => {
  const appJs = read("public/js/app.js");

  [
    "cfg-fad-p-min",
    "cfg-fad-p-max",
    "cfg-fad-alpha",
    "cfg-fad-c",
    "cfg-fad-curriculum-start",
    "cfg-fad-curriculum-end",
    "cfg-fad-curriculum-beta",
    "cfg-fad-step-start",
    "cfg-fad-step-end",
  ].forEach((id) => {
    assert.match(appJs, new RegExp(`\\$\\("${id}"\\)\\.value\\s*=`));
  });

  [
    "fad_p_min",
    "fad_p_max",
    "fad_alpha",
    "fad_c",
    "fad_curriculum_start",
    "fad_curriculum_end",
    "fad_curriculum_beta",
    "fad_step_start",
    "fad_step_end",
  ].forEach((key) => {
    assert.match(appJs, new RegExp(`${key}:\\s*safeFloat`));
  });
});

test("UI loads and saves subset-level FAD switches", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /enable_fad:\s*s\.enable_fad\s*\?\?\s*false/);
  assert.match(appJs, /fad_curriculum:\s*s\.fad_curriculum\s*\?\?\s*false/);
  assert.match(appJs, /class="sub-enable-fad"/);
  assert.match(appJs, /class="sub-fad-curriculum"/);
  assert.match(appJs, /enable_fad:\s*s\.enable_fad/);
  assert.match(appJs, /fad_curriculum:\s*s\.fad_curriculum/);
});
