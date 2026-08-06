const fs = require("fs");
const path = require("path");
const test = require("node:test");
const assert = require("node:assert/strict");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "public", "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(root, "public", "js", "app.js"), "utf8");
const serverJs = fs.readFileSync(path.join(root, "server.js"), "utf8");

test("removed experimental controls are absent from the UI", () => {
  [
    "cfg-diff-output-preservation",
    "cfg-use-cdc-fm",
    "cfg-use-self-flow",
    "cfg-model-guidance-weight",
    "cfg-ciop-prob",
    "cfg-progressive-reso",
    "cfg-disable-bucket-shuffle",
  ].forEach((id) => {
    assert.doesNotMatch(html, new RegExp(`id="${id}"`));
    assert.doesNotMatch(appJs, new RegExp(id));
  });
});

test("stale hidden experimental settings are stripped before training", () => {
  assert.match(serverJs, /function stripRemovedExperimentalArgs\(merged\)/);
  [
    "use_cdc_fm",
    "use_self_flow",
    "diff_output_preservation",
    "resolution_schedule",
    "disable_bucket_shuffle",
    "model_guidance_weight",
    "ciop_prob",
  ].forEach((key) => assert.match(serverJs, new RegExp(`'${key}'`)));
  assert.match(serverJs, /stripRemovedExperimentalArgs\(merged\)/);
});
