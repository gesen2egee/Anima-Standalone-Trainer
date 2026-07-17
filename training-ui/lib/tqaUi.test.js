const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "public", "index.html"), "utf8");
const appJs = fs.readFileSync(path.join(root, "public", "js", "app.js"), "utf8");

test("TQA controls and Autoshift option are exposed", () => {
  assert.match(html, /<option value="autoshift_tqa">Autoshift \(TQA: DBAES − Quality\)<\/option>/);
  assert.match(html, /id="cfg-tqa-loss-weighting"/);
  assert.match(html, /id="cfg-tqa-loss-weighting-schedule"/);
});

test("TQA settings force latent caching and round-trip", () => {
  assert.match(appJs, /tqa_loss_weighting:\s*\$\("cfg-tqa-loss-weighting"\)/);
  assert.match(appJs, /tqa_loss_weighting_schedule:\s*\$\("cfg-tqa-loss-weighting-schedule"\)/);
  assert.match(appJs, /t\.tqa_loss_weighting/);
  assert.match(appJs, /cfg-timestep-method"\)\.value === "autoshift_tqa"/);
});
