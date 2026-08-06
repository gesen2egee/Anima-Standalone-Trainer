const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("Contrastive and Differential Guidance are exposed in Advanced UI", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  [
    "cfg-contrastive-guidance",
    "cfg-guidance-loss-target",
    "cfg-guidance-loss-schedule",
    "cfg-differential-guidance",
    "cfg-differential-guidance-scale",
  ].forEach((id) => {
    assert.match(html, new RegExp("id=\"" + id + "\""));
    assert.match(appJs, new RegExp(id));
  });
  assert.match(html, /σ-schedule/);
});

test("guidance settings are serialized only when enabled", () => {
  const appJs = read("public/js/app.js");
  assert.match(appJs, /do_guidance_loss: true/);
  assert.match(appJs, /guidance_loss_schedule:/);
  assert.match(appJs, /do_differential_guidance: true/);
  assert.match(appJs, /differential_guidance_scale:/);
});

test("server keeps Differential Guidance settings and upgrades legacy scale-only jobs", () => {
  const serverJs = read("server.js");
  assert.match(serverJs, /function normalizeAnimaArgs\(merged\)/);
  assert.match(serverJs, /do_differential_guidance === undefined/);
  assert.doesNotMatch(
    serverJs,
    /'differential_guidance_scale',\s*\n\s*'ciop_prob'/
  );
});
