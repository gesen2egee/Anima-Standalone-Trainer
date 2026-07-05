const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("LR scheduler UI exposes warmup stable decay fields", () => {
  const html = read("public/index.html");

  assert.match(html, /<option value="warmup_stable_decay">Warmup Stable Decay<\/option>/);
  assert.match(html, /id="group-lr-decay-steps"/);
  assert.match(html, /id="cfg-lr-decay-steps"/);
  assert.match(html, /id="cfg-lr-decay-steps" value="0\.1" min="0" step="0\.01"/);
  assert.match(html, /lr_decay_steps/);
});

test("LR scheduler mapping loads and saves warmup stable decay options", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /\$\("cfg-lr-decay-steps"\)\.value = t\.lr_decay_steps \?\? 0\.1/);
  assert.match(appJs, /const isWarmupStableDecay = scheduler === "warmup_stable_decay"/);
  assert.match(appJs, /scheduler !== "cosine_with_min_lr" && !isWarmupStableDecay/);
  assert.match(appJs, /scheduler !== "warmup_stable_decay"/);
  assert.match(appJs, /\$\("cfg-lr-scheduler"\)\.value === "cosine_with_min_lr" \|\|\s*\$\("cfg-lr-scheduler"\)\.value === "warmup_stable_decay"/);
  assert.match(appJs, /lr_decay_steps:\s*\$\("cfg-lr-scheduler"\)\.value === "warmup_stable_decay"[\s\S]*?safeFloat\(\$\("cfg-lr-decay-steps"\)\.value\)/);
});
