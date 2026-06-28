const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("UI exposes SNR-Aware Huber Wavelet loss controls", () => {
  const html = read("public/index.html");
  assert.match(html, /value="snr_aware_huber_wavelet"/);
  assert.match(html, /value="wavelet_l2"/);
  [
    "cfg-wavelet-loss-c-min",
    "cfg-wavelet-loss-c-max",
    "cfg-wavelet-loss-alpha",
    "cfg-wavelet-loss-beta",
    "cfg-wavelet-loss-gamma",
    "cfg-wavelet-loss-weight",
    "cfg-wavelet-loss-prediction-type",
    "group-wavelet-loss-fields",
  ].forEach((id) => assert.match(html, new RegExp(`id="${id}"`)));
});

test("UI loads and saves SNR-Aware Huber Wavelet config values", () => {
  const appJs = read("public/js/app.js");
  assert.match(appJs, /wavelet_l2/);
  [
    "wavelet_loss_c_min",
    "wavelet_loss_c_max",
    "wavelet_loss_alpha",
    "wavelet_loss_beta",
    "wavelet_loss_gamma",
    "wavelet_loss_weight",
    "wavelet_loss_prediction_type",
  ].forEach((key) => assert.match(appJs, new RegExp(key)));
});

test("default training template contains SNR-Aware Huber Wavelet defaults", () => {
  const template = read("templates/config_template.toml");
  assert.match(template, /wavelet_loss_c_min = 0\.2/);
  assert.match(template, /wavelet_loss_beta = 0\.5/);
  assert.match(template, /wavelet_loss_prediction_type = "velocity"/);
});
