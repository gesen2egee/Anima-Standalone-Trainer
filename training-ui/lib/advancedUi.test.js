const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function extractElementById(html, id) {
  const marker = `id="${id}"`;
  const markerIndex = html.indexOf(marker);
  assert.notStrictEqual(markerIndex, -1, `missing id ${id}`);

  const tagStart = html.lastIndexOf("<", markerIndex);
  const tagEnd = html.indexOf(">", markerIndex);
  const openTag = html.slice(tagStart, tagEnd + 1);
  const tagName = openTag.match(/^<([a-z0-9-]+)/i)?.[1];
  assert.ok(tagName, `cannot determine tag for ${id}`);

  if (/\/>$/.test(openTag) || ["input", "br", "hr", "img"].includes(tagName)) {
    return openTag;
  }

  const tokenPattern = new RegExp(`<\\/?${tagName}\\b[^>]*>`, "gi");
  tokenPattern.lastIndex = tagStart;
  let depth = 0;
  let match;
  while ((match = tokenPattern.exec(html)) !== null) {
    if (match[0].startsWith(`</`)) {
      depth -= 1;
      if (depth === 0) {
        return html.slice(tagStart, tokenPattern.lastIndex);
      }
    } else if (!/\/>$/.test(match[0])) {
      depth += 1;
    }
  }

  assert.fail(`missing close tag for ${id}`);
}

test("advanced tab contains research-oriented training controls", () => {
  const html = read("public/index.html");
  assert.match(html, /data-tab="advanced"[^>]*>Advanced<\/button>/);

  const advanced = extractElementById(html, "tab-advanced");
  [
    "cfg-knn-noise-k",
    "cfg-cep-noise",
    "cfg-loss-type",
    "cfg-pnp-loss-weight",
    "group-cwmi-fields",
    "group-wavelet-loss-fields",
    "cfg-model-guidance-weight",
    "cfg-model-guidance-warmup-steps",
    "cfg-model-guidance-prob",
    "cfg-model-guidance-end-step",
    "cfg-model-guidance-timestep-scaling",
    "cfg-model-guidance-min-weight",
    "cfg-model-guidance-cfg-zero",
    "cfg-model-guidance-zero-init-threshold",
    "cfg-differential-guidance-scale",
    "cfg-ciop-prob",
    "cfg-ciop-noise-magnitude",
    "cfg-ciop-noise-type",
  ].forEach((id) => assert.match(advanced, new RegExp(`id="${id}"`)));

  assert.match(advanced, /Batch and Noise/);
  assert.match(advanced, /Loss Experiments/);
  assert.match(advanced, /Model Guidance/);
  assert.match(advanced, /Differential Guidance Scale/);
  assert.match(advanced, /CIOP/);

  const dataset = extractElementById(html, "tab-dataset");
  assert.match(dataset, /id="cfg-batch-size"/);
  assert.doesNotMatch(advanced, /id="cfg-batch-size"/);
});

test("resume training is shown in training tab below checkpoint management", () => {
  const html = read("public/index.html");
  const training = extractElementById(html, "tab-training");
  const checkpointIndex = training.indexOf("Checkpoint Management");
  const resumeIndex = training.indexOf("Resume Training");

  assert.ok(checkpointIndex >= 0, "missing Checkpoint Management section");
  assert.ok(resumeIndex > checkpointIndex, "Resume Training should follow Checkpoint Management");

  const network = extractElementById(html, "tab-network");
  assert.strictEqual(network.includes("Resume Training"), false);
});

test("HTML ids are unique", () => {
  const html = read("public/index.html");
  const counts = new Map();
  for (const match of html.matchAll(/id="([^"]+)"/g)) {
    counts.set(match[1], (counts.get(match[1]) || 0) + 1);
  }

  const duplicates = [...counts.entries()]
    .filter(([, count]) => count > 1)
    .map(([id, count]) => `${id} x${count}`);
  assert.deepStrictEqual(duplicates, []);
});

test("Krea 2 controls expose real token, Flow Shift, and Text Encoder behavior", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  const serverJs = read("server.js");
  const kreaTrainer = read("../krea2_train_network.py");
  const kreaEncoder = read("../library/krea2/krea2_encoder.py");
  const kreaUtils = read("../library/krea2/krea2_utils.py");

  assert.doesNotMatch(html, /value="lumina"/i);
  assert.match(html, /id="group-t5-max-tokens"/);
  assert.match(html, /id="krea2-dynamic-text-encoder-warning"/);
  assert.match(appJs, /group-t5-max-tokens"\)\?\.classList\.toggle\("hidden", architecture === "krea2"\)/);
  assert.match(appJs, /architecture === "krea2" \? "Qwen3-VL Max Tokens"/);
  assert.match(appJs, /\.\.\.\(!isKrea2 && \{[\s\S]*qwen3_max_token_length:[\s\S]*t5_max_token_length:/);
  assert.match(serverJs, /stripKrea2ShadowedAnimaArgs\(merged\.anima_arguments\)/);
  assert.match(appJs, /filterNetworkModuleOptions\(\$\("cfg-network-module"\), architecture\)/);
  assert.match(appJs, /krea2DynamicTextEncoder = isKrea2 && \([\s\S]*!\$\("cfg-unet-only"\)\.checked/);
  assert.match(serverJs, /network_module && !requestedArchitecture\.network_modules\.includes\(network_module\)/);
  assert.match(kreaTrainer, /def is_train_text_encoder\(self, args\):[\s\S]*return not args\.network_train_unet_only/);
  assert.doesNotMatch(kreaTrainer, /args\.network_train_unet_only = True/);
  assert.match(kreaEncoder, /def set_gradient_enabled\(self, enabled: bool\)/);
  assert.doesNotMatch(kreaUtils, /@torch\.no_grad\(\)\s*\ndef get_krea2_prompt_embeds/);
});
