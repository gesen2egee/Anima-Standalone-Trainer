const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");
const TOML = require("@iarna/toml");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function parseTemplate(relPath) {
  return TOML.parse(read(relPath));
}

test("default config template follows my_job_v1 training and network defaults", () => {
  const config = parseTemplate("templates/config_template.toml");

  assert.strictEqual(config.training_arguments.learning_rate, 0.0001);
  assert.strictEqual(config.training_arguments.lr_scheduler, "cosine");
  assert.strictEqual(config.training_arguments.train_batch_size, 2);
  assert.strictEqual(config.training_arguments.knn_noise_k, 64);
  assert.strictEqual(config.training_arguments.cep_noise, 0.01);
  assert.strictEqual(config.training_arguments.loss_type, "wavelet");
  assert.strictEqual(config.training_arguments.pnp_loss_weight, 0.000001);
  assert.strictEqual(config.training_arguments.cache_latents_to_disk, false);
  assert.strictEqual(config.network_arguments.network_module, "networks.cdka");
  assert.deepStrictEqual(config.network_arguments.network_args, [
    'exclude_patterns=[".*"]',
    "network_reg_lrs=.*self_attn.*=5e-5",
    'include_patterns=[".*(self_attn|cross_attn)\\\\.(v_proj|output_proj)"]',
    "allora=True",
  ]);
  assert.strictEqual(config.anima_arguments.timestep_sampling, "uniform");
  assert.strictEqual(config.anima_arguments.ciop_noise_magnitude, 0);
});

test("default dataset template follows my_job_v1 dataset defaults", () => {
  const dataset = parseTemplate("templates/dataset_template.toml");
  const firstDataset = dataset.datasets[0];
  const firstSubset = firstDataset.subsets[0];

  assert.strictEqual(dataset.general.min_bucket_reso, 384);
  assert.strictEqual(dataset.general.max_bucket_reso, 1536);
  assert.deepStrictEqual(firstDataset.resolution, [768, 768]);
  assert.strictEqual(firstDataset.batch_size, 2);
  assert.strictEqual(firstDataset.fad_curriculum_start, 0.05);
  assert.strictEqual(firstDataset.fad_curriculum_end, 0.4);
  assert.strictEqual(firstSubset.keep_tokens, 2);
  assert.strictEqual(firstSubset.enable_fad, true);
  assert.strictEqual(firstSubset.fad_curriculum, true);
  assert.strictEqual(firstSubset.caption_tag_dropout_rate, 0);
});

test("new job modal defaults to the template network module", () => {
  const appJs = read("public/js/app.js");
  const newJobBlock = appJs.match(/\$\("btn-new-job"\)\.addEventListener\("click", \(\) => \{[\s\S]*?\n\}\);/);

  assert.ok(newJobBlock, "new job click handler should exist");
  assert.match(newJobBlock[0], /\$\("new-job-network-module"\)\.value = "networks\.cdka"/);
});
