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

  assert.strictEqual(config.training_arguments.learning_rate, 0.0003);
  assert.strictEqual(config.training_arguments.lr_scheduler, "warmup_stable_decay");
  assert.strictEqual(config.training_arguments.lr_scheduler_min_lr_ratio, 0.1);
  assert.strictEqual(config.training_arguments.lr_decay_steps, 0.1);
  assert.strictEqual(config.training_arguments.train_batch_size, 1);
  assert.strictEqual(config.training_arguments.knn_noise_k, 64);
  assert.strictEqual(config.training_arguments.cep_noise, 0.01);
  assert.strictEqual(config.training_arguments.loss_type, "wavelet");
  assert.strictEqual(config.training_arguments.pnp_loss_weight, 0.000001);
  assert.strictEqual(config.training_arguments.cache_latents_to_disk, true);
  assert.strictEqual(config.training_arguments.automask, true);
  assert.strictEqual(config.training_arguments.automask_alpha, 128);
  assert.strictEqual(config.training_arguments.automask_shrink, 1);
  assert.strictEqual(config.training_arguments.automask_blur, 3);
  assert.strictEqual(config.training_arguments.automask_model, "base-nightly");
  assert.strictEqual(config.network_arguments.network_module, "networks.cdka");
  assert.deepStrictEqual(config.network_arguments.network_args, [
    'exclude_patterns=[".*"]',
    "network_reg_lrs=.*self_attn.*=5e-5",
    'include_patterns=[".*(self_attn|cross_attn)\\\\.(v_proj|output_proj)"]',
    "allora=True",
  ]);
  assert.strictEqual(config.anima_arguments.timestep_sampling, "autoshift");
  assert.strictEqual(config.training_arguments.sample_at_first, true);
  assert.strictEqual(config.training_arguments.sample_every_n_steps, 200);
  assert.strictEqual(config.training_arguments.cache_text_encoder_outputs_to_disk, false);
  assert.strictEqual(config.training_arguments.masked_loss_random_strength, 0);
  assert.strictEqual(Object.hasOwn(config.anima_arguments, "differential_guidance_scale"), false);
  assert.strictEqual(Object.hasOwn(config.anima_arguments, "ciop_prob"), false);
  assert.strictEqual(Object.hasOwn(config.anima_arguments, "model_guidance_weight"), false);
  assert.strictEqual(Object.hasOwn(config.training_arguments, "use_cdc_fm"), false);
  assert.strictEqual(Object.hasOwn(config.training_arguments, "use_self_flow"), false);
  assert.strictEqual(Object.hasOwn(config.training_arguments, "diff_output_preservation"), false);
});

test("default dataset template follows my_job_v1 dataset defaults", () => {
  const dataset = parseTemplate("templates/dataset_template.toml");
  const firstDataset = dataset.datasets[0];
  const firstSubset = firstDataset.subsets[0];

  assert.strictEqual(dataset.general.min_bucket_reso, 384);
  assert.strictEqual(dataset.general.max_bucket_reso, 1536);
  assert.deepStrictEqual(firstDataset.resolution, [768, 768]);
  assert.strictEqual(firstDataset.batch_size, 1);
  assert.strictEqual(firstDataset.fad_curriculum_start, 0.1);
  assert.strictEqual(firstDataset.fad_curriculum_end, 1);
  assert.strictEqual(firstDataset.folder_shift_curriculum, true);
  assert.strictEqual(firstSubset.keep_tokens, 1);
  assert.strictEqual(firstSubset.enable_fad, true);
  assert.strictEqual(firstSubset.fad_curriculum, true);
  assert.strictEqual(firstSubset.fad_timestep, true);
  assert.strictEqual(firstSubset.alpha_mask, true);
  assert.strictEqual(firstSubset.caption_tag_dropout_rate, 0);
});

test("new job modal defaults to the template network module", () => {
  const appJs = read("public/js/app.js");
  const newJobBlock = appJs.match(/\$\("btn-new-job"\)\.addEventListener\("click", \(\) => \{[\s\S]*?\n\}\);/);

  assert.ok(newJobBlock, "new job click handler should exist");
  assert.match(newJobBlock[0], /\$\("new-job-network-module"\)\.value = "networks\.cdka"/);
});

test("architecture registry exposes Anima, Krea 2, and Krea 2 Bypass presets", () => {
  const registry = JSON.parse(read("architectures.json"));
  assert.deepStrictEqual(Object.keys(registry.architectures), ["anima", "krea2", "krea2_bypass"]);

  const anima = registry.architectures.anima;
  assert.strictEqual(anima.job_defaults.anima_arguments.timestep_sampling, "autoshift");
  assert.strictEqual(anima.dataset_defaults.resolution, 768);
  assert.strictEqual(anima.job_defaults.network_arguments.network_module, "networks.cdka");

  const krea2 = registry.architectures.krea2;
  assert.strictEqual(krea2.job_defaults.krea2_arguments.timestep_sampling, "autoshift");
  assert.strictEqual(krea2.job_defaults.krea2_arguments.discrete_flow_shift, 2.5);
  assert.strictEqual(krea2.job_defaults.training_arguments.blocks_to_swap, 20);
  assert.strictEqual(krea2.dataset_defaults.resolution, 512);
  assert.match(krea2.job_defaults.network_arguments.network_args.join(" "), /self_attn/);

  const bypass = registry.architectures.krea2_bypass;
  assert.strictEqual(bypass.job_defaults.krea2_arguments.krea2_bypass, true);
  assert.strictEqual(bypass.scripts.train_network, krea2.scripts.train_network);
  assert.strictEqual(bypass.scripts.generate, krea2.scripts.generate);
  assert.strictEqual(bypass.global_paths.krea2_dit_path.cli_flag, krea2.global_paths.krea2_dit_path.cli_flag);
});
