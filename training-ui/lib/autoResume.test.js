const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");

const {
  findAutoResumeSource,
  findLatestCheckpointFile,
  findLatestStateDir,
} = require("./autoResume");

function makeOutputDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anima-auto-resume-"));
  const outputDir = path.join(dir, "output");
  fs.mkdirSync(outputDir, { recursive: true });
  return outputDir;
}

function touch(target, mtimeMs) {
  if (path.extname(target)) {
    fs.writeFileSync(target, "x", "utf8");
  } else {
    fs.mkdirSync(target, { recursive: true });
  }
  const time = new Date(mtimeMs);
  fs.utimesSync(target, time, time);
}

test("findLatestStateDir picks the newest state for the output name", () => {
  const outputDir = makeOutputDir();
  const older = path.join(outputDir, "hero-step00001000-state");
  const newer = path.join(outputDir, "hero-step00002000-state");
  const other = path.join(outputDir, "other-step00003000-state");
  touch(older, 1000);
  touch(newer, 2000);
  touch(other, 3000);

  assert.strictEqual(findLatestStateDir(outputDir, "hero"), newer);
});

test("findLatestCheckpointFile falls back to newest save for the output name", () => {
  const outputDir = makeOutputDir();
  const older = path.join(outputDir, "hero-step00001000.safetensors");
  const newer = path.join(outputDir, "hero-step00002000.safetensors");
  const other = path.join(outputDir, "other-step00003000.safetensors");
  touch(older, 1000);
  touch(newer, 2000);
  touch(other, 3000);

  assert.strictEqual(findLatestCheckpointFile(outputDir, "hero"), newer);
});

test("findLatestCheckpointFile supports common save formats", () => {
  const outputDir = makeOutputDir();
  const older = path.join(outputDir, "hero-step00001000.ckpt");
  const newer = path.join(outputDir, "hero-step00002000.pt");
  touch(older, 1000);
  touch(newer, 2000);

  assert.strictEqual(findLatestCheckpointFile(outputDir, "hero"), newer);
});

test("auto resume prefers state over checkpoint", () => {
  const outputDir = makeOutputDir();
  const state = path.join(outputDir, "hero-step00001000-state");
  const checkpoint = path.join(outputDir, "hero-step00002000.safetensors");
  touch(state, 1000);
  touch(checkpoint, 2000);

  assert.deepStrictEqual(findAutoResumeSource(outputDir, "hero", { allowCheckpoint: true }), {
    type: "state",
    path: state,
  });
});

test("auto resume uses checkpoint only when state is missing and checkpoint is allowed", () => {
  const outputDir = makeOutputDir();
  const checkpoint = path.join(outputDir, "hero-step00002000.safetensors");
  touch(checkpoint, 2000);

  assert.deepStrictEqual(findAutoResumeSource(outputDir, "hero", { allowCheckpoint: true }), {
    type: "checkpoint",
    path: checkpoint,
  });
  assert.strictEqual(findAutoResumeSource(outputDir, "hero", { allowCheckpoint: false }), null);
});

test("auto resume returns null when no state or checkpoint exists", () => {
  const outputDir = makeOutputDir();

  assert.strictEqual(findAutoResumeSource(outputDir, "hero", { allowCheckpoint: true }), null);
});
