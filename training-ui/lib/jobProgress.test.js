const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");

const {
  calculateJobProgress,
  findLatestProgressSave,
} = require("./jobProgress");

function makeOutputDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anima-job-progress-"));
  const outputDir = path.join(dir, "output");
  fs.mkdirSync(outputDir, { recursive: true });
  return outputDir;
}

function touchFile(target, mtimeMs) {
  fs.writeFileSync(target, "x", "utf8");
  const time = new Date(mtimeMs);
  fs.utimesSync(target, time, time);
}

function touchDir(target, mtimeMs) {
  fs.mkdirSync(target, { recursive: true });
  const time = new Date(mtimeMs);
  fs.utimesSync(target, time, time);
}

test("findLatestProgressSave picks the newest step save or state for the output name", () => {
  const outputDir = makeOutputDir();
  touchFile(path.join(outputDir, "other-step00005000.safetensors"), 5000);
  touchFile(path.join(outputDir, "hero-step00001000.safetensors"), 1000);
  touchDir(path.join(outputDir, "hero-step00002000-state"), 2000);

  assert.deepStrictEqual(findLatestProgressSave(outputDir, "hero", "steps"), {
    unit: "steps",
    value: 2000,
    name: "hero-step00002000-state",
  });
});

test("findLatestProgressSave picks the newest epoch save or state for the output name", () => {
  const outputDir = makeOutputDir();
  touchFile(path.join(outputDir, "hero-000001.safetensors"), 1000);
  touchDir(path.join(outputDir, "hero-000003-state"), 3000);
  touchFile(path.join(outputDir, "other-000009.safetensors"), 9000);

  assert.deepStrictEqual(findLatestProgressSave(outputDir, "hero", "epochs"), {
    unit: "epochs",
    value: 3,
    name: "hero-000003-state",
  });
});

test("calculateJobProgress clamps progress percentage at 100", () => {
  const outputDir = makeOutputDir();
  touchFile(path.join(outputDir, "hero-step00004000.safetensors"), 1000);

  assert.deepStrictEqual(calculateJobProgress(outputDir, {
    outputName: "hero",
    maxTrainSteps: 3000,
  }), {
    unit: "steps",
    current: 4000,
    target: 3000,
    percent: 100,
    label: "step 4000 / 3000",
    saveName: "hero-step00004000.safetensors",
  });
});

test("calculateJobProgress returns an empty progress model when no numbered save exists", () => {
  const outputDir = makeOutputDir();
  touchFile(path.join(outputDir, "hero.safetensors"), 1000);

  assert.deepStrictEqual(calculateJobProgress(outputDir, {
    outputName: "hero",
    maxTrainEpochs: 20,
  }), {
    unit: "epochs",
    current: 0,
    target: 20,
    percent: 0,
    label: "",
    saveName: "",
  });
});
