const assert = require("assert");
const test = require("node:test");

const { isSuccessfulTrainingExit } = require("./trainingExit");

test("code zero is successful when stop was not requested", () => {
  assert.strictEqual(isSuccessfulTrainingExit({
    code: 0,
    stoppedByRequest: false,
    logText: "",
  }), true);
});

test("user-requested stop is never treated as successful completion", () => {
  assert.strictEqual(isSuccessfulTrainingExit({
    code: 0,
    stoppedByRequest: true,
    logText: "steps: 100%| saving checkpoint: model.safetensors",
  }), false);
});

test("Windows access violation after full progress and checkpoint save counts as completed", () => {
  const logText = [
    "steps: 100%|██████████| 3000/3000 [1:40:14<00:00,  2.00s/it]",
    "saving checkpoint: D:\\models\\job\\output\\job-step00003000.safetensors",
    "saving checkpoint: D:\\models\\job\\output\\job.safetensors",
  ].join("\n");

  assert.strictEqual(isSuccessfulTrainingExit({
    code: 3221225477,
    stoppedByRequest: false,
    logText,
  }), true);
});

test("partial progress with non-zero code remains failed", () => {
  assert.strictEqual(isSuccessfulTrainingExit({
    code: 1,
    stoppedByRequest: false,
    logText: "steps:  55%|█████▌    | 5507/10000",
  }), false);
});
