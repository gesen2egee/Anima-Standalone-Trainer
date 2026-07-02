const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("jobs API attaches calculated save progress to each job", () => {
  const serverJs = read("server.js");

  assert.match(serverJs, /calculateJobProgress/);
  assert.match(serverJs, /progress:\s*calculateJobProgress\(/);
  assert.match(serverJs, /maxTrainSteps:\s*trainingArgs\.max_train_steps/);
  assert.match(serverJs, /maxTrainEpochs:\s*trainingArgs\.max_train_epochs/);
});

test("job item renders a thin save progress bar", () => {
  const appJs = read("public/js/app.js");
  const css = read("public/css/style.css");

  assert.match(appJs, /renderJobProgress\(job\.progress\)/);
  assert.match(appJs, /job-progress/);
  assert.match(appJs, /job-progress-fill/);
  assert.match(css, /\.job-progress\s*\{/);
  assert.match(css, /\.job-progress-fill\s*\{/);
  assert.match(css, /#39ff14|57,\s*255,\s*20/);
});
