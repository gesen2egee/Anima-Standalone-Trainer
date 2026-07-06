const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function getQueueStartHelper(appJs) {
  const start = appJs.indexOf("async function runCurrentJobFromTopButton(");
  const end = appJs.indexOf("\nasync function pauseQueue()", start);

  assert.ok(start >= 0, "queue-aware top training helper should exist");
  assert.ok(end > start, "queue-aware top training helper should be isolated before pauseQueue");
  return appJs.slice(start, end);
}

test("top training start uses queue APIs instead of direct train start", () => {
  const appJs = read("public/js/app.js");
  const trainBlock = appJs.match(/\$\("btn-run"\)\.addEventListener\("click", async \(\) => \{[\s\S]*?\n\}\);/);
  const helperBlock = getQueueStartHelper(appJs);

  assert.ok(trainBlock, "btn-run click handler should exist");
  assert.match(trainBlock[0], /runCurrentJobFromTopButton\(warningMsg\)/);
  assert.match(helperBlock, /\/api\/queue/);
  assert.match(helperBlock, /\/api\/queue\/jobs\/\$\{encodeURIComponent\(currentJob\)\}/);
  assert.match(helperBlock, /\/api\/queue\/jobs\/\$\{encodeURIComponent\(currentJob\)\}\/move/);
  assert.match(helperBlock, /index: 0/);
  assert.match(helperBlock, /\/api\/queue\/start/);
  assert.doesNotMatch(trainBlock[0], /\/train\/start/);
});

test("top training start only queues selected job when another job is already running", () => {
  const appJs = read("public/js/app.js");
  const helperBlock = getQueueStartHelper(appJs);
  const runningBranch = helperBlock.match(/if \(runningJob\) \{[\s\S]*?return;\n  \}/);

  assert.match(helperBlock, /const jobs = await loadJobs\(\)/);
  assert.match(helperBlock, /const runningJob = jobs\.find\(\(job\) => job\.running \|\| job\.queueActive\)/);
  assert.ok(runningBranch, "running branch should return before starting queue");
  assert.match(runningBranch[0], /\/api\/queue\/jobs\/\$\{encodeURIComponent\(currentJob\)\}/);
  assert.match(runningBranch[0], /return;/);
  assert.doesNotMatch(runningBranch[0], /\/api\/queue\/start/);
});

test("top training stop stops the queue instead of only the selected job", () => {
  const appJs = read("public/js/app.js");
  const stopBlock = appJs.match(/\$\("btn-stop"\)\.addEventListener\("click", \(\) => \{[\s\S]*?\n\}\);/);

  assert.ok(stopBlock, "btn-stop click handler should exist");
  assert.match(stopBlock[0], /\/api\/queue\/stop/);
  assert.doesNotMatch(stopBlock[0], /\/train\/stop/);
});

test("websocket queue refresh updates job lists even for non-selected jobs", () => {
  const appJs = read("public/js/app.js");
  const wsBlock = appJs.match(/ws\.onmessage = \(event\) => \{[\s\S]*?\n  \};/);

  assert.ok(wsBlock, "websocket message handler should exist");
  assert.match(wsBlock[0], /msg\.type === "queue"/);
  assert.match(wsBlock[0], /loadJobs\(\)/);
});

test("selected training status follows jobs polling without extra status polling", () => {
  const appJs = read("public/js/app.js");
  const loadJobsBlock = appJs.match(/async function loadJobs\(\) \{[\s\S]*?\n\}/);

  assert.ok(loadJobsBlock, "loadJobs helper should exist");
  assert.match(loadJobsBlock[0], /const selectedJob = jobs\.find\(\(job\) => job\.name === currentJob\)/);
  assert.match(loadJobsBlock[0], /updateRunningState\(selectedJob\.running \|\| selectedJob\.queueActive\)/);
  assert.doesNotMatch(appJs, /function refreshCurrentJobTrainingStatus/);
  assert.doesNotMatch(appJs, /setInterval\(refreshCurrentJobTrainingStatus/);
});
