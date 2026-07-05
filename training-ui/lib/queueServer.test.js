const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function routeBlock(source, route) {
  const start = source.indexOf(`app.post('${route}'`);
  assert.notStrictEqual(start, -1, `${route} route should exist`);
  const nextRoute = source.indexOf("\napp.", start + 1);
  return source.slice(start, nextRoute === -1 ? source.length : nextRoute);
}

test("queue stop interrupts the current training process and preserves queue state", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/queue/stop");

  assert.match(serverJs, /function stopTrainingJob\(/);
  assert.match(serverJs, /function stopRunningTrainingForQueue\(/);
  assert.match(block, /stopRunningTrainingForQueue\(\)/);
  assert.match(block, /queueAutoRunning\s*=\s*false/);
});

test("queue start uses an in-flight guard so duplicate starts cannot stop auto-run", () => {
  const serverJs = read("server.js");
  const block = serverJs.match(/async function runNextQueuedJob\(\) \{[\s\S]*?\n\}/);

  assert.ok(block, "runNextQueuedJob should exist");
  assert.match(serverJs, /let queueStartInFlight\s*=\s*false/);
  assert.match(block[0], /queueStartInFlight/);
  assert.match(block[0], /finally/);
});

test("status changes broadcast a queue refresh to every websocket client", () => {
  const serverJs = read("server.js");
  const block = serverJs.match(/function broadcastStatus\(jobName, status\) \{[\s\S]*?\n\}/);

  assert.ok(block, "broadcastStatus should exist");
  assert.match(serverJs, /function broadcastQueueChanged\(/);
  assert.match(block[0], /broadcastQueueChanged\(\)/);
});

test("direct training stop uses the shared queue-aware stop helper", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs/:name/train/stop");

  assert.match(block, /stopTrainingJob\(jobName/);
  assert.doesNotMatch(block, /runningJobs\.delete\(jobName\)/);
});

test("training status falls back to real process detection when memory state is stale", () => {
  const serverJs = read("server.js");
  const jobsBlock = serverJs.match(/app\.get\('\/api\/jobs'[\s\S]*?\n\}\);/);
  const statusBlock = serverJs.match(/app\.get\('\/api\/jobs\/:name\/train\/status'[\s\S]*?\n\}\);/);
  const detectorBlock = serverJs.match(/function refreshDetectedTrainingProcesses\(\) \{[\s\S]*?\n\}/);

  assert.ok(jobsBlock, "jobs list route should exist");
  assert.ok(statusBlock, "train status route should exist");
  assert.ok(detectorBlock, "async detector refresh should exist");
  assert.match(serverJs, /function getDetectedTrainingProcesses\(/);
  assert.match(serverJs, /function getDetectedTrainingProcessesFresh\(/);
  assert.match(serverJs, /_merged_config\.toml/);
  assert.match(detectorBlock[0], /spawn\('powershell\.exe'/);
  assert.doesNotMatch(detectorBlock[0], /execFileSync/);
  assert.match(serverJs, /detectedTrainingRefreshPromise/);
  assert.match(serverJs, /function isJobTraining\(/);
  assert.match(serverJs, /async function isJobTrainingFresh\(jobName\)/);
  assert.match(serverJs, /memoryJob\?\.type === 'training'/);
  assert.match(jobsBlock[0], /await getDetectedTrainingProcessesFresh\(\)/);
  assert.match(jobsBlock[0], /running:\s*isJobTraining\(d\.name/);
  assert.match(statusBlock[0], /const isRunning = await isJobTrainingFresh\(jobName\)/);
  assert.match(serverJs, /function getRunningTrainingJobName\(\)[\s\S]*getDetectedTrainingProcesses\(\{ refresh: false \}\)/);
  assert.match(serverJs, /async function getRunningTrainingJobNameFresh\(\)/);
  assert.match(serverJs, /function stopTrainingJob\(jobName\)[\s\S]*getTrainingProcessInfo\(jobName\)/);
  assert.match(serverJs, /const runningTraining = await getRunningTrainingJobNameFresh\(\)/);
  assert.match(serverJs, /Another training job is already running/);
});
