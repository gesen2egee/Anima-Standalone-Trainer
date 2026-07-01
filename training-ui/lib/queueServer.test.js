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
