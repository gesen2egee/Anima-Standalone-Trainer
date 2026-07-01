const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("top training start uses queue APIs instead of direct train start", () => {
  const appJs = read("public/js/app.js");
  const trainBlock = appJs.match(/\$\("btn-run"\)\.addEventListener\("click", async \(\) => \{[\s\S]*?\n\}\);/);

  assert.ok(trainBlock, "btn-run click handler should exist");
  assert.match(trainBlock[0], /\/api\/queue\/jobs\/\$\{encodeURIComponent\(currentJob\)\}/);
  assert.match(trainBlock[0], /\/api\/queue\/start/);
  assert.doesNotMatch(trainBlock[0], /\/train\/start/);
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
