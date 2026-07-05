const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("samples refresh is scheduled when training or generation logs arrive", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /let samplesRefreshTimer = null/);
  assert.match(appJs, /function scheduleSamplesRefresh\(/);
  assert.match(appJs, /appendConsole\(msg\.data\);\s*scheduleSamplesRefresh\(\)/);
  assert.match(appJs, /msg\.data === "completed"[\s\S]*scheduleSamplesRefresh\(0\)/);
});

test("console output keeps width stable and always scrolls to the newest line", () => {
  const appJs = read("public/js/app.js");
  const css = read("public/css/style.css");

  assert.match(appJs, /consoleOutput\.scrollTop = consoleOutput\.scrollHeight/);
  assert.doesNotMatch(appJs, /const wasNearBottom/);
  assert.match(css, /\.console-output\s*\{[\s\S]*max-width:\s*100%/);
  assert.match(css, /\.console-output\s*\{[\s\S]*overflow-wrap:\s*anywhere/);
  assert.match(css, /\.console-output\s*\{[\s\S]*box-sizing:\s*border-box/);
});
