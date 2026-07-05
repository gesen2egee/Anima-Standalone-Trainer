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

test("new job modal exposes trigger words and auto tag action", () => {
  const html = read("public/index.html");

  assert.match(html, /id="new-job-trigger-words"/);
  assert.match(html, /id="btn-create-job-auto-tag"/);
  assert.match(html, /Create \+ Auto Tag/);
});

test("new job creation sends trigger words and can auto tag with defaults", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /async function createJobFromModal\(\{ autoTag = false \} = \{\}\)/);
  assert.match(appJs, /trigger_words:\s*\$\("new-job-trigger-words"\)\.value\.trim\(\)/);
  assert.match(appJs, /\$\("btn-create-job-auto-tag"\)\.addEventListener\("click", \(\) => createJobFromModal\(\{ autoTag: true \}\)\)/);
  assert.match(appJs, /async function runNewJobDefaultTagger\(jobName, imageDir\)/);
  assert.match(appJs, /\/api\/jobs\/\$\{encodeURIComponent\(jobName\)\}\/tag-captions/);
  assert.match(appJs, /caption_extension:\s*"\.txt"/);
  assert.match(appJs, /include_char:\s*true/);
  assert.match(appJs, /include_rating:\s*true/);
  assert.match(appJs, /include_general:\s*true/);
});

test("server writes trigger words into first subset caption prefix", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs");

  assert.match(serverJs, /function normalizeCaptionPrefixFromTriggerWords\(value\)/);
  assert.match(block, /trigger_words/);
  assert.match(block, /normalizeCaptionPrefixFromTriggerWords\(trigger_words\)/);
  assert.match(block, /caption_prefix\s*=\s*triggerCaptionPrefix/);
});
