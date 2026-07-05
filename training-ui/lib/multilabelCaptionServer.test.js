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

test("multilabel caption API launches the project venv python tagger", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "/api/jobs/:name/tag-captions");

  assert.match(block, /getGlobalConfig\(\)/);
  assert.match(block, /getVenvPaths\(venvPath\)/);
  assert.match(block, /tools[\\/]tag_images_by_multilabel_timm\.py/);
  assert.match(block, /caption_extension/);
  assert.match(block, /include_char/);
  assert.match(block, /include_rating/);
  assert.match(block, /include_general/);
});
