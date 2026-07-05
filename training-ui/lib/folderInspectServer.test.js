const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

function routeBlock(source, method, route) {
  const start = source.indexOf(`app.${method}('${route}'`);
  assert.notStrictEqual(start, -1, `${method.toUpperCase()} ${route} route should exist`);
  const nextRoute = source.indexOf("\napp.", start + 1);
  return source.slice(start, nextRoute === -1 ? source.length : nextRoute);
}

test("server can select and inspect image folders for caption coverage", () => {
  const serverJs = read("server.js");
  const selectBlock = routeBlock(serverJs, "post", "/api/system/select-folder");
  const inspectBlock = routeBlock(serverJs, "post", "/api/system/inspect-image-folder");
  const pickerBlock = serverJs.match(/function selectFolderDialog\([\s\S]*?\n\}/)?.[0] || "";

  assert.match(serverJs, /function selectFolderDialog\(/);
  assert.match(serverJs, /function inspectImageFolder\(/);
  assert.match(pickerBlock, /new Promise/);
  assert.match(serverJs, /spawn\(/);
  assert.match(serverJs, /IFileOpenDialog/);
  assert.match(serverJs, /FOS_PICKFOLDERS/);
  assert.doesNotMatch(serverJs, /FolderBrowserDialog/);
  assert.doesNotMatch(pickerBlock, /execFileSync/);
  assert.match(selectBlock, /selectFolderDialog\(/);
  assert.match(selectBlock, /await selectFolderDialog\(/);
  assert.match(inspectBlock, /inspectImageFolder\(/);
  assert.match(inspectBlock, /caption_extension/);
  assert.match(serverJs, /image_count/);
  assert.match(serverJs, /missing_caption/);
  assert.match(serverJs, /empty_caption/);
});
