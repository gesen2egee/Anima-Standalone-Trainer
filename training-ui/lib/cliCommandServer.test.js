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

test("server exposes CLI command preview from the shared launch builder", () => {
  const serverJs = read("server.js");
  const block = routeBlock(serverJs, "get", "/api/jobs/:name/cli-command");

  assert.match(serverJs, /function normalizeCustomCliArgs\(value\)/);
  assert.match(serverJs, /function buildTrainingLaunchCommand\(/);
  assert.match(block, /buildTrainingLaunchCommand\(/);
  assert.match(block, /base_command/);
  assert.match(block, /toml/);
  assert.match(block, /TOML\.stringify\(mergedConfig\)/);
  assert.match(block, /custom_cli_args/);
});

test("training launch appends custom CLI args after the config file argument", () => {
  const serverJs = read("server.js");
  const helperBlock = serverJs.match(/function buildTrainingLaunchCommand\([\s\S]*?\n\}/);

  assert.ok(helperBlock, "buildTrainingLaunchCommand helper should exist");
  assert.match(helperBlock[0], /ui_arguments\?\.custom_cli_args/);
  assert.match(helperBlock[0], /normalizeCustomCliArgs/);
  assert.match(helperBlock[0], /baseTrainCmd/);
  assert.match(helperBlock[0], /customCliArgs \? `\$\{baseTrainCmd\} \$\{customCliArgs\}` : baseTrainCmd/);
});
