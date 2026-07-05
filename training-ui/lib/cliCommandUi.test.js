const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("CLI tab exposes command preview and TOML custom args editor", () => {
  const html = read("public/index.html");

  assert.match(html, /data-tab="cli">CLI<\/button>/);
  assert.match(html, /id="tab-cli"/);
  assert.match(html, /id="cfg-cli-command"/);
  assert.match(html, /id="cfg-custom-cli-args-toml"/);
  assert.match(html, /custom_cli_args/);
});

test("CLI custom args are saved in ui_arguments and update command preview", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /function renderCliCustomArgsToml\(value = ""\)/);
  assert.match(appJs, /function parseCliCustomArgsToml\(value\)/);
  assert.match(appJs, /\$\("cfg-custom-cli-args-toml"\)\.value = renderCliCustomArgsToml\(ui\.custom_cli_args \|\| ""\)/);
  assert.match(appJs, /ui_arguments:\s*\{\s*custom_cli_args:\s*parseCliCustomArgsToml\(\$\("cfg-custom-cli-args-toml"\)\.value\)/);
  assert.match(appJs, /\/api\/jobs\/\$\{encodeURIComponent\(currentJob\)\}\/cli-command/);
  assert.match(appJs, /currentCliBaseCommand \+ \(customCliArgs \? ` \$\{customCliArgs\}` : ""\)/);
});
