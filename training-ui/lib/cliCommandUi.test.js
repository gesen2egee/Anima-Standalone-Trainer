const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("CLI tab exposes command and TOML sub-tabs plus custom args editor", () => {
  const html = read("public/index.html");
  const consoleTabIndex = html.indexOf('data-tab="console"');
  const cliTabIndex = html.indexOf('data-tab="cli"');
  const generateTabIndex = html.indexOf('data-tab="generate"');

  assert.match(html, /data-tab="console">Console<\/button>[\s\S]*data-tab="cli">CLI<\/button>/);
  assert.ok(cliTabIndex > consoleTabIndex, "CLI tab should be to the right of Console");
  assert.match(html, /data-tab="cli">CLI<\/button>\s*<button class="tab" data-tab="generate">產生圖片<\/button>/);
  assert.ok(generateTabIndex > cliTabIndex, "Generate tab should be to the right of CLI");
  assert.match(html, /id="tab-cli"/);
  assert.match(html, /data-cli-subtab="command"/);
  assert.match(html, /data-cli-subtab="config-toml"/);
  assert.match(html, /data-cli-subtab="dataset-toml"/);
  assert.match(html, /data-cli-subtab="sample-txt"/);
  assert.match(html, /id="cli-subtab-command"/);
  assert.match(html, /id="cli-subtab-config-toml"/);
  assert.match(html, /id="cli-subtab-dataset-toml"/);
  assert.match(html, /id="cli-subtab-sample-txt"/);
  assert.match(html, /id="cfg-cli-command"/);
  assert.match(html, /id="cfg-cli-toml"/);
  assert.match(html, /id="cfg-cli-dataset-toml"/);
  assert.match(html, /id="cfg-cli-sample-txt"/);
  assert.match(html, /id="cfg-custom-cli-args-toml"/);
  assert.match(html, /custom_cli_args/);
});

test("CLI custom args are saved in ui_arguments and update command/TOML previews", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /let currentCliToml = ""/);
  assert.match(appJs, /function showCliSubtab\(name\)/);
  assert.match(appJs, /function renderCliCustomArgsToml\(value = ""\)/);
  assert.match(appJs, /function parseCliCustomArgsToml\(value\)/);
  assert.match(appJs, /\$\("cfg-custom-cli-args-toml"\)\.value = renderCliCustomArgsToml\(ui\.custom_cli_args \|\| ""\)/);
  assert.match(appJs, /ui_arguments:\s*\{\s*custom_cli_args:\s*parseCliCustomArgsToml\(\$\("cfg-custom-cli-args-toml"\)\.value\)/);
  assert.match(appJs, /\/api\/jobs\/\$\{encodeURIComponent\(currentJob\)\}\/cli-command/);
  assert.match(appJs, /currentCliToml = data\.toml \|\| ""/);
  assert.match(appJs, /currentCliDatasetToml = data\.dataset_toml \|\| ""/);
  assert.match(appJs, /currentCliSampleText = data\.sample_prompts \|\| ""/);
  assert.match(appJs, /\$\("cfg-cli-toml"\)\.value = currentCliToml/);
  assert.match(appJs, /\$\("cfg-cli-dataset-toml"\)\.value = currentCliDatasetToml/);
  assert.match(appJs, /\$\("cfg-cli-sample-txt"\)\.value = currentCliSampleText/);
  assert.match(appJs, /currentCliBaseCommand \+ \(customCliArgs \? ` \$\{customCliArgs\}` : ""\)/);
});
