const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("server wires automatic resume into training resume and LoRA checkpoint fallback", () => {
  const serverJs = read("server.js");

  assert.match(serverJs, /findAutoResumeSource/);
  assert.match(serverJs, /autoResumeEnabled\s*=\s*jobConfig\.network_arguments\?\.auto_resume_last_state\s*!==\s*false/);
  assert.match(serverJs, /findAutoResumeSource\(outputDir,\s*outputName/);
  assert.match(serverJs, /source\??\.type\s*===\s*['"]state['"][\s\S]*merged\.training_arguments\.resume\s*=\s*source\.path/);
  assert.match(serverJs, /source\??\.type\s*===\s*['"]checkpoint['"][\s\S]*autoResumeNetworkWeights\s*=\s*source\.path/);
  assert.match(serverJs, /merged\.network_arguments\.network_weights\s*=\s*autoResumeNetworkWeights/);
});

test("UI exposes automatic resume and defaults it on for new or legacy configs", () => {
  const html = read("public/index.html");
  const appJs = read("public/js/app.js");
  const templateToml = read("templates/config_template.toml");

  assert.match(html, /id="group-resume-training"/);
  assert.doesNotMatch(html, /id="group-resume-training"\s+style="display:\s*none;"/);
  assert.match(html, /id="cfg-auto-resume"[^>]*checked/);
  assert.match(appJs, /cfg-auto-resume"\)\.checked\s*=\s*n\.auto_resume_last_state\s*\?\?\s*true/);
  assert.match(templateToml, /auto_resume_last_state\s*=\s*true/);
});
