const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("server treats legacy false auto-resume values as default-on unless user opt-out is marked", () => {
  const serverJs = read("server.js");

  assert.match(serverJs, /auto_resume_last_state_user_set/);
  assert.match(serverJs, /autoResumeUserSet\s*=\s*jobConfig\.network_arguments\?\.auto_resume_last_state_user_set\s*===\s*true/);
  assert.match(serverJs, /autoResumeEnabled\s*=\s*!\(\s*autoResumeUserSet\s*&&\s*jobConfig\.network_arguments\?\.auto_resume_last_state\s*===\s*false\s*\)/);
  assert.match(serverJs, /delete merged\.network_arguments\.auto_resume_last_state_user_set/);
});

test("UI saves an explicit auto-resume choice marker with the checkbox value", () => {
  const appJs = read("public/js/app.js");

  assert.match(appJs, /auto_resume_last_state:\s*\$\("cfg-auto-resume"\)\.checked/);
  assert.match(appJs, /auto_resume_last_state_user_set:\s*true/);
  assert.match(appJs, /cfg-auto-resume"\)\.checked\s*=\s*!\(n\.auto_resume_last_state_user_set\s*===\s*true\s*&&\s*n\.auto_resume_last_state\s*===\s*false\)/);
});
