const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..");

function read(relPath) {
  return fs.readFileSync(path.join(ROOT, relPath), "utf8");
}

test("web UI exposes a recognizable app icon", () => {
  const html = read("public/index.html");
  const css = read("public/css/style.css");
  const iconPath = path.join(ROOT, "public", "app-icon.png");

  assert.ok(fs.existsSync(iconPath), "app icon file should exist");
  assert.match(html, /rel="icon"[^>]+href="app-icon\.png"/);
  assert.match(html, /rel="apple-touch-icon"[^>]+href="app-icon\.png"/);
  assert.match(html, /src="app-icon\.png"[^>]+class="app-title-icon"/);
  assert.match(css, /\.app-title-icon/);
});
