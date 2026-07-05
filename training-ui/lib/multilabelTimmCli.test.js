const assert = require("assert");
const fs = require("fs");
const path = require("path");
const test = require("node:test");

const ROOT = path.join(__dirname, "..", "..");

test("multilabel timm CLI can tag an explicit sd-scripts image list", () => {
  const source = fs.readFileSync(path.join(ROOT, "library", "multilabel_timm.py"), "utf8");
  const iterStart = source.indexOf("def iter_image_paths");
  const iterEnd = source.indexOf("\ndef write_captions_for_directory", iterStart);
  const iterBlock = source.slice(iterStart, iterEnd);

  assert.match(source, /--image-list/);
  assert.match(source, /image_list=args\.image_list/);
  assert.match(source, /Path\(image_list\)\.read_text\(encoding="utf-8"\)/);
  assert.match(iterBlock, /root\.iterdir\(\)/);
  assert.doesNotMatch(iterBlock, /rglob/);
});
