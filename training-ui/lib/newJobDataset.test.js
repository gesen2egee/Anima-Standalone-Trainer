const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");

const { buildNewJobSubsets } = require("./newJobDataset");

function makeTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "anima-new-job-dataset-"));
}

function writeImage(dir, name = "image.png") {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, name), "");
}

test("batch import reads only first-level numbered folders into subsets", () => {
  const root = makeTempDir();
  writeImage(path.join(root, "12_alice"), "a.png");
  writeImage(path.join(root, "03_bob"), "b.jpg");
  writeImage(path.join(root, "12_alice", "7_nested"), "nested.png");
  writeImage(path.join(root, "05_nested_only", "nested"), "skip.png");
  writeImage(path.join(root, "06_gif_only"), "skip.gif");
  writeImage(path.join(root, "not_a_subset"), "skip.png");

  const subsets = buildNewJobSubsets({
    imageDir: root,
    batchImport: true,
    autoBalanceRepeats: false,
    triggerCaptionPrefix: "root,",
  });

  assert.deepStrictEqual(
    subsets.map((s) => ({
      image_dir: path.basename(s.image_dir),
      num_repeats: s.num_repeats,
      caption_prefix: s.caption_prefix,
    })),
    [
      { image_dir: "03_bob", num_repeats: 3, caption_prefix: "bob," },
      { image_dir: "12_alice", num_repeats: 12, caption_prefix: "alice," },
    ],
  );
});

test("auto balance uses ceil of 100 divided by image count for batch folders", () => {
  const root = makeTempDir();
  writeImage(path.join(root, "5_three_images"), "a.png");
  writeImage(path.join(root, "5_three_images"), "b.png");
  writeImage(path.join(root, "5_three_images"), "c.png");
  for (let i = 0; i < 101; i += 1) {
    writeImage(path.join(root, "9_many_images"), `${i}.png`);
  }

  const subsets = buildNewJobSubsets({
    imageDir: root,
    batchImport: true,
    autoBalanceRepeats: true,
  });

  assert.deepStrictEqual(
    subsets.map((s) => ({ name: path.basename(s.image_dir), repeats: s.num_repeats })),
    [
      { name: "5_three_images", repeats: 34 },
      { name: "9_many_images", repeats: 1 },
    ],
  );
});

test("single child folder fallback is used when selected folder has no images", () => {
  const root = makeTempDir();
  const child = path.join(root, "images");
  writeImage(child, "a.webp");

  const subsets = buildNewJobSubsets({
    imageDir: root,
    batchImport: false,
    triggerCaptionPrefix: "hero,",
  });

  assert.strictEqual(subsets.length, 1);
  assert.strictEqual(subsets[0].image_dir, child);
  assert.strictEqual(subsets[0].num_repeats, 1);
  assert.strictEqual(subsets[0].caption_prefix, "hero,");
});

test("single folder keeps current folder when it has direct images", () => {
  const root = makeTempDir();
  const child = path.join(root, "images");
  writeImage(root, "root.png");
  writeImage(child, "child.png");

  const subsets = buildNewJobSubsets({
    imageDir: root,
    batchImport: false,
    triggerCaptionPrefix: "hero,",
  });

  assert.strictEqual(subsets.length, 1);
  assert.strictEqual(subsets[0].image_dir, root);
});
