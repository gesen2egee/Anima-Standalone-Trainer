const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const test = require("node:test");

const {
  listSdScriptsImages,
  resolveDatasetImageFolders,
  inspectDatasetImageFolders,
} = require("./datasetImageMatch");

function makeTempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "anima-dataset-match-"));
}

function writeFile(filePath, content = "") {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}

test("sd-scripts image matching reads only direct supported image files", () => {
  const root = makeTempDir();
  writeFile(path.join(root, "a.png"));
  writeFile(path.join(root, "b.JPG"));
  writeFile(path.join(root, "c.gif"));
  writeFile(path.join(root, "nested", "d.png"));

  const images = listSdScriptsImages(root).map((file) => path.basename(file));

  assert.deepStrictEqual(images, ["a.png", "b.JPG"]);
});

test("non-batch selection uses current folder before single-child fallback", () => {
  const root = makeTempDir();
  writeFile(path.join(root, "root.png"));
  writeFile(path.join(root, "only_child", "child.png"));

  const result = resolveDatasetImageFolders({ imageDir: root });

  assert.strictEqual(result.mode, "single");
  assert.strictEqual(result.folders.length, 1);
  assert.strictEqual(result.folders[0].imageDir, root);
  assert.strictEqual(result.folders[0].imageCount, 1);
});

test("non-batch selection falls back to only child when current folder has no direct images", () => {
  const root = makeTempDir();
  const child = path.join(root, "only_child");
  writeFile(path.join(child, "child.webp"));

  const result = resolveDatasetImageFolders({ imageDir: root });

  assert.strictEqual(result.mode, "single");
  assert.strictEqual(result.folders.length, 1);
  assert.strictEqual(result.folders[0].imageDir, child);
  assert.strictEqual(result.totalImages, 1);
});

test("batch selection matches only numbered first-level folders with direct images", () => {
  const root = makeTempDir();
  writeFile(path.join(root, "10_alpha", "a.png"));
  writeFile(path.join(root, "03_beta", "nested", "b.png"));
  writeFile(path.join(root, "7_gamma", "g.gif"));
  writeFile(path.join(root, "2_delta", "D.BMP"));

  const result = resolveDatasetImageFolders({ imageDir: root, batchImport: true });

  assert.strictEqual(result.mode, "batch");
  assert.strictEqual(result.matchedFolderCount, 2);
  assert.strictEqual(result.totalImages, 2);
  assert.deepStrictEqual(
    result.folders.map((folder) => ({
      name: path.basename(folder.imageDir),
      repeats: folder.repeats,
      triggerWords: folder.triggerWords,
      imageCount: folder.imageCount,
    })),
    [
      { name: "2_delta", repeats: 2, triggerWords: "delta", imageCount: 1 },
      { name: "10_alpha", repeats: 10, triggerWords: "alpha", imageCount: 1 },
    ],
  );
});

test("folder inspection reports matched folders, total images, and caption coverage", async () => {
  const root = makeTempDir();
  writeFile(path.join(root, "5_alpha", "a.png"));
  writeFile(path.join(root, "5_alpha", "a.txt"), "alpha");
  writeFile(path.join(root, "5_alpha", "b.jpg"));
  writeFile(path.join(root, "8_beta", "c.png"));
  writeFile(path.join(root, "8_beta", "c.txt"), "");

  const summary = await inspectDatasetImageFolders({
    imageDir: root,
    batchImport: true,
    captionExtension: ".txt",
  });

  assert.strictEqual(summary.matched_folder_count, 2);
  assert.strictEqual(summary.image_count, 3);
  assert.strictEqual(summary.missing_caption, 1);
  assert.strictEqual(summary.empty_caption, 1);
});
