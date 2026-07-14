const { randomInt } = require("crypto");

const DEFAULT_NEGATIVE_PROMPT =
  "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, low quality, worst quality, blurry, bad anatomy, extra limbs, deformed, watermark, text, signature, bareness, artifacts, hands, copyrights name, jpeg_artifacts, scan_artifacts, bad hands, missing fingers, extra digit, fewer digits, artistic error, ye-pop, deviantart, logo, patreon logo";

function buildNewJobSamplePrompts(triggerWords, nextSeed = () => randomInt(0, 2_147_483_647)) {
  const prompt = String(triggerWords || "").replace(/[\r\n]+/g, " ").trim() || "1girl";

  const firstSeed = nextSeed();
  const candidateSeed = nextSeed();
  const secondSeed = candidateSeed === firstSeed
    ? (firstSeed + 1) % 2_147_483_647
    : candidateSeed;

  return [firstSeed, secondSeed]
    .map(
      (seed) =>
        `${prompt} --w 832 --h 1216 --s 28 --d ${seed} --l 3.5 --n ${DEFAULT_NEGATIVE_PROMPT}`,
    )
    .join("\n");
}

module.exports = { buildNewJobSamplePrompts, DEFAULT_NEGATIVE_PROMPT };
