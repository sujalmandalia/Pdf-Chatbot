/**
 * Gemini embeddings.
 *
 * Three things here are easy to get wrong, and each costs you recall silently:
 *
 * 1. TASK TYPE. RAG embeds two different KINDS of text. Passages get
 *    RETRIEVAL_DOCUMENT; questions get RETRIEVAL_QUERY. The model projects them
 *    into a shared space where "how do I take time off" lands near a passage
 *    about "annual leave entitlement" despite sharing no words. Use one task type
 *    for both and retrieval measurably degrades - with no error anywhere.
 *
 * 2. NORMALIZATION. gemini-embedding-001 returns unit vectors only at its native
 *    3072 dims. We truncate to 768, so we MUST re-normalize ourselves or cosine
 *    distance is wrong.
 *
 * 3. BATCHING. Sending N texts in one request is the real fix for a tight
 *    free-tier RPM limit. But some Gemini embedding models AGGREGATE multiple
 *    inputs into a single vector. Silently. The length assertion below is the
 *    only thing standing between you and an index of averaged nonsense.
 */
import { ai, sleep, parseQuotaError, isQuotaError } from "./gemini.js";
import { config } from "../config.js";

const { embeddingModel: MODEL, embeddingDimensions: DIMENSIONS } = config.gemini;

const BATCH_SIZE = 15;        // texts per request
const MIN_GAP_MS = 1500;      // starting floor between requests; grows on every 429
const MAX_GAP_MS = 15_000;

/* ------------------------------------------------------------------- math --- */

/** L2-normalize, so cosine similarity reduces to a plain dot product. */
export function normalize(vector) {
  let sumSquares = 0;
  for (const v of vector) sumSquares += v * v;
  const magnitude = Math.sqrt(sumSquares);
  return magnitude === 0 ? vector : vector.map((v) => v / magnitude);
}

/* ----------------------------------------------------------- rate limits --- */

// Google does not publish free-tier RPM/TPM numbers - they are per-project and
// visible only in AI Studio. So don't hardcode a guess: start conservative and let
// every 429 widen the gap. The limiter converges on the real limit by itself.
let gapMs = MIN_GAP_MS;
let lastCallAt = 0;

async function pace() {
  const wait = lastCallAt + gapMs - Date.now();
  if (wait > 0) await sleep(wait);
  lastCallAt = Date.now();
}

/* ------------------------------------------------------------- embedding --- */

async function embedBatch(texts, taskType, attempt = 0) {
  await pace();
  try {
    const response = await ai.models.embedContent({
      model: MODEL,
      contents: texts,
      config: { taskType, outputDimensionality: DIMENSIONS },
    });

    const embeddings = response.embeddings ?? [];
    if (embeddings.length !== texts.length) {
      // Load-bearing. See note 3 at the top of this file.
      throw new Error(
        `AGGREGATION: sent ${texts.length} texts, got ${embeddings.length} embeddings. ` +
          `${MODEL} is aggregating inputs - set BATCH_SIZE = 1 in services/embeddings.js.`,
      );
    }
    return embeddings.map((e) => normalize(e.values));
  } catch (error) {
    if (String(error.message).startsWith("AGGREGATION:")) throw error;

    if (isQuotaError(error)) {
      const { retryAfterMs, quota, perDay } = parseQuotaError(error);

      if (perDay) {
        const err = new Error(
          `Gemini daily quota exhausted${quota ? ` (${quota})` : ""}. Waiting will not help today. ` +
            `Check https://aistudio.google.com/rate-limit`,
        );
        err.status = 429;
        throw err;
      }

      // Per-minute limit: widen the gap permanently so we stop tripping it.
      gapMs = Math.min(Math.round(gapMs * 1.5), MAX_GAP_MS);
      if (attempt >= 8) throw error;
      const wait = retryAfterMs ?? Math.min(2 ** attempt * 2000, 60_000);
      console.warn(
        `  429${quota ? ` ${quota}` : ""} - waiting ${(wait / 1000).toFixed(1)}s; ` +
          `pacing now ${(gapMs / 1000).toFixed(1)}s/req`,
      );
      await sleep(wait + 250);
      return embedBatch(texts, taskType, attempt + 1);
    }

    if (attempt >= 5) throw error;
    const wait = Math.min(2 ** attempt * 1000, 32_000);
    console.warn(`  retry ${attempt + 1} in ${wait}ms (${String(error.message).slice(0, 120)})`);
    await sleep(wait);
    return embedBatch(texts, taskType, attempt + 1);
  }
}

/**
 * Embed document passages (ingestion side).
 * @param {string[]} texts
 * @param {(done:number,total:number)=>void} [onProgress]
 */
export async function embedDocuments(texts, onProgress) {
  const vectors = [];
  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    const batch = texts.slice(start, start + BATCH_SIZE);
    vectors.push(...(await embedBatch(batch, "RETRIEVAL_DOCUMENT")));
    onProgress?.(vectors.length, texts.length);
  }
  return vectors;
}

/** Embed one search question (query side). Phase 5 uses this. */
export async function embedQuery(text) {
  const [vector] = await embedBatch([text], "RETRIEVAL_QUERY");
  return vector;
}
