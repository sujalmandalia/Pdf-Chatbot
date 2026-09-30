/**
 * Shared Gemini client and quota handling.
 *
 * This file exists because of a bug found while testing Phase 6's streaming:
 * embeddings.js had careful 429 handling and generation.js had none, so the
 * first rate limit on the chat model killed the request outright. Free-tier
 * limits apply to BOTH models, so the handling belongs in one place.
 */
import { GoogleGenAI } from "@google/genai";
import { config } from "../config.js";

export const ai = new GoogleGenAI({ apiKey: config.gemini.apiKey });

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * A 429 body carries the answer: a RetryInfo with `retryDelay`, and a quota
 * metric naming which limit you hit. The metric matters most - it tells you
 * whether waiting helps at all (per-minute) or not (per-day).
 */
export function parseQuotaError(error) {
  const raw = error?.message ?? String(error);
  let retryAfterMs = null;
  let quota = null;

  try {
    const details = JSON.parse(raw.slice(raw.indexOf("{")))?.error?.details ?? [];
    for (const d of details) {
      const type = d["@type"] ?? "";
      if (type.includes("RetryInfo") && d.retryDelay) {
        retryAfterMs = Math.ceil(parseFloat(d.retryDelay) * 1000);
      }
      if (type.includes("QuotaFailure")) {
        const violation = d.violations?.[0];
        quota = violation?.quotaId ?? violation?.quotaMetric ?? null;
      }
    }
  } catch {
    const match = /retryDelay"?:\s*"?(\d+(?:\.\d+)?)s/.exec(raw);
    if (match) retryAfterMs = Math.ceil(parseFloat(match[1]) * 1000);
  }

  const perDay = quota ? /PerDay/i.test(quota) : /per day/i.test(raw);
  return { retryAfterMs, quota, perDay };
}

/**
 * Turn a Gemini failure into something a person can read.
 *
 * The SDK's ApiError message is the provider's JSON body, double-encoded -
 * shown as-is it reaches the chat window as a wall of escaped braces. Our own
 * quota errors (thrown by withQuotaRetry) are already readable and pass through.
 *
 * @returns {{status:number, message:string, retryable:boolean}}
 */
export function describeError(err) {
  const status = Number(err?.status) || 500;
  const raw = err?.message ?? "";
  const fromProvider = err?.name === "ApiError" || raw.trimStart().startsWith("{");

  if (!fromProvider) {
    return { status, message: raw || "Something went wrong.", retryable: status === 429 && !/daily/i.test(raw) };
  }

  const known = {
    429: "The AI model is rate-limited right now. Wait a few seconds and try again.",
    500: "The AI model hit an internal error. Try again in a moment.",
    503: "The AI model is busy right now. This is usually temporary - try again in a few seconds.",
    504: "The AI model took too long to respond. Try again.",
    400: "The AI model rejected the request.",
    403: "The AI model refused the request - check the API key and its permissions.",
  };
  return {
    status,
    message: known[status] ?? "The AI model could not answer right now. Try again shortly.",
    retryable: [429, 500, 503, 504].includes(status),
  };
}

export const isQuotaError = (error) =>
  /\b429\b|RESOURCE_EXHAUSTED|quota/i.test(error?.message ?? "") || error?.status === 429;

/**
 * Retry a Gemini call through per-minute rate limits.
 *
 * Waits for however long the server asked for, because a guessed backoff is
 * either too short (and trips the limit again) or needlessly slow. Gives up
 * immediately on a per-DAY quota: no amount of waiting fixes that today, and
 * pretending otherwise just hangs the request.
 *
 * @param {() => Promise<T>} fn
 * @param {string} label  appears in the log line, e.g. "generate"
 * @template T
 */
export async function withQuotaRetry(fn, label, maxAttempts = 4) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (!isQuotaError(error)) throw error;

      const { retryAfterMs, quota, perDay } = parseQuotaError(error);

      if (perDay) {
        const err = new Error(
          `Gemini daily quota exhausted${quota ? ` (${quota})` : ""}. Waiting will not help today. ` +
            `Check https://aistudio.google.com/rate-limit`,
        );
        err.status = 429;
        throw err;
      }

      if (attempt >= maxAttempts - 1) {
        const err = new Error(
          `Gemini rate limit on ${label} after ${maxAttempts} attempts${quota ? ` (${quota})` : ""}. Try again shortly.`,
        );
        err.status = 429;
        throw err;
      }

      const wait = retryAfterMs ?? Math.min(2 ** attempt * 2000, 30_000);
      console.warn(`  429 on ${label}${quota ? ` ${quota}` : ""} - waiting ${(wait / 1000).toFixed(1)}s`);
      await sleep(wait + 250);
    }
  }
}
