/**
 * A single shared Postgres connection pool.
 *
 * Why a pool and not a client: opening a TCP connection + auth to Postgres costs
 * ~5-20ms. A pool keeps a few open and hands them out. `pg` also serialises
 * queries on a single client, so one client = no concurrency at all.
 *
 * This module is imported everywhere; the pool is created once because ESM
 * modules are singletons.
 */
import pg from "pg";
import { config } from "../config.js";

export const pool = new pg.Pool({
  connectionString: config.databaseUrl,
  max: 10,                       // plenty; the bottleneck in RAG is the Gemini API
  idleTimeoutMillis: 30_000,
});

// A pooled client can die in the background (server restart, network blip). Without
// this listener that emits an 'error' event with no handler and crashes the process.
pool.on("error", (err) => console.error("idle postgres client error:", err.message));

/** Thin wrapper so call sites read as `query(sql, params)`. */
export const query = (sql, params) => pool.query(sql, params);

/**
 * Run several statements inside one transaction on ONE client.
 *
 * Needed in Phase 4: inserting a document row plus its N chunk rows must be
 * all-or-nothing. A half-ingested PDF is the worst state to be in - retrieval
 * would silently answer from a partial document with no way to tell.
 */
export async function withTransaction(fn) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (err) {
    await client.query("ROLLBACK");
    throw err;
  } finally {
    client.release();            // back to the pool, NOT closed
  }
}

/**
 * Serialise a JS number[] into pgvector's text input format: "[0.1,0.2,...]".
 *
 * pgvector accepts this string for a vector column, so we need no extra driver
 * plugin. Note it is NOT a Postgres array ('{...}') - a common first bug.
 * Passing a plain JS array would make `pg` send '{...}' and Postgres would reject it.
 */
export function toVector(values) {
  return `[${values.join(",")}]`;
}
