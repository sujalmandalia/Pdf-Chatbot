/**
 * Apply schema.sql. Run with: npm run migrate
 *
 * Not a real migration framework on purpose - the whole schema is two tables and
 * every statement is IF NOT EXISTS, so re-running is safe and idempotent. Adding
 * Knex/Prisma here would add a lot of machinery and hide the SQL, which is the
 * part actually worth understanding.
 *
 * The one interesting job it has: substituting the embedding dimension from
 * config, so `vector(N)` and EMBEDDING_DIMENSIONS cannot disagree.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pool, query } from "./pool.js";
import { config } from "../config.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const dimensions = config.gemini.embeddingDimensions;

const sql = fs
  .readFileSync(path.join(here, "schema.sql"), "utf8")
  .replaceAll(":dimensions", String(dimensions));

try {
  await query(sql);
  console.log(`schema applied  ·  embedding column: vector(${dimensions})`);

  // Read the schema back from Postgres rather than trusting that it worked.
  // If a `chunks` table already existed at a different width, this is where
  // you find out - before ingesting anything.
  const { rows } = await query(`
    SELECT extversion AS pgvector_version,
           (SELECT atttypmod FROM pg_attribute
             WHERE attrelid = 'chunks'::regclass AND attname = 'embedding') AS column_dimensions
      FROM pg_extension WHERE extname = 'vector'
  `);
  const actual = rows[0]?.column_dimensions;
  console.log(`pgvector ${rows[0]?.pgvector_version}  ·  chunks.embedding is vector(${actual})`);

  if (actual !== dimensions) {
    console.error(
      `\nMISMATCH: config says ${dimensions} dims but the existing column is ${actual}.\n` +
      `Postgres will not widen a vector column in place. Either set EMBEDDING_DIMENSIONS=${actual},\n` +
      `or drop the table (losing its embeddings) and migrate again.`,
    );
    process.exitCode = 1;
  }
} finally {
  await pool.end();              // otherwise node hangs on the open pool
}
