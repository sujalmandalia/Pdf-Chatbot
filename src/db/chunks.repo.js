/**
 * All SQL touching `chunks`.
 *
 * Every function here takes an optional `client` so it can run either on the pool
 * (a standalone read) or inside a transaction (the ingestion write). That is the
 * usual shape for a repository that participates in transactions.
 */
import { pool, toVector } from "./pool.js";

/**
 * Insert every chunk of a document in ONE statement.
 *
 * Why one statement rather than a loop of inserts: each round-trip to Postgres is
 * ~1ms, so 500 chunks is half a second of pure waiting. More importantly, a
 * multi-row VALUES list is a single atomic statement - there is no window in
 * which half the chunks exist.
 *
 * The placeholder maths is the fiddly part. With 6 columns per row, row i uses
 * $(i*6+1) ... $(i*6+6). We build that string rather than interpolating values,
 * because interpolating is how you get SQL injection - and pgvector's text format
 * ("[0.1,0.2]") looks alarmingly like something you could just paste in.
 */
export async function insertChunks(client, documentId, chunks, embeddings) {
  if (chunks.length === 0) return 0;

  const COLUMNS = 6;
  const values = [];
  const placeholders = chunks.map((chunk, i) => {
    values.push(
      documentId,
      chunk.chunkIndex,
      chunk.pageNumber,
      chunk.content,
      chunk.tokenCount,
      toVector(embeddings[i]),     // number[] -> "[0.1,0.2,...]"
    );
    const base = i * COLUMNS;
    return `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6})`;
  });

  const { rowCount } = await client.query(
    `INSERT INTO chunks (document_id, chunk_index, page_number, content, token_count, embedding)
     VALUES ${placeholders.join(", ")}`,
    values,
  );
  return rowCount;
}

/**
 * Read a document's chunks back for inspection.
 *
 * Note what is NOT selected: `embedding`. 768 floats per row would be ~3KB of
 * JSON per chunk and tells you nothing you can read. We return its length as
 * proof it exists and is the right size.
 */
export async function listChunksByDocument(documentId) {
  const { rows } = await pool.query(
    `SELECT id, chunk_index, page_number, content, token_count,
            vector_dims(embedding) AS embedding_dimensions,
            -- The vector's magnitude. Should be ~1 for every row: proof that
            -- normalization actually happened before insert.
            --
            -- Derived from the inner-product operator rather than l2_norm(),
            -- which pgvector defines only for halfvec/sparsevec, not vector.
            -- The <#> operator returns the NEGATIVE inner product, so a vector
            -- against itself gives -(v.v) = -||v||^2, hence sqrt of its negation.
            round(sqrt(-1 * (embedding <#> embedding))::numeric, 6) AS embedding_norm
       FROM chunks
      WHERE document_id = $1
      ORDER BY chunk_index`,
    [documentId],
  );
  return rows;
}

export async function countChunks(documentId) {
  const { rows } = await pool.query(
    `SELECT count(*)::int AS count FROM chunks WHERE document_id = $1`,
    [documentId],
  );
  return rows[0].count;
}
