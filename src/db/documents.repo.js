/**
 * All SQL touching `documents` lives here.
 *
 * Why a repository file at all: it keeps SQL out of route handlers, so a route
 * reads as a sequence of steps ("find, extract, insert") and the shape of the
 * database can change without touching HTTP code. In Phase 4 the chunk inserts
 * join this same layer.
 */
import { query } from "./pool.js";

export async function findById(id) {
  const { rows } = await query(
    `SELECT id, filename, page_count, chunk_count, created_at
       FROM documents WHERE id = $1`,
    [id],
  );
  return rows[0] ?? null;
}

export async function findByHash(contentHash) {
  const { rows } = await query(
    `SELECT id, filename, page_count, chunk_count, created_at
       FROM documents WHERE content_hash = $1`,
    [contentHash],
  );
  return rows[0] ?? null;
}

/**
 * Insert the document row.
 *
 * Takes a `client` rather than using the pool: this runs inside the ingestion
 * transaction, alongside the chunk inserts. Using the pool here would put the
 * document row on a DIFFERENT connection - outside the transaction - so a later
 * rollback would leave the document behind with zero chunks.
 */
export async function insertDocument(client, { filename, pageCount, chunkCount, contentHash }) {
  const { rows } = await client.query(
    `INSERT INTO documents (filename, page_count, chunk_count, content_hash)
     VALUES ($1, $2, $3, $4)
     RETURNING id, filename, page_count, chunk_count, created_at`,
    [filename, pageCount, chunkCount, contentHash],
  );
  return rows[0];
}

export async function listDocuments() {
  const { rows } = await query(
    `SELECT id, filename, page_count, chunk_count, created_at
       FROM documents ORDER BY created_at DESC`,
  );
  return rows;
}

export async function deleteDocument(id) {
  // chunks go with it via ON DELETE CASCADE
  const { rowCount } = await query(`DELETE FROM documents WHERE id = $1`, [id]);
  return rowCount > 0;
}
