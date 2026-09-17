-- ============================================================================
-- Schema for a custom RAG store. Two tables only.
--
--   documents : one row per uploaded PDF   (the thing a user uploaded)
--   chunks    : many rows per document     (the thing we actually search)
--
-- Everything a citation needs (page number, chunk position, filename) lives on
-- the chunk row. Retrieval returns chunks, so anything not stored here cannot
-- be shown as a source later.
-- ============================================================================

-- pgvector adds the `vector` column type and the distance operators (<=>, <->, <#>).
-- IF NOT EXISTS makes this file safe to re-run.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id            BIGSERIAL PRIMARY KEY,
  filename      TEXT        NOT NULL,
  page_count    INTEGER     NOT NULL,
  chunk_count   INTEGER     NOT NULL DEFAULT 0,
  -- sha256 of the file bytes. Lets us detect a re-upload of the same PDF and
  -- refuse to pay for embedding it twice.
  content_hash  TEXT        NOT NULL UNIQUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,

  -- ON DELETE CASCADE: deleting a document must delete its chunks. Orphaned
  -- embeddings are worse than useless - they keep answering questions about a
  -- document the user thinks is gone.
  document_id  BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

  -- Ordinal within the document (0,1,2...). Used to fetch neighbouring chunks
  -- and to make retrieval results reproducible/orderable.
  chunk_index  INTEGER     NOT NULL,

  -- THE citation field. Carried from PDF extraction all the way to the answer.
  page_number  INTEGER     NOT NULL,

  -- The exact text that was embedded. Must be stored verbatim: this is what we
  -- feed the LLM as context, and what the user sees as the quoted source.
  content      TEXT        NOT NULL,

  token_count  INTEGER     NOT NULL,

  -- The embedding. :dimensions is substituted by migrate.js from config, so the
  -- column width and the embedding model can never drift apart.
  -- vector(N) is a FIXED-WIDTH type: Postgres rejects an insert
  -- of any other length. That strictness is a feature - it makes a model/dimension
  -- mismatch a loud error at insert time instead of silently wrong search results.
  --
  -- Stored as float4 (4 bytes/dim) + header, so 768 dims is ~3KB per chunk row.
  embedding    vector(:dimensions) NOT NULL,

  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE (document_id, chunk_index)
);

-- A plain btree index for the ordinary lookups (list chunks of a document).
-- This does NOT help vector search at all.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- NOTE: no ANN (HNSW/IVFFlat) index yet - on purpose. See Phase 7.
-- Without one, pgvector does an exact sequential scan: 100% correct recall,
-- and fast enough for thousands of chunks. An ANN index is an APPROXIMATION;
-- adding it before you can measure retrieval quality means you can never tell
-- whether a bad answer came from your pipeline or from the index missing a row.
