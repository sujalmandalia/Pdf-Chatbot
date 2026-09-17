import { Router } from "express";
import { query } from "../db/pool.js";
import { config } from "../config.js";

export const healthRouter = Router();

/**
 * Liveness + readiness.
 *
 * Worth doing properly in a RAG service: the two things that break are the DB
 * and the embedding config, and both fail in confusing ways downstream. Checking
 * pgvector's presence and the actual column width here turns "search returns
 * nothing" into "pgvector is missing" at the one endpoint you'd hit first.
 */
healthRouter.get("/health", async (req, res) => {
  const { rows } = await query(`
    SELECT (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector,
           (SELECT atttypmod FROM pg_attribute
             WHERE attrelid = to_regclass('chunks') AND attname = 'embedding') AS column_dimensions,
           (SELECT count(*) FROM documents) AS documents,
           (SELECT count(*) FROM chunks)    AS chunks
  `);
  const db = rows[0];

  res.json({
    status: "ok",
    uptimeSeconds: Math.round(process.uptime()),
    database: {
      pgvector: db.pgvector,
      columnDimensions: db.column_dimensions,
      documents: Number(db.documents),      // pg returns bigint as string
      chunks: Number(db.chunks),
    },
    embedding: {
      model: config.gemini.embeddingModel,
      dimensions: config.gemini.embeddingDimensions,
      matchesColumn: db.column_dimensions === config.gemini.embeddingDimensions,
    },
  });
});
