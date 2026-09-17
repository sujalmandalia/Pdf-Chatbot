/**
 * The ingestion endpoint.
 *
 * Phase 2 scope: upload -> extract text -> store the document row.
 * Phases 3 and 4 will insert themselves between extraction and the response
 * (chunk the pages, embed the chunks, store them). The route stays a readable
 * list of steps; each phase adds one line.
 */
import { Router } from "express";
import crypto from "node:crypto";
import { uploadPdf } from "../middleware/upload.js";
import { httpError } from "../middleware/errorHandler.js";
import { extractPages } from "../services/pdf.js";
import { chunkPages, DEFAULTS, POLICY_PRESET } from "../services/chunker.js";
import { embedDocuments } from "../services/embeddings.js";
import { withTransaction } from "../db/pool.js";
import * as chunksRepo from "../db/chunks.repo.js";
import * as documents from "../db/documents.repo.js";
import { config } from "../config.js";

export const documentsRouter = Router();

/** POST /documents  (multipart/form-data, field name: file) */
documentsRouter.post("/documents", uploadPdf, async (req, res) => {
  if (!req.file) throw httpError(400, 'No file uploaded. Send multipart/form-data with field "file".');

  // Hash the BYTES, not the filename. Two people uploading the same handbook
  // under different names is the same document, and embedding it twice costs
  // real API quota and pollutes retrieval with duplicate chunks.
  const contentHash = crypto.createHash("sha256").update(req.file.buffer).digest("hex");

  const existing = await documents.findByHash(contentHash);
  if (existing) {
    return res.status(200).json({
      alreadyIngested: true,
      document: existing,
      message: "This exact PDF was already uploaded; nothing was re-embedded.",
    });
  }

  // ---- 1. extract ---------------------------------------------------------
  const { pageCount, pages } = await extractPages(req.file.buffer);

  // ---- 2. chunk -----------------------------------------------------------
  const chunks = chunkPages(pages);
  if (chunks.length === 0) throw httpError(422, "PDF produced no chunks.");

  // ---- 3. embed -----------------------------------------------------------
  // Deliberately BEFORE the transaction. This is the slow, failure-prone part:
  // a 200-page PDF is dozens of API calls over several minutes, any of which can
  // 429. Holding a Postgres connection open through that would pin a pool slot
  // and risk a transaction timeout for no benefit - nothing is written yet, so
  // there is nothing to roll back if it fails.
  const startedAt = Date.now();
  const embeddings = await embedDocuments(
    chunks.map((c) => c.content),
    (done, total) => console.log(`  embedded ${done}/${total} chunks`),
  );
  const embedMs = Date.now() - startedAt;

  // ---- 4. store, all or nothing -------------------------------------------
  // The document row and every chunk row go in together. A partial ingest is the
  // worst state this system can be in: retrieval would answer from an incomplete
  // document and give you no signal that anything was missing.
  const document = await withTransaction(async (client) => {
    const doc = await documents.insertDocument(client, {
      filename: req.file.originalname,
      pageCount,
      chunkCount: chunks.length,
      contentHash,
    });
    await chunksRepo.insertChunks(client, doc.id, chunks, embeddings);
    return doc;
  });

  res.status(201).json({
    document,
    ingestion: {
      pageCount,
      chunkCount: chunks.length,
      totalTokens: chunks.reduce((n, c) => n + c.tokenCount, 0),
      embeddingModel: config.gemini.embeddingModel,
      dimensions: config.gemini.embeddingDimensions,
      embedMs,
      chunks: chunks.map((c) => ({
        chunkIndex: c.chunkIndex,
        pageNumber: c.pageNumber,
        tokenCount: c.tokenCount,
        preview: c.content.slice(0, 120),
      })),
    },
  });
});

/**
 * POST /documents/preview-chunks  (multipart/form-data, field name: file)
 *
 * A DRY RUN: extract, chunk, return - nothing is stored, no API is called.
 *
 * This endpoint exists because chunk size is the one RAG parameter you cannot
 * reason about in the abstract. You have to look at real chunks from a real
 * document and ask "could an LLM answer from this alone?". Being able to do that
 * for free, before spending embedding quota, is worth one extra route.
 *
 * Query params (all optional): ?preset=policy&maxTokens=300&minTokens=50&overlapTokens=40
 */
documentsRouter.post("/documents/preview-chunks", uploadPdf, async (req, res) => {
  if (!req.file) throw httpError(400, 'No file uploaded. Send multipart/form-data with field "file".');

  const base = req.query.preset === "policy" ? POLICY_PRESET : DEFAULTS;
  const options = {
    ...base,
    // Query params arrive as strings; only override what was actually supplied.
    ...(req.query.maxTokens ? { maxTokens: Number(req.query.maxTokens) } : {}),
    ...(req.query.minTokens ? { minTokens: Number(req.query.minTokens) } : {}),
    ...(req.query.overlapTokens ? { overlapTokens: Number(req.query.overlapTokens) } : {}),
  };

  const { pageCount, pages } = await extractPages(req.file.buffer);
  const chunks = chunkPages(pages, options);
  const tokens = chunks.map((c) => c.tokenCount);

  res.json({
    filename: req.file.originalname,
    options,
    stats: {
      pageCount,
      chunkCount: chunks.length,
      // The distribution is what you are really inspecting. A min far below
      // minTokens means runt merging could not help; a max at the ceiling means
      // blocks are being split rather than packed.
      tokens: {
        total: tokens.reduce((a, b) => a + b, 0),
        min: Math.min(...tokens),
        max: Math.max(...tokens),
        average: Math.round(tokens.reduce((a, b) => a + b, 0) / tokens.length),
      },
      chunksPerPage: pages.map((p) => ({
        pageNumber: p.pageNumber,
        chunks: chunks.filter((c) => c.pageNumber === p.pageNumber).length,
      })),
    },
    chunks,
  });
});

/** GET /documents - what is in the store. */
documentsRouter.get("/documents", async (req, res) => {
  res.json({ documents: await documents.listDocuments() });
});

/**
 * GET /documents/:id/chunks - what actually landed in Postgres.
 *
 * The point of this endpoint is verification: it reports each chunk's stored
 * vector dimensions and its L2 norm, read back out of pgvector. A norm of ~1
 * proves normalization happened; a wrong dimension count would have failed at
 * insert time, so seeing 768 here closes the loop end to end.
 */
documentsRouter.get("/documents/:id/chunks", async (req, res) => {
  const id = Number(req.params.id);
  const document = await documents.findById(id);
  if (!document) throw httpError(404, `No document with id ${id}`);

  const chunks = await chunksRepo.listChunksByDocument(id);
  res.json({ document, chunkCount: chunks.length, chunks });
});

/** DELETE /documents/:id - removes the document and (later) all its chunks. */
documentsRouter.delete("/documents/:id", async (req, res) => {
  const deleted = await documents.deleteDocument(Number(req.params.id));
  if (!deleted) throw httpError(404, `No document with id ${req.params.id}`);
  res.status(204).end();
});
