/**
 * POST /search - retrieval with no LLM.
 *
 * The most important endpoint in the project while you are learning. Every bad
 * RAG answer is either bad retrieval or bad generation, and this is the only
 * place you can observe the first one in isolation.
 */
import { Router } from "express";
import { retrieve, DEFAULTS } from "../services/retrieval.js";
import { httpError } from "../middleware/errorHandler.js";

export const searchRouter = Router();

searchRouter.post("/search", async (req, res) => {
  const { question, topK, minSimilarity, neighbours, documentId } = req.body ?? {};

  if (typeof question !== "string" || question.trim() === "") {
    throw httpError(400, 'Body must be JSON with a non-empty "question" string.');
  }

  const result = await retrieve(question.trim(), {
    // Only override a default when the caller actually supplied a value -
    // `?? undefined` would pass undefined through and `{...DEFAULTS, topK: undefined}`
    // would overwrite the default with undefined.
    ...(topK !== undefined ? { topK: Number(topK) } : {}),
    ...(minSimilarity !== undefined ? { minSimilarity: Number(minSimilarity) } : {}),
    ...(neighbours !== undefined ? { neighbours: Number(neighbours) } : {}),
    ...(documentId !== undefined ? { documentId: Number(documentId) } : {}),
  });

  res.json({
    ...result,
    matchCount: result.matches.length,
    // The shape Phase 6 will actually consume: what to show the user as a source.
    sources: result.matches.map((m) => ({
      filename: m.filename,
      page: m.page_number,
      similarity: Number(m.similarity.toFixed(4)),
    })),
  });
});
