/**
 * Semantic retrieval: question -> ranked chunks.
 *
 * This is the read half of RAG, and it is deliberately free of any LLM. If the
 * chunks coming out of here are wrong, no amount of prompt engineering in
 * Phase 6 will produce a right answer - it will produce a FLUENT wrong answer,
 * which is worse. Keeping retrieval behind its own endpoint is what lets you
 * tell the two failure modes apart.
 */
import { pool, toVector } from "../db/pool.js";
import { embedQuery } from "./embeddings.js";

export const DEFAULTS = {
  topK: 5,
  // CALIBRATED, not guessed. Measured against samples/employee-handbook-full.pdf:
  //
  //   answerable questions,   top-1 similarity:  0.62 - 0.72
  //   unanswerable question,  top-1 similarity:  0.51   ("recipe for chocolate cake")
  //
  // The usual tutorial default of 0.3 would have ADMITTED the cake question -
  // handing Phase 6 five plausible-looking chunks about laptop encryption and
  // inviting a confident, sourced, entirely wrong answer. 0.6 sits in the gap.
  //
  // Note how narrow that gap is (0.62 vs 0.51). It is document-specific and must
  // be re-measured for a new corpus - which is what the `rejected` array in the
  // response is for.
  minSimilarity: 0.6,
  // Chunks either side of a match to include as context. See expandNeighbours().
  neighbours: 1,
};

/**
 * The search itself. One operator and an ORDER BY is the whole of semantic search.
 *
 *   <=> is COSINE DISTANCE: 0 = identical, 1 = unrelated, 2 = opposite.
 *   So this is ORDER BY ... ASC. Using DESC returns your LEAST relevant chunks
 *   and is the single most common pgvector bug.
 *
 * No index is used - pgvector falls back to an exact sequential scan, which is
 * 100% recall and fine at this scale. Phase 7 adds an ANN index and shows how to
 * verify it is actually being used.
 */
async function searchChunks(queryVector, { topK, documentId }) {
  const vector = toVector(queryVector);

  const { rows } = await pool.query(
    `SELECT c.id, c.document_id, c.chunk_index, c.page_number, c.content, c.token_count,
            d.filename,
            -- The raw distance, and the similarity a human can read. We ORDER BY
            -- the distance expression rather than the alias so the planner sees
            -- the operator directly - that is what an ANN index binds to later.
            (c.embedding <=> $1) AS distance,
            1 - (c.embedding <=> $1) AS similarity
       FROM chunks c
       JOIN documents d ON d.id = c.document_id
      WHERE ($2::bigint IS NULL OR c.document_id = $2)
      ORDER BY c.embedding <=> $1
      LIMIT $3`,
    [vector, documentId ?? null, topK],
  );
  return rows;
}

/**
 * Fetch the chunks immediately either side of each match.
 *
 * The problem this solves: a paragraph that starts at the bottom of page 4 and
 * finishes on page 5 becomes two chunks, and overlap deliberately stops at page
 * boundaries (so page_number stays a fact). The page-5 chunk can therefore begin
 * mid-explanation with nothing before it.
 *
 * The fix is to retrieve differently rather than chunk differently: search small,
 * precise chunks for good discrimination, then widen the window for generation.
 * chunk_index is a document-wide ordinal, so the neighbours are a plain range
 * query - no re-embedding, and it fixes EVERY boundary, not just page seams.
 *
 * Citations still come from the matched chunk. Neighbours are context only.
 */
async function expandNeighbours(matches, neighbours) {
  if (neighbours <= 0 || matches.length === 0) return matches;

  // One query for all matches: build an OR of ranges rather than N round-trips.
  const conditions = [];
  const values = [];
  for (const match of matches) {
    const base = values.length;
    values.push(match.document_id, match.chunk_index - neighbours, match.chunk_index + neighbours);
    conditions.push(`(document_id = $${base + 1} AND chunk_index BETWEEN $${base + 2} AND $${base + 3})`);
  }

  const { rows } = await pool.query(
    `SELECT document_id, chunk_index, page_number, content
       FROM chunks
      WHERE ${conditions.join(" OR ")}
      ORDER BY document_id, chunk_index`,
    values,
  );

  const byKey = new Map(rows.map((r) => [`${r.document_id}:${r.chunk_index}`, r]));

  return matches.map((match) => {
    const window = [];
    for (let i = match.chunk_index - neighbours; i <= match.chunk_index + neighbours; i++) {
      const row = byKey.get(`${match.document_id}:${i}`);
      if (row) window.push(row);
    }
    return {
      ...match,
      // The text to hand the LLM: the match plus its neighbours, in document order.
      contextText: window.map((r) => r.content).join("\n\n"),
      // The same window, but keeping each piece's OWN page number.
      //
      // Needed because a neighbour can come from an adjacent page. Handing the
      // model one joined blob under a single page label makes it cite that page
      // for a fact that came from the neighbour - a citation that looks right and
      // is wrong. generation.js labels each piece with its own page instead.
      contextWindow: window.map((r) => ({
        chunkIndex: r.chunk_index,
        pageNumber: r.page_number,
        content: r.content,
        isMatch: r.chunk_index === match.chunk_index,
      })),
      contextChunkIndexes: window.map((r) => r.chunk_index),
      // Pages the context spans. The CITATION is still match.page_number - these
      // are only here so you can see what the wider window pulled in.
      contextPages: [...new Set(window.map((r) => r.page_number))].sort((a, b) => a - b),
    };
  });
}

/**
 * @param {string} question
 * @param {{topK?:number, minSimilarity?:number, neighbours?:number, documentId?:number}} [options]
 */
export async function retrieve(question, options = {}) {
  const { topK, minSimilarity, neighbours } = { ...DEFAULTS, ...options };

  // RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT. Same model, same 768 dims, different
  // projection - the model places a question near its answer only when told which
  // is which. Getting this wrong degrades recall with no error anywhere.
  const startedAt = Date.now();
  const queryVector = await embedQuery(question);
  const embedMs = Date.now() - startedAt;

  const searchedAt = Date.now();
  // Fetch topK BEFORE thresholding. Filtering in SQL with a WHERE on similarity
  // would prevent the ORDER BY from being index-friendly later, and knowing what
  // the rejected matches scored is exactly the information you need to calibrate
  // the threshold. So we filter in JS and report both sets.
  const all = await searchChunks(queryVector, { topK, documentId: options.documentId });
  const searchMs = Date.now() - searchedAt;

  const matches = all.filter((r) => r.similarity >= minSimilarity);
  const rejected = all.filter((r) => r.similarity < minSimilarity);

  return {
    question,
    options: { topK, minSimilarity, neighbours, documentId: options.documentId ?? null },
    timings: { embedMs, searchMs },
    matches: await expandNeighbours(matches, neighbours),
    // Below-threshold results, so you can SEE what the threshold rejected.
    // Without this, calibrating minSimilarity is guesswork.
    rejected: rejected.map((r) => ({
      chunk_index: r.chunk_index,
      page_number: r.page_number,
      similarity: r.similarity,
      preview: r.content.slice(0, 100),
    })),
  };
}
