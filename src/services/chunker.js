/**
 * Page text -> chunks.
 *
 * This is the phase where a RAG system is quietly won or lost. A chunk is the
 * unit of BOTH retrieval and citation, so its boundaries decide two things:
 * whether the right passage can be found at all, and whether the answer can
 * point at a page.
 *
 * Strategy, in order:
 *   0. The PAGE is a hard boundary. A chunk never spans two pages, so
 *      page_number is always a fact rather than a guess.
 *   1. Split the page into blocks on "\n\n" - the paragraph breaks that
 *      services/pdf.js reconstructed from the glyph y-coordinates.
 *   2. Greedily pack whole blocks up to maxTokens.
 *   3. A single block bigger than maxTokens is split recursively, trying
 *      separators in descending order of meaning: "\n\n" -> "\n" -> ". " -> " ".
 *   4. Merge runt chunks (< minTokens) into a neighbour on the same page.
 *   5. Prefix each chunk with the tail of the previous one (overlap).
 */

export const DEFAULTS = {
  maxTokens: 500,       // big enough for a full clause with context, small enough
                        // that the embedding is about ONE topic
  minTokens: 100,       // below this a chunk lacks the context to answer anything
  overlapTokens: 80,    // ~1-2 sentences of recall insurance at each boundary
};

/**
 * Preset for clause-numbered documents: policies, contracts, standards.
 * A short numbered clause is a complete, CITABLE unit - merging it into a
 * neighbour destroys the ability to say "per Section 4.2", which for this kind
 * of document is most of the value of the answer.
 */
export const POLICY_PRESET = {
  maxTokens: 400,
  minTokens: 40,
  overlapTokens: 40,
};

/**
 * Cheap offline token estimate: English runs ~4 characters per token.
 *
 * Deliberately not a real tokenizer. The budget it feeds (500) is itself a
 * heuristic, so precision here buys nothing - and a real tokenizer would add a
 * dependency plus a per-chunk cost during ingestion. Swap it if you ever need
 * exactness; nothing else in this file changes.
 */
export function countTokens(text) {
  return Math.ceil(text.length / 4);
}

/** Separators to try, most meaningful first. */
const SEPARATORS = ["\n\n", "\n", ". ", " "];

/**
 * Step 3 - split one oversized block.
 *
 * Recursive because the first separator that exists in the text may still leave
 * pieces that are too big (one enormous line with no sentence punctuation, say),
 * and each piece then gets the same treatment with the next separator down.
 */
function splitBlock(text, maxTokens, separatorIndex = 0) {
  if (countTokens(text) <= maxTokens) return [text];

  // Ran out of separators: the text is one unbroken token soup (a URL, a table
  // dump). Hard-slice as a last resort - correctness beats elegance here, and an
  // oversized chunk would break the embedding API's input limit.
  if (separatorIndex >= SEPARATORS.length) {
    const maxChars = maxTokens * 4;
    const pieces = [];
    for (let i = 0; i < text.length; i += maxChars) pieces.push(text.slice(i, i + maxChars));
    return pieces;
  }

  const separator = SEPARATORS[separatorIndex];
  const parts = text.split(separator);
  if (parts.length === 1) return splitBlock(text, maxTokens, separatorIndex + 1);

  // Re-pack the parts: splitting on ". " could produce 40 sentences, and we want
  // as few chunks as possible, each as close to the budget as possible.
  const chunks = [];
  let current = "";
  for (const part of parts) {
    const candidate = current ? current + separator + part : part;
    if (countTokens(candidate) <= maxTokens) {
      current = candidate;
      continue;
    }
    if (current) chunks.push(current);
    // The part alone may STILL be too big -> next separator down.
    if (countTokens(part) > maxTokens) {
      chunks.push(...splitBlock(part, maxTokens, separatorIndex + 1));
      current = "";
    } else {
      current = part;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

/** Steps 1-3: one page's text -> packed chunk texts, no overlap yet. */
function packPage(text, maxTokens) {
  const blocks = text
    .split("\n\n")
    .map((b) => b.trim())
    .filter(Boolean);

  const chunks = [];
  let current = "";

  for (const block of blocks) {
    // An oversized block cannot be packed - flush what we have and split it.
    if (countTokens(block) > maxTokens) {
      if (current) {
        chunks.push(current);
        current = "";
      }
      chunks.push(...splitBlock(block, maxTokens));
      continue;
    }

    const candidate = current ? `${current}\n\n${block}` : block;
    if (countTokens(candidate) <= maxTokens) {
      current = candidate;
    } else {
      chunks.push(current);
      current = block;
    }
  }
  if (current) chunks.push(current);
  return chunks;
}

/**
 * Step 4 - fold runts into a neighbour.
 *
 * Prefer merging BACKWARD (into the previous chunk): a heading like
 * "Section 3 - Remote Work" is a runt, and it belongs with the text that follows
 * it, not the section that preceded it. So a leading runt merges forward and
 * everything else merges backward.
 */
function mergeRunts(chunks, { minTokens, maxTokens }) {
  if (chunks.length < 2) return chunks;
  const out = [];

  for (const chunk of chunks) {
    const previous = out[out.length - 1];
    const isRunt = countTokens(chunk) < minTokens;

    // Merge only if the result still respects the budget; otherwise a runt at the
    // end of a full page would push a chunk over the limit.
    if (previous && isRunt && countTokens(previous) + countTokens(chunk) <= maxTokens * 1.2) {
      out[out.length - 1] = `${previous}\n\n${chunk}`;
    } else {
      out.push(chunk);
    }
  }

  // A leading runt has no previous chunk, so handle it after the fact.
  if (out.length > 1 && countTokens(out[0]) < minTokens) {
    out[1] = `${out[0]}\n\n${out[1]}`;
    out.shift();
  }
  return out;
}

/** Take roughly the last `tokens` worth of text, cut at a word boundary. */
function tail(text, tokens) {
  const chars = tokens * 4;
  if (text.length <= chars) return text;
  const slice = text.slice(-chars);
  const space = slice.indexOf(" ");
  return space === -1 ? slice : slice.slice(space + 1);
}

/**
 * Chunk a document's extracted pages.
 *
 * @param {{pageNumber:number, text:string}[]} pages  from services/pdf.js
 * @param {Partial<typeof DEFAULTS>} [options]
 * @returns {{chunkIndex:number, pageNumber:number, content:string, tokenCount:number}[]}
 */
export function chunkPages(pages, options = {}) {
  const { maxTokens, minTokens, overlapTokens } = { ...DEFAULTS, ...options };
  const chunks = [];

  for (const page of pages) {
    if (!page.text.trim()) continue;               // blank page: nothing to embed

    const packed = mergeRunts(packPage(page.text, maxTokens), { minTokens, maxTokens });

    packed.forEach((content, i) => {
      // Step 5 - overlap, WITHIN the page only. Overlapping across a page break
      // would put page 3's words in a chunk labelled page 4 and reintroduce
      // exactly the citation ambiguity that step 0 exists to prevent.
      const withOverlap =
        i > 0 && overlapTokens > 0
          ? `${tail(packed[i - 1], overlapTokens)}\n\n${content}`
          : content;

      chunks.push({
        // Ordinal across the WHOLE document, not per page - it is the chunk's
        // address, used for neighbour lookups and for a stable UNIQUE constraint.
        chunkIndex: chunks.length,
        pageNumber: page.pageNumber,
        content: withOverlap,
        tokenCount: countTokens(withOverlap),
      });
    });
  }

  return chunks;
}
