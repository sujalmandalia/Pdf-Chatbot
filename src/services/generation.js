/**
 * Generation: retrieved chunks + question -> a grounded answer.
 *
 * The whole job of this file is to stop the model doing what it is best at.
 * Gemini knows plenty about annual leave in general; we want it to answer only
 * from THIS document, and to admit when the document is silent. Everything below
 * serves that one goal.
 */
import { ai, withQuotaRetry } from "./gemini.js";
import { config } from "../config.js";

const MODEL = config.gemini.chatModel;

/**
 * The grounding contract, sent as a system instruction rather than glued onto the
 * user turn. System instructions carry more weight and, more practically, they
 * cannot be overridden by a user typing "ignore the document and just tell me".
 *
 * Each rule below exists because of a specific failure:
 *   - "only the context"      -> stops answering from pretraining
 *   - "say you cannot answer" -> the alternative is a confident invention
 *   - "cite [Page N]"         -> makes the answer auditable; the user can check
 *   - "do not merge"          -> stops it blending two policies into one wrong rule
 *   - "quote figures exactly" -> numbers are what people act on; paraphrasing 24
 *                                days into "about three weeks" is a real harm
 *   - "you MAY apply a rule"  -> added after measuring. Without it, "can I claim
 *                                a 300 unit monitor?" was refused even with the
 *                                allowance rules in context: the model would not
 *                                compare 300 against the document's 100 limit.
 *                                Strict grounding, taken too literally, costs you
 *                                the reasoning that makes retrieval useful.
 */
const SYSTEM_INSTRUCTION = `You answer questions about a specific document using ONLY the context provided in the user message.

Rules:
1. Use only the numbered context passages. Never use outside knowledge, even if you are confident it is correct.
2. If the context does not contain the answer, say exactly: "The document does not contain the answer to that." Do not guess, and do not offer general advice instead.
3. You MAY apply a rule stated in the context to the specifics of the question - for example comparing a figure the questioner gives against a limit in the document, or combining two stated rules that both apply. What you may not do is introduce any fact that is not in the context. If applying a rule leaves part of the question unanswered, answer the part you can and say which part the document does not cover.
4. Cite the page for every fact you state, in the form [Page N]. Each passage is made up of pieces, and every piece is prefixed with its own page marker like "(Page 3)". Use the page marker of the piece the fact actually came from - not the other pieces in the same passage.
5. If passages conflict or cover different cases, present them separately with their own citations rather than merging them into one rule.
6. Quote numbers, dates, limits and durations exactly as written. Never round, convert, or approximate them.
7. Be concise. Answer the question asked and stop.`;

/**
 * Assemble the context block.
 *
 * Structure matters more than it looks, and getting it wrong produces the worst
 * class of bug in a RAG system: a citation that looks right and is wrong.
 *
 * First attempt labelled each passage with the matched chunk's page and noted
 * "context spans pages 1, 2" alongside. The model dutifully answered a page-1
 * fact with "[Page 1, Page 2]". The label was ambiguous, so the citation was too.
 *
 * The fix: every piece of text carries its OWN page marker, immediately before
 * the text it describes. Neighbour chunks pulled in from an adjacent page are
 * labelled with that page, so the model can attribute each fact correctly.
 */
function buildContext(matches) {
  return matches
    .map((match, i) => {
      const window = match.contextWindow ?? [
        { pageNumber: match.page_number, content: match.content },
      ];
      const body = window.map((piece) => `(Page ${piece.pageNumber}) ${piece.content}`).join("\n\n");
      return `[Passage ${i + 1}]\n${body}`;
    })
    .join("\n\n---\n\n");
}

/** The prompt payload, shared by the buffered and streaming paths. */
function buildRequest(question, matches) {
  const contents = `Context passages:\n\n${buildContext(matches)}\n\n---\n\nQuestion: ${question}`;
  return {
    model: MODEL,
    contents,
    config: {
      systemInstruction: SYSTEM_INSTRUCTION,
      // Near-zero temperature. This is an extraction task, not a writing task:
      // we want the same answer every time for the same context, and creativity
      // here is indistinguishable from making things up.
      temperature: 0,
    },
  };
}

/**
 * Stream the answer token by token.
 *
 * @param {string} question
 * @param {Array} matches
 * @param {(text:string)=>void} onToken  called with each incremental piece
 * @returns {Promise<{answer:string, model:string, generateMs:number, promptCharacters:number}>}
 *          the FULL answer, still needed after streaming - see the note in
 *          routes/chat.js about why grounding can only be decided at the end.
 */
export async function streamAnswer(question, matches, onToken) {
  const request = buildRequest(question, matches);
  const startedAt = Date.now();

  // The retry wraps only the CALL that opens the stream, not the loop that
  // consumes it. Retrying mid-stream would re-emit tokens the client already
  // has - a 429 arriving after the first token is unrecoverable and must
  // surface as an error event instead.
  const stream = await withQuotaRetry(() => ai.models.generateContentStream(request), "generate");

  let answer = "";
  for await (const chunk of stream) {
    const text = chunk.text ?? "";
    if (!text) continue;          // some chunks carry only metadata
    answer += text;
    onToken(text);
  }

  return {
    answer: answer.trim(),
    model: MODEL,
    generateMs: Date.now() - startedAt,
    promptCharacters: request.contents.length + SYSTEM_INSTRUCTION.length,
  };
}

/**
 * @param {string} question
 * @param {Array} matches  from services/retrieval.js
 */
export async function generateAnswer(question, matches) {
  const contents = `Context passages:\n\n${buildContext(matches)}\n\n---\n\nQuestion: ${question}`;

  const startedAt = Date.now();
  const response = await withQuotaRetry(
    () => ai.models.generateContent(buildRequest(question, matches)),
    "generate",
  );

  return {
    answer: response.text?.trim() ?? "",
    model: MODEL,
    generateMs: Date.now() - startedAt,
    // The exact prompt payload, so a wrong answer can be diagnosed by reading
    // what the model was actually shown rather than what you assume it was shown.
    promptCharacters: contents.length + SYSTEM_INSTRUCTION.length,
  };
}
