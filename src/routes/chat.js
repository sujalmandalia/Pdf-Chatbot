/**
 * POST /chat - the full RAG loop, streamed.
 *
 *   question -> embed -> pgvector search -> threshold -> LLM -> answer + sources
 *
 * Everything here is composition; the work is in retrieval.js and generation.js.
 * That is the payoff of building the phases separately: when an answer is wrong
 * you already have /search to tell you which half broke.
 *
 * The response is Server-Sent Events. The event order is fixed and meaningful:
 *
 *   meta     retrieval finished - match count, pages, timings. Arrives BEFORE the
 *            model has produced anything, so a UI can show "found 3 passages"
 *            during the ~2-3s the LLM takes.
 *   token    one incremental piece of the answer. Many of these.
 *   sources  the citation panel, plus `grounded`. LAST, and it has to be:
 *            whether the answer is grounded can only be known once the answer is
 *            complete, and the first token is out the door long before that.
 *   done     terminator, carrying final timings.
 *
 * Pass {"stream": false} to get the whole thing as one JSON object instead. Kept
 * because Postman renders raw SSE frames but cannot show you a `sources` object
 * you can read - and reading sources against excerpts is the habit that catches
 * wrong citations.
 */
import { Router } from "express";
import { retrieve } from "../services/retrieval.js";
import { generateAnswer, streamAnswer } from "../services/generation.js";
import { httpError } from "../middleware/errorHandler.js";

export const chatRouter = Router();

/** The one answer we are allowed to give with no context. Kept identical to the
 *  wording in the system instruction so the API has a single "I don't know". */
const NO_ANSWER = "The document does not contain the answer to that.";

/** Serialise one SSE frame. The blank line is the frame terminator - omit it and
 *  the client buffers forever waiting for the rest of the event. */
function sendEvent(res, event, data) {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/** Sources are built identically on both paths, so they cannot drift apart. */
function buildSources(matches) {
  return matches.map((m, i) => ({
    passage: i + 1,
    filename: m.filename,
    page: m.page_number,
    similarity: Number(m.similarity.toFixed(4)),
    contextPages: m.contextPages,
    excerpt: m.content.slice(0, 200),
  }));
}

chatRouter.post("/chat", async (req, res) => {
  const { question, topK, minSimilarity, neighbours, documentId, stream = true } = req.body ?? {};

  if (typeof question !== "string" || question.trim() === "") {
    throw httpError(400, 'Body must be JSON with a non-empty "question" string.');
  }

  const retrieval = await retrieve(question.trim(), {
    ...(topK !== undefined ? { topK: Number(topK) } : {}),
    ...(minSimilarity !== undefined ? { minSimilarity: Number(minSimilarity) } : {}),
    ...(neighbours !== undefined ? { neighbours: Number(neighbours) } : {}),
    ...(documentId !== undefined ? { documentId: Number(documentId) } : {}),
  });

  // Nothing cleared the threshold: answer WITHOUT calling the LLM.
  //
  // Two reasons, and the second is the important one. It saves a request - but
  // more than that, an LLM given zero context and asked to refuse still
  // sometimes answers from pretraining. Not calling it makes the refusal a
  // guarantee of the system rather than a hope about the model.
  const noContext = retrieval.matches.length === 0;

  /* ------------------------------------------------------------ buffered --- */
  if (!stream) {
    if (noContext) {
      return res.json({
        question: retrieval.question,
        answer: NO_ANSWER,
        sources: [],
        grounded: false,
        retrieval: {
          matchCount: 0,
          rejected: retrieval.rejected,
          options: retrieval.options,
          timings: retrieval.timings,
        },
      });
    }

    const generation = await generateAnswer(retrieval.question, retrieval.matches);

    // Context clearing the threshold does NOT mean the question was answered. The
    // model can be handed five relevant-scoring passages and still correctly find
    // that none of them contain the answer. Reporting grounded:true there would be
    // a lie a frontend then builds a "Sources" panel on top of.
    const answered = !generation.answer.startsWith(NO_ANSWER);

    return res.json({
      question: retrieval.question,
      answer: generation.answer,
      // No sources on a refusal: citing passages the answer did not use invites
      // the reader to assume the document said something it did not.
      sources: answered ? buildSources(retrieval.matches) : [],
      grounded: answered,
      retrieval: {
        matchCount: retrieval.matches.length,
        // Kept even on a refusal: "retrieval found 4 passages but none answered
        // it" is a completely different diagnosis from "retrieval found nothing".
        retrievedPages: retrieval.matches.map((m) => m.page_number),
        options: retrieval.options,
        timings: { ...retrieval.timings, generateMs: generation.generateMs },
      },
      model: generation.model,
      promptCharacters: generation.promptCharacters,
    });
  }

  /* ------------------------------------------------------------- stream ---- */
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    // Tells nginx and friends not to buffer the response. Without it a proxy
    // happily collects the whole stream and delivers it at once, which looks
    // exactly like streaming being broken.
    "X-Accel-Buffering": "no",
  });
  res.flushHeaders();

  // If the client hangs up we must stop writing. Express will not tell us via an
  // error - writes to a dead socket simply vanish, and the Gemini stream keeps
  // being consumed (and billed) to the end.
  let aborted = false;
  req.on("close", () => {
    aborted = true;
  });

  sendEvent(res, "meta", {
    question: retrieval.question,
    matchCount: retrieval.matches.length,
    retrievedPages: retrieval.matches.map((m) => m.page_number),
    options: retrieval.options,
    timings: retrieval.timings,
    ...(noContext ? { rejected: retrieval.rejected } : {}),
  });

  if (noContext) {
    sendEvent(res, "token", { text: NO_ANSWER });
    sendEvent(res, "sources", { grounded: false, sources: [] });
    sendEvent(res, "done", { timings: retrieval.timings });
    return res.end();
  }

  try {
    const generation = await streamAnswer(retrieval.question, retrieval.matches, (text) => {
      if (!aborted) sendEvent(res, "token", { text });
    });

    const answered = !generation.answer.startsWith(NO_ANSWER);

    sendEvent(res, "sources", {
      grounded: answered,
      sources: answered ? buildSources(retrieval.matches) : [],
      model: generation.model,
      promptCharacters: generation.promptCharacters,
    });
    sendEvent(res, "done", {
      timings: { ...retrieval.timings, generateMs: generation.generateMs },
    });
    res.end();
  } catch (err) {
    // Headers went out with the first byte, so a 500 is no longer available - the
    // status line is already sent. The error has to travel as an event, and this
    // must NOT be rethrown: errorHandler would try to set a status on a response
    // that is already streaming and crash on ERR_HTTP_HEADERS_SENT.
    console.error("stream failed:", err);
    sendEvent(res, "error", { error: err.message });
    res.end();
  }
});
