/**
 * Express app wiring, deliberately separate from server.js.
 *
 * The app knows how to handle a request; server.js knows how to open a port.
 * Keeping them apart means tests (or a script) can import the app without
 * anything starting to listen.
 *
 * Route order matters and mirrors the two halves of a RAG system:
 *
 *   INGESTION  (write path)  POST /documents   pdf -> text -> chunks -> embeddings -> DB
 *   QUERY      (read path)   POST /search      question -> embedding -> similarity -> chunks
 *                            POST /chat        the above + an LLM answering from those chunks
 *
 * Those two paths share exactly one thing: the embedding model. Everything else
 * is independent, which is why we can build and test them in separate phases.
 */
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { healthRouter } from "./routes/health.js";
import { documentsRouter } from "./routes/documents.js";
import { searchRouter } from "./routes/search.js";
import { chatRouter } from "./routes/chat.js";
import { errorHandler } from "./middleware/errorHandler.js";

export function createApp() {
  const app = express();

  // Parses application/json bodies. Multipart (the PDF upload) is handled by
  // multer on that one route in Phase 2 - not globally, because buffering every
  // request as multipart would be wasteful and surprising.
  app.use(express.json());

  // The frontend. Served from the API itself, which means no CORS configuration
  // and no second process - the browser and the API share an origin.
  const here = path.dirname(fileURLToPath(import.meta.url));
  app.use(express.static(path.join(here, "..", "public")));

  app.use(healthRouter);
  app.use(documentsRouter);
  app.use(searchRouter);
  app.use(chatRouter);

  // 404 for anything unmatched, so the client always gets JSON, never Express' HTML.
  app.use((req, res) => {
    res.status(404).json({ error: `No route for ${req.method} ${req.originalUrl}` });
  });

  // Error middleware must be registered LAST - Express identifies it by arity (4 args).
  app.use(errorHandler);

  return app;
}
