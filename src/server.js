/** Entry point: build the app, open the port. */
import { createApp } from "./app.js";
import { config } from "./config.js";

const app = createApp();

app.listen(config.port, () => {
  console.log(`pdf-rag-api listening on http://localhost:${config.port}`);
  console.log(`  embeddings: ${config.gemini.embeddingModel} @ ${config.gemini.embeddingDimensions}d`);
});
