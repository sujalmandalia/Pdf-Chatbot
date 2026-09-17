/**
 * One place that reads process.env, validated once at boot.
 *
 * Why bother: a RAG system has config that MUST agree with data already written
 * to disk. EMBEDDING_DIMENSIONS is the clearest case - the `vector(768)` column
 * is created from this number. Change it later and every stored embedding becomes
 * garbage. Centralising it means there is exactly one truth to change.
 *
 * We fail fast on a missing key instead of discovering it 30 seconds into an
 * upload, halfway through embedding a 40-page PDF.
 */

function required(name) {
  const value = process.env[name];
  if (!value) {
    console.error(`Missing required env var: ${name}. Copy .env.example to .env and fill it in.`);
    process.exit(1);
  }
  return value;
}

export const config = {
  port: Number(process.env.PORT ?? 3000),

  // Postgres. A single connection string keeps this portable (local / docker / cloud).
  databaseUrl: required("DATABASE_URL"),

  gemini: {
    apiKey: required("GEMINI_API_KEY"),

    // Embedding model + dimensions are a matched pair with the DB schema.
    embeddingModel: process.env.EMBEDDING_MODEL ?? "gemini-embedding-001",
    embeddingDimensions: Number(process.env.EMBEDDING_DIMENSIONS ?? 768),

    // Generation model, used only in Phase 6.
    chatModel: process.env.CHAT_MODEL ?? "gemini-3.6-flash",
  },
};
