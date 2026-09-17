/**
 * Frontend for the RAG API. No framework, no build step.
 *
 * The one genuinely interesting part is readStream(): consuming Server-Sent
 * Events from a POST request. EventSource - the browser's built-in SSE client -
 * only does GET and cannot send a body, so it is useless for /chat. The stream
 * has to be read off fetch()'s response body and framed by hand, which is why
 * that function is longer than everything else here.
 */

const $ = (id) => document.getElementById(id);

const el = {
  status: $("status"),
  file: $("file"),
  dropzone: $("dropzone"),
  uploadLabel: $("uploadLabel"),
  uploadHint: $("uploadHint"),
  documents: $("documents"),
  messages: $("messages"),
  empty: $("empty"),
  form: $("form"),
  question: $("question"),
  send: $("send"),
};

/* ══════════════════════════════════════════════════════════ SSE parsing ════ */

/**
 * Read an SSE stream from a POST response and dispatch each event.
 *
 * The wire format is text frames separated by a BLANK line:
 *
 *   event: token\n
 *   data: {"text":"hello"}\n
 *   \n
 *
 * Two things make this fiddly:
 *
 * 1. A network chunk is NOT a frame. One read can deliver half a frame, or three
 *    and a half frames. So we keep a buffer and only consume up to the last
 *    complete "\n\n", leaving the remainder for the next read. Skipping this is
 *    the classic SSE bug: it works locally and corrupts JSON in production,
 *    where packets fragment differently.
 *
 * 2. The stream is bytes, not text. TextDecoder with `stream: true` carries
 *    partial multi-byte characters across chunk boundaries - without it a UTF-8
 *    character split down the middle becomes a replacement character.
 */
async function readStream(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop();                     // possibly incomplete - keep it

    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      handlers[event]?.(JSON.parse(data));
    }
  }
}

/* ══════════════════════════════════════════════════════════ rendering ═════ */

/**
 * Turn [Page 3] into a highlighted span.
 *
 * Escaping first is not optional: the answer text is model output built from PDF
 * content, and injecting it as raw HTML would let a crafted PDF run script in
 * this page. Escape, then add only the markup we generate ourselves.
 */
function renderAnswer(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return (
    escaped
      // Just enough markdown for what the model actually emits: bullets and bold.
      // Not a markdown parser - a real one is a dependency and a whole class of
      // XSS bugs, for a model whose output we already know the shape of.
      .replace(/^\s*[*-]\s+/gm, "• ")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      // Citations, highlighted so the eye can find them. The trailing space is
      // swallowed because the chip has its own padding - without this you get
      // "1 week [Page 6] ." with a visible gap before the full stop.
      .replace(/\[Page\s+\d+(?:\s*,\s*(?:Page\s+)?\d+)*\]\s*/g, (m) => `<span class="cite">${m.trim()}</span>`)
  );
}

function scrollDown() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

/** Build the DOM for one turn and return handles the stream will write into. */
function addTurn(question) {
  el.empty?.remove();

  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="q"></div>
    <div class="answer"><span class="dots"></span></div>
    <div class="sourcesSlot"></div>`;
  turn.querySelector(".q").textContent = question;
  el.messages.append(turn);
  scrollDown();

  return {
    answer: turn.querySelector(".answer"),
    sourcesSlot: turn.querySelector(".sourcesSlot"),
  };
}

function renderSources(slot, { sources, model }) {
  // A refusal sends an empty array on purpose: showing citations the answer did
  // not use invites the reader to assume the document said something it did not.
  if (!sources?.length) return;

  const details = document.createElement("details");
  details.className = "sources";
  // Collapsed by default. The citation is already inline in the answer as
  // [Page N]; this is here for when the reader wants to check it against the
  // actual text, which is a deliberate action rather than something to stare at.
  const pages = [...new Set(sources.map((s) => s.page))].sort((a, b) => a - b);
  details.innerHTML = `<summary>Sources · page${pages.length > 1 ? "s" : ""} ${pages.join(", ")}</summary>`;

  for (const s of sources) {
    const div = document.createElement("div");
    div.className = "source";
    div.innerHTML = `
      <div class="head"><span>Page ${s.page}</span></div>
      <div class="excerpt"></div>`;
    // The excerpt is the point: it lets you check the citation against the real
    // text without leaving the page. textContent, never innerHTML - see above.
    div.querySelector(".excerpt").textContent = `${s.excerpt}…`;
    details.append(div);
  }
  slot.append(details);
}

/* ══════════════════════════════════════════════════════════ chat ══════════ */

async function ask(question) {
  const view = addTurn(question);
  let answer = "";

  // No retrieval options sent: the server's calibrated defaults (topK 5,
  // minSimilarity 0.6, neighbours 1) apply. Tune them with /search in Postman,
  // where you can see the rejected scores that justify a change.
  const body = { question };

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    // A 400 or 429 arrives as ordinary JSON, before any stream starts.
    if (!response.ok) {
      const { error } = await response.json().catch(() => ({ error: response.statusText }));
      view.answer.innerHTML = `<span class="error">${error}</span>`;
      return;
    }

    // The `meta` and `done` events still arrive - they are just not rendered.
    // A handler that is not listed is silently ignored by readStream().
    await readStream(response, {
      token: (t) => {
        answer += t.text;
        view.answer.innerHTML = renderAnswer(answer);
        scrollDown();
      },

      // Last, necessarily: `grounded` is only knowable once the answer is
      // complete, and the first token left long before that.
      // Last, necessarily: `grounded` is only knowable once the answer is
      // complete, and the first token left long before that.
      sources: (s) => {
        view.answer.classList.toggle("refused", s.grounded === false);
        renderSources(view.sourcesSlot, s);
        scrollDown();
      },

      // Mid-stream failure. The HTTP status was already 200, so this is the only
      // channel left for an error.
      error: (e) => {
        view.answer.insertAdjacentHTML("beforeend", `<span class="error"> — ${e.error}</span>`);
      },
    });
  } catch (err) {
    view.answer.innerHTML = `<span class="error">${err.message}</span>`;
  }
}

el.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (!question) return;

  el.question.value = "";
  el.send.disabled = true;
  await ask(question);
  el.send.disabled = false;
  el.question.focus();
});

/* ══════════════════════════════════════════════════════════ documents ═════ */

async function loadDocuments() {
  const { documents } = await fetch("/documents").then((r) => r.json());
  el.documents.innerHTML = "";

  for (const doc of documents) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="name"></span>
      <span class="meta">${doc.page_count}p · ${doc.chunk_count} chunks</span>
      <button title="Delete">×</button>`;
    li.querySelector(".name").textContent = doc.filename;
    li.querySelector("button").onclick = async () => {
      // Deleting means re-embedding to get it back, so this one gets a confirm.
      if (!confirm(`Delete ${doc.filename}? Re-uploading will cost embedding quota again.`)) return;
      await fetch(`/documents/${doc.id}`, { method: "DELETE" });
      loadDocuments();
      refreshStatus();
    };
    el.documents.append(li);
  }
}

async function upload(file) {
  el.dropzone.classList.add("busy");
  el.uploadLabel.textContent = file.name;
  el.uploadHint.textContent = "extracting, chunking, embedding…";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/documents", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      el.uploadHint.innerHTML = `<span class="error">${data.error}</span>`;
    } else if (data.alreadyIngested) {
      el.uploadHint.textContent = "already ingested — nothing re-embedded";
    } else {
      const i = data.ingestion;
      el.uploadHint.textContent = `${i.pageCount} pages → ${i.chunkCount} chunks → embedded in ${i.embedMs}ms`;
    }
    loadDocuments();
    refreshStatus();
  } catch (err) {
    el.uploadHint.innerHTML = `<span class="error">${err.message}</span>`;
  } finally {
    el.dropzone.classList.remove("busy");
    el.uploadLabel.textContent = "Choose a PDF";
    el.file.value = "";
  }
}

el.file.addEventListener("change", () => el.file.files[0] && upload(el.file.files[0]));

// Drag and drop. preventDefault on dragover is what stops the browser from
// navigating away to the dropped file - without it the drop never reaches us.
for (const type of ["dragenter", "dragover"]) {
  el.dropzone.addEventListener(type, (e) => {
    e.preventDefault();
    el.dropzone.classList.add("over");
  });
}
for (const type of ["dragleave", "drop"]) {
  el.dropzone.addEventListener(type, () => el.dropzone.classList.remove("over"));
}
el.dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) upload(file);
});

/* ══════════════════════════════════════════════════════════ status ════════ */

async function refreshStatus() {
  try {
    const h = await fetch("/health").then((r) => r.json());
    const d = h.database;
    el.status.textContent = `pgvector ${d.pgvector} · ${d.documents} docs · ${d.chunks} chunks · ${h.embedding.model} @ ${h.embedding.dimensions}d`;
    if (!h.embedding.matchesColumn) {
      el.status.innerHTML += ` <span class="error">dimension mismatch</span>`;
    }
  } catch {
    el.status.innerHTML = `<span class="error">API unreachable</span>`;
  }
}

/* ══════════════════════════════════════════════════════════ init ══════════ */

refreshStatus();
loadDocuments();
