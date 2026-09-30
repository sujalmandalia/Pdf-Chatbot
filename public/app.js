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
  docCount: $("docCount"),
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
    turn,
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

/**
 * Replace the answer area with an error card. A partial answer, if any tokens
 * arrived before the failure, is kept above the card rather than thrown away.
 */
function renderError(view, question, { error, retryable }) {
  view.answer.querySelector(".dots")?.remove();
  if (!view.answer.textContent.trim()) view.answer.remove();

  const card = document.createElement("div");
  card.className = "error-card";
  card.setAttribute("role", "alert");
  card.innerHTML = `
    <svg class="error-icon" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path d="M12 8v5m0 3.5v.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
            fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div class="error-body">
      <div class="error-title">Couldn't get an answer</div>
      <div class="error-text"></div>
    </div>`;
  // textContent: the message comes from the server.
  card.querySelector(".error-text").textContent = error || "Something went wrong.";

  if (retryable !== false) {
    const retry = document.createElement("button");
    retry.className = "retry";
    retry.textContent = "Try again";
    retry.onclick = () => {
      view.turn.remove();
      submit(question);
    };
    card.append(retry);
  }

  view.sourcesSlot.replaceChildren(card);
  scrollDown();
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
      const data = await response.json().catch(() => ({ error: response.statusText }));
      // 400 is a bad question, not a transient failure - retrying won't help.
      renderError(view, question, { retryable: response.status !== 400, ...data });
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
      sources: (s) => {
        view.answer.classList.toggle("refused", s.grounded === false);
        renderSources(view.sourcesSlot, s);
        scrollDown();
      },

      // Mid-stream failure. The HTTP status was already 200, so this is the only
      // channel left for an error.
      error: (e) => renderError(view, question, e),
    });
  } catch (err) {
    // fetch itself failed: the server is down or the network dropped.
    renderError(view, question, { error: "Couldn't reach the server. Check that it's running and try again." });
  }
}

async function submit(question) {
  if (el.send.disabled) return;   // one question in flight at a time, retries included
  el.send.disabled = true;
  await ask(question);
  el.send.disabled = false;
  el.question.focus();
}

el.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (!question) return;
  el.question.value = "";
  submit(question);
});

/* ══════════════════════════════════════════════════════════ documents ═════ */

async function loadDocuments() {
  const { documents } = await fetch("/documents").then((r) => r.json());
  el.documents.innerHTML = "";
  el.docCount.textContent = documents.length || "";

  for (const doc of documents) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="doc-icon" aria-hidden="true">PDF</span>
      <div class="doc-body">
        <span class="name"></span>
        <span class="meta">${doc.page_count} pages · ${doc.chunk_count} chunks</span>
      </div>
      <button class="delete" aria-label="Delete document" title="Delete">
        <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
          <path d="M4 7h16M10 11v6m4-6v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"
                fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>`;
    li.querySelector(".name").textContent = doc.filename;
    li.title = doc.filename;
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

/** Upload result line. textContent, not innerHTML: error text comes from the server. */
function setHint(kind, text) {
  el.uploadHint.className = `hint ${kind}`;
  el.uploadHint.textContent = text;
}

async function upload(file) {
  el.dropzone.classList.add("busy");
  el.uploadLabel.textContent = file.name;
  el.uploadHint.className = "hint working";
  el.uploadHint.textContent = "Extracting, chunking, embedding…";

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/documents", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      setHint("error", data.error);
    } else if (data.alreadyIngested) {
      setHint("info", "Already uploaded — nothing re-embedded");
    } else {
      const i = data.ingestion;
      setHint("ok", `${i.pageCount} pages · ${i.chunkCount} chunks · embedded in ${(i.embedMs / 1000).toFixed(1)}s`);
    }
    loadDocuments();
    refreshStatus();
  } catch (err) {
    setHint("error", err.message);
  } finally {
    el.dropzone.classList.remove("busy");
    el.uploadLabel.innerHTML = "Drop a PDF or <u>browse</u>";
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
