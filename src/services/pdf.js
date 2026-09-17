/**
 * PDF -> per-page plain text.
 *
 * We use pdfjs-dist (Mozilla's PDF.js) rather than pdf-parse for one reason:
 * it gives us a page-at-a-time API. pdf-parse hands back the whole document as
 * a single string, and at that moment page numbers are gone permanently - no
 * clever prompting later can recover them. Citations are an EXTRACTION-time
 * decision, not a feature you bolt on in the answer step.
 *
 * The other thing to understand: a PDF does not store paragraphs, or even lines.
 * It stores positioned glyph runs - "draw 'leave' at x=72,y=531". PDF.js gives us
 * those runs in `items[]` with a transform matrix each. Reconstructing lines and
 * paragraph breaks from the y-coordinates is OUR job, and it matters because
 * Phase 3's chunker splits on exactly those breaks. Join everything with spaces
 * and you hand the chunker one giant undifferentiated blob.
 */

// The `legacy` build is the one meant for Node (no DOM, no worker assumptions).
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import path from "node:path";
import { createRequire } from "node:module";
import { httpError } from "../middleware/errorHandler.js";

// PDF.js needs the 14 standard PDF fonts on disk to map glyphs in documents that
// rely on them. In a browser it fetches them over HTTP; in Node we must point at
// the copy inside the installed package or every page logs a warning and some
// text comes back with the wrong characters. Resolved from the package itself so
// it survives npm hoisting.
const STANDARD_FONTS = path.join(
  path.dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json")),
  "standard_fonts/",
);

/** Vertical gap larger than this multiple of the normal line gap = paragraph break. */
const PARAGRAPH_GAP_RATIO = 1.6;

/**
 * Turn one page's glyph runs into text with line and paragraph breaks preserved.
 */
function itemsToText(items) {
  // ---- 1. group runs into lines by their y position -----------------------
  // transform is [a,b,c,d,e,f]; e = x, f = y. y is measured from the page BOTTOM,
  // so a *decreasing* y means moving down the page.
  const lines = [];
  let current = null;

  for (const item of items) {
    if (typeof item.str !== "string") continue;
    const y = item.transform[5];

    // Same line if the baseline is within a point or so (sub/superscripts wobble).
    if (current && Math.abs(current.y - y) < 2) {
      current.text += item.str;
    } else {
      current = { y, text: item.str };
      lines.push(current);
    }
  }

  const kept = lines.filter((l) => l.text.trim() !== "");
  if (kept.length === 0) return "";

  // ---- 2. find the document's "normal" line spacing ----------------------
  // The MODE of the baseline gaps is the body line height, almost by definition:
  // on a page of prose, consecutive-lines-within-a-paragraph is the most common
  // relationship there is.
  //
  // The median looks tempting but breaks on short paragraphs - a page of 2-line
  // paragraphs has as many paragraph gaps as line gaps, so the median lands
  // between the two and detects nothing. The mode is unaffected by how many
  // paragraph breaks there are.
  const counts = new Map();
  for (let i = 1; i < kept.length; i++) {
    const gap = Math.round(kept[i - 1].y - kept[i].y);
    if (gap > 0) counts.set(gap, (counts.get(gap) ?? 0) + 1);
  }
  let lineGap = 0;
  let best = 0;
  for (const [gap, count] of counts) {
    // Tie-break towards the SMALLER gap: line spacing is never larger than
    // paragraph spacing, so the smaller candidate is the safer guess.
    if (count > best || (count === best && gap < lineGap)) {
      best = count;
      lineGap = gap;
    }
  }

  // ---- 3. join lines, upgrading unusually large gaps to paragraph breaks --
  let text = kept[0].text;
  for (let i = 1; i < kept.length; i++) {
    const gap = kept[i - 1].y - kept[i].y;
    const isParagraph = lineGap > 0 && gap > lineGap * PARAGRAPH_GAP_RATIO;
    text += (isParagraph ? "\n\n" : "\n") + kept[i].text;
  }
  return text;
}

/**
 * Cleanup that is safe to do BEFORE chunking.
 *
 * De-hyphenation is the important one: PDFs break words across lines as
 * "entitle-\nment". Left alone, the chunk contains two nonsense tokens and the
 * embedding of that passage is measurably worse.
 */
function cleanPageText(text) {
  return text
    .replace(/\r\n?/g, "\n")
    .replace(/(\w)-\n(\w)/g, "$1$2")      // re-join hyphenated line breaks
    .replace(/[ \t]+/g, " ")              // collapse runs of spaces
    .replace(/ *\n */g, "\n")             // trim each line
    .replace(/\n{3,}/g, "\n\n")           // at most one blank line
    .trim();
}

/**
 * @param {Buffer} buffer  raw PDF bytes
 * @returns {Promise<{ pageCount: number, pages: {pageNumber:number, text:string}[] }>}
 */
export async function extractPages(buffer) {
  // getDocument returns a *loading task*, not the document. The task owns the
  // parse and is the thing you destroy; the document proxy only has cleanup().
  let task;
  let doc;
  try {
    task = pdfjs.getDocument({
      // PDF.js wants a Uint8Array and TAKES OWNERSHIP of it - it detaches the
      // buffer while parsing. Copying first keeps our Buffer usable afterwards
      // (we still need it to compute the content hash).
      data: new Uint8Array(buffer),
      isEvalSupported: false,             // no need to run PDF-embedded JS
      useSystemFonts: false,              // we want text, not rendering fidelity
      standardFontDataUrl: STANDARD_FONTS,
    });
    doc = await task.promise;
  } catch (err) {
    // Encrypted, truncated, or not actually a PDF - the client's problem, not a bug.
    throw httpError(400, `Could not parse PDF: ${err.message}`);
  }

  try {
  const pages = [];
  for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber++) {
    const page = await doc.getPage(pageNumber);
    const { items } = await page.getTextContent();
    pages.push({ pageNumber, text: cleanPageText(itemsToText(items)) });
    page.cleanup();                       // release per-page memory as we go
  }
  // A scanned/image-only PDF extracts to nothing and raises NO error. Storing it
  // would give you a document that silently answers every question with "not in
  // the context". Fail loudly instead.
  const characters = pages.reduce((sum, p) => sum + p.text.length, 0);
  if (characters === 0) {
    throw httpError(
      422,
      "No extractable text found. This PDF is probably scanned images; OCR would be required.",
    );
  }

  return { pageCount: doc.numPages, pages };
  } finally {
    // Frees the parsed document and, on a worker-backed build, terminates it.
    await task.destroy();
  }
}
