/**
 * Multipart handling for the one route that needs it.
 *
 * memoryStorage, not diskStorage: we extract text from the bytes and then throw
 * the PDF away. Writing it to disk would mean managing an uploads/ directory,
 * cleaning up temp files on every error path, and having no benefit - nothing in
 * the RAG pipeline ever reads the original file again.
 *
 * The tradeoff is that the whole file sits in RAM, which is why the size limit
 * below is not optional.
 */
import multer from "multer";
import { httpError } from "./errorHandler.js";

const MAX_BYTES = 25 * 1024 * 1024;   // 25MB

export const uploadPdf = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_BYTES, files: 1 },
  fileFilter: (req, file, cb) => {
    // Checked before the body is buffered, so a wrong file type costs nothing.
    // Note this is the CLIENT'S claimed mimetype - trivially spoofed. The real
    // validation is PDF.js failing to parse it, which we already handle as a 400.
    if (file.mimetype !== "application/pdf") {
      return cb(httpError(400, `Expected a PDF, got ${file.mimetype}`));
    }
    cb(null, true);
  },
}).single("file");                    // form field name the client must use
