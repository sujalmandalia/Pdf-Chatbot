/**
 * Central error handling.
 *
 * Express 5 forwards a rejected promise from an async route handler to the error
 * middleware automatically (Express 4 did not - that is what all those
 * `asyncHandler` wrappers in older tutorials were for). So route handlers here
 * can just `throw` and this is the single place that turns an error into JSON.
 *
 * `status` on the error lets a service say "this is the client's fault" (400/404)
 * without importing anything HTTP-related. Everything else is a 500.
 */
export function errorHandler(err, req, res, next) {
  // Multer throws its own error class (file too large, too many files) with no
  // `status`. Untranslated it would surface as a 500, blaming the server for
  // the client sending a 40MB file.
  if (err.name === "MulterError") err.status = 400;

  const status = err.status ?? 500;

  if (status >= 500) console.error(err);           // real bugs get a stack trace
  else console.warn(`${status} ${req.method} ${req.originalUrl} - ${err.message}`);

  res.status(status).json({
    error: err.message || "Internal Server Error",
    ...(err.details ? { details: err.details } : {}),
  });
}

/** Small helper so services can signal an HTTP status without knowing about Express. */
export function httpError(status, message, details) {
  const err = new Error(message);
  err.status = status;
  if (details) err.details = details;
  return err;
}
