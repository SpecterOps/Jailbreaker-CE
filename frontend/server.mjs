import { createServer, request as httpRequest } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";

const publicDir = join(process.cwd(), "public");
const port = Number(process.env.PORT || 3000);
const apiBaseUrl = process.env.API_BASE_URL || "http://backend:8000";

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

createServer((req, res) => {
  if ((req.url || "").startsWith("/api/")) {
    proxyToApi(req, res);
    return;
  }

  if (req.url === "/analysis" || req.url === "/analysis/") {
    serveStatic("/analysis.html", res);
    return;
  }

  if (req.url === "/experiments" || req.url === "/experiments/") {
    serveStatic("/experiments.html", res);
    return;
  }

  if (req.url === "/history" || req.url === "/history/") {
    serveStatic("/history.html", res);
    return;
  }

  const url = req.url === "/" ? "/index.html" : req.url || "/index.html";
  serveStatic(url, res);
}).listen(port, "0.0.0.0", () => {
  console.log(`Frontend listening on http://0.0.0.0:${port}`);
});

function proxyToApi(req, res) {
  const upstream = new URL(req.url || "/", apiBaseUrl);
  const proxyReq = httpRequest(
    upstream,
    {
      method: req.method,
      headers: {
        ...req.headers,
        host: upstream.host,
      },
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
      proxyRes.pipe(res);
    },
  );

  proxyReq.on("error", () => {
    res.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: "API unavailable" }));
  });

  req.pipe(proxyReq);
}

function serveStatic(url, res) {
  const cleanPath = normalize(url).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(publicDir, cleanPath);

  if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Not found");
    return;
  }

  res.writeHead(200, {
    "Content-Type": contentTypes[extname(filePath)] || "application/octet-stream",
  });
  createReadStream(filePath).pipe(res);
}
