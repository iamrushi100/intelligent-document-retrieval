const $ = (selector) => document.querySelector(selector);
const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function request(url, options) {
  const response = await fetch(url, options);
  const type = response.headers.get("content-type") || "";
  const body = type.includes("json") ? await response.json() : null;
  if (!response.ok) throw new Error(body?.detail || `Request failed (${response.status}).`);
  return body;
}

function showMessage(target, message, kind = "") {
  target.textContent = message;
  target.className = `message ${kind}`.trim();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function loadDocuments() {
  const list = $("#documents");
  list.replaceChildren(element("p", "muted", "Loading documents…"));
  try {
    const { documents } = await request("/documents");
    $("#document-count").textContent = documents.length;
    list.replaceChildren();
    if (!documents.length) {
      list.append(element("p", "muted", "No documents yet. Upload a PDF, DOCX, or TXT file to get started."));
      return;
    }
    for (const doc of documents) {
      const card = element("article", "doc-card");
      const top = element("div", "doc-top");
      top.append(element("span", "doc-name", doc.name), element("span", "doc-type", doc.file_type.toUpperCase()));
      card.append(top, element("p", "doc-meta", `${doc.status} · ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"} · ${formatBytes(doc.size_bytes)}`));
      const detail = element("details", "doc-detail");
      const summary = element("summary", "", "View document metadata and chunks");
      detail.append(summary);
      detail.addEventListener("toggle", async () => {
        if (!detail.open || detail.dataset.loaded) return;
        summary.textContent = "Loading document…";
        try {
          const full = await request(`/documents/${encodeURIComponent(doc.id)}`);
          detail.append(element("p", "doc-meta", `ID: ${full.id} · Created: ${new Date(full.created_at).toLocaleString()}`));
          for (const chunk of full.chunks) {
            const preview = element("div", "chunk", chunk.text);
            if (chunk.page_number) preview.prepend(element("strong", "", `Page ${chunk.page_number} · `));
            detail.append(preview);
          }
          detail.dataset.loaded = "true";
          summary.textContent = `Metadata and ${full.chunks.length} chunk${full.chunks.length === 1 ? "" : "s"}`;
        } catch (error) {
          summary.textContent = "Could not load details";
          detail.append(element("p", "message error", error.message));
        }
      });
      card.append(detail);
      list.append(card);
    }
  } catch (error) {
    $("#document-count").textContent = "—";
    list.replaceChildren(element("p", "message error", `Could not load documents: ${error.message}`));
  }
}

$("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#file-input");
  const button = $("#upload-button");
  const message = $("#upload-message");
  const file = input.files[0];
  if (!file) return;
  button.disabled = true;
  button.textContent = "Uploading…";
  showMessage(message, "Processing document and building search indexes…");
  try {
    const form = new FormData();
    form.append("file", file);
    const doc = await request("/documents", { method: "POST", body: form });
    showMessage(message, `Uploaded ${doc.name} (${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}).`, "success");
    input.value = "";
    await loadDocuments();
  } catch (error) {
    showMessage(message, error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Upload document";
  }
});

$("#search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("#query").value.trim();
  if (!query) return;
  const button = event.currentTarget.querySelector("button[type=submit]");
  const message = $("#search-message");
  const results = $("#results");
  const mode = $("#mode").value;
  button.disabled = true;
  button.textContent = "Searching…";
  showMessage(message, mode === "semantic" ? "First semantic search may take a moment while the model loads…" : "Searching…");
  results.replaceChildren();
  try {
    const params = new URLSearchParams({ q: query, top_k: $("#top-k").value });
    const suffix = mode === "semantic" ? "" : `/${mode}`;
    const data = await request(`/search${suffix}?${params}`);
    showMessage(message, `${data.results.length} result${data.results.length === 1 ? "" : "s"} · ${mode === "lexical" ? "BM25 / Lexical" : mode[0].toUpperCase() + mode.slice(1)} retrieval`, "success");
    if (!data.results.length) {
      results.append(element("div", "empty-state", "No matching passages found. Try different words or another retrieval method."));
    } else {
      for (const item of data.results) {
        const card = element("article", "result-card");
        const title = element("div", "result-title");
        title.append(element("strong", "", item.document_name));
        const score = mode === "semantic" ? item.similarity : mode === "lexical" ? item.bm25_score : item.hybrid_score;
        const label = mode === "semantic" ? "Cosine" : mode === "lexical" ? "BM25" : "RRF";
        title.append(element("span", "score", `${label} ${Number(score).toFixed(4)}`));
        card.append(title, element("p", "result-meta", `${item.chunk_id}${item.page_number ? ` · Page ${item.page_number}` : ""}${item.section ? ` · ${item.section}` : ""}`), element("p", "result-text", item.text));
        results.append(card);
      }
    }
  } catch (error) {
    showMessage(message, error.message, "error");
    results.append(element("div", "empty-state", "Search could not be completed. Check that the API is running and try again."));
  } finally {
    button.disabled = false;
    button.textContent = "Search";
  }
});

async function loadEvaluation() {
  const target = $("#evaluation");
  try {
    const report = await request("/evaluation/results/evaluation.json");
    const config = report.configuration;
    const counts = report.aggregate.semantic[String(config.evaluation_ks[0])];
    const summary = element("p", "eval-summary", `${report.per_question.length} questions (${counts.answerable_case_count} answerable, ${counts.unanswerable_case_count} unanswerable) · ${report.inputs.corpus_chunk_count} corpus chunks · Answerable-only macro averages · K = ${config.evaluation_ks.join(", ")}`);
    const tableWrap = element("div", "table-wrap");
    const table = element("table");
    const head = element("thead");
    const heading = element("tr");
    for (const text of ["Method", "K", "Recall", "Precision", "MRR"]) heading.append(element("th", "", text));
    head.append(heading);
    const body = element("tbody");
    for (const [method, byK] of Object.entries(report.aggregate)) {
      for (const k of config.evaluation_ks) {
        const metrics = byK[String(k)].answerable_only_macro;
        const row = element("tr");
        const name = method === "bm25" ? "BM25 / Lexical" : method[0].toUpperCase() + method.slice(1);
        for (const text of [name, `@${k}`, metrics.recall.toFixed(4), metrics.precision.toFixed(4), metrics.mrr.toFixed(4)]) row.append(element("td", "", text));
        body.append(row);
      }
    }
    table.append(head, body);
    tableWrap.append(table);
    target.replaceChildren(summary, tableWrap);
  } catch (error) {
    target.replaceChildren(element("p", "message error", `Could not load the saved evaluation report: ${error.message}`));
  }
}

loadDocuments();
loadEvaluation();