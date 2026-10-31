"use strict";

const $ = (selector) => document.querySelector(selector);
const PAGE_SIZE = 10;
const MODEL_NAMES = { bm25: "BM25", tfidf: "TF-IDF" };

const state = { q: "", model: "bm25", category: "", offset: 0, total: 0, busy: false };
let evalLoaded = false;

/* ---------- helpers ---------- */

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== "" && value !== null && value !== undefined) url.searchParams.set(key, value);
  }
  const response = await fetch(url);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : "Invalid request";
    } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return response.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(message, isError = false) {
  const status = $("#status");
  status.textContent = message;
  status.classList.toggle("error", isError);
}

/* ---------- search ---------- */

function renderSnippet(segments) {
  const p = el("p", "snippet");
  for (const segment of segments) {
    if (segment.match) p.appendChild(el("mark", "", segment.text));
    else p.appendChild(document.createTextNode(segment.text));
  }
  return p;
}

function renderResult(result) {
  const item = el("li", "result");

  const meta = el("div", "meta");
  meta.append(
    el("span", "rank", `#${result.rank}`),
    el("span", "badge", result.category),
    el("span", "score", `score ${result.score.toFixed(3)}`)
  );

  const heading = el("h2", "title");
  const link = el("button", "link", result.title);
  link.type = "button";
  link.addEventListener("click", () => openDocument(result.doc_id));
  heading.appendChild(link);

  item.append(meta, heading, renderSnippet(result.snippet));
  return item;
}

function syncUrl() {
  const params = new URLSearchParams();
  if (state.q) params.set("q", state.q);
  if (state.model !== "bm25") params.set("model", state.model);
  if (state.category) params.set("category", state.category);
  history.replaceState(null, "", params.toString() ? `?${params}` : window.location.pathname);
}

async function runSearch(append = false) {
  if (state.busy) return;
  if (!append) {
    state.offset = 0;
    $("#results").replaceChildren();
    $("#more").hidden = true;
  }
  if (!state.q) { setStatus(""); return; }

  state.busy = true;
  setStatus("Searching...");
  try {
    const data = await api("/api/search", {
      q: state.q, model: state.model, category: state.category, k: PAGE_SIZE, offset: state.offset,
    });
    const list = $("#results");
    data.results.forEach((result) => list.appendChild(renderResult(result)));

    state.total = data.total;
    state.offset += data.results.length;

    if (data.total === 0) {
      setStatus(`No results for "${data.query}". Try different or fewer words.`);
    } else {
      const shown = Math.min(state.offset, data.total);
      setStatus(`${data.total} matching articles, showing ${shown} (${MODEL_NAMES[data.model]}, ${data.took_ms} ms). Search terms after stemming: ${data.terms.join(", ") || "none"}.`);
    }
    $("#more").hidden = state.offset >= data.total;
    syncUrl();
  } catch (error) {
    setStatus(`Search failed: ${error.message}`, true);
  } finally {
    state.busy = false;
  }
}

/* ---------- reader dialog ---------- */

async function openDocument(docId) {
  try {
    const doc = await api(`/api/doc/${docId}`);
    $("#reader-title").textContent = doc.title;
    $("#reader-meta").textContent = `${doc.category} \u00b7 ${doc.name}`;
    const body = $("#reader-body");
    body.replaceChildren();
    doc.body.split(/\n\s*\n/).filter(Boolean).forEach((paragraph) => body.appendChild(el("p", "", paragraph.trim())));
    $("#reader").showModal();
  } catch (error) {
    setStatus(`Could not open the article: ${error.message}`, true);
  }
}

/* ---------- evaluation tab ---------- */

function renderEvaluation(report) {
  const container = $("#eval-content");
  container.replaceChildren();

  const models = Object.keys(report.models);
  const wrap = el("div", "table-wrap");
  const table = el("table");

  const headRow = el("tr");
  headRow.appendChild(el("th", "", "Metric"));
  models.forEach((m) => headRow.appendChild(el("th", "", MODEL_NAMES[m] || m)));
  table.appendChild(el("thead")).appendChild(headRow);

  const tbody = el("tbody");
  for (const metric of report.metrics) {
    const row = el("tr");
    row.appendChild(el("td", "", metric));
    const values = models.map((m) => report.models[m].mean[metric]);
    const best = Math.max(...values);
    values.forEach((value) => {
      const cell = el("td", value === best ? "num best" : "num", value.toFixed(3));
      const bar = el("span", "bar");
      bar.style.width = `${Math.round(value * 100)}%`;
      cell.appendChild(bar);
      row.appendChild(cell);
    });
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  wrap.appendChild(table);

  const note = el(
    "p", "muted small",
    `${report.n_queries} queries, ranking depth ${report.depth}. Relevance: ${report.relevance === "category" ? "an article is relevant if it belongs to the query's BBC category" : "explicit judgments"}. Best value in each row is highlighted.`
  );
  container.append(wrap, note);
}

async function loadEvaluation() {
  if (evalLoaded) return;
  try {
    renderEvaluation(await api("/api/evaluate"));
    evalLoaded = true;
  } catch (error) {
    $("#eval-content").textContent = `Evaluation unavailable: ${error.message}`;
  }
}

/* ---------- tabs ---------- */

function showTab(name) {
  const searchActive = name === "search";
  $("#tab-search").setAttribute("aria-selected", String(searchActive));
  $("#tab-eval").setAttribute("aria-selected", String(!searchActive));
  $("#tab-search").tabIndex = searchActive ? 0 : -1;
  $("#tab-eval").tabIndex = searchActive ? -1 : 0;
  $("#panel-search").hidden = !searchActive;
  $("#panel-eval").hidden = searchActive;
  if (!searchActive) loadEvaluation();
}

/* ---------- init ---------- */

async function init() {
  $("#tab-search").addEventListener("click", () => showTab("search"));
  $("#tab-eval").addEventListener("click", () => showTab("eval"));

  $("#search-form").addEventListener("submit", (event) => {
    event.preventDefault();
    state.q = $("#q").value.trim();
    runSearch();
  });
  document.querySelectorAll('input[name="model"]').forEach((radio) =>
    radio.addEventListener("change", () => { state.model = radio.value; if (state.q) runSearch(); })
  );
  $("#category").addEventListener("change", (event) => { state.category = event.target.value; if (state.q) runSearch(); });
  $("#more").addEventListener("click", () => runSearch(true));
  $("#reader-close").addEventListener("click", () => $("#reader").close());
  $("#reader").addEventListener("click", (event) => { if (event.target === $("#reader")) $("#reader").close(); });

  try {
    const stats = await api("/api/stats");
    $("#stats").textContent = `${stats.documents.toLocaleString()} articles, ${stats.vocabulary.toLocaleString()} indexed terms.`;
    const select = $("#category");
    Object.entries(stats.categories).forEach(([name, count]) => {
      const option = el("option", "", `${name} (${count})`);
      option.value = name;
      select.appendChild(option);
    });
  } catch (error) {
    setStatus(`Could not reach the server: ${error.message}`, true);
  }

  const params = new URLSearchParams(window.location.search);
  state.q = params.get("q") || "";
  state.model = params.get("model") === "tfidf" ? "tfidf" : "bm25";
  state.category = params.get("category") || "";
  $("#q").value = state.q;
  document.querySelector(`input[name="model"][value="${state.model}"]`).checked = true;
  $("#category").value = state.category;
  if (state.q) runSearch();
}

init();
