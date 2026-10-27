/* AquaVision dashboard — vanilla JS, no external assets or telemetry. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    stream: $("stream"),
    videoCard: $("videoCard"),
    liveBadge: $("liveBadge"),
    liveText: $("liveText"),
    resolutionBadge: $("resolutionBadge"),
    deviceBadge: $("deviceBadge"),
    clock: $("clock"),

    connectionPill: $("connectionPill"),
    connectionText: $("connectionText"),
    startBtn: $("startBtn"),
    stopBtn: $("stopBtn"),
    retryBtn: $("retryBtn"),

    errorBanner: $("errorBanner"),
    errorTitle: $("errorTitle"),
    errorText: $("errorText"),
    errorHelpBtn: $("errorHelpBtn"),

    resultPanel: $("resultPanel"),
    resultLine1: $("resultLine1"),
    resultLine2: $("resultLine2"),
    modelChip: $("modelChip"),
    modelChipText: $("modelChipText"),
    confidenceValue: $("confidenceValue"),
    confidenceBar: $("confidenceBar"),
    stabilityText: $("stabilityText"),
    reasonText: $("reasonText"),
    cameraFps: $("cameraFps"),
    analysisFps: $("analysisFps"),
    inferenceMs: $("inferenceMs"),
    deviceName: $("deviceName"),
    statusText: $("statusText"),

    speciesList: $("speciesList"),
    runtimePill: $("runtimePill"),
    runtimeCamera: $("runtimeCamera"),
    runtimeResolution: $("runtimeResolution"),
    runtimeDevice: $("runtimeDevice"),
    runtimeModels: $("runtimeModels"),
    runtimeSpecies: $("runtimeSpecies"),
    runtimeUptime: $("runtimeUptime"),
    deviceWarning: $("deviceWarning"),

    diagBtn: $("diagBtn"),
    helpBtn: $("helpBtn"),
    diagModal: $("diagModal"),
    helpModal: $("helpModal"),
    speciesModal: $("speciesModal"),
    diagBody: $("diagBody"),
    copyDiagBtn: $("copyDiagBtn"),

    speciesBtn: $("speciesBtn"),
    speciesGrid: $("speciesGrid"),
    speciesCount: $("speciesCount"),
    speciesSearch: $("speciesSearch"),

    toast: $("toast"),
  };

  const reasons = {
    face_present: "Face detected · reported as Human",
    fish_confident: "Fish evidence confirmed",
    bioclip_confident: "Species confirmed by BioCLIP-2",
    imagenet_tiebreak: "Near-tie resolved by ImageNet verifier",
    species_uncertain: "Fish-like subject, species too uncertain",
    verifier_rejected: "Fish verifier rejected the scene",
    no_evidence: "No fish or face detected",
    low_fish_evidence: "Fish evidence too weak",
    transitioning: "Updating observation",
    initializing: "Initializing",
    starting: "Starting service",
    models_loading: "Local models are loading",
    models_unavailable: "Recognition models unavailable",
    camera_starting: "Opening camera",
    camera_stopped: "Camera stopped",
    camera_not_running: "Camera not running",
    camera_unavailable: "Camera unavailable",
    stale_observation: "Waiting for a fresh frame",
    fallback: "No reliable observation",
  };

  const state = {
    last: null,
    online: false,
    pollInFlight: false,
    cameraBusy: false,
    species: null,
    speciesLoaded: false,
    diagnostics: null,
    speciesSignature: "",
    toastTimer: null,
    streamTimer: null,
    lastSubject: null,
  };

  /* -------------------------------------------------------------- utilities */
  function setText(el, value) {
    if (!el) return;
    const text = value == null ? "—" : String(value);
    if (el.textContent !== text) el.textContent = text;
  }

  function pct(value) {
    return `${Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 100)}%`;
  }

  function oneDecimal(value) {
    const n = Number(value);
    return Number.isFinite(n) && n > 0 ? n.toFixed(1) : "—";
  }

  function duration(seconds) {
    const n = Math.max(0, Number(seconds) || 0);
    if (n < 60) return `${Math.round(n)}s`;
    if (n < 3600) {
      const m = Math.floor(n / 60);
      return `${m}m ${Math.floor(n % 60)}s`;
    }
    const h = Math.floor(n / 3600);
    return `${h}h ${String(Math.floor((n % 3600) / 60)).padStart(2, "0")}m`;
  }

  function shortDevice(name) {
    if (!name) return "";
    return String(name)
      .replace(/^NVIDIA\s+/i, "")
      .replace(/\s+Laptop GPU$/i, "")
      .replace(/^GeForce\s+/i, "")
      .replace(/^AMD\s+/i, "")
      .replace(/\s+Processor$/i, "");
  }

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function toast(message, type = "info") {
    if (!els.toast) return;
    els.toast.textContent = message;
    els.toast.dataset.state = type;
    els.toast.hidden = false;
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {
      els.toast.hidden = true;
    }, type === "error" ? 6000 : 3200);
  }

  async function fetchJSON(url, options = {}, timeoutMs = 6000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        cache: "no-store",
        ...options,
        signal: controller.signal,
        headers: { Accept: "application/json", ...(options.headers || {}) },
      });
      let body = null;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      if (!response.ok) {
        const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
        throw new Error(typeof detail === "string" ? detail : "Request failed");
      }
      return body;
    } finally {
      clearTimeout(timer);
    }
  }

  /* -------------------------------------------------------------- rendering */
  function setConnection(kind, text) {
    els.connectionPill.dataset.state = kind;
    setText(els.connectionText, text);
  }

  function renderResult(result) {
    if (!result || !result.display) return;
    // Both lines are updated in the same task, from one atomic result object.
    setText(els.resultLine1, result.display.line1);
    setText(els.resultLine2, result.display.line2);
    els.resultPanel.dataset.subject = result.subject || "not_fish";

    const liveObservation =
      state.last &&
      state.last.service &&
      state.last.service.ready &&
      state.last.camera &&
      state.last.camera.running &&
      !result.stale;

    const hasReading =
      Boolean(liveObservation) &&
      (result.subject !== "not_fish" || Number(result.confidence) > 0);

    setText(els.confidenceValue, hasReading ? pct(result.confidence) : "—");
    els.confidenceBar.style.width = hasReading ? pct(result.confidence) : "0%";
    setText(
      els.stabilityText,
      hasReading ? `Stability ${pct(result.stability)}` : "Stability —"
    );
    setText(els.reasonText, reasons[result.reason] || "Live observation");
  }

  function renderSpecies(result) {
    const phase = state.last?.service?.phase;
    const cameraRunning = Boolean(state.last?.camera?.running);
    const candidates = result && result.fish ? result.species_top3 || [] : [];
    const signature = `${result?.subject}|${phase}|${cameraRunning}|${candidates
      .map((c) => `${c.name}:${c.probability}`)
      .join(",")}`;
    if (signature === state.speciesSignature) return;
    state.speciesSignature = signature;

    if (!candidates.length) {
      let message = "No fish detected — no species ranking to show.";
      if (phase === "loading_models" || phase === "starting") {
        message = "Waiting for the recognition models to finish loading.";
      } else if (!cameraRunning) {
        message = "Camera is stopped. Start the camera for a live ranking.";
      } else if (result && result.subject === "human") {
        message = "A person is the primary subject. Species ranking is paused.";
      }
      els.speciesList.replaceChildren(make("p", "empty-state", message));
      return;
    }

    const fragment = document.createDocumentFragment();
    candidates.slice(0, 3).forEach((candidate, index) => {
      const row = make("div", "species-row");
      const name = make("div", "species-name");
      const rank = make("span", "rank", String(index + 1).padStart(2, "0"));
      name.append(rank, document.createTextNode(candidate.name));
      const value = make("div", "species-prob", pct(candidate.probability));
      const bar = make("div", "species-bar");
      const fill = make("span");
      fill.style.width = pct(candidate.probability);
      bar.append(fill);
      row.append(name, value, bar);
      fragment.append(row);
    });
    els.speciesList.replaceChildren(fragment);
  }

  function renderErrorBanner(data) {
    let title = "";
    let text = "";
    if (data.service && data.service.phase === "error") {
      title = "Recognition unavailable";
      text = data.service.error || "The local models could not be loaded.";
    } else if (data.camera && data.camera.status === "error") {
      title = "Camera unavailable";
      text = data.camera.message || "The camera could not be opened.";
    }

    if (!title) {
      els.errorBanner.hidden = true;
      return;
    }
    els.errorBanner.hidden = false;
    setText(els.errorTitle, title);
    setText(els.errorText, text);
  }

  function renderState(data) {
    state.last = data;
    state.online = true;

    const phase = data.service?.phase || "starting";
    const camera = data.camera || {};
    const stats = data.stats || {};
    const models = data.models || {};
    const result = data.result;
    const running = Boolean(camera.running);
    const ready = phase === "ready";

    let connection = { kind: "idle", text: "Camera stopped" };
    if (phase === "error") {
      connection = { kind: "error", text: "Setup required" };
    } else if (phase === "starting" || phase === "loading_models" || phase === "stopping") {
      connection = { kind: "warn", text: "Loading models" };
    } else if (camera.status === "error") {
      connection = { kind: "error", text: "Camera unavailable" };
    } else if (!running) {
      connection = { kind: "idle", text: "Camera stopped" };
    } else if (result && result.stale) {
      connection = { kind: "warn", text: "Waiting for frames" };
    } else {
      connection = { kind: "live", text: "Live" };
    }
    setConnection(connection.kind, connection.text);

    els.videoCard.dataset.live = running ? "true" : "false";
    els.liveBadge.dataset.live = running ? "true" : "false";
    setText(
      els.liveText,
      running ? "LIVE" : camera.status === "error" ? "CAMERA ERROR" : "OFFLINE"
    );

    if (running && camera.width && camera.height) {
      setText(
        els.resolutionBadge,
        `${camera.width}×${camera.height}${camera.fps ? ` · ${Math.round(camera.fps)} fps` : ""}`
      );
    } else if (phase === "error") {
      setText(els.resolutionBadge, "Recognition offline");
    } else {
      setText(els.resolutionBadge, "Camera not started");
    }

    setText(els.deviceBadge, camera.device || "/dev/video0");
    setText(els.cameraFps, running ? oneDecimal(stats.camera_fps) : "—");
    setText(els.analysisFps, ready && running ? oneDecimal(stats.analysis_fps) : "—");
    setText(
      els.inferenceMs,
      ready && Number(stats.inference_ms) > 0 ? `${Math.round(stats.inference_ms)} ms` : "—"
    );

    const device = String(models.device || "").toUpperCase();
    setText(els.deviceName, ready && device ? device : "—");
    if (els.deviceName) {
      els.deviceName.title = models.device_name || "";
    }

    if (phase === "error") {
      els.modelChip.dataset.state = "error";
      setText(els.modelChipText, "Setup required");
      els.modelChip.title = data.service?.error || "";
    } else if (ready) {
      els.modelChip.dataset.state = "ready";
      const label = shortDevice(models.device_name);
      setText(els.modelChipText, `${device || "LOCAL"}${label ? ` · ${label}` : ""}`);
      els.modelChip.title = models.device_name || "";
    } else {
      els.modelChip.dataset.state = "loading";
      setText(els.modelChipText, "Loading models");
      els.modelChip.title = "";
    }

    setText(els.statusText, data.status_text || connection.text);
    els.statusText.parentElement.dataset.state = connection.kind;

    renderResult(result);
    renderSpecies(result);
    renderErrorBanner(data);

    // Controls: stop is only offered while the camera is actually running.
    els.startBtn.disabled = running || state.cameraBusy;
    els.stopBtn.disabled = !running || state.cameraBusy;
    els.retryBtn.disabled = state.cameraBusy;

    // Runtime card.
    setText(els.runtimeCamera, camera.device || "—");
    setText(
      els.runtimeResolution,
      running && camera.width ? `${camera.width}×${camera.height}` : "—"
    );
    setText(els.runtimeDevice, ready ? models.device_name || device : "—");
    if (els.runtimeDevice) els.runtimeDevice.title = models.device_name || "";
    setText(
      els.runtimeModels,
      ready ? "4 models ready" : phase === "error" ? "Unavailable" : "Loading…"
    );
    setText(els.runtimeSpecies, models.species_count ? String(models.species_count) : "—");
    document.querySelectorAll(".species-total").forEach((node) => {
      if (models.species_count) setText(node, String(models.species_count));
    });
    setText(els.runtimeUptime, data.service?.uptime ? duration(data.service.uptime) : "—");

    const runtimeKind =
      phase === "error" ? "error" : ready ? (running ? "live" : "idle") : "warn";
    els.runtimePill.dataset.state = runtimeKind === "live" ? "live" : runtimeKind;
    setText(
      els.runtimePill,
      phase === "error"
        ? "Error"
        : ready
          ? running
            ? "Live"
            : "Ready"
          : "Loading"
    );

    if (models.warning) {
      els.deviceWarning.hidden = false;
      setText(els.deviceWarning, models.warning);
    } else {
      els.deviceWarning.hidden = true;
    }
  }

  function renderOffline() {
    state.online = false;
    setConnection("error", "Server offline");
    els.videoCard.dataset.live = "false";
    els.liveBadge.dataset.live = "false";
    setText(els.liveText, "OFFLINE");
    setText(els.resolutionBadge, "No connection");
    setText(els.resultLine1, "Fish: —");
    setText(els.resultLine2, 'Type of Fish: "—"');
    els.resultPanel.dataset.subject = "not_fish";
    setText(els.confidenceValue, "—");
    els.confidenceBar.style.width = "0%";
    setText(els.stabilityText, "Stability —");
    setText(els.reasonText, "Backend unreachable");
    setText(els.statusText, "Waiting for the local server to return");
    els.statusText.parentElement.dataset.state = "error";
    setText(els.cameraFps, "—");
    setText(els.analysisFps, "—");
    setText(els.inferenceMs, "—");
    setText(els.deviceName, "—");
    els.modelChip.dataset.state = "error";
    setText(els.modelChipText, "Server offline");
    setText(els.runtimePill, "Offline");
    els.runtimePill.dataset.state = "error";
    els.startBtn.disabled = true;
    els.stopBtn.disabled = true;
    els.retryBtn.disabled = true;
    state.speciesSignature = "";
    els.speciesList.replaceChildren(
      make("p", "empty-state", "The local recognition server is not responding.")
    );
  }

  /* -------------------------------------------------------------- polling */
  async function pollState() {
    if (state.pollInFlight) return;
    state.pollInFlight = true;
    try {
      const data = await fetchJSON("/api/state", {}, 5000);
      renderState(data);
    } catch {
      if (state.online || !state.last) renderOffline();
    } finally {
      state.pollInFlight = false;
    }
  }

  /* -------------------------------------------------------------- camera */
  async function cameraAction(action, button) {
    if (state.cameraBusy) return;
    state.cameraBusy = true;
    button.classList.add("is-busy");
    button.disabled = true;
    try {
      const data = await fetchJSON(
        `/api/camera/${action}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        },
        25000
      );
      if (data && data.state) renderState(data.state);
      toast(
        action === "start"
          ? "Camera start requested."
          : action === "stop"
            ? "Camera stopped and released."
            : "Camera retry requested."
      );
    } catch (error) {
      const message = error.name === "AbortError" ? "Camera request timed out." : error.message;
      toast(message || "Camera request failed.", "error");
      pollState();
    } finally {
      state.cameraBusy = false;
      button.classList.remove("is-busy");
      const running = Boolean(state.last?.camera?.running);
      els.startBtn.disabled = running;
      els.stopBtn.disabled = !running;
      els.retryBtn.disabled = false;
    }
  }

  /* -------------------------------------------------------------- modals */
  let lastFocused = null;

  function openModal(modal) {
    if (!modal) return;
    lastFocused = document.activeElement;
    modal.hidden = false;
    document.body.classList.add("modal-open");
    const focusable = modal.querySelector("button, input, [href]");
    if (focusable) setTimeout(() => focusable.focus(), 30);
  }

  function closeModal(modal) {
    if (!modal || modal.hidden) return;
    modal.hidden = true;
    if (![...document.querySelectorAll(".modal")].some((m) => !m.hidden)) {
      document.body.classList.remove("modal-open");
    }
    if (lastFocused && typeof lastFocused.focus === "function") lastFocused.focus();
  }

  function closeTopModal() {
    const open = [...document.querySelectorAll(".modal")].filter((m) => !m.hidden);
    if (open.length) closeModal(open[open.length - 1]);
  }

  document.addEventListener("click", (event) => {
    const closeId = event.target.closest("[data-close]")?.dataset.close;
    if (closeId) closeModal($(closeId));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeTopModal();
  });

  /* -------------------------------------------------------------- diagnostics */
  function diagItem(label, value, tone) {
    const wrap = make("div", `diag-item${tone ? ` ${tone}` : ""}`);
    wrap.append(make("dt", "", label), make("dd", "", value == null ? "—" : String(value)));
    return wrap;
  }

  function diagSection(title, items) {
    const section = make("section", "diag-section");
    section.append(make("h3", "", title));
    const grid = make("div", "diag-grid");
    items.forEach((item) => grid.append(diagItem(item[0], item[1], item[2])));
    section.append(grid);
    return section;
  }

  function renderDiagnostics(data) {
    state.diagnostics = data;
    const fragment = document.createDocumentFragment();
    const env = data.environment || {};
    const models = data.models || {};
    const camera = data.camera || {};
    const files = data.model_files || {};
    const service = data.service || {};
    const settings = data.settings || {};

    fragment.append(
      diagSection("Service", [
        ["Phase", service.phase],
        ["Status", data.status_text],
        ["Uptime", duration(service.uptime || 0)],
        ["Error", service.error || "None", service.error ? "bad" : "good"],
      ])
    );

    fragment.append(
      diagSection("Camera", [
        ["Status", camera.status],
        ["Device", camera.device],
        [
          "Resolution",
          camera.width ? `${camera.width}×${camera.height}` : "—",
        ],
        ["Capture FPS", camera.fps ? `${camera.fps} fps` : "—"],
        ["Detected nodes", (camera.devices || []).join(", ") || "None"],
        ["Mirror", camera.config?.mirror ? "On" : "Off"],
      ])
    );

    fragment.append(
      diagSection("Models", [
        ["Ready", models.ready ? "Yes" : "No", models.ready ? "good" : "bad"],
        ["Device", models.device_name || models.device],
        ["Load time", models.load_seconds != null ? `${models.load_seconds}s` : "—"],
        ["Species set", data.species_count],
        ["Warning", models.warning || "None"],
        ["Face detector", models.face_model],
        ["Fish detector", models.fish_detector],
        ["Verifier", models.fish_verifier],
        ["Species model", models.species_model],
      ])
    );

    fragment.append(
      diagSection(
        "Model files",
        Object.entries(files).map(([name, info]) => [
          name.replaceAll("_", " "),
          info.present ? `present · ${info.path}` : `MISSING · ${info.path}`,
          info.present ? "good" : "bad",
        ])
      )
    );

    fragment.append(
      diagSection("Environment", [
        ["Platform", env.platform],
        ["Python", env.python],
        ["OpenCV", env.opencv],
        ["PyTorch", env.torch],
        ["Torchvision", env.torchvision],
        ["Ultralytics", env.ultralytics],
        ["OpenCLIP", env.open_clip],
        ["HF offline", env.offline_hub === "1" ? "Yes" : env.offline_hub],
      ])
    );

    fragment.append(
      diagSection("Thresholds", [
        ["Face min", settings.thresholds?.face_min_confidence],
        ["Fish gate", settings.thresholds?.fish_gate_p],
        ["Fish top-1", settings.thresholds?.fish_gate_top1],
        ["Species min", settings.thresholds?.species_min_p],
        ["Species margin", settings.thresholds?.species_min_margin],
        ["Smoothing window", `${settings.smoothing?.window_seconds}s`],
      ])
    );

    if (camera.message) {
      const section = make("section", "diag-section");
      section.append(make("h3", "", "Camera message"));
      section.append(make("p", "diag-message", camera.message));
      fragment.append(section);
    }

    els.diagBody.replaceChildren(fragment);
  }

  async function openDiagnostics() {
    openModal(els.diagModal);
    els.diagBody.replaceChildren(make("p", "empty-state", "Loading diagnostics…"));
    try {
      const data = await fetchJSON("/api/diagnostics", {}, 8000);
      renderDiagnostics(data);
    } catch (error) {
      els.diagBody.replaceChildren(
        make("p", "empty-state", `Diagnostics unavailable: ${error.message}`)
      );
    }
  }

  async function copyDiagnostics() {
    if (!state.diagnostics) {
      toast("Diagnostics are not loaded yet.", "error");
      return;
    }
    const text = JSON.stringify(state.diagnostics, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      toast("Diagnostics copied to clipboard.");
    } catch {
      toast("Clipboard access was blocked by the browser.", "error");
    }
  }

  /* -------------------------------------------------------------- species */
  function renderSpeciesLibrary(filter = "") {
    if (!state.species) return;
    const needle = filter.trim().toLowerCase();
    const all = state.species.species || [];
    const matches = needle
      ? all.filter(
          (item) =>
            item.name.toLowerCase().includes(needle) ||
            item.scientific.toLowerCase().includes(needle)
        )
      : all;
    const fragment = document.createDocumentFragment();
    matches.forEach((item) => {
      const tile = make("div", "species-tile");
      tile.append(
        make("strong", "", item.name),
        make("span", "", item.scientific)
      );
      fragment.append(tile);
    });
    if (!matches.length) {
      fragment.append(make("p", "empty-state", "No species match that filter."));
    }
    els.speciesGrid.replaceChildren(fragment);
    setText(els.speciesCount, `${matches.length} of ${all.length} species`);
  }

  async function openSpeciesLibrary() {
    openModal(els.speciesModal);
    if (!state.speciesLoaded) {
      els.speciesCount.textContent = "Loading species list…";
      try {
        const data = await fetchJSON("/api/species", {}, 8000);
        state.species = data;
        state.speciesLoaded = true;
        renderSpeciesLibrary(els.speciesSearch.value);
      } catch (error) {
        els.speciesCount.textContent = "Unavailable";
        els.speciesGrid.replaceChildren(
          make("p", "empty-state", `Species list unavailable: ${error.message}`)
        );
      }
    } else {
      renderSpeciesLibrary(els.speciesSearch.value);
    }
  }

  /* -------------------------------------------------------------- stream */
  function connectStream() {
    clearTimeout(state.streamTimer);
    const separator = els.stream.src.includes("?") ? "&" : "?";
    els.stream.src = `/api/stream${separator}t=${Date.now()}`;
  }

  els.stream.addEventListener("error", () => {
    state.streamTimer = setTimeout(connectStream, 1800);
  });

  /* -------------------------------------------------------------- events */
  els.startBtn.addEventListener("click", () => cameraAction("start", els.startBtn));
  els.stopBtn.addEventListener("click", () => cameraAction("stop", els.stopBtn));
  els.retryBtn.addEventListener("click", () => cameraAction("retry", els.retryBtn));
  els.helpBtn.addEventListener("click", () => openModal(els.helpModal));
  els.errorHelpBtn.addEventListener("click", () => openModal(els.helpModal));
  els.diagBtn.addEventListener("click", openDiagnostics);
  els.copyDiagBtn.addEventListener("click", copyDiagnostics);
  els.speciesBtn.addEventListener("click", openSpeciesLibrary);
  els.speciesSearch.addEventListener("input", (event) =>
    renderSpeciesLibrary(event.target.value)
  );

  /* -------------------------------------------------------------- boot */
  function tickClock() {
    const now = new Date();
    setText(
      els.clock,
      now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    );
  }

  tickClock();
  setInterval(tickClock, 1000);
  connectStream();
  pollState();
  setInterval(pollState, 500);
})();
