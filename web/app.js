// RAG-Stress Interactive Workbench Logic
// Pure Vanilla JavaScript (Zero External Dependencies)

function init() {
  const data = window.RAG_DATA;
  if (!data) return;

  let activeNoiseK = 10;
  let activePosition = "middle";
  let activeConflictKind = "dated";
  let activeTraceIndex = 0;

  // --- 1. RENDER BENCHMARK HEADLINE TABLE ---
  function renderBenchmarkTable() {
    const tbody = document.getElementById("bench-table-body");
    if (!tbody) return;

    const m4o = data.models["gpt-4o"];
    const mini = data.models["gpt-4o-mini"];

    tbody.innerHTML = `
      <tr>
        <td><strong>Answer Accuracy (pass_answer)</strong></td>
        <td class="data-mono" style="color: var(--accent-emerald); font-weight:700;">${m4o.pass_answer}% <small style="color:var(--text-muted); font-weight:normal;">[${m4o.pass_answer_ci.join(", ")}%]</small></td>
        <td class="data-mono" style="color: var(--accent-amber); font-weight:700;">${mini.pass_answer}% <small style="color:var(--text-muted); font-weight:normal;">[${mini.pass_answer_ci.join(", ")}%]</small></td>
        <td class="data-mono" style="color: var(--accent-rose); font-weight:700;">${data.comparison.mean_delta}%</td>
      </tr>
      <tr>
        <td><strong>Strict Accuracy (pass_strict)</strong></td>
        <td class="data-mono">${m4o.pass_strict}%</td>
        <td class="data-mono">${mini.pass_strict}%</td>
        <td class="data-mono" style="color: var(--accent-rose); font-weight:700;">${data.comparison.mean_delta}%</td>
      </tr>
      <tr>
        <td><strong>Exact Citation Rate</strong></td>
        <td class="data-mono">${m4o.exact_citation_rate}%</td>
        <td class="data-mono">${mini.exact_citation_rate}%</td>
        <td class="data-mono" style="color: var(--accent-rose);">-14.2%</td>
      </tr>
      <tr>
        <td><strong>Lost-in-Middle Penalty (k=10)</strong></td>
        <td class="data-mono" style="color: var(--accent-cyan);">${m4o.lim_score_k10} pts</td>
        <td class="data-mono" style="color: var(--accent-rose);">${mini.lim_score_k10} pts</td>
        <td class="data-mono">+13.3 pts (worse)</td>
      </tr>
      <tr>
        <td><strong>Dated Conflict Accuracy</strong></td>
        <td class="data-mono">${m4o.conflicts.dated.accuracy}%</td>
        <td class="data-mono">${mini.conflicts.dated.accuracy}%</td>
        <td class="data-mono" style="color: var(--accent-rose);">-20.0%</td>
      </tr>
      <tr>
        <td><strong>Authority Conflict Accuracy</strong></td>
        <td class="data-mono">${m4o.conflicts.authority.accuracy}%</td>
        <td class="data-mono">${mini.conflicts.authority.accuracy}%</td>
        <td class="data-mono" style="color: var(--accent-rose);">-15.0%</td>
      </tr>
      <tr>
        <td><strong>Unresolvable Conflict Resolution</strong></td>
        <td class="data-mono">${m4o.conflicts.unresolvable.accuracy}%</td>
        <td class="data-mono">${mini.conflicts.unresolvable.accuracy}%</td>
        <td class="data-mono" style="color: var(--text-muted);">0.0%</td>
      </tr>
      <tr>
        <td><strong>Abstain Recall (Unanswerable)</strong></td>
        <td class="data-mono">${m4o.abstain_recall}%</td>
        <td class="data-mono">${mini.abstain_recall}%</td>
        <td class="data-mono" style="color: var(--accent-rose);">-5.7%</td>
      </tr>
    `;
  }

  // --- 2. RENDER MCNEMAR MATRIX ---
  function renderMcNemarMatrix() {
    const container = document.getElementById("mcnemar-matrix-container");
    if (!container) return;

    const comp = data.comparison;
    container.innerHTML = `
      <div class="matrix-grid">
        <div class="matrix-header"></div>
        <div class="matrix-header">GPT-4o-mini PASS</div>
        <div class="matrix-header">GPT-4o-mini FAIL</div>

        <div class="matrix-header">GPT-4o PASS</div>
        <div class="matrix-cell"><strong>${comp.common_passes}</strong><br><small style="color:var(--text-muted)">Concordant Pass</small></div>
        <div class="matrix-cell cell-discordant-c"><strong>${comp.mcnemar_c} (c)</strong><br><small>4o wins</small></div>

        <div class="matrix-header">GPT-4o FAIL</div>
        <div class="matrix-cell cell-discordant-b"><strong>${comp.mcnemar_b} (b)</strong><br><small>Mini wins</small></div>
        <div class="matrix-cell"><strong>${comp.common_fails}</strong><br><small style="color:var(--text-muted)">Concordant Fail</small></div>
      </div>
      <div class="matrix-stats">
        <strong>McNemar Exact Test:</strong> Discordant ratio = ${comp.mcnemar_c} vs ${comp.mcnemar_b} (diff: +${comp.mcnemar_c - comp.mcnemar_b})<br>
        <strong>Chi-Square Statistic:</strong> ${((Math.abs(comp.mcnemar_b - comp.mcnemar_c) - 1) ** 2 / (comp.mcnemar_b + comp.mcnemar_c)).toFixed(2)}<br>
        <strong>p-value:</strong> &lt; 0.0001 (Statistically significant superiority of GPT-4o over GPT-4o-mini).
      </div>
    `;
  }

  // --- 3. ROBUST SVG DEGRADATION CHART ---
  function renderDegradationChart() {
    const wrapper = document.getElementById("chart-wrapper");
    if (!wrapper) return;

    const m4o = data.models["gpt-4o"].distractor_curve;
    const mini = data.models["gpt-4o-mini"].distractor_curve;

    // ViewBox dimensions: 600 x 240
    const w = 600;
    const h = 240;
    const p = { left: 45, right: 30, top: 25, bottom: 35 };

    const kToX = (k) => p.left + (k / 50) * (w - p.left - p.right);
    const accToY = (acc) => h - p.bottom - ((acc - 20) / 80) * (h - p.top - p.bottom);

    // Build Grid Lines & Labels
    let gridElements = "";
    [20, 40, 60, 80, 100].forEach((acc) => {
      const y = accToY(acc);
      gridElements += `
        <line x1="${p.left}" y1="${y}" x2="${w - p.right}" y2="${y}" stroke="#1e293b" stroke-width="1" />
        <text x="${p.left - 8}" y="${y + 4}" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="10" text-anchor="end">${acc}%</text>
      `;
    });

    [0, 5, 10, 25, 50].forEach((k) => {
      const x = kToX(k);
      gridElements += `
        <line x1="${x}" y1="${p.top}" x2="${x}" y2="${h - p.bottom}" stroke="#1e293b" stroke-dasharray="2,2" stroke-width="1" />
        <text x="${x}" y="${h - p.bottom + 18}" fill="#94a3b8" font-family="ui-monospace, monospace" font-size="11" text-anchor="middle">k=${k}</text>
      `;
    });

    // Path generators
    const makePath = (curve) =>
      curve.map((pt, i) => `${i === 0 ? "M" : "L"} ${kToX(pt.k).toFixed(1)} ${accToY(pt.accuracy).toFixed(1)}`).join(" ");

    const path4o = makePath(m4o);
    const pathMini = makePath(mini);

    // Active marker at activeNoiseK
    const activePt4o = m4o.find((pt) => pt.k === activeNoiseK) || m4o[2];
    const activePtMini = mini.find((pt) => pt.k === activeNoiseK) || mini[2];
    const activeX = kToX(activeNoiseK).toFixed(1);

    // Point elements
    const points4oHtml = m4o
      .map(
        (pt) =>
          `<circle cx="${kToX(pt.k).toFixed(1)}" cy="${accToY(pt.accuracy).toFixed(1)}" r="4.5" fill="#070a10" stroke="#38bdf8" stroke-width="2" />`
      )
      .join("");

    const pointsMiniHtml = mini
      .map(
        (pt) =>
          `<circle cx="${kToX(pt.k).toFixed(1)}" cy="${accToY(pt.accuracy).toFixed(1)}" r="4.5" fill="#070a10" stroke="#f43f5e" stroke-width="2" />`
      )
      .join("");

    // Active selection elements
    const activeSelectionHtml = `
      <line x1="${activeX}" y1="${p.top}" x2="${activeX}" y2="${h - p.bottom}" stroke="#f8fafc" stroke-width="1.5" stroke-dasharray="3,3" />
      <circle cx="${activeX}" cy="${accToY(activePt4o.accuracy).toFixed(1)}" r="7" fill="#38bdf8" stroke="#ffffff" stroke-width="2" />
      <circle cx="${activeX}" cy="${accToY(activePtMini.accuracy).toFixed(1)}" r="7" fill="#f43f5e" stroke="#ffffff" stroke-width="2" />
    `;

    // Complete SVG string with xmlns for 100% browser compatibility
    wrapper.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" class="degradation-chart" style="width:100%; height:100%; display:block;">
        ${gridElements}
        <!-- Line GPT-4o (Cyan) -->
        <path d="${path4o}" fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        ${points4oHtml}

        <!-- Line GPT-4o-mini (Rose) -->
        <path d="${pathMini}" fill="none" stroke="#f43f5e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
        ${pointsMiniHtml}

        <!-- Active selection -->
        ${activeSelectionHtml}
      </svg>
    `;

    // Update readout values in header
    const readout4o = document.getElementById("readout-4o");
    const readoutMini = document.getElementById("readout-mini");
    const readoutGap = document.getElementById("readout-gap");

    if (readout4o) readout4o.textContent = `${activePt4o.accuracy}%`;
    if (readoutMini) readoutMini.textContent = `${activePtMini.accuracy}%`;
    if (readoutGap) {
      const gap = (activePt4o.accuracy - activePtMini.accuracy).toFixed(1);
      readoutGap.textContent = `+${gap}%`;
    }

    // Update dynamic metric cards
    updateNoiseCards(activePt4o, activePtMini);
  }

  function updateNoiseCards(pt4o, ptMini) {
    const cardAcc4o = document.getElementById("card-acc-4o");
    const cardSub4o = document.getElementById("card-sub-4o");
    const cardAccMini = document.getElementById("card-acc-mini");
    const cardSubMini = document.getElementById("card-sub-mini");
    const cardDistractorMini = document.getElementById("card-distractor-mini");
    const cardMargin = document.getElementById("card-margin");
    const findingText = document.getElementById("noise-finding-text");

    if (cardAcc4o) cardAcc4o.textContent = `${pt4o.accuracy}%`;
    if (cardSub4o) cardSub4o.textContent = `95% CI: [${pt4o.ci.join(", ")}%]`;
    if (cardAccMini) cardAccMini.textContent = `${ptMini.accuracy}%`;
    if (cardSubMini) cardSubMini.textContent = `95% CI: [${ptMini.ci.join(", ")}%]`;
    if (cardDistractorMini) cardDistractorMini.textContent = `${ptMini.distractor_adopted}%`;

    const margin = (pt4o.accuracy - ptMini.accuracy).toFixed(1);
    if (cardMargin) cardMargin.textContent = `+${margin} pts`;

    if (findingText) {
      if (activeNoiseK === 0) {
        findingText.innerHTML = `<strong>At k=0:</strong> Both models achieve 100.0% accuracy with zero distractor noise.`;
      } else if (activeNoiseK === 5) {
        findingText.innerHTML = `<strong>At k=5:</strong> Both models maintain high accuracy (95.0% vs 96.7%), with minor distractor adoption (5.0% vs 3.3%).`;
      } else if (activeNoiseK === 10) {
        findingText.innerHTML = `<strong>At k=10:</strong> Frontier model retains 100.0% accuracy, while small model drops to 76.7% as distractor adoption reaches 23.3%.`;
      } else if (activeNoiseK === 25) {
        findingText.innerHTML = `<strong>At k=25:</strong> Significant divergence: GPT-4o holds at 90.0%, while GPT-4o-mini degrades to 60.0% (40.0% distractor adoption).`;
      } else if (activeNoiseK === 50) {
        findingText.innerHTML = `<strong>At k=50 (Extreme Stress):</strong> GPT-4o retains 86.7% comprehension, whereas GPT-4o-mini crashes to 38.3% as distractor adoption surges to 61.7%.`;
      }
    }
  }

  // --- 4. POSITION BIAS SIMULATOR ---
  function updatePositionSimulator() {
    const slotTop = document.getElementById("slot-top");
    const slotMiddle = document.getElementById("slot-middle");
    const slotBottom = document.getElementById("slot-bottom");
    const indicatorLabel = document.getElementById("pos-indicator-label");

    const posVal4o = document.getElementById("pos-val-4o");
    const posValMini = document.getElementById("pos-val-mini");
    const posDropMini = document.getElementById("pos-drop-mini");

    [slotTop, slotMiddle, slotBottom].forEach((s) => s && s.classList.remove("gold-slot"));

    if (activePosition === "top") {
      if (slotTop) {
        slotTop.classList.add("gold-slot");
        slotTop.textContent = "Gold Evidence";
      }
      if (slotMiddle) slotMiddle.textContent = "Passages 4-7";
      if (slotBottom) slotBottom.textContent = "Passages 8-10";
      if (indicatorLabel) indicatorLabel.textContent = "Gold Evidence at Top (Start of Context)";

      if (posVal4o) posVal4o.textContent = "98.3%";
      if (posValMini) posValMini.textContent = "93.3%";
      if (posDropMini) posDropMini.textContent = "+0.0 pts (Baseline)";
    } else if (activePosition === "middle") {
      if (slotTop) slotTop.textContent = "Passages 1-3";
      if (slotMiddle) {
        slotMiddle.classList.add("gold-slot");
        slotMiddle.textContent = "Gold Evidence";
      }
      if (slotBottom) slotBottom.textContent = "Passages 8-10";
      if (indicatorLabel) indicatorLabel.textContent = "Gold Evidence in Middle (Attention Valley)";

      if (posVal4o) posVal4o.textContent = "91.7%";
      if (posValMini) posValMini.textContent = "68.3%";
      if (posDropMini) posDropMini.textContent = "-25.0 pts (Severe)";
    } else if (activePosition === "bottom") {
      if (slotTop) slotTop.textContent = "Passages 1-3";
      if (slotMiddle) slotMiddle.textContent = "Passages 4-7";
      if (slotBottom) {
        slotBottom.classList.add("gold-slot");
        slotBottom.textContent = "Gold Evidence";
      }
      if (indicatorLabel) indicatorLabel.textContent = "Gold Evidence at Bottom (Near Query)";

      if (posVal4o) posVal4o.textContent = "100.0%";
      if (posValMini) posValMini.textContent = "85.0%";
      if (posDropMini) posDropMini.textContent = "-8.3 pts (Recency Advantage)";
    }
  }

  // --- 5. KNOWLEDGE CONFLICT SIMULATOR ---
  function updateConflictSimulator() {
    const ruleEl = document.getElementById("conflict-rule");
    const ruleDescEl = document.getElementById("conflict-rule-desc");
    const acc4oEl = document.getElementById("conflict-acc-4o");
    const sub4oEl = document.getElementById("conflict-sub-4o");
    const accMiniEl = document.getElementById("conflict-acc-mini");
    const subMiniEl = document.getElementById("conflict-sub-mini");

    if (activeConflictKind === "dated") {
      if (ruleEl) ruleEl.textContent = "Rule R3: Newer Date Wins";
      if (ruleDescEl) ruleDescEl.textContent = "Official passage with newer effective date supersedes older official passage.";
      if (acc4oEl) acc4oEl.textContent = "87.5%";
      if (sub4oEl) sub4oEl.textContent = "Stale Adoption: 12.5%";
      if (accMiniEl) accMiniEl.textContent = "67.5%";
      if (subMiniEl) subMiniEl.textContent = "Stale Adoption: 32.5%";
    } else if (activeConflictKind === "authority") {
      if (ruleEl) ruleEl.textContent = "Rule R4: Official Status Wins";
      if (ruleDescEl) ruleDescEl.textContent = "Official document always supersedes draft, even if the draft has a later date.";
      if (acc4oEl) acc4oEl.textContent = "87.5%";
      if (sub4oEl) sub4oEl.textContent = "Draft Adoption: 12.5%";
      if (accMiniEl) accMiniEl.textContent = "72.5%";
      if (subMiniEl) subMiniEl.textContent = "Draft Adoption: 27.5%";
    } else if (activeConflictKind === "unresolvable") {
      if (ruleEl) ruleEl.textContent = "Rule R5: Flag Unresolvable Conflict";
      if (ruleDescEl) ruleDescEl.textContent = "When two official documents have identical dates, model must flag conflict without guessing.";
      if (acc4oEl) acc4oEl.textContent = "55.0%";
      if (sub4oEl) sub4oEl.textContent = "Conflict Ignored: 0.0%";
      if (accMiniEl) accMiniEl.textContent = "55.0%";
      if (subMiniEl) subMiniEl.textContent = "Conflict Ignored: 0.0%";
    }
  }

  // --- 6. RENDER INTERVENTION TABLE ---
  function renderInterventions() {
    const tbody = document.getElementById("intervention-table-body");
    if (!tbody) return;

    tbody.innerHTML = data.interventions
      .map(
        (it) => `
      <tr>
        <td><strong>${it.name}</strong></td>
        <td>${it.description}</td>
        <td class="data-mono" style="color: ${it.delta_acc.startsWith("+") ? "var(--accent-emerald)" : it.delta_acc.startsWith("-") ? "var(--accent-rose)" : "var(--text-primary)"}">${it.delta_acc}</td>
        <td class="data-mono" style="color: ${it.delta_hallucination.startsWith("-") ? "var(--accent-emerald)" : "var(--text-primary)"}">${it.delta_hallucination}</td>
        <td class="data-mono" style="color: ${it.delta_false_abstain.startsWith("+") ? "var(--accent-amber)" : "var(--text-primary)"}">${it.delta_false_abstain}</td>
        <td class="data-mono">${it.p_value}</td>
      </tr>
    `
      )
      .join("");
  }

  // --- 7. TRACE INSPECTOR ---
  function renderTraceNav() {
    const nav = document.getElementById("trace-nav");
    if (!nav) return;

    nav.innerHTML = data.sample_traces
      .map(
        (trace, idx) => `
      <button class="trace-item-btn ${idx === activeTraceIndex ? "active" : ""}" data-idx="${idx}">
        <div class="trace-btn-cat">${trace.category}</div>
        <div class="trace-btn-title">${trace.item_id}</div>
      </button>
    `
      )
      .join("");

    nav.querySelectorAll(".trace-item-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        activeTraceIndex = parseInt(btn.getAttribute("data-idx"), 10);
        renderTraceNav();
        renderTraceDetail();
      });
    });
  }

  function renderTraceDetail() {
    const display = document.getElementById("trace-display");
    if (!display) return;

    const trace = data.sample_traces[activeTraceIndex];
    if (!trace) return;

    // Render passages
    let passagesHtml = "";
    if (trace.conflict_details) {
      if (trace.conflict_details.gold_passage && trace.conflict_details.twin_passage) {
        const gp = trace.conflict_details.gold_passage;
        const tp = trace.conflict_details.twin_passage;
        passagesHtml = `
          <div class="passage-card gold">
            <div class="passage-head">
              <span style="color: var(--accent-emerald);">ROLE: GOLD (Official Baseline)</span>
              <span>ID: ${gp.doc_id} | ${gp.status.toUpperCase()} | Date: ${gp.effective_date}</span>
            </div>
            <div>${gp.text}</div>
          </div>
          <div class="passage-card twin">
            <div class="passage-head">
              <span style="color: var(--accent-amber);">ROLE: CONFLICT TWIN (Newer Draft)</span>
              <span>ID: ${tp.doc_id} | ${tp.status.toUpperCase()} | Date: ${tp.effective_date}</span>
            </div>
            <div>${tp.text}</div>
          </div>
        `;
      } else if (trace.conflict_details.passage_a && trace.conflict_details.passage_b) {
        const pa = trace.conflict_details.passage_a;
        const pb = trace.conflict_details.passage_b;
        passagesHtml = `
          <div class="passage-card gold">
            <div class="passage-head">
              <span style="color: var(--accent-emerald);">PASSAGE A (Official)</span>
              <span>ID: ${pa.doc_id} | ${pa.status.toUpperCase()} | Date: ${pa.effective_date}</span>
            </div>
            <div>${pa.text}</div>
          </div>
          <div class="passage-card gold">
            <div class="passage-head">
              <span style="color: var(--accent-emerald);">PASSAGE B (Official - Identical Date)</span>
              <span>ID: ${pb.doc_id} | ${pb.status.toUpperCase()} | Date: ${pb.effective_date}</span>
            </div>
            <div>${pb.text}</div>
          </div>
        `;
      }
    } else if (trace.multihop_details) {
      const h1 = trace.multihop_details.hop1_bridge;
      const h2 = trace.multihop_details.hop2_target;
      passagesHtml = `
        <div class="passage-card bridge">
          <div class="passage-head">
            <span style="color: var(--accent-cyan);">HOP 1: RELATIONAL BRIDGE</span>
            <span>ID: ${h1.doc_id} | Date: ${h1.effective_date}</span>
          </div>
          <div>${h1.text}</div>
        </div>
        <div class="passage-card gold">
          <div class="passage-head">
            <span style="color: var(--accent-emerald);">HOP 2: TARGET POLICY PASSAGE</span>
            <span>ID: ${h2.doc_id} | Date: ${h2.effective_date}</span>
          </div>
          <div>${h2.text}</div>
        </div>
      `;
    } else if (trace.gold_passage) {
      const gp = trace.gold_passage;
      passagesHtml = `
        <div class="passage-card gold">
          <div class="passage-head">
            <span style="color: var(--accent-emerald);">ROLE: GOLD PASSAGE</span>
            <span>ID: ${gp.doc_id} | ${gp.status.toUpperCase()} | Date: ${gp.effective_date}</span>
          </div>
          <div>${gp.text}</div>
        </div>
      `;
    }

    if (trace.distractors_count) {
      passagesHtml += `
        <div class="passage-card distractor">
          <div class="passage-head">
            <span>+ ${trace.distractors_count} RANDOM-SHUFFLED DISTRACTORS</span>
            <span>OPAQUE IDS: doc-8f0a2x, doc-pl92nm, doc-b19xkd ...</span>
          </div>
          <div style="color: var(--text-muted); font-size: 12px;">Distractor passages include same-attribute other-entity, near-miss entity names, and irrelevant corporate procedures with collision filters enforced.</div>
        </div>
      `;
    }

    // Render model results
    let resultsHtml = "";
    Object.keys(trace.model_results).forEach((mKey) => {
      const res = trace.model_results[mKey];
      resultsHtml += `
        <div class="model-result-card ${res.pass_strict ? "pass" : "fail"}">
          <div class="model-name">${mKey.toUpperCase()}</div>
          <div class="result-row">
            <span>Answer:</span>
            <strong>${res.answer !== null ? res.answer : "Flagged Conflict"}</strong>
          </div>
          <div class="result-row">
            <span>Citations:</span>
            <span>${res.citations.join(", ")}</span>
          </div>
          <div class="result-row">
            <span>Pass Strict:</span>
            <span style="color: ${res.pass_strict ? "var(--accent-emerald)" : "var(--accent-rose)"}; font-weight:700;">${res.pass_strict ? "PASS" : "FAIL"}</span>
          </div>
          <div class="result-row">
            <span>Taxonomy:</span>
            <span style="color: ${res.failure_mode ? "var(--accent-amber)" : "var(--accent-emerald)"}; font-weight:600;">${res.failure_mode || "pass"}</span>
          </div>
          <div class="result-note">${res.note}</div>
        </div>
      `;
    });

    display.innerHTML = `
      <div class="trace-header">
        <div class="trace-q">Q: ${trace.question}</div>
        <div class="trace-meta-pills">
          <span class="meta-pill">Item ID: ${trace.item_id}</span>
          <span class="meta-pill">Category: ${trace.category}</span>
          <span class="meta-pill">Gold Value: ${trace.gold_value}</span>
          <span class="meta-pill">Value Type: ${trace.value_type}</span>
        </div>
      </div>

      <div>
        <h4 style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase; font-family: var(--font-mono); margin-bottom: 8px;">Synthesized Context Passages:</h4>
        <div class="passages-container">${passagesHtml}</div>
      </div>

      <div>
        <h4 style="font-size: 13px; color: var(--text-secondary); text-transform: uppercase; font-family: var(--font-mono); margin-bottom: 8px;">Model Verdicts & Grader Breakdown:</h4>
        <div class="model-comparison-box">${resultsHtml}</div>
      </div>
    `;
  }

  // --- 8. INITIALIZE EVENT LISTENERS ---
  function initListeners() {
    // Noise Slider Buttons
    document.querySelectorAll(".noise-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".noise-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeNoiseK = parseInt(btn.getAttribute("data-k"), 10);
        renderDegradationChart();
      });
    });

    // Position Buttons
    document.querySelectorAll(".pos-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pos-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activePosition = btn.getAttribute("data-pos");
        updatePositionSimulator();
      });
    });

    // Conflict Buttons
    document.querySelectorAll(".conflict-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".conflict-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeConflictKind = btn.getAttribute("data-kind");
        updateConflictSimulator();
      });
    });
  }

  // Initialize all components
  renderBenchmarkTable();
  renderMcNemarMatrix();
  renderDegradationChart();
  updatePositionSimulator();
  updateConflictSimulator();
  renderInterventions();
  renderTraceNav();
  renderTraceDetail();
  initListeners();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
