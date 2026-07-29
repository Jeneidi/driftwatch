"use strict";

const EPSILON = 1e-6;
const PH_OPTIONS = Object.freeze({
  delta: 0.005,
  lambda: 50,
  alpha: 0.9999,
  minInstances: 30,
});
const QUALITY_LIMITS = Object.freeze({
  accuracyDrop: 0.08,
  aucDrop: 0.08,
  brierRise: 0.05,
});
const CHI_CRITICAL_005 = Object.freeze([
  null, 3.841, 5.991, 7.815, 9.488, 11.07, 12.592, 14.067, 15.507,
  16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
  27.587, 28.869, 30.144, 31.41,
]);

function shares(counts) {
  const total = counts.reduce((sum, value) => sum + value, 0);
  return counts.map((value) => value / total);
}

function psi(referenceCounts, batchCounts) {
  const reference = shares(referenceCounts);
  const batch = shares(batchCounts);
  return reference.reduce((sum, value, index) => {
    const p = Math.max(value, EPSILON);
    const q = Math.max(batch[index], EPSILON);
    return sum + (q - p) * Math.log(q / p);
  }, 0);
}

function jsDivergence(referenceCounts, batchCounts) {
  const reference = shares(referenceCounts);
  const batch = shares(batchCounts);
  return reference.reduce((sum, p, index) => {
    const q = batch[index];
    const midpoint = (p + q) / 2;
    const left = p === 0 ? 0 : p * Math.log2(p / midpoint);
    const right = q === 0 ? 0 : q * Math.log2(q / midpoint);
    return sum + 0.5 * (left + right);
  }, 0);
}

function ksStatistic(referenceValues, batchValues) {
  const reference = [...referenceValues].sort((a, b) => a - b);
  const batch = [...batchValues].sort((a, b) => a - b);
  let i = 0;
  let j = 0;
  let maximum = 0;
  while (i < reference.length || j < batch.length) {
    const nextReference = i < reference.length ? reference[i] : Infinity;
    const nextBatch = j < batch.length ? batch[j] : Infinity;
    const value = Math.min(nextReference, nextBatch);
    while (i < reference.length && reference[i] <= value) i += 1;
    while (j < batch.length && batch[j] <= value) j += 1;
    maximum = Math.max(
      maximum,
      Math.abs(i / reference.length - j / batch.length),
    );
  }
  return maximum;
}

function ksCritical(referenceSize, batchSize, alpha = 0.05) {
  const constant = Math.sqrt(-0.5 * Math.log(alpha / 2));
  return constant * Math.sqrt((referenceSize + batchSize) / (referenceSize * batchSize));
}

function auc(labels, scores) {
  const pairs = scores
    .map((score, index) => ({ score, label: labels[index] }))
    .sort((a, b) => a.score - b.score);
  let positiveRanks = 0;
  let positives = 0;
  for (let start = 0; start < pairs.length;) {
    let end = start + 1;
    while (end < pairs.length && pairs[end].score === pairs[start].score) end += 1;
    const averageRank = (start + 1 + end) / 2;
    for (let index = start; index < end; index += 1) {
      if (pairs[index].label === 1) {
        positives += 1;
        positiveRanks += averageRank;
      }
    }
    start = end;
  }
  const negatives = labels.length - positives;
  if (positives === 0 || negatives === 0) return NaN;
  return (positiveRanks - positives * (positives + 1) / 2) / (positives * negatives);
}

function performance(labels, probabilities) {
  let correct = 0;
  let brier = 0;
  let positives = 0;
  labels.forEach((label, index) => {
    const probability = probabilities[index];
    if (Number(probability >= 0.5) === label) correct += 1;
    brier += (probability - label) ** 2;
    positives += label;
  });
  return {
    accuracy: correct / labels.length,
    auc: auc(labels, probabilities),
    brier: brier / labels.length,
    positiveRate: positives / labels.length,
  };
}

function pageHinkley(values, options = PH_OPTIONS) {
  const { delta, lambda, alpha, minInstances } = options;
  let mean = 0;
  let cumulative = 0;
  let minimum = 0;
  for (let index = 0; index < values.length; index += 1) {
    const count = index + 1;
    mean += (values[index] - mean) / count;
    cumulative = alpha * cumulative + values[index] - mean - delta;
    minimum = Math.min(minimum, cumulative);
    if (count >= minInstances && cumulative - minimum > lambda) return index;
  }
  return -1;
}

function chiSquare(referenceCounts, batchCounts) {
  const referenceSize = referenceCounts.reduce((sum, value) => sum + value, 0);
  const batchSize = batchCounts.reduce((sum, value) => sum + value, 0);
  const total = referenceSize + batchSize;
  let statistic = 0;
  let categories = 0;
  referenceCounts.forEach((referenceValue, index) => {
    const columnTotal = referenceValue + batchCounts[index];
    if (columnTotal === 0) return;
    categories += 1;
    const expectedReference = referenceSize * columnTotal / total;
    const expectedBatch = batchSize * columnTotal / total;
    statistic += (referenceValue - expectedReference) ** 2 / expectedReference;
    statistic += (batchCounts[index] - expectedBatch) ** 2 / expectedBatch;
  });
  return { statistic, degreesOfFreedom: Math.max(0, categories - 1) };
}

function chiCritical(degreesOfFreedom) {
  if (degreesOfFreedom < CHI_CRITICAL_005.length) {
    return CHI_CRITICAL_005[degreesOfFreedom] ?? NaN;
  }
  // ponytail: Wilson-Hilferty is sufficient above the shipped table; use a
  // statistics library only if future data has many more category levels.
  const z = 1.644854;
  return degreesOfFreedom * (
    1 - 2 / (9 * degreesOfFreedom) + z * Math.sqrt(2 / (9 * degreesOfFreedom))
  ) ** 3;
}

function alignedDistribution(referenceFeature, batchFeature) {
  if (referenceFeature.type === "numeric") {
    return {
      labels: referenceFeature.edges.slice(0, -1).map((edge, index) => {
        const suffix = index === referenceFeature.edges.length - 2 ? " +" : "";
        return `${formatNumber(edge, 3)} to ${formatNumber(referenceFeature.edges[index + 1], 3)}${suffix}`;
      }),
      referenceCounts: referenceFeature.counts,
      batchCounts: batchFeature.counts,
    };
  }
  const labels = [...new Set([
    ...Object.keys(referenceFeature.counts),
    ...Object.keys(batchFeature.counts),
  ])].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return {
    labels,
    referenceCounts: labels.map((label) => referenceFeature.counts[label] || 0),
    batchCounts: labels.map((label) => batchFeature.counts[label] || 0),
  };
}

function featureStatistics(referenceFeature, batchFeature, alpha) {
  const aligned = alignedDistribution(referenceFeature, batchFeature);
  const result = {
    ...aligned,
    psi: psi(aligned.referenceCounts, aligned.batchCounts),
    js: jsDivergence(aligned.referenceCounts, aligned.batchCounts),
    ks: NaN,
    ksCritical: NaN,
    chiSquare: NaN,
    chiCritical: NaN,
    chiDf: NaN,
  };
  if (referenceFeature.type === "numeric") {
    result.ks = ksStatistic(referenceFeature.values, batchFeature.values);
    result.ksCritical = ksCritical(
      referenceFeature.values.length,
      batchFeature.values.length,
      alpha,
    );
  } else {
    const chi = chiSquare(aligned.referenceCounts, aligned.batchCounts);
    result.chiSquare = chi.statistic;
    result.chiDf = chi.degreesOfFreedom;
    result.chiCritical = chiCritical(chi.degreesOfFreedom);
  }
  return result;
}

function errorStream(labels, probabilities) {
  return labels.map((label, index) => Number(Number(probabilities[index] >= 0.5) !== label));
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "n/a";
  if (!formatNumber.cache.has(digits)) {
    formatNumber.cache.set(digits, new Intl.NumberFormat(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
      useGrouping: false,
    }));
  }
  return formatNumber.cache.get(digits).format(value);
}
formatNumber.cache = new Map();

function buildDashboard(reference, stream, thresholds) {
  const batches = stream.batches.map((batch) => {
    const featureMetrics = {};
    reference.meta.features.forEach(({ name }) => {
      featureMetrics[name] = featureStatistics(
        reference.features[name],
        batch.features[name],
        thresholds.ksAlpha,
      );
    });
    return {
      source: batch,
      performance: performance(batch.labels, batch.predictions),
      featureMetrics,
    };
  });

  const referenceErrors = errorStream(reference.labels, reference.predictions);
  const streamErrors = batches.flatMap(({ source }) => errorStream(source.labels, source.predictions));
  const changeIndex = pageHinkley([...referenceErrors, ...streamErrors]);
  const streamChangeIndex = changeIndex - referenceErrors.length;
  let phBatchIndex = null;
  if (streamChangeIndex >= 0) {
    let seen = 0;
    phBatchIndex = batches.findIndex(({ source }) => {
      seen += source.size;
      return streamChangeIndex < seen;
    });
  }

  const alerts = [];
  const addAlert = (batchIndex, metric, value, threshold, driftType, action) => {
    alerts.push({
      batchIndex,
      batch: batches[batchIndex].source.label,
      metric,
      value,
      threshold,
      driftType,
      action,
    });
  };

  batches.forEach((batch, batchIndex) => {
    reference.meta.features.forEach(({ name, label, type }) => {
      const metrics = batch.featureMetrics[name];
      if (metrics.psi >= thresholds.psi) {
        addAlert(
          batchIndex,
          `${label} / PSI`,
          metrics.psi,
          `>= ${formatNumber(thresholds.psi, 3)}`,
          "Data drift",
          "Inspect the feature pipeline and compare affected segments.",
        );
      }
      if (type === "numeric" && metrics.ks > metrics.ksCritical) {
        addAlert(
          batchIndex,
          `${label} / KS`,
          metrics.ks,
          `> ${formatNumber(metrics.ksCritical, 3)} at alpha ${thresholds.ksAlpha}`,
          "Data drift",
          "Review the ECDF shift and upstream measurement changes.",
        );
      }
      if (type === "categorical" && metrics.chiSquare > metrics.chiCritical) {
        addAlert(
          batchIndex,
          `${label} / chi-square`,
          metrics.chiSquare,
          `> ${formatNumber(metrics.chiCritical, 3)} at alpha 0.05`,
          "Data drift",
          "Review category mix, missing levels, and acquisition channels.",
        );
      }
    });

    const quality = batch.performance;
    if (quality.accuracy <= reference.performance.accuracy - QUALITY_LIMITS.accuracyDrop) {
      addAlert(
        batchIndex,
        "Accuracy decay",
        quality.accuracy,
        `<= ${formatNumber(reference.performance.accuracy - QUALITY_LIMITS.accuracyDrop, 3)}`,
        "Concept drift signal",
        "Validate labels and compare conditional error slices before retraining.",
      );
    }
    if (
      Number.isFinite(quality.auc)
      && quality.auc <= reference.performance.auc - QUALITY_LIMITS.aucDrop
    ) {
      addAlert(
        batchIndex,
        "AUC decay",
        quality.auc,
        `<= ${formatNumber(reference.performance.auc - QUALITY_LIMITS.aucDrop, 3)}`,
        "Concept drift signal",
        "Check ranking failures by segment and test a fresh temporal model.",
      );
    }
    if (quality.brier >= reference.performance.brier + QUALITY_LIMITS.brierRise) {
      addAlert(
        batchIndex,
        "Brier increase",
        quality.brier,
        `>= ${formatNumber(reference.performance.brier + QUALITY_LIMITS.brierRise, 3)}`,
        "Concept drift signal",
        "Inspect calibration and recent label relationships.",
      );
    }
    if (batchIndex === phBatchIndex) {
      addAlert(
        batchIndex,
        "Page-Hinkley error change",
        1,
        `lambda ${PH_OPTIONS.lambda}`,
        "Concept drift signal",
        "Quarantine the change window and compare pre/post error cohorts.",
      );
    }
  });

  alerts.sort((a, b) => (
    b.batchIndex - a.batchIndex || a.metric.localeCompare(b.metric)
  ));
  return { batches, alerts, phBatchIndex, streamChangeIndex };
}

const state = {
  reference: null,
  streams: {},
  streamName: "real",
  dashboard: null,
  selected: { batchIndex: 0, featureName: "" },
};

const byId = (id) => document.getElementById(id);

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function tableCell(row, text, tag = "td") {
  const cell = element(tag, text);
  row.appendChild(cell);
  return cell;
}

function currentThresholds() {
  const psiValue = Number(byId("psi-threshold").value);
  const alphaValue = Number(byId("ks-alpha").value);
  return {
    psi: Number.isFinite(psiValue) ? Math.max(0.01, psiValue) : 0.25,
    ksAlpha: Number.isFinite(alphaValue) ? Math.min(0.2, Math.max(0.001, alphaValue)) : 0.05,
  };
}

function psiBand(value) {
  if (value >= 0.25) return "alert";
  if (value >= 0.1) return "warn";
  return "stable";
}

function chooseLatestFeature() {
  const batchIndex = state.dashboard.batches.length - 1;
  const featureName = state.reference.meta.features
    .map(({ name }) => name)
    .sort((a, b) => (
      state.dashboard.batches[batchIndex].featureMetrics[b].psi
      - state.dashboard.batches[batchIndex].featureMetrics[a].psi
    ))[0];
  state.selected = { batchIndex, featureName };
}

function renderTimeline() {
  const table = byId("timeline-table");
  table.replaceChildren();
  const caption = element("caption", "Population Stability Index by feature and temporal batch", "sr-only");
  const head = element("thead");
  const headRow = element("tr");
  const corner = tableCell(headRow, "FEATURE / PSI", "th");
  corner.scope = "col";
  state.dashboard.batches.forEach(({ source }) => {
    const th = tableCell(headRow, source.label, "th");
    th.scope = "col";
  });
  head.appendChild(headRow);

  const body = element("tbody");
  state.reference.meta.features.forEach(({ name, label }) => {
    const row = element("tr");
    const heading = tableCell(row, label, "th");
    heading.scope = "row";
    state.dashboard.batches.forEach((batch, batchIndex) => {
      const metric = batch.featureMetrics[name].psi;
      const cell = element("td");
      const button = element("button", formatNumber(metric, 2), `heat-cell ${psiBand(metric)}`);
      button.type = "button";
      button.setAttribute(
        "aria-label",
        `${label}, ${batch.source.label}, PSI ${formatNumber(metric, 4)}`,
      );
      const selected = (
        state.selected.batchIndex === batchIndex
        && state.selected.featureName === name
      );
      button.setAttribute("aria-pressed", String(selected));
      button.addEventListener("click", () => {
        state.selected = { batchIndex, featureName: name };
        renderTimeline();
        renderDrillDown();
        syncUrl();
      });
      cell.appendChild(button);
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  table.append(caption, head, body);
}

function addChartFrame(svg, width, height, padding, maximum, yTicks = 4) {
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  for (let tick = 0; tick <= yTicks; tick += 1) {
    const ratio = tick / yTicks;
    const y = padding.top + innerHeight * (1 - ratio);
    svg.append(
      svgElement("line", {
        x1: padding.left,
        x2: width - padding.right,
        y1: y,
        y2: y,
        class: tick === 0 ? "axis" : "grid",
      }),
      svgElement("text", {
        x: padding.left - 8,
        y: y + 4,
        "text-anchor": "end",
      }, formatNumber(maximum * ratio, 2)),
    );
  }
  return { innerWidth, innerHeight };
}

function renderDistributionChart(featureMeta, batch, metrics) {
  const svg = byId("distribution-chart");
  svg.replaceChildren(
    svgElement("title", {}, `${featureMeta.label} distribution for ${batch.source.label}`),
    svgElement(
      "desc",
      {},
      "June reference and selected batch distributions. Exact shares follow in a table.",
    ),
  );
  const width = 760;
  const height = 280;
  const padding = { left: 54, right: 22, top: 42, bottom: 72 };
  const reference = shares(metrics.referenceCounts);
  const current = shares(metrics.batchCounts);
  const maximum = Math.max(...reference, ...current, 0.01) * 1.12;
  const { innerWidth, innerHeight } = addChartFrame(
    svg,
    width,
    height,
    padding,
    maximum,
    4,
  );
  const count = metrics.labels.length;
  const xAt = (index) => padding.left + innerWidth * (index + 0.5) / count;
  const yAt = (value) => padding.top + innerHeight * (1 - value / maximum);

  if (featureMeta.type === "numeric") {
    const referencePoints = reference.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" ");
    const batchPoints = current.map((value, index) => `${xAt(index)},${yAt(value)}`).join(" ");
    svg.append(
      svgElement("polyline", { points: referencePoints, class: "reference-line" }),
      svgElement("polyline", { points: batchPoints, class: "batch-line" }),
    );
  } else {
    const groupWidth = innerWidth / count;
    const barWidth = Math.max(3, groupWidth * 0.32);
    reference.forEach((value, index) => {
      svg.append(
        svgElement("rect", {
          x: xAt(index) - barWidth,
          y: yAt(value),
          width: barWidth,
          height: padding.top + innerHeight - yAt(value),
          class: "reference-bar",
        }),
        svgElement("rect", {
          x: xAt(index),
          y: yAt(current[index]),
          width: barWidth,
          height: padding.top + innerHeight - yAt(current[index]),
          class: "batch-bar",
        }),
      );
    });
  }

  const labelStep = count > 12 ? 2 : 1;
  metrics.labels.forEach((label, index) => {
    if (index % labelStep !== 0) return;
    const shortLabel = label.length > 18 ? `${label.slice(0, 16)}…` : label;
    svg.appendChild(svgElement("text", {
      x: xAt(index),
      y: height - padding.bottom + 18,
      "text-anchor": "end",
      transform: `rotate(-38 ${xAt(index)} ${height - padding.bottom + 18})`,
    }, shortLabel));
  });
  svg.append(
    svgElement("line", { x1: 58, x2: 76, y1: 19, y2: 19, class: "reference-line" }),
    svgElement("text", { x: 82, y: 23 }, "June reference"),
    svgElement("line", { x1: 190, x2: 208, y1: 19, y2: 19, class: "batch-line" }),
    svgElement("text", { x: 214, y: 23 }, batch.source.label),
  );
}

function renderDrillDown() {
  const { batchIndex, featureName } = state.selected;
  const batch = state.dashboard.batches[batchIndex];
  const featureMeta = state.reference.meta.features.find(({ name }) => name === featureName);
  const metrics = batch.featureMetrics[featureName];
  byId("drill-selection").textContent = `${featureMeta.label} / ${batch.source.label} / n=${batch.source.size}`;
  renderDistributionChart(featureMeta, batch, metrics);

  const metricRow = element("tr");
  [
    formatNumber(metrics.psi, 6),
    formatNumber(metrics.js, 6),
    formatNumber(metrics.ks, 6),
    formatNumber(metrics.ksCritical, 6),
    formatNumber(metrics.chiSquare, 6),
    Number.isFinite(metrics.chiCritical)
      ? `${formatNumber(metrics.chiCritical, 6)} (df ${metrics.chiDf})`
      : "n/a",
  ].forEach((value) => tableCell(metricRow, value));
  byId("metric-row").replaceChildren(metricRow);

  const rows = document.createDocumentFragment();
  const referenceValues = shares(metrics.referenceCounts);
  const batchValues = shares(metrics.batchCounts);
  metrics.labels.forEach((label, index) => {
    const row = element("tr");
    tableCell(row, label);
    tableCell(row, formatNumber(referenceValues[index], 6));
    tableCell(row, formatNumber(batchValues[index], 6));
    rows.appendChild(row);
  });
  byId("distribution-rows").replaceChildren(rows);
}

function performancePath(values, xAt, yAt) {
  return values.map((value, index) => `${index === 0 ? "M" : "L"} ${xAt(index)} ${yAt(value)}`).join(" ");
}

function renderPerformance() {
  const svg = byId("performance-chart");
  svg.replaceChildren(
    svgElement("title", {}, "Classifier performance across temporal batches"),
    svgElement(
      "desc",
      {},
      "Accuracy, AUC, and Brier score with the first Page-Hinkley error change marked.",
    ),
  );
  const width = 980;
  const height = 320;
  const padding = { left: 56, right: 24, top: 46, bottom: 70 };
  const { innerWidth, innerHeight } = addChartFrame(svg, width, height, padding, 1, 4);
  const batches = state.dashboard.batches;
  const xAt = (index) => (
    batches.length === 1
      ? padding.left + innerWidth / 2
      : padding.left + innerWidth * index / (batches.length - 1)
  );
  const yAt = (value) => padding.top + innerHeight * (1 - value);
  const series = [
    ["accuracy", "accuracy-line"],
    ["auc", "auc-line"],
    ["brier", "brier-line"],
  ];
  series.forEach(([key, className]) => {
    const values = batches.map(({ performance: values }) => values[key]);
    svg.appendChild(svgElement("path", {
      d: performancePath(values, xAt, yAt),
      class: className,
    }));
    const baseline = state.reference.performance[key];
    svg.appendChild(svgElement("line", {
      x1: padding.left,
      x2: width - padding.right,
      y1: yAt(baseline),
      y2: yAt(baseline),
      class: className,
      opacity: "0.25",
      "stroke-dasharray": "3 5",
    }));
  });

  batches.forEach(({ source }, index) => {
    if (index % 2 !== 0 && index !== batches.length - 1) return;
    svg.appendChild(svgElement("text", {
      x: xAt(index),
      y: height - padding.bottom + 18,
      "text-anchor": "end",
      transform: `rotate(-38 ${xAt(index)} ${height - padding.bottom + 18})`,
    }, source.label));
  });

  if (state.dashboard.phBatchIndex !== null) {
    const x = xAt(state.dashboard.phBatchIndex);
    svg.append(
      svgElement("line", {
        x1: x,
        x2: x,
        y1: padding.top,
        y2: height - padding.bottom,
        class: "ph-line",
      }),
      svgElement("text", {
        x: Math.min(x + 6, width - 132),
        y: padding.top + 14,
        class: "ph-text",
      }, "PH error change"),
    );
  }
  svg.appendChild(svgElement("text", { x: padding.left, y: 22 }, (
    `June baselines: accuracy ${formatNumber(state.reference.performance.accuracy)}  `
    + `AUC ${formatNumber(state.reference.performance.auc)}  `
    + `Brier ${formatNumber(state.reference.performance.brier)}`
  )));

  const rows = document.createDocumentFragment();
  batches.forEach(({ source, performance: values }) => {
    const row = element("tr");
    [
      source.label,
      String(source.size),
      formatNumber(values.accuracy, 4),
      formatNumber(values.auc, 4),
      formatNumber(values.brier, 4),
      formatNumber(values.positiveRate, 4),
    ].forEach((value) => tableCell(row, value));
    rows.appendChild(row);
  });
  byId("performance-rows").replaceChildren(rows);
}

function renderAlerts() {
  const rows = document.createDocumentFragment();
  state.dashboard.alerts.forEach((alert) => {
    const row = element("tr");
    tableCell(row, alert.batch);
    tableCell(row, alert.metric);
    tableCell(row, alert.metric.startsWith("Page-Hinkley") ? "fired" : formatNumber(alert.value, 4));
    tableCell(row, alert.threshold);
    const typeCell = tableCell(row, alert.driftType);
    typeCell.className = `drift-type ${alert.driftType.startsWith("Data") ? "data" : "concept"}`;
    tableCell(row, alert.action);
    rows.appendChild(row);
  });
  if (state.dashboard.alerts.length === 0) {
    const row = element("tr");
    const cell = tableCell(row, "No alerts at the current thresholds.");
    cell.colSpan = 6;
    rows.appendChild(row);
  }
  byId("alert-rows").replaceChildren(rows);
  byId("alert-count").textContent = String(state.dashboard.alerts.length);
}

function renderSummary(thresholds) {
  const latestIndex = state.dashboard.batches.length - 1;
  const latest = state.dashboard.batches[latestIndex];
  const maximumPsi = Math.max(
    ...Object.values(latest.featureMetrics).map((metrics) => metrics.psi),
  );
  const latestAlerts = state.dashboard.alerts.filter(({ batchIndex }) => batchIndex === latestIndex);
  const hasConcept = latestAlerts.some(({ driftType }) => driftType.startsWith("Concept"));
  const hasPsiAlert = maximumPsi >= thresholds.psi;
  const hasDataTest = latestAlerts.some(({ driftType }) => driftType === "Data drift");
  const status = hasConcept || hasPsiAlert ? "alert" : (hasDataTest || maximumPsi >= 0.1 ? "warn" : "stable");
  byId("status-pill").dataset.status = status;
  byId("status-text").textContent = status.toUpperCase();
  byId("latest-batch").textContent = latest.source.label;
  byId("latest-psi").textContent = formatNumber(maximumPsi, 4);
  byId("latest-accuracy").textContent = formatNumber(latest.performance.accuracy, 4);
  byId("latest-auc").textContent = formatNumber(latest.performance.auc, 4);
  byId("latest-brier").textContent = formatNumber(latest.performance.brier, 4);
  byId("latest-ph").textContent = state.dashboard.phBatchIndex === null
    ? "none"
    : state.dashboard.batches[state.dashboard.phBatchIndex].source.label;
}

function renderAll({ resetSelection = false } = {}) {
  const stream = state.streams[state.streamName];
  const thresholds = currentThresholds();
  state.dashboard = buildDashboard(state.reference, stream, thresholds);
  if (resetSelection || !state.selected.featureName) chooseLatestFeature();
  renderSummary(thresholds);
  renderTimeline();
  renderDrillDown();
  renderPerformance();
  renderAlerts();
}

function restoreUrlState() {
  const params = new URLSearchParams(window.location.search);
  if (["real", "injected"].includes(params.get("stream"))) {
    state.streamName = params.get("stream");
  }
  const psiValue = Number(params.get("psi"));
  const alphaValue = Number(params.get("alpha"));
  if (Number.isFinite(psiValue) && psiValue >= 0.05 && psiValue <= 1) {
    byId("psi-threshold").value = String(psiValue);
  }
  if (Number.isFinite(alphaValue) && alphaValue >= 0.001 && alphaValue <= 0.2) {
    byId("ks-alpha").value = String(alphaValue);
  }
  const feature = params.get("feature");
  const batchId = params.get("batch");
  const validFeature = state.reference.meta.features.some(({ name }) => name === feature);
  const batchIndex = state.streams[state.streamName].batches.findIndex(({ id }) => id === batchId);
  if (validFeature && batchIndex >= 0) state.selected = { batchIndex, featureName: feature };
}

function syncUrl() {
  const params = new URLSearchParams(window.location.search);
  const thresholds = currentThresholds();
  const selectedBatch = state.dashboard.batches[state.selected.batchIndex].source.id;
  params.set("stream", state.streamName);
  params.set("psi", String(thresholds.psi));
  params.set("alpha", String(thresholds.ksAlpha));
  params.set("batch", selectedBatch);
  params.set("feature", state.selected.featureName);
  history.replaceState(null, "", `${window.location.pathname}?${params}`);
}

function csvCell(value) {
  const text = String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

function exportAlerts() {
  const headers = ["Batch", "Metric", "Value", "Threshold", "Drift type", "Recommended action"];
  const lines = [
    headers.map(csvCell).join(","),
    ...state.dashboard.alerts.map((alert) => [
      alert.batch,
      alert.metric,
      alert.metric.startsWith("Page-Hinkley") ? "fired" : formatNumber(alert.value, 6),
      alert.threshold,
      alert.driftType,
      alert.action,
    ].map(csvCell).join(",")),
  ];
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" }));
  const link = element("a");
  link.href = url;
  link.download = `driftwatch-${state.streamName}-alerts.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

async function loadData() {
  const paths = ["reference", "real", "injected"];
  const responses = await Promise.all(paths.map((name) => fetch(`data/${name}.json`)));
  responses.forEach((response, index) => {
    if (!response.ok) throw new Error(`Could not load data/${paths[index]}.json`);
  });
  const [reference, real, injected] = await Promise.all(responses.map((response) => response.json()));
  state.reference = reference;
  state.streams = { real, injected };
}

function wireControls() {
  byId("stream-select").addEventListener("change", (event) => {
    state.streamName = event.target.value;
    const stream = state.streams[state.streamName];
    byId("stream-note").textContent = stream.description;
    byId("stream-note").dataset.injected = String(state.streamName === "injected");
    renderAll({ resetSelection: true });
    syncUrl();
  });
  ["psi-threshold", "ks-alpha"].forEach((id) => {
    byId(id).addEventListener("input", () => {
      renderAll();
      syncUrl();
    });
  });
  byId("export-alerts").addEventListener("click", exportAlerts);
}

async function init() {
  try {
    await loadData();
    const { meta } = state.reference;
    byId("model-name").textContent = meta.model;
    byId("fit-window").textContent = `${meta.fitWindow} / n=${meta.fitRows}`;
    byId("reference-window").textContent = `${meta.referenceWindow} / n=${meta.referenceRows}`;
    restoreUrlState();
    byId("stream-select").value = state.streamName;
    byId("stream-note").textContent = state.streams[state.streamName].description;
    byId("stream-note").dataset.injected = String(state.streamName === "injected");
    wireControls();
    renderAll({ resetSelection: !state.selected.featureName });
    syncUrl();
  } catch (error) {
    byId("status-pill").dataset.status = "alert";
    byId("status-text").textContent = "ERROR";
    byId("load-error").hidden = false;
    byId("load-error").textContent = `${error.message}. Serve this directory over HTTP and rebuild the artifacts if needed.`;
  }
}

function runSelfTests() {
  let passed = 0;
  let failed = 0;
  const check = (name, condition) => {
    console.assert(condition, `DriftWatch self-test failed: ${name}`);
    if (condition) passed += 1;
    else failed += 1;
  };
  check("PSI identical", Math.abs(psi([20, 30, 50], [20, 30, 50])) < 1e-12);
  // Hand calculation: (-.25)ln(.5) + (.25)ln(1.5) = 0.274653072.
  check("PSI hand example", Math.abs(psi([50, 50], [25, 75]) - 0.274653072) < 1e-9);
  check("JS identical", Math.abs(jsDivergence([2, 3], [2, 3])) < 1e-12);
  check("JS disjoint", Math.abs(jsDivergence([1, 0], [0, 1]) - 1) < 1e-12);
  check("KS identical", Math.abs(ksStatistic([1, 2, 3], [1, 2, 3])) < 1e-12);
  check("AUC perfect", Math.abs(auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1) < 1e-12);
  check("AUC random tie", Math.abs(auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12);
  const step = [...Array(400).fill(0), ...Array(400).fill(1)];
  check("Page-Hinkley step", pageHinkley(step) >= 400);
  let seed = 17;
  const stationary = Array.from({ length: 1000 }, () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32 < 0.2 ? 1 : 0;
  });
  check("Page-Hinkley stationary", pageHinkley(stationary) === -1);
  const summary = `DriftWatch self-test: ${passed}/${passed + failed} passed`;
  if (failed) console.error(summary);
  else console.info(summary);
  return failed === 0;
}

if (typeof window === "undefined") {
  if (process.argv.includes("--selftest") && !runSelfTests()) process.exitCode = 1;
} else {
  document.addEventListener("DOMContentLoaded", () => {
    if (new URLSearchParams(window.location.search).get("selftest") === "1") runSelfTests();
    init();
  });
}
