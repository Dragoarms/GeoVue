const CONFIG = {
  holeId: "BA0007",
  firstMeter: 1,
  maxDepthMeter: 102,
  textureWidthPx: 900,
  resampledHeightPx: 500,
  metresPerResampledPixel: 0.002,
  dataUrl: "./BA0007_viewer_data.json",
  chipTrayManifestUrl: "./chip_tray_manifest.json",
  annotationsUrl: "",
  annotationsSaveUrl: "",
  rawDir: "../BA0007_meter_segments_raw_by_record",
  resampledDir: "../BA0007_meter_segments_depth_resampled_500px_per_m",
};

const MINERAL_KEYS = [
  "HEMATITE",
  "GOETHITE",
  "CHERT",
  "KAOLINITE",
  "GIBBSITE",
  "PHOSPHOSIDERITE",
  "APATITE",
  "PYRITE",
  "ANATASE",
  "KFELDSP",
  "MANGANESE",
  "DOLOMITE",
  "CALCITE",
  "MAGNESITE",
  "MANJORITE",
  "XSLOI",
];

const MINERAL_COLORS = {
  HEMATITE: "#c95f55",
  GOETHITE: "#c89555",
  CHERT: "#d9d4bd",
  KAOLINITE: "#c6a1d8",
  GIBBSITE: "#83c7c7",
  PHOSPHOSIDERITE: "#a48cd8",
  APATITE: "#7fbf8e",
  PYRITE: "#d8bd62",
  ANATASE: "#8fa7e6",
  KFELDSP: "#e0a97c",
  MANGANESE: "#8b7b9d",
  DOLOMITE: "#a4c7a1",
  CALCITE: "#bcd8d8",
  MAGNESITE: "#8bb3a7",
  MANJORITE: "#a06e82",
  XSLOI: "#9ea49a",
  OTHER: "#656b61",
};

const PLANE_CATEGORY_COLORS = {
  Bedding: "#77d6ff",
  Fracture: "#f2c46d",
};

const canvas = document.getElementById("glCanvas");
const trackCanvas = document.getElementById("trackCanvas");
const overviewCanvas = document.getElementById("overviewCanvas");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
const trackCtx = trackCanvas.getContext("2d");
const overviewCtx = overviewCanvas.getContext("2d");

if (!gl) {
  document.body.innerHTML = "<p>WebGL2 is required for this viewer.</p>";
  throw new Error("WebGL2 unavailable");
}

const ui = {
  segment: document.getElementById("segmentInput"),
  range: document.getElementById("rangeInput"),
  prev: document.getElementById("prevSegment"),
  next: document.getElementById("nextSegment"),
  mode: document.getElementById("modeSelect"),
  geometry: document.getElementById("geometrySelect"),
  diameter: document.getElementById("diameterInput"),
  sceneScale: document.getElementById("sceneScaleInput"),
  traceSample: document.getElementById("traceSampleInput"),
  view: document.getElementById("viewSelect"),
  wire: document.getElementById("wireToggle"),
  trace: document.getElementById("traceToggle"),
  spin: document.getElementById("spinToggle"),
  seam: document.getElementById("seamToggle"),
  chip: document.getElementById("chipToggle"),
  squarePixel: document.getElementById("squarePixelButton"),
  hudSegment: document.getElementById("hudSegment"),
  hudScale: document.getElementById("hudScale"),
  hudStatus: document.getElementById("hudStatus"),
  hoverReadout: document.getElementById("hoverReadout"),
  metricTexture: document.getElementById("metricTexture"),
  metricVertical: document.getElementById("metricVertical"),
  metricCircumference: document.getElementById("metricCircumference"),
  metricRadius: document.getElementById("metricRadius"),
  metricCoverage: document.getElementById("metricCoverage"),
  metricLayers: document.getElementById("metricLayers"),
  metricAgreement: document.getElementById("metricAgreement"),
  metricFocus: document.getElementById("metricFocus"),
  traceSummary: document.getElementById("traceSummary"),
  trackLegend: document.getElementById("trackLegend"),
  planeToolbar: document.getElementById("planeToolbar"),
  chipOverlay: document.getElementById("chipOverlay"),
  contextMenu: document.getElementById("contextMenu"),
  contextAddPoint: document.getElementById("contextAddPoint"),
  contextPlaneMode: document.getElementById("contextPlaneMode"),
  overview: overviewCanvas,
  overviewStatus: document.getElementById("overviewStatus"),
  addDrillhole: document.getElementById("addDrillholeButton"),
  fitHole: document.getElementById("fitHoleButton"),
  trackList: document.getElementById("trackList"),
  leftTrackList: document.getElementById("leftTrackList"),
  rightTrackList: document.getElementById("rightTrackList"),
  addTrack: document.getElementById("addTrackButton"),
  styleTrackName: document.getElementById("styleTrackName"),
  styleSourceName: document.getElementById("styleSourceName"),
  trackDisplayType: document.getElementById("trackDisplayType"),
  trackSide: document.getElementById("trackSide"),
  trackColorMode: document.getElementById("trackColorMode"),
  trackScale: document.getElementById("trackScale"),
  trackOffset: document.getElementById("trackOffset"),
  trackLineWidth: document.getElementById("trackLineWidth"),
  trackFlatColor: document.getElementById("trackFlatColor"),
  trackClamp: document.getElementById("trackClamp"),
  trackLog: document.getElementById("trackLog"),
  trackLabels: document.getElementById("trackLabels"),
  trackRounding: document.getElementById("trackRounding"),
  trackSmoothing: document.getElementById("trackSmoothing"),
  trackNormalise: document.getElementById("trackNormalise"),
  trackMin: document.getElementById("trackMin"),
  trackMax: document.getElementById("trackMax"),
  geometryToggle: document.getElementById("geometryToggleButton"),
  zoomOut: document.getElementById("zoomOutButton"),
  zoomIn: document.getElementById("zoomInButton"),
  fitView: document.getElementById("fitViewButton"),
  planeSelect: document.getElementById("planeSelectTool"),
  planeNew: document.getElementById("planeNewButton"),
  planeFinish: document.getElementById("planeFinishButton"),
  planeUndo: document.getElementById("planeUndoButton"),
  planeRedo: document.getElementById("planeRedoButton"),
  planeDelete: document.getElementById("planeDeleteButton"),
  planeCategory: document.getElementById("planeCategorySelect"),
  planeAzimuth: document.getElementById("planeAzimuthInput"),
  planeDip: document.getElementById("planeDipInput"),
};

const state = {
  data: null,
  meter: 9,
  rangeCount: 4,
  mode: "resampled",
  geometry: "wrapped",
  diameterM: 0.15,
  sceneScale: 1,
  traceSampleM: 0.5,
  meshes: [],
  traceLine: null,
  basePoint: null,
  focusDepth: 11,
  orbitYaw: -0.4,
  surfaceYaw: 0,
  distance: 7.2,
  lateralPanM: 0,
  depthScrollCarryM: 0,
  trackHeaderHitboxes: [],
  dragTrackId: null,
  dragging: false,
  dragMode: "pan",
  pointerMoved: false,
  lastPointer: { x: 0, y: 0 },
  lastTime: 0,
  camera: null,
  textureCache: new Map(),
  chipTrayRows: [],
  chipImageCache: new Map(),
  chipTrayStatus: "No chip tray manifest loaded",
  tracks: [],
  selectedTrackId: "gamma",
  observations: [],
  planeMode: false,
  planeTool: "select",
  planeCategory: "Bedding",
  planePoints: [],
  planes: [],
  selectedPlaneRef: null,
  planeHistory: [],
  planeRedo: [],
  manualDraftOrientation: null,
  contextDepth: NaN,
  annotationStatus: "Annotations not loaded",
  annotationSaveTimer: null,
};

const surfaceVertexShader = `#version 300 es
precision highp float;
in vec3 aPosition;
in vec2 aUv;
uniform mat4 uMvp;
out vec2 vUv;
void main() {
  vUv = aUv;
  gl_Position = uMvp * vec4(aPosition, 1.0);
}`;

const surfaceFragmentShader = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTexture;
uniform float uUOffset;
out vec4 outColor;
void main() {
  vec2 uv = vec2(fract(vUv.x + uUOffset), vUv.y);
  vec4 texel = texture(uTexture, uv);
  outColor = vec4(texel.rgb, 1.0);
}`;

const lineVertexShader = `#version 300 es
precision highp float;
in vec3 aPosition;
uniform mat4 uMvp;
void main() {
  gl_Position = uMvp * vec4(aPosition, 1.0);
}`;

const lineFragmentShader = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 outColor;
void main() {
  outColor = vec4(uColor, 1.0);
}`;

const surfaceProgram = createProgram(surfaceVertexShader, surfaceFragmentShader);
const lineProgram = createProgram(lineVertexShader, lineFragmentShader);

const surfaceLoc = {
  position: gl.getAttribLocation(surfaceProgram, "aPosition"),
  uv: gl.getAttribLocation(surfaceProgram, "aUv"),
  mvp: gl.getUniformLocation(surfaceProgram, "uMvp"),
  texture: gl.getUniformLocation(surfaceProgram, "uTexture"),
  uOffset: gl.getUniformLocation(surfaceProgram, "uUOffset"),
};

const lineLoc = {
  position: gl.getAttribLocation(lineProgram, "aPosition"),
  mvp: gl.getUniformLocation(lineProgram, "uMvp"),
  color: gl.getUniformLocation(lineProgram, "uColor"),
};

applyInitialUrlState();
initEvents();
loadData();
requestAnimationFrame(render);
window.__otvViewerSnapshot = viewerSnapshot;

function applyInitialConfigFromUrl(params) {
  const textKeys = ["holeId", "dataUrl", "chipTrayManifestUrl", "annotationsUrl", "annotationsSaveUrl", "rawDir", "resampledDir"];
  for (const key of textKeys) {
    const value = params.get(key);
    if (value) CONFIG[key] = value;
  }

  const numberKeys = [
    "firstMeter",
    "maxDepthMeter",
    "textureWidthPx",
    "resampledHeightPx",
    "metresPerResampledPixel",
  ];
  for (const key of numberKeys) {
    const value = parseFloat(params.get(key));
    if (Number.isFinite(value)) CONFIG[key] = value;
  }

  ui.segment.min = CONFIG.firstMeter;
  ui.segment.max = Math.max(CONFIG.firstMeter, CONFIG.maxDepthMeter - 1);
  ui.range.max = Math.max(1, CONFIG.maxDepthMeter - CONFIG.firstMeter);
  updateHoleLabels();
}

function updateHoleLabels() {
  const holeId = CONFIG.holeId || "Televiewer";
  document.title = `${holeId} OTV Viewer`;
  const eyebrow = document.querySelector(".panel-heading .eyebrow");
  if (eyebrow) eyebrow.textContent = holeId;
  const chipText = ui.chipOverlay?.querySelector("span");
  if (chipText) chipText.textContent = `No ${holeId} chip tray image in cache`;
  if (ui.hudSegment && ui.hudSegment.textContent.includes("BA0007")) {
    ui.hudSegment.textContent = ui.hudSegment.textContent.replace("BA0007", holeId);
  }
}

function applyInitialUrlState() {
  const params = new URLSearchParams(window.location.search);
  applyInitialConfigFromUrl(params);
  const start = parseInt(params.get("start") || params.get("meter"), 10);
  const range = parseInt(params.get("range"), 10);
  const depth = parseFloat(params.get("depth"));
  const geometry = params.get("geometry");
  const mode = params.get("mode");
  const planeMode = params.get("planeMode") || params.get("plane");
  const chip = params.get("chip") || params.get("chipTray");

  if (Number.isFinite(range)) {
    state.rangeCount = clamp(range, 1, CONFIG.maxDepthMeter - CONFIG.firstMeter);
    ui.range.value = state.rangeCount;
  }
  if (Number.isFinite(start)) {
    state.meter = clamp(start, CONFIG.firstMeter, maxStartForRange());
    ui.segment.value = state.meter;
  }
  if (Number.isFinite(depth)) {
    state.focusDepth = clamp(depth, state.meter, state.meter + state.rangeCount);
  } else {
    state.focusDepth = viewportCenterDepth();
  }
  if (geometry === "flat" || geometry === "wrapped") {
    state.geometry = geometry;
    ui.geometry.value = geometry;
  }
  if (mode === "raw" || mode === "resampled") {
    state.mode = mode;
    ui.mode.value = mode;
  }
  if (planeMode === "1" || planeMode === "true") {
    state.planeMode = true;
    state.planeTool = "draw";
  }
  if (chip === "1" || chip === "true") {
    ui.chip.checked = true;
  }
}

function initEvents() {
  ui.segment.addEventListener("change", () => setMeter(parseInt(ui.segment.value, 10)));
  ui.prev.addEventListener("click", () => setMeter(state.meter - 1));
  ui.next.addEventListener("click", () => setMeter(state.meter + 1));

  ui.range.addEventListener("change", () => {
    state.rangeCount = clamp(parseInt(ui.range.value, 10) || 1, 1, CONFIG.maxDepthMeter - CONFIG.firstMeter);
    state.meter = clamp(state.meter, CONFIG.firstMeter, maxStartForRange());
    state.focusDepth = clamp(state.focusDepth, state.meter, state.meter + state.rangeCount);
    rebuildScene();
  });

  ui.mode.addEventListener("change", () => {
    state.mode = ui.mode.value;
    rebuildScene();
  });

  ui.geometry.addEventListener("change", () => {
    setGeometryMode(ui.geometry.value);
  });

  ui.diameter.addEventListener("input", () => {
    state.diameterM = clamp(parseFloat(ui.diameter.value) || 0.15, 0.03, 1.5);
    rebuildScene();
  });

  ui.sceneScale.addEventListener("input", () => {
    state.sceneScale = clamp(parseFloat(ui.sceneScale.value) || 1, 0.25, 5);
    rebuildScene();
  });

  ui.traceSample.addEventListener("input", () => {
    state.traceSampleM = clamp(parseFloat(ui.traceSample.value) || 0.5, 0.25, 2);
    rebuildScene();
  });

  ui.view.addEventListener("change", () => {
    state.lateralPanM = 0;
    fitCameraToScene({ preserveDistance: false });
  });
  ui.chip.addEventListener("change", () => {
    ui.chipOverlay.classList.remove("visible");
    ui.hudStatus.textContent = ui.chip.checked ? chipTrayStatusText() : "Chip tray track off";
  });

  ui.squarePixel.addEventListener("click", () => {
    const circumference = CONFIG.textureWidthPx * CONFIG.metresPerResampledPixel;
    const diameter = circumference / Math.PI;
    state.diameterM = diameter;
    ui.diameter.value = diameter.toFixed(3);
    rebuildScene();
  });

  for (const list of [ui.trackList, ui.leftTrackList, ui.rightTrackList]) {
    list.addEventListener("click", handleTrackListClick);
    list.addEventListener("change", handleTrackListChange);
  }

  ui.addTrack.addEventListener("click", addTrack);
  ui.geometryToggle.addEventListener("click", () => {
    setGeometryMode(state.geometry === "flat" ? "wrapped" : "flat");
  });
  ui.zoomIn.addEventListener("click", () => zoomByFactor(0.72));
  ui.zoomOut.addEventListener("click", () => zoomByFactor(1.32));
  ui.fitView.addEventListener("click", () => {
    state.lateralPanM = 0;
    fitCameraToScene({ preserveDistance: false });
    updateMetrics();
  });
  ui.planeSelect.addEventListener("click", selectPlaneTool);
  ui.planeNew.addEventListener("click", startNewPlane);
  ui.planeFinish.addEventListener("click", finishPlane);
  ui.planeUndo.addEventListener("click", undoPlaneAction);
  ui.planeRedo.addEventListener("click", redoPlaneAction);
  ui.planeDelete.addEventListener("click", deleteSelectedPlaneItem);
  ui.planeCategory.addEventListener("change", () => {
    recordPlaneHistory();
    state.planeCategory = ui.planeCategory.value;
    if (state.planePoints.length) scheduleAnnotationSave();
    updatePlaneToolbar();
  });
  ui.planeAzimuth.addEventListener("input", applyPlaneOrientationInputs);
  ui.planeAzimuth.addEventListener("change", applyPlaneOrientationInputs);
  ui.planeDip.addEventListener("input", applyPlaneOrientationInputs);
  ui.planeDip.addEventListener("change", applyPlaneOrientationInputs);
  ui.addDrillhole.addEventListener("click", () => {
    ui.overviewStatus.textContent = "Add drillhole stub: future picker will load another trace.";
  });
  ui.fitHole.addEventListener("click", () => {
    setViewport(CONFIG.firstMeter, CONFIG.maxDepthMeter - CONFIG.firstMeter);
    ui.overviewStatus.textContent = "Showing full logged OTV depth range.";
  });

  ui.overview.addEventListener("click", (event) => {
    const depth = overviewDepthFromEvent(event);
    if (!Number.isFinite(depth)) return;
    state.focusDepth = depth;
    const half = Math.max(0.5, state.rangeCount / 2);
    setViewport(Math.round(depth - half), state.rangeCount);
    ui.overviewStatus.textContent = `Jumped main viewport to ${depth.toFixed(2)} m.`;
  });

  ui.contextAddPoint.addEventListener("click", () => {
    const depth = Number.isFinite(state.contextDepth) ? state.contextDepth : state.focusDepth;
    state.observations.push({ depth, kind: "point" });
    scheduleAnnotationSave();
    hideContextMenu();
    ui.hudStatus.textContent = `Point stub added at ${depth.toFixed(2)} m`;
    updateMetrics();
  });

  ui.contextPlaneMode.addEventListener("click", () => {
    if (state.planeMode) {
      selectPlaneTool();
    } else {
      startNewPlane();
    }
    hideContextMenu();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("#contextMenu")) hideContextMenu();
  });

  /*
  ui.trackList.addEventListener("click", (event) => {
    const item = event.target.closest(".track-item");
    if (!item) return;
    state.selectedTrackId = item.dataset.trackId;
    renderTrackList();
    updateStyleEditor();
  });

  ui.trackList.addEventListener("change", (event) => {
    if (!event.target.matches("input[type='checkbox']")) return;
    const item = event.target.closest(".track-item");
    const track = getTrack(item.dataset.trackId);
    track.visible = event.target.checked;
    renderTrackList();
    updateMetrics();
  });
  */

  [
    ui.trackDisplayType,
    ui.trackSide,
    ui.trackColorMode,
    ui.trackScale,
    ui.trackOffset,
    ui.trackLineWidth,
    ui.trackFlatColor,
    ui.trackClamp,
    ui.trackLog,
    ui.trackLabels,
    ui.trackRounding,
    ui.trackSmoothing,
    ui.trackNormalise,
    ui.trackMin,
    ui.trackMax,
  ].forEach((control) => {
    control.addEventListener("input", applyStyleEditor);
    control.addEventListener("change", applyStyleEditor);
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault();
    const headerTrack = event.button === 0 ? trackHeaderAtEvent(event) : null;
    state.dragging = true;
    state.pointerMoved = false;
    state.dragTrackId = headerTrack?.id || null;
    if (headerTrack) {
      state.dragMode = "track-header";
      state.selectedTrackId = headerTrack.id;
      updateStyleEditor();
      renderTrackList();
    } else {
      state.dragMode = event.button === 1 || event.shiftKey
        ? "pan"
        : (state.geometry === "wrapped" ? "wrapped-roll-depth" : "pan");
    }
    state.lastPointer = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (state.dragging) {
      const dx = event.clientX - state.lastPointer.x;
      const dy = event.clientY - state.lastPointer.y;
      state.pointerMoved = state.pointerMoved || Math.hypot(dx, dy) > 2;
      state.lastPointer = { x: event.clientX, y: event.clientY };
      if (state.dragMode === "track-header") {
        dragTrackHeader(dx);
      } else if (state.dragMode === "wrapped-roll-depth") {
        rotateWrappedSurface(dx);
        panViewport(0, dy, { direct: true });
      } else {
        panViewport(dx, dy, { direct: true });
      }
      updateHover(event);
      return;
    }
    canvas.style.cursor = trackHeaderAtEvent(event) ? "grab" : "";
    updateHover(event);
  });

  canvas.addEventListener("pointerup", (event) => {
    state.dragging = false;
    state.dragTrackId = null;
    canvas.releasePointerCapture(event.pointerId);
    if (!state.pointerMoved) {
      if (state.geometry === "flat" && state.planeTool === "draw") addPlanePointFromEvent(event);
      else if (state.planeTool === "select") selectPlaneItemFromEvent(event);
    }
    updateHover(event);
  });

  canvas.addEventListener("pointerleave", () => {
    ui.hoverReadout.textContent = "Hover cylinder or graph track for depth";
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const wheelDelta = primaryWheelDelta(event);
    if (event.shiftKey) {
      zoomViewport(-wheelDelta, event);
    } else if (event.altKey) {
      zoomViewport(wheelDelta, event);
    } else {
      scrollDepth(wheelDelta * wheelDepthMetersPerDelta());
    }
    updateMetrics();
    updateHover(event);
  }, { passive: false });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const depth = nearestTraceDepthAtPointer(event);
    state.contextDepth = depth;
    showContextMenu(event, depth);
  });
}

async function loadData() {
  try {
    ui.hudStatus.textContent = `Loading ${CONFIG.holeId} televiewer bundle`;
    const response = await fetch(CONFIG.dataUrl);
    if (!response.ok) throw new Error(`Could not load ${CONFIG.dataUrl}`);
    state.data = await response.json();
    normalizeBundle(state.data);
    await loadAnnotations();
    buildTracks();
    renderTrackList();
    updateStyleEditor();
    rebuildScene();
    void loadChipTrayManifest();
  } catch (error) {
    ui.hudStatus.textContent = "Data load failed";
    ui.traceSummary.textContent = error.message;
    throw error;
  }
}

async function loadChipTrayManifest() {
  try {
    const response = await fetch(CONFIG.chipTrayManifestUrl, { cache: "no-store" });
    if (!response.ok) {
      state.chipTrayStatus = `No ${CONFIG.holeId} chip tray image manifest found`;
      updateMetrics();
      return;
    }
    const manifest = await response.json();
    state.chipTrayRows = normalizeChipTrayManifest(manifest);
    state.chipTrayStatus = state.chipTrayRows.length
      ? `${state.chipTrayRows.length} chip tray images indexed`
      : "Chip tray manifest is empty";
    if (ui.chip.checked) ui.hudStatus.textContent = chipTrayStatusText();
    updateMetrics();
  } catch {
    state.chipTrayStatus = `No ${CONFIG.holeId} chip tray image manifest found`;
    updateMetrics();
  }
}

async function loadAnnotations() {
  if (!CONFIG.annotationsUrl) {
    state.annotationStatus = "Annotation register unavailable";
    return;
  }
  try {
    const response = await fetch(CONFIG.annotationsUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.observations = normalizeObservations(payload.observations);
    state.planePoints = normalizePlanePoints(payload.planePoints || payload.draftPoints);
    state.planes = normalizePlanes(payload.planes);
    state.annotationStatus = payload.register?.updated_at
      ? `Annotations loaded (${payload.register.updated_at})`
      : "Annotations loaded";
  } catch (error) {
    state.annotationStatus = `Annotation load failed: ${error.message}`;
  }
}

function normalizeObservations(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((row) => ({
    ...row,
    depth: toNumber(row.depth),
    kind: row.kind || "point",
  })).filter((row) => Number.isFinite(row.depth));
}

function normalizePlanePoints(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((point) => ({
    ...point,
    depth: toNumber(point.depth),
    angleDeg: Number.isFinite(toNumber(point.angleDeg)) ? toNumber(point.angleDeg) : 180,
    category: point.category || state.planeCategory,
  })).filter((point) => Number.isFinite(point.depth));
}

function normalizePlanes(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map((plane) => {
    const points = normalizePlanePoints(plane.points);
    const fitted = plane.fitted
      ? {
          center: toNumber(plane.fitted.center),
          cosCoeff: toNumber(plane.fitted.cosCoeff),
          sinCoeff: toNumber(plane.fitted.sinCoeff),
          amplitude: toNumber(plane.fitted.amplitude),
          phaseDeg: toNumber(plane.fitted.phaseDeg),
        }
      : (points.length >= 3 ? fitPlaneSinusoid(points) : null);
    const orientation = plane.orientation
      ? { azimuth: toNumber(plane.orientation.azimuth), dip: toNumber(plane.orientation.dip) }
      : (fitted ? orientationFromFitted(fitted) : null);
    return {
      id: plane.id || `plane-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
      category: plane.category || "Bedding",
      points,
      fitted,
      orientation,
    };
  }).filter((plane) => plane.points.length || plane.fitted);
}

function annotationPayload() {
  return {
    schema: "televiewer_annotations.v1",
    hole_id: CONFIG.holeId,
    observations: state.observations.map((point) => ({ ...point })),
    planePoints: clonePlanePoints(state.planePoints),
    planes: state.planes.map((plane) => ({
      id: plane.id,
      category: plane.category,
      points: clonePlanePoints(plane.points),
      fitted: plane.fitted ? { ...plane.fitted } : null,
      orientation: plane.orientation ? { ...plane.orientation } : null,
    })),
  };
}

function scheduleAnnotationSave() {
  if (!CONFIG.annotationsSaveUrl) {
    state.annotationStatus = "Annotation register read-only";
    return;
  }
  if (state.annotationSaveTimer) clearTimeout(state.annotationSaveTimer);
  state.annotationStatus = "Annotation changes pending";
  state.annotationSaveTimer = setTimeout(() => {
    state.annotationSaveTimer = null;
    void saveAnnotations();
  }, 450);
}

async function saveAnnotations() {
  if (!CONFIG.annotationsSaveUrl) return;
  try {
    const response = await fetch(CONFIG.annotationsSaveUrl, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(annotationPayload()),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.annotationStatus = "Annotations saved";
  } catch (error) {
    state.annotationStatus = `Annotation save failed: ${error.message}`;
  }
  updateMetrics({ skipCameraSync: true });
}

function normalizeChipTrayManifest(manifest) {
  const rows = Array.isArray(manifest)
    ? manifest
    : (manifest.images || manifest.rows || manifest.chip_trays || []);
  return rows.map((row) => {
    const from = toNumber(row.from_m ?? row.from ?? row.start_m ?? row.start_depth ?? row.depth_from);
    const to = toNumber(row.to_m ?? row.to ?? row.end_m ?? row.end_depth ?? row.depth_to);
    const meter = toNumber(row.meter ?? row.depth_m ?? row.depth);
    return {
      from_m: Number.isFinite(from) ? from : meter,
      to_m: Number.isFinite(to) ? to : (Number.isFinite(meter) ? meter + 1 : NaN),
      url: row.url || row.path || row.file || row.src || "",
      label: row.label || row.name || "",
    };
  }).filter((row) => Number.isFinite(row.from_m) && Number.isFinite(row.to_m) && row.url);
}

function chipTrayStatusText() {
  return state.chipTrayRows.length
    ? `Chip tray column on: ${state.chipTrayStatus}`
    : `Chip tray placeholder on: ${state.chipTrayStatus}`;
}

function normalizeBundle(bundle) {
  bundle.geophysics = (bundle.geophysics || []).map((row) => ({
    ...row,
    depth_m: toNumber(row.DEPTH),
    gamma_cps: toNumber(row.GAMMA_CPS),
    caliper_mm: toNumber(row.CALIPER_MM),
    speed: toNumber(row.SPEED),
    temp: toNumber(row.TEMP),
    voltage: toNumber(row.VOLTAGE),
  }));

  bundle.mineralogy = (bundle.mineralogy || []).map((row) => ({
    ...row,
    from_m: toNumber(row.SAMPFROM),
    to_m: toNumber(row.SAMPTO),
  }));

  bundle.assays = (bundle.assays || []).map((row) => ({
    ...row,
    from_m: toNumber(row.SAMPFROM),
    to_m: toNumber(row.SAMPTO),
    fe_pct: toNumber(row.FE_PCT_BEST),
    sio2_pct: toNumber(row.SIO2_PCT_BEST),
    al2o3_pct: toNumber(row.AL2O3_PCT_BEST),
    p_pct: toNumber(row.P_PCT_BEST),
  }));

  bundle.tfdAps544 = (bundle.tfdAps544 || []).map((row) => ({
    ...row,
    depth_m: toNumber(row.depth_m),
    roll_deg: toNumber(row.roll_deg),
    mroll_deg: toNumber(row.mroll_deg),
    tilt_deg: toNumber(row.tilt_deg),
    azimuth_deg: toNumber(row.azimuth_deg),
    gravity_g: toNumber(row.gravity_g),
    valid: row.valid === true,
  }));

  bundle.trace = (bundle.trace || []).map((row) => ({
    ...row,
    depth_m: toNumber(row.depth_m),
    x: toNumber(row.x),
    y: toNumber(row.y),
    z: toNumber(row.z),
  }));
}

function buildTracks() {
  const gammaRange = valueRange(state.data.geophysics, "gamma_cps");
  const feRange = valueRange(state.data.assays, "fe_pct");
  const sio2Range = valueRange(state.data.assays, "sio2_pct");

  state.tracks = [
    {
      id: "gamma",
      label: "Gamma_CPS",
      source: "GEOPHYSICSDETAILS",
      type: "point",
      rows: state.data.geophysics,
      key: "gamma_cps",
      displayType: "line",
      side: "left",
      scale: 0.8,
      offset: 0.24,
      lineWidth: 1.2,
      flatColor: "#9caaff",
      colorMode: "flat",
      colormap: "gamma",
      dataMin: gammaRange.min,
      dataMax: gammaRange.max,
      displayMin: gammaRange.min,
      displayMax: gammaRange.max,
      clamp: false,
      log: false,
      labels: true,
      rounding: false,
      smoothing: "none",
      normalise: "none",
      decimals: 1,
      visible: true,
    },
    {
      id: "fe",
      label: "Fe_pct_BEST",
      source: "SUMMARY_LOGGING_ASSAYS",
      type: "interval",
      rows: state.data.assays,
      key: "fe_pct",
      displayType: "bars",
      side: "right",
      scale: 1.0,
      offset: 0.28,
      lineWidth: 1,
      flatColor: "#f06d53",
      colorMode: "colormap",
      colormap: "iron",
      dataMin: feRange.min,
      dataMax: feRange.max,
      displayMin: feRange.min,
      displayMax: feRange.max,
      clamp: false,
      log: false,
      labels: true,
      rounding: false,
      smoothing: "none",
      normalise: "none",
      decimals: 1,
      visible: true,
    },
    {
      id: "sio2",
      label: "SiO2_pct_BEST",
      source: "SUMMARY_LOGGING_ASSAYS",
      type: "interval",
      rows: state.data.assays,
      key: "sio2_pct",
      displayType: "bars",
      side: "right",
      scale: 0.72,
      offset: 0.98,
      lineWidth: 1,
      flatColor: "#d9d4bd",
      colorMode: "colormap",
      colormap: "silica",
      dataMin: sio2Range.min,
      dataMax: sio2Range.max,
      displayMin: sio2Range.min,
      displayMax: sio2Range.max,
      clamp: false,
      log: false,
      labels: false,
      rounding: false,
      smoothing: "none",
      normalise: "none",
      decimals: 1,
      visible: false,
    },
    {
      id: "mineralogy",
      label: "Normative mineralogy",
      source: "NORMATIVE_MINERALOGY",
      type: "stacked",
      rows: state.data.mineralogy,
      displayType: "stacked",
      side: "left",
      scale: 0.55,
      offset: 0.72,
      lineWidth: 1,
      flatColor: "#8ccf9d",
      colorMode: "colormap",
      colormap: "mineralogy",
      dataMin: 0,
      dataMax: 100,
      displayMin: 0,
      displayMax: 100,
      clamp: false,
      log: false,
      labels: false,
      rounding: false,
      smoothing: "none",
      normalise: "none",
      decimals: 1,
      visible: true,
    },
  ];
}

function renderTrackList() {
  const renderSide = (side) => state.tracks.filter((track) => track.side === side).map((track) => `
    <div class="track-item${track.id === state.selectedTrackId ? " active" : ""}" data-track-id="${track.id}">
      <input type="checkbox" ${track.visible ? "checked" : ""} aria-label="Show ${track.label}" />
      <span>${track.label}</span>
      <i class="track-chip" style="--track-colour:${track.flatColor}"></i>
      <span class="track-order">
        <button type="button" data-order="up" title="Draw earlier">▲</button>
        <button type="button" data-order="down" title="Draw later">▼</button>
      </span>
    </div>
  `).join("");
  ui.leftTrackList.innerHTML = renderSide("left");
  ui.rightTrackList.innerHTML = renderSide("right");
  ui.trackList.innerHTML = state.tracks.map((track) => `<div class="track-item" data-track-id="${track.id}"></div>`).join("");
  renderTrackLegend();
}

function renderTrackLegend() {
  ui.trackLegend.innerHTML = "";
}

function handleTrackListClick(event) {
  const orderButton = event.target.closest("[data-order]");
  const item = event.target.closest(".track-item");
  if (!item) return;
  const track = getTrack(item.dataset.trackId);
  if (!track) return;

  if (orderButton) {
    moveTrack(track.id, orderButton.dataset.order === "up" ? -1 : 1);
    return;
  }

  state.selectedTrackId = track.id;
  renderTrackList();
  updateStyleEditor();
}

function handleTrackListChange(event) {
  if (!event.target.matches("input[type='checkbox']")) return;
  const item = event.target.closest(".track-item");
  const track = getTrack(item.dataset.trackId);
  if (!track) return;
  track.visible = event.target.checked;
  renderTrackList();
  updateMetrics();
}

function moveTrack(trackId, direction) {
  const track = getTrack(trackId);
  if (!track) return;
  const sameSide = state.tracks.filter((candidate) => candidate.side === track.side);
  const sideIndex = sameSide.findIndex((candidate) => candidate.id === trackId);
  const targetSideIndex = clamp(sideIndex + direction, 0, sameSide.length - 1);
  if (targetSideIndex === sideIndex) return;
  const targetTrack = sameSide[targetSideIndex];
  const a = state.tracks.findIndex((candidate) => candidate.id === track.id);
  const b = state.tracks.findIndex((candidate) => candidate.id === targetTrack.id);
  [state.tracks[a], state.tracks[b]] = [state.tracks[b], state.tracks[a]];
  renderTrackList();
}

function addTrack() {
  const candidates = state.tracks.filter((track) => !track.visible);
  const track = candidates[0] || state.tracks.find((candidate) => candidate.id === "sio2");
  if (!track) return;
  track.visible = true;
  state.selectedTrackId = track.id;
  renderTrackList();
  updateStyleEditor();
  ui.hudStatus.textContent = `Track added: ${track.label}`;
  updateMetrics();
}

function updateStyleEditor() {
  const track = selectedTrack();
  if (!track) return;
  ui.styleTrackName.textContent = track.label;
  ui.styleSourceName.textContent = track.source;
  ui.trackDisplayType.value = track.displayType;
  ui.trackDisplayType.disabled = track.type === "stacked";
  ui.trackSide.value = track.side;
  ui.trackColorMode.value = track.colorMode;
  ui.trackScale.value = track.scale;
  ui.trackOffset.value = track.offset;
  ui.trackLineWidth.value = track.lineWidth;
  ui.trackFlatColor.value = track.flatColor;
  ui.trackClamp.checked = track.clamp;
  ui.trackLog.checked = track.log;
  ui.trackLabels.checked = track.labels;
  ui.trackRounding.checked = track.rounding;
  ui.trackSmoothing.value = track.smoothing || "none";
  ui.trackNormalise.value = track.normalise || "none";
  ui.trackMin.value = formatInputNumber(track.displayMin);
  ui.trackMax.value = formatInputNumber(track.displayMax);
}

function applyStyleEditor() {
  const track = selectedTrack();
  if (!track) return;
  track.displayType = track.type === "stacked" ? "stacked" : ui.trackDisplayType.value;
  track.side = ui.trackSide.value;
  track.colorMode = ui.trackColorMode.value;
  track.scale = toNumber(ui.trackScale.value) || track.scale;
  track.offset = toNumber(ui.trackOffset.value) || track.offset;
  track.lineWidth = toNumber(ui.trackLineWidth.value) || track.lineWidth;
  track.flatColor = ui.trackFlatColor.value || track.flatColor;
  track.clamp = ui.trackClamp.checked;
  track.log = ui.trackLog.checked;
  track.labels = ui.trackLabels.checked;
  track.rounding = ui.trackRounding.checked;
  track.smoothing = ui.trackSmoothing.value;
  track.normalise = ui.trackNormalise.value;
  track.displayMin = toNumber(ui.trackMin.value);
  track.displayMax = toNumber(ui.trackMax.value);
  renderTrackList();
  updateMetrics();
}

function setViewport(startMeter, rangeMeters) {
  state.rangeCount = clamp(Math.round(rangeMeters) || 1, 1, CONFIG.maxDepthMeter - CONFIG.firstMeter);
  state.meter = clamp(Math.round(startMeter) || CONFIG.firstMeter, CONFIG.firstMeter, maxStartForRange());
  state.focusDepth = clamp(state.focusDepth, state.meter, state.meter + state.rangeCount);
  ui.range.value = state.rangeCount;
  ui.segment.value = state.meter;
  rebuildScene({ preserveView: false });
}

function setGeometryMode(mode) {
  if (mode !== "flat" && mode !== "wrapped") return;
  state.geometry = mode;
  ui.geometry.value = mode;
  state.lateralPanM = 0;
  if (mode === "flat") {
    state.orbitYaw = 0;
    ui.spin.checked = false;
  } else if (state.planeTool === "draw") {
    selectPlaneTool();
  }
  rebuildScene({ preserveView: false });
  updatePlaneToolbar();
}

function panViewport(dx, dy, options = {}) {
  panLateral(dx);
  const direction = options.direct ? -1 : 1;
  const depthPerPixel = Math.max(0.002, state.rangeCount / Math.max(1, canvas.clientHeight * 0.82));
  scrollDepth(dy * depthPerPixel * direction);
}

function scrollDepth(depthDelta) {
  if (!Number.isFinite(depthDelta) || Math.abs(depthDelta) < 0.000001) {
    updateMetrics();
    return;
  }
  const bounds = focusDepthBounds();
  const desiredFocus = state.focusDepth + depthDelta;
  const boundedFocus = clamp(desiredFocus, bounds.min, bounds.max);
  const overflow = desiredFocus - boundedFocus;
  state.focusDepth = boundedFocus;

  if (Math.abs(overflow) > 0.000001) {
    handoffDepthOverflow(overflow);
    return;
  }

  state.depthScrollCarryM = 0;
  updateMetrics();
}

function handoffDepthOverflow(overflow) {
  if (Math.sign(overflow) !== Math.sign(state.depthScrollCarryM || overflow)) {
    state.depthScrollCarryM = 0;
  }
  state.depthScrollCarryM += overflow;
  const thresholdM = 0.25;
  if (Math.abs(state.depthScrollCarryM) < thresholdM) {
    updateMetrics();
    return;
  }

  const step = Math.sign(state.depthScrollCarryM);
  const nextMeter = clamp(state.meter + step, CONFIG.firstMeter, maxStartForRange());
  if (nextMeter === state.meter) {
    state.depthScrollCarryM = 0;
    updateMetrics();
    return;
  }

  const actualStep = nextMeter - state.meter;
  state.meter = nextMeter;
  state.focusDepth = clampFocusDepthToVisibleRange(state.focusDepth + actualStep);
  state.depthScrollCarryM = 0;
  rebuildScene({ preserveView: true });
}
function focusDepthBounds() {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  const range = Math.max(0.001, end - start);
  const halfVisible = Math.min(range / 2, visibleHalfDepthM(range));
  const min = start + halfVisible;
  const max = end - halfVisible;
  if (min > max) {
    const center = (start + end) / 2;
    return { min: center, max: center, halfVisible };
  }
  return { min, max, halfVisible };
}

function visibleHalfDepthM(range = getRangeMeters().length) {
  const fitDistance = Math.max(minZoomInDistance(), maxZoomOutDistance());
  const zoomRatio = clamp(state.distance / fitDistance, 0.04, 1);
  return Math.max(0.05, (range / 2) * zoomRatio);
}

function clampFocusDepthToVisibleRange(depth = state.focusDepth) {
  const bounds = focusDepthBounds();
  return clamp(depth, bounds.min, bounds.max);
}

function isPlaneFocusMode() {
  return state.geometry === "flat" && (
    state.planeMode ||
    state.planeTool === "draw" ||
    state.planePoints.length > 0 ||
    Boolean(state.selectedPlaneRef)
  );
}

function sideDataVisibleDepthM(range = getRangeMeters().length) {
  return visibleHalfDepthM(range) * 2;
}

function sideDataZoomThresholdM(range = getRangeMeters().length) {
  return clamp(range * 0.55, 2.75, 4.5);
}

function shouldDrawSideData() {
  if (isPlaneFocusMode()) return false;
  const range = getRangeMeters().length;
  if (sideDataVisibleDepthM(range) < sideDataZoomThresholdM(range)) return false;
  return sideDataFitsViewport();
}

function shouldDrawChipTrack() {
  return ui.chip.checked && shouldDrawSideData();
}

function sideDataFitsViewport() {
  if (!canvas.clientWidth || !state.camera) return true;
  const range = getRangeMeters().length;
  const probeDepth = clamp(state.focusDepth, state.meter + 0.05, state.meter + Math.max(0.05, range - 0.05));
  const frame = screenFrameAtDepth(probeDepth);
  if (!frame) return false;

  const chipReserve = ui.chip.checked ? chipColumnInnerPx() + chipStripWidthPx() + 10 : 0;
  const clearance = trackClearancePx("left");
  const leftExtent = Math.max(
    ui.chip.checked ? chipReserve : 0,
    ...state.tracks
      .filter((track) => track.visible && track.side === "left")
      .map((track) => chipReserve + trackOffsetPx(track) + maxTrackDrawWidthPx(track) + clearance)
  );
  const rightExtent = Math.max(
    0,
    ...state.tracks
      .filter((track) => track.visible && track.side === "right")
      .map((track) => trackOffsetPx(track) + maxTrackDrawWidthPx(track) + clearance)
  );
  const leftOuter = leftExtent > 0 ? lateralPoint(frame, "left", frame.radiusPx + leftExtent) : null;
  const rightOuter = rightExtent > 0 ? lateralPoint(frame, "right", frame.radiusPx + rightExtent) : null;
  const margin = 32;
  return (!leftOuter || leftOuter.x >= -margin) && (!rightOuter || rightOuter.x <= canvas.clientWidth + margin);
}

function panLateral(dx) {
  if (!Number.isFinite(dx) || Math.abs(dx) < 0.001) return;
  const worldDelta = -dx * worldUnitsPerScreenPixel();
  state.lateralPanM = clamp(state.lateralPanM + worldDelta, -maxLateralPanM(), maxLateralPanM());
}

function zoomViewport(deltaY, event) {
  const depth = nearestTraceDepthAtPointer(event);
  if (Number.isFinite(depth)) state.focusDepth = depth;
  const factor = Math.exp(deltaY * 0.001);
  zoomByFactor(factor);
}

function primaryWheelDelta(event) {
  return Math.abs(event.deltaX) > Math.abs(event.deltaY) ? event.deltaX : event.deltaY;
}

function zoomByFactor(factor) {
  state.distance = clamp(state.distance * factor, minZoomInDistance(), maxZoomOutDistance());
  state.focusDepth = clampFocusDepthToVisibleRange();
  state.lateralPanM = clamp(state.lateralPanM, -maxLateralPanM(), maxLateralPanM());
  updateMetrics();
}

function wheelDepthMetersPerDelta() {
  return Math.max(0.0015, state.rangeCount / Math.max(1, canvas.clientHeight * 1.35));
}

function worldUnitsPerScreenPixel() {
  const fov = currentFovDeg();
  return Math.max(0.00001, 2 * state.distance * Math.tan(degToRad(fov / 2)) / Math.max(1, canvas.clientHeight));
}

function maxLateralPanM() {
  const viewHalfWidth = worldUnitsPerScreenPixel() * canvas.clientWidth * 0.5;
  const contentWidthM = Math.max(
    state.diameterM * state.sceneScale,
    ...state.meshes.map((mesh) => mesh.flatWidth || mesh.radius * 2)
  );
  const meshHalfWidth = contentWidthM * 0.5;
  const trackAllowance = worldUnitsPerScreenPixel() * maxScreenTrackExtentPx();
  return Math.max(0.05, meshHalfWidth + trackAllowance - viewHalfWidth * 0.55);
}

function maxScreenTrackExtentPx() {
  if (!shouldDrawSideData()) return 120;
  const visible = state.tracks.filter((track) => track.visible);
  const trackExtent = visible.reduce((maxExtent, track) => {
    return Math.max(maxExtent, trackReservePx(track.side) + trackOffsetPx(track) + trackScalePx(track));
  }, 0);
  const chipExtent = shouldDrawChipTrack() ? chipColumnLayout().outer : 0;
  return Math.max(trackExtent, chipExtent, 120);
}

function maxZoomOutDistance() {
  const fov = currentFovDeg();
  const rangeLength = vectorLength(sub(localPointAtDepth(state.meter + getRangeMeters().length), localPointAtDepth(state.meter)));
  const fillFactor = state.geometry === "flat" ? 1.62 : 1.38;
  const fit = rangeLength / Math.max(0.001, 2 * Math.tan(degToRad(fov / 2)) * fillFactor);
  return Math.max(1.3, fit + state.diameterM * state.sceneScale * 2.5);
}

function currentFovDeg() {
  return ui.view.value === "inside" ? 78 : 38;
}

function minZoomInDistance() {
  return state.geometry === "flat" ? 0.35 : 0.08;
}

function clampCameraDistance() {
  state.distance = clamp(state.distance, minZoomInDistance(), maxZoomOutDistance());
}

function rotateWrappedSurface(dx) {
  if (state.geometry !== "wrapped") return;
  state.surfaceYaw = mod(state.surfaceYaw + dx * 0.008, Math.PI * 2);
}

function surfaceUOffset() {
  return mod(state.surfaceYaw / (Math.PI * 2), 1);
}

function planeSnapshot() {
  return {
    planeMode: state.planeMode,
    planeTool: state.planeTool,
    planeCategory: state.planeCategory,
    planePoints: clonePlanePoints(state.planePoints),
    planes: state.planes.map((plane) => ({
      ...plane,
      points: clonePlanePoints(plane.points),
      fitted: plane.fitted ? { ...plane.fitted } : null,
      orientation: plane.orientation ? { ...plane.orientation } : null,
    })),
    selectedPlaneRef: state.selectedPlaneRef ? { ...state.selectedPlaneRef } : null,
    manualDraftOrientation: state.manualDraftOrientation ? { ...state.manualDraftOrientation } : null,
  };
}

function restorePlaneSnapshot(snapshot) {
  state.planeMode = snapshot.planeMode;
  state.planeTool = snapshot.planeTool;
  state.planeCategory = snapshot.planeCategory;
  state.planePoints = clonePlanePoints(snapshot.planePoints);
  state.planes = snapshot.planes.map((plane) => ({
    ...plane,
    points: clonePlanePoints(plane.points),
    fitted: plane.fitted ? { ...plane.fitted } : null,
    orientation: plane.orientation ? { ...plane.orientation } : null,
  }));
  state.selectedPlaneRef = snapshot.selectedPlaneRef ? { ...snapshot.selectedPlaneRef } : null;
  state.manualDraftOrientation = snapshot.manualDraftOrientation ? { ...snapshot.manualDraftOrientation } : null;
  ui.planeCategory.value = state.planeCategory;
  updatePlaneToolbar();
  updateMetrics();
}

function clonePlanePoints(points) {
  return points.map((point) => ({ ...point }));
}

function recordPlaneHistory() {
  state.planeHistory.push(planeSnapshot());
  if (state.planeHistory.length > 80) state.planeHistory.shift();
  state.planeRedo = [];
}

function selectPlaneTool() {
  state.planeTool = "select";
  state.planeMode = false;
  ui.hudStatus.textContent = "Plane select mode";
  updatePlaneToolbar();
  updateMetrics();
}

function startNewPlane() {
  if (state.geometry !== "flat") {
    setGeometryMode("flat");
  }
  recordPlaneHistory();
  state.planePoints = [];
  state.selectedPlaneRef = null;
  state.planeCategory = ui.planeCategory.value || state.planeCategory;
  state.manualDraftOrientation = null;
  state.planeTool = "draw";
  state.planeMode = true;
  ui.hudStatus.textContent = `New ${state.planeCategory.toLowerCase()} plane draft`;
  updatePlaneToolbar();
  updateMetrics();
}

function finishPlane() {
  if (state.planePoints.length < 3) {
    ui.hudStatus.textContent = "Add at least 3 points to finish a plane";
    updatePlaneToolbar();
    return;
  }
  const rawFitted = fitPlaneSinusoid(state.planePoints);
  if (!rawFitted) {
    ui.hudStatus.textContent = "Plane fit failed; add points around more of the borehole";
    return;
  }
  const fitted = state.manualDraftOrientation
    ? fittedFromOrientation(state.manualDraftOrientation, rawFitted.center)
    : rawFitted;
  recordPlaneHistory();
  const id = `plane-${Date.now().toString(36)}`;
  state.planes.push({
    id,
    category: state.planeCategory,
    points: clonePlanePoints(state.planePoints),
    fitted,
    orientation: state.manualDraftOrientation || orientationFromFitted(fitted),
  });
  state.planePoints = [];
  state.manualDraftOrientation = null;
  state.selectedPlaneRef = null;
  state.planeTool = "select";
  state.planeMode = false;
  ui.hudStatus.textContent = `${state.planeCategory} plane finished (${state.planes.length})`;
  scheduleAnnotationSave();
  updatePlaneToolbar();
  updateMetrics();
}

function undoPlaneAction() {
  const snapshot = state.planeHistory.pop();
  if (!snapshot) return;
  state.planeRedo.push(planeSnapshot());
  restorePlaneSnapshot(snapshot);
  ui.hudStatus.textContent = "Plane edit undone";
  scheduleAnnotationSave();
}

function redoPlaneAction() {
  const snapshot = state.planeRedo.pop();
  if (!snapshot) return;
  state.planeHistory.push(planeSnapshot());
  restorePlaneSnapshot(snapshot);
  ui.hudStatus.textContent = "Plane edit redone";
  scheduleAnnotationSave();
}

function deleteSelectedPlaneItem() {
  if (!state.selectedPlaneRef && state.planePoints.length) {
    recordPlaneHistory();
    state.planePoints.pop();
    ui.hudStatus.textContent = "Last draft point deleted";
    scheduleAnnotationSave();
    updatePlaneToolbar();
    updateMetrics();
    return;
  }
  if (!state.selectedPlaneRef) return;

  recordPlaneHistory();
  const ref = state.selectedPlaneRef;
  if (ref.kind === "draft-point") {
    state.planePoints.splice(ref.index, 1);
  } else if (ref.kind === "plane") {
    state.planes = state.planes.filter((plane) => plane.id !== ref.planeId);
  } else if (ref.kind === "plane-point") {
    const plane = state.planes.find((candidate) => candidate.id === ref.planeId);
    if (plane) {
      plane.points.splice(ref.index, 1);
      plane.fitted = plane.points.length >= 3 ? fitPlaneSinusoid(plane.points) : null;
      plane.orientation = plane.fitted ? orientationFromFitted(plane.fitted) : plane.orientation;
    }
  }
  state.selectedPlaneRef = null;
  ui.hudStatus.textContent = "Plane item deleted";
  scheduleAnnotationSave();
  updatePlaneToolbar();
  updateMetrics();
}

function updatePlaneToolbar() {
  if (!ui.planeSelect) return;
  ui.planeToolbar.classList.toggle("wrapped", state.geometry !== "flat");
  ui.geometryToggle.textContent = state.geometry === "flat" ? "Wrapped view" : "Flat view";
  ui.geometryToggle.classList.toggle("active", state.geometry === "flat");
  ui.planeSelect.classList.toggle("active", state.planeTool === "select");
  ui.planeNew.classList.toggle("active", state.planeTool === "draw");
  ui.planeFinish.disabled = state.planePoints.length < 3;
  ui.planeUndo.disabled = state.planeHistory.length === 0;
  ui.planeRedo.disabled = state.planeRedo.length === 0;
  ui.planeDelete.disabled = !state.selectedPlaneRef && state.planePoints.length === 0;
  ui.planeCategory.value = state.planeCategory;
  const orientation = currentPlaneOrientation();
  if (document.activeElement !== ui.planeAzimuth) {
    ui.planeAzimuth.value = Number.isFinite(orientation?.azimuth) ? String(Math.round(orientation.azimuth)) : "0";
  }
  if (document.activeElement !== ui.planeDip) {
    ui.planeDip.value = Number.isFinite(orientation?.dip) ? String(Math.round(orientation.dip)) : "0";
  }
}

function currentPlaneOrientation() {
  if (state.selectedPlaneRef?.planeId) {
    const plane = state.planes.find((candidate) => candidate.id === state.selectedPlaneRef.planeId);
    if (plane?.orientation) return plane.orientation;
    if (plane?.fitted) return orientationFromFitted(plane.fitted);
  }
  if (state.manualDraftOrientation) return state.manualDraftOrientation;
  if (state.planePoints.length >= 3) {
    const fitted = fitPlaneSinusoid(state.planePoints);
    if (fitted) return orientationFromFitted(fitted);
  }
  return { azimuth: 0, dip: 0 };
}

function applyPlaneOrientationInputs() {
  const orientation = {
    azimuth: mod(toNumber(ui.planeAzimuth.value), 360),
    dip: clamp(toNumber(ui.planeDip.value), 0, 89),
  };
  if (!Number.isFinite(orientation.azimuth) || !Number.isFinite(orientation.dip)) return;
  recordPlaneHistory();
  if (state.selectedPlaneRef?.planeId) {
    const plane = state.planes.find((candidate) => candidate.id === state.selectedPlaneRef.planeId);
    if (plane) {
      plane.orientation = orientation;
      if (plane.fitted) plane.fitted = fittedFromOrientation(orientation, plane.fitted.center);
    }
  } else {
    state.manualDraftOrientation = orientation;
  }
  scheduleAnnotationSave();
  updatePlaneToolbar();
  updateMetrics();
}

function orientationFromFitted(fitted) {
  const radius = Math.max(0.001, state.diameterM / 2);
  const dip = clamp(radToDeg(Math.atan(Math.abs(fitted.amplitude) / radius)), 0, 89);
  const azimuth = mod(traceAzimuthAtDepth(fitted.center) + fitted.phaseDeg, 360);
  return { azimuth, dip };
}

function fittedFromOrientation(orientation, centerDepth) {
  const radius = Math.max(0.001, state.diameterM / 2);
  const amplitude = Math.tan(degToRad(orientation.dip)) * radius;
  const phase = degToRad(mod(orientation.azimuth - traceAzimuthAtDepth(centerDepth), 360));
  return {
    center: centerDepth,
    cosCoeff: amplitude * Math.cos(phase),
    sinCoeff: amplitude * Math.sin(phase),
    amplitude,
    phaseDeg: radToDeg(phase),
  };
}

function showContextMenu(event, depth) {
  const rect = canvas.getBoundingClientRect();
  ui.contextPlaneMode.textContent = state.geometry !== "flat"
    ? "Switch to flat and start plane"
    : state.planeMode ? "Stop plane drawing" : "Start new plane";
  ui.contextAddPoint.textContent = Number.isFinite(depth)
    ? `Add point at ${depth.toFixed(2)} m`
    : "Add point downhole";
  ui.contextMenu.style.left = `${event.clientX - rect.left}px`;
  ui.contextMenu.style.top = `${event.clientY - rect.top}px`;
  ui.contextMenu.classList.add("visible");
  ui.contextMenu.setAttribute("aria-hidden", "false");
}

function hideContextMenu() {
  ui.contextMenu.classList.remove("visible");
  ui.contextMenu.setAttribute("aria-hidden", "true");
}

function setMeter(value) {
  state.meter = clamp(value || CONFIG.firstMeter, CONFIG.firstMeter, maxStartForRange());
  state.focusDepth = viewportCenterDepth();
  rebuildScene();
}

function viewportCenterDepth() {
  return clamp(state.meter + state.rangeCount / 2, state.meter, state.meter + state.rangeCount);
}

function rebuildScene(options = {}) {
  if (!state.data) return;
  const preserveView = Boolean(options.preserveView);

  syncControls();
  disposeScene();

  const meters = getRangeMeters();
  state.basePoint = tracePointAtDepth(state.meter);
  state.meshes = shouldRenderOtvTextures() ? meters.map((meter) => createSegmentMesh(meter, meter + 1)) : [];
  state.traceLine = createTraceLine(state.meter, state.meter + meters.length);

  fitCameraToScene({ preserveDistance: preserveView });
  if (!preserveView) state.lateralPanM = 0;
  updateTraceSummary();
  updateMetrics();
  loadTexturesForMeshes();
}

function syncControls() {
  state.rangeCount = clamp(parseInt(ui.range.value, 10) || state.rangeCount, 1, CONFIG.maxDepthMeter - CONFIG.firstMeter);
  state.meter = clamp(state.meter, CONFIG.firstMeter, maxStartForRange());
  state.focusDepth = clamp(state.focusDepth, state.meter, state.meter + state.rangeCount);
  state.mode = ui.mode.value;
  state.geometry = ui.geometry.value;
  ui.range.value = state.rangeCount;
  ui.segment.value = state.meter;
  ui.segment.max = maxStartForRange();
  ui.diameter.value = state.diameterM.toFixed(3);
  ui.sceneScale.value = state.sceneScale.toFixed(2);
  ui.traceSample.value = state.traceSampleM.toFixed(2);
}

function disposeScene() {
  for (const mesh of state.meshes) deleteMeshBuffers(mesh);
  state.meshes = [];

  if (state.traceLine) {
    gl.deleteBuffer(state.traceLine.buffer);
    gl.deleteVertexArray(state.traceLine.vao);
  }
  state.traceLine = null;
}

function deleteMeshBuffers(mesh) {
  gl.deleteBuffer(mesh.surface.vertex);
  gl.deleteBuffer(mesh.surface.index);
  gl.deleteVertexArray(mesh.surface.vao);
  gl.deleteBuffer(mesh.rings.buffer);
  gl.deleteVertexArray(mesh.rings.vao);
  gl.deleteBuffer(mesh.seam.buffer);
  gl.deleteVertexArray(mesh.seam.vao);
}

function createSegmentMesh(startDepth, endDepth) {
  const pTop = localPointAtDepth(startDepth);
  const pBottom = localPointAtDepth(endDepth);
  const axis = sub(pBottom, pTop);
  const length = vectorLength(axis) || 1;
  const tangent = scale(axis, 1 / length);
  const reference = Math.abs(dot(tangent, [0, 1, 0])) > 0.92 ? [0, 0, 1] : [0, 1, 0];
  const n1 = normalize(cross(reference, tangent));
  const n2 = normalize(cross(tangent, n1));
  const radius = (state.diameterM * state.sceneScale) / 2;
  if (state.geometry === "flat") {
    const flatFrame = flatBillboardFrame(tangent);
    return createFlatSegmentMesh(startDepth, endDepth, pTop, pBottom, tangent, flatFrame.n1, flatFrame.n2, radius);
  }
  const imageSize = { width: CONFIG.textureWidthPx, height: CONFIG.resampledHeightPx };
  const radialSegments = 192;
  const verticalSegments = 10;
  const vertices = [];
  const indices = [];

  for (let yIndex = 0; yIndex <= verticalSegments; yIndex++) {
    const v = yIndex / verticalSegments;
    const center = lerpVec(pTop, pBottom, v);
    for (let i = 0; i <= radialSegments; i++) {
      const u = i / radialSegments;
      const theta = u * Math.PI * 2;
      const radial = add(scale(n1, Math.cos(theta) * radius), scale(n2, Math.sin(theta) * radius));
      const position = add(center, radial);
      vertices.push(position[0], position[1], position[2], u, v);
    }
  }

  const row = radialSegments + 1;
  for (let yIndex = 0; yIndex < verticalSegments; yIndex++) {
    for (let i = 0; i < radialSegments; i++) {
      const a = yIndex * row + i;
      const b = a + 1;
      const c = a + row;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const rings = buildRingLines(pTop, pBottom, n1, n2, radius);
  const seam = buildLineBuffer([
    ...add(pTop, scale(n1, radius * 1.012)),
    ...add(pBottom, scale(n1, radius * 1.012)),
  ]);

  return {
    meter: startDepth,
    startDepth,
    endDepth,
    pTop,
    pBottom,
    center: lerpVec(pTop, pBottom, 0.5),
    length,
    tangent,
    n1,
    n2,
    radius,
    imageSize,
    texture: null,
    path: segmentPath(startDepth),
    surface: buildSurfaceBuffer(vertices, indices),
    rings,
    seam,
  };
}

function flatBillboardFrame(tangent) {
  const axis = viewAxis();
  const lockedViewNormal = viewRadialFrame(axis).right;
  let normal = sub(lockedViewNormal, scale(tangent, dot(lockedViewNormal, tangent)));
  if (vectorLength(normal) < 0.0001) normal = viewRadialFrame(axis).forward;
  const n2 = normalize(normal);
  const n1 = normalize(cross(n2, tangent));
  return { n1, n2 };
}

function createFlatSegmentMesh(startDepth, endDepth, pTop, pBottom, tangent, n1, n2, radius) {
  const length = vectorLength(sub(pBottom, pTop)) || 1;
  const width = Math.max(radius * Math.PI * 2, 0.55 * state.sceneScale);
  const imageSize = { width: CONFIG.textureWidthPx, height: CONFIG.resampledHeightPx };
  const horizontalSegments = 64;
  const verticalSegments = 10;
  const vertices = [];
  const indices = [];

  for (let yIndex = 0; yIndex <= verticalSegments; yIndex++) {
    const v = yIndex / verticalSegments;
    const center = lerpVec(pTop, pBottom, v);
    for (let i = 0; i <= horizontalSegments; i++) {
      const u = i / horizontalSegments;
      const position = add(center, scale(n1, (u - 0.5) * width));
      vertices.push(position[0], position[1], position[2], u, v);
    }
  }

  const row = horizontalSegments + 1;
  for (let yIndex = 0; yIndex < verticalSegments; yIndex++) {
    for (let i = 0; i < horizontalSegments; i++) {
      const a = yIndex * row + i;
      const b = a + 1;
      const c = a + row;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const rings = buildFlatDepthLines(pTop, pBottom, n1, width);
  const seam = buildLineBuffer([
    ...add(pTop, scale(n1, -width / 2)),
    ...add(pBottom, scale(n1, -width / 2)),
    ...add(pTop, scale(n1, width / 2)),
    ...add(pBottom, scale(n1, width / 2)),
  ]);

  return {
    meter: startDepth,
    startDepth,
    endDepth,
    pTop,
    pBottom,
    center: lerpVec(pTop, pBottom, 0.5),
    length,
    tangent,
    n1,
    n2,
    radius,
    flatWidth: width,
    imageSize,
    texture: null,
    path: segmentPath(startDepth),
    surface: buildSurfaceBuffer(vertices, indices),
    rings,
    seam,
  };
}

function buildFlatDepthLines(pTop, pBottom, n1, width) {
  const vertices = [];
  const ringCount = 4;
  for (let r = 0; r <= ringCount; r++) {
    const center = lerpVec(pTop, pBottom, r / ringCount);
    const left = add(center, scale(n1, -width / 2));
    const right = add(center, scale(n1, width / 2));
    vertices.push(...left, ...right);
  }
  const line = buildLineBuffer(vertices);
  line.ringSize = 2;
  line.ringCount = ringCount + 1;
  return line;
}

function buildSurfaceBuffer(vertices, indices) {
  const vao = gl.createVertexArray();
  const vertex = gl.createBuffer();
  const index = gl.createBuffer();

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vertex);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(surfaceLoc.position);
  gl.vertexAttribPointer(surfaceLoc.position, 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(surfaceLoc.uv);
  gl.vertexAttribPointer(surfaceLoc.uv, 2, gl.FLOAT, false, 20, 12);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(indices), gl.STATIC_DRAW);

  return { vao, vertex, index, indexCount: indices.length };
}

function buildRingLines(pTop, pBottom, n1, n2, radius) {
  const vertices = [];
  const radialSegments = 144;
  const ringCount = 4;

  for (let r = 0; r <= ringCount; r++) {
    const center = lerpVec(pTop, pBottom, r / ringCount);
    for (let i = 0; i <= radialSegments; i++) {
      const theta = (i / radialSegments) * Math.PI * 2;
      const radial = add(scale(n1, Math.cos(theta) * radius * 1.006), scale(n2, Math.sin(theta) * radius * 1.006));
      const position = add(center, radial);
      vertices.push(position[0], position[1], position[2]);
    }
  }

  const line = buildLineBuffer(vertices);
  line.ringSize = radialSegments + 1;
  line.ringCount = ringCount + 1;
  return line;
}

function buildLineBuffer(vertices) {
  const vao = gl.createVertexArray();
  const buffer = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(lineLoc.position);
  gl.vertexAttribPointer(lineLoc.position, 3, gl.FLOAT, false, 12, 0);
  return { vao, buffer, count: vertices.length / 3 };
}

function createTraceLine(startDepth, endDepth) {
  const vertices = [];
  for (let depth = startDepth; depth <= endDepth + 0.0001; depth += state.traceSampleM) {
    vertices.push(...localPointAtDepth(Math.min(depth, endDepth)));
  }
  if (vertices.length < 6) vertices.push(...localPointAtDepth(startDepth), ...localPointAtDepth(endDepth));
  return buildLineBuffer(vertices);
}

function loadTexturesForMeshes() {
  let loaded = 0;
  const total = state.meshes.length;
  if (!total) {
    ui.hudStatus.textContent = shouldRenderOtvTextures() ? "No OTV textures in range" : "Summary range: OTV textures paused";
    updateMetrics();
    return;
  }
  ui.hudStatus.textContent = `Loading textures 0/${total}`;

  for (const mesh of state.meshes) {
    loadTexture(mesh.path)
      .then(({ texture, width, height }) => {
        mesh.texture = texture;
        mesh.imageSize = { width, height };
        loaded += 1;
        updateMetrics();
        ui.hudStatus.textContent = loaded === total ? "Ready" : `Loading textures ${loaded}/${total}`;
      })
      .catch(() => {
        loaded += 1;
        ui.hudStatus.textContent = `Missing texture ${pad3(mesh.meter)}-${pad3(mesh.meter + 1)} m`;
      });
  }
}

function shouldRenderOtvTextures() {
  return state.rangeCount <= 18;
}

function loadTexture(path) {
  const cached = state.textureCache.get(path);
  if (cached) return cached;

  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const texture = makeTexture(image);
      resolve({ texture, width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = reject;
    image.src = path;
  });

  state.textureCache.set(path, promise);
  return promise;
}

function makeTexture(image) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  return texture;
}

function segmentPath(meter) {
  const start = pad3(meter);
  const end = pad3(meter + 1);
  if (state.mode === "raw") return `${CONFIG.rawDir}/${CONFIG.holeId}_${start}_${end}m_raw.jpg`;
  return `${CONFIG.resampledDir}/${CONFIG.holeId}_${start}_${end}m_resampled.jpg`;
}

function fitCameraToScene({ preserveDistance } = { preserveDistance: false }) {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  state.focusDepth = clamp(state.focusDepth, start, end);
  if (!preserveDistance) {
    state.distance = maxZoomOutDistance();
  } else {
    clampCameraDistance();
  }
  state.focusDepth = clampFocusDepthToVisibleRange();
}

function render(time) {
  resizeCanvas();
  const dt = Math.min(0.05, (time - state.lastTime) / 1000 || 0);
  state.lastTime = time;
  if (ui.spin.checked && state.geometry === "wrapped") state.surfaceYaw = mod(state.surfaceYaw + dt * 0.35, Math.PI * 2);
  if (state.geometry === "flat") state.orbitYaw = 0;

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.025, 0.032, 0.025, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);

  const mvp = computeMvp();

  for (const mesh of state.meshes) {
    if (!mesh.texture) continue;
    gl.useProgram(surfaceProgram);
    gl.uniformMatrix4fv(surfaceLoc.mvp, false, mvp);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, mesh.texture);
    gl.uniform1i(surfaceLoc.texture, 0);
    gl.uniform1f(surfaceLoc.uOffset, state.geometry === "wrapped" ? surfaceUOffset() : 0);
    gl.bindVertexArray(mesh.surface.vao);
    gl.drawElements(gl.TRIANGLES, mesh.surface.indexCount, gl.UNSIGNED_INT, 0);
  }

  gl.useProgram(lineProgram);
  gl.uniformMatrix4fv(lineLoc.mvp, false, mvp);
  if (ui.wire.checked) {
    gl.uniform3f(lineLoc.color, 0.77, 0.82, 0.74);
    for (const mesh of state.meshes) {
      gl.bindVertexArray(mesh.rings.vao);
      for (let ring = 0; ring < mesh.rings.ringCount; ring++) {
        gl.drawArrays(gl.LINE_STRIP, ring * mesh.rings.ringSize, mesh.rings.ringSize);
      }
    }
  }

  if (ui.seam.checked) {
    gl.uniform3f(lineLoc.color, 0.92, 0.62, 0.3);
    for (const mesh of state.meshes) {
      gl.bindVertexArray(mesh.seam.vao);
      gl.drawArrays(gl.LINES, 0, mesh.seam.count);
    }
  }

  if (ui.trace.checked && state.traceLine) {
    gl.disable(gl.DEPTH_TEST);
    gl.uniform3f(lineLoc.color, 0.78, 0.8, 0.78);
    gl.bindVertexArray(state.traceLine.vao);
    gl.drawArrays(gl.LINE_STRIP, 0, state.traceLine.count);
    gl.enable(gl.DEPTH_TEST);
  }

  drawTrackOverlay();
  drawOverview();
  updateMetrics({ skipCameraSync: true });
  requestAnimationFrame(render);
}

function computeMvp() {
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const projection = mat4Perspective(degToRad(ui.view.value === "inside" ? 78 : 38), aspect, 0.01, 140);
  const depthTarget = localPointAtDepth(state.focusDepth);
  const axis = viewAxis();
  const radialFrame = viewRadialFrame(axis);
  const target = add(depthTarget, scale(radialFrame.right, state.lateralPanM));
  const radial = add(scale(radialFrame.right, Math.cos(state.orbitYaw)), scale(radialFrame.forward, Math.sin(state.orbitYaw)));

  let eye = add(target, scale(radial, state.distance));
  let lookAt = target;
  let up = scale(axis, -1);

  if (ui.view.value === "inside") {
    const radius = Math.max((state.diameterM * state.sceneScale) / 2, 0.02);
    eye = add(target, scale(radial, radius * 0.15));
    lookAt = add(target, scale(axis, 0.85 * state.sceneScale));
    up = scale(axis, -1);
  }

  const view = mat4LookAt(eye, lookAt, up);
  const vp = mat4Multiply(projection, view);
  state.camera = { eye, target: lookAt, projection, view, vp, inverseVp: mat4Invert(vp), axis, radial };
  return vp;
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  if (trackCanvas.width !== width || trackCanvas.height !== height) {
    trackCanvas.width = width;
    trackCanvas.height = height;
  }
  const overviewDpr = Math.min(window.devicePixelRatio || 1, 2);
  const overviewWidth = Math.max(1, Math.floor(ui.overview.clientWidth * overviewDpr));
  const overviewHeight = Math.max(1, Math.floor(ui.overview.clientHeight * overviewDpr));
  if (ui.overview.width !== overviewWidth || ui.overview.height !== overviewHeight) {
    ui.overview.width = overviewWidth;
    ui.overview.height = overviewHeight;
  }
}

function drawTrackOverlay() {
  const ctx = trackCtx;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = trackCanvas.clientWidth;
  const height = trackCanvas.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!state.data || !state.camera) return;

  state.trackHeaderHitboxes = [];
  drawDepthTicks(ctx);
  if (shouldDrawSideData()) {
    if (shouldDrawChipTrack()) drawChipTrack(ctx);
    for (const track of state.tracks) {
      if (!track.visible) continue;
      if (track.displayType === "stacked") drawStackedTrack(ctx, track);
      else if (track.displayType === "bars") drawBarTrack(ctx, track);
      else if (track.displayType === "midpoints") drawMidpointTrack(ctx, track);
      else drawLineTrack(ctx, track);
    }
    drawStickyTrackHeaders(ctx);
  }
  drawPlaneDrawing(ctx);
  if (!isPlaneFocusMode()) drawObservationPoints(ctx);
}

function drawDepthTicks(ctx) {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(230, 235, 220, 0.28)";
  ctx.fillStyle = "rgba(230, 235, 220, 0.76)";
  ctx.font = "11px Inter, sans-serif";

  let lastPoint = null;
  for (let depth = start; depth <= end + 0.001; depth += 0.25) {
    const center = projectTraceDepth(depth);
    if (!center) continue;
    if (lastPoint) {
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(center.x, center.y);
      ctx.stroke();
    }
    lastPoint = center;
  }

  for (let depth = Math.ceil(start); depth <= end; depth += 1) {
    const frame = screenFrameAtDepth(depth);
    if (!frame) continue;
    const tick = depth % 5 === 0 ? 10 : 6;
    const a = offsetPoint(frame.center, frame.right, -tick);
    const b = offsetPoint(frame.center, frame.right, tick);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    if (depth % 5 === 0 || depth === start || depth === end) {
      const label = offsetPoint(frame.center, frame.right, -26);
      ctx.fillText(String(depth), label.x - 10, label.y + 4);
    }
  }
  ctx.restore();
}

function drawChipTrack(ctx) {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  const layout = chipColumnLayout();
  const left = [];
  const right = [];
  for (let depth = start; depth <= end + 0.001; depth += 0.1) {
    const frame = screenFrameAtDepth(depth);
    if (!frame) continue;
    left.push(lateralPoint(frame, "left", frame.radiusPx + layout.outer));
    right.push(lateralPoint(frame, "left", frame.radiusPx + layout.inner));
  }
  if (left.length < 2) return;
  ctx.save();
  ctx.fillStyle = "rgba(20, 24, 20, 0.72)";
  ctx.strokeStyle = "rgba(230, 235, 220, 0.18)";
  ctx.beginPath();
  ctx.moveTo(left[0].x, left[0].y);
  for (const point of left.slice(1)) ctx.lineTo(point.x, point.y);
  for (const point of right.slice().reverse()) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  const rows = state.chipTrayRows.filter((row) => row.from_m < end && row.to_m > start);
  for (const row of rows) {
    const from = clamp(row.from_m, start, end);
    const to = clamp(row.to_m, start, end);
    const span = Math.max(0.001, row.to_m - row.from_m);
    const cropTop = clamp((from - row.from_m) / span, 0, 1);
    const cropBottom = clamp((to - row.from_m) / span, cropTop, 1);
    const quad = chipColumnQuad(from, to, layout);
    if (!quad) continue;
    const image = chipImage(row);
    if (image) drawChipImage(ctx, image, quad, cropTop, cropBottom);
    else drawChipPlaceholderRow(ctx, quad, row.label || `${from.toFixed(0)}-${to.toFixed(0)} m`);
  }

  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  for (let depth = Math.ceil(start); depth <= end; depth += 1) {
    const frame = screenFrameAtDepth(depth);
    if (!frame) continue;
    const a = lateralPoint(frame, "left", frame.radiusPx + layout.inner);
    const b = lateralPoint(frame, "left", frame.radiusPx + layout.outer);
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  const topFrame = screenFrameAtDepth(start + 0.1);
  if (topFrame) {
    const labelPoint = lateralPoint(topFrame, "left", topFrame.radiusPx + layout.outer + 4);
    ctx.fillStyle = "rgba(230, 235, 220, 0.72)";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(rows.length ? "Chip tray" : "No chip tray images", labelPoint.x, 34);
  }
  ctx.restore();
}

function chipColumnQuad(fromDepth, toDepth, layout = chipColumnLayout()) {
  const topFrame = screenFrameAtDepth(fromDepth);
  const bottomFrame = screenFrameAtDepth(toDepth);
  if (!topFrame || !bottomFrame) return null;
  return {
    topOuter: lateralPoint(topFrame, "left", topFrame.radiusPx + layout.outer),
    topInner: lateralPoint(topFrame, "left", topFrame.radiusPx + layout.inner),
    bottomInner: lateralPoint(bottomFrame, "left", bottomFrame.radiusPx + layout.inner),
    bottomOuter: lateralPoint(bottomFrame, "left", bottomFrame.radiusPx + layout.outer),
  };
}

function chipColumnLayout() {
  const inner = chipColumnInnerPx();
  const width = chipStripWidthPx();
  return { inner, outer: inner + width, width };
}

function chipColumnInnerPx() {
  return state.geometry === "flat" ? 8 : 18;
}

function chipStripWidthPx() {
  return state.geometry === "wrapped" ? 108 : 128;
}

function projectedDepthPixelsPerMeter() {
  const range = getRangeMeters().length;
  const depth = clamp(state.focusDepth, state.meter + 0.05, state.meter + Math.max(0.05, range - 1.05));
  const a = screenFrameAtDepth(depth);
  const b = screenFrameAtDepth(Math.min(state.meter + range, depth + 1));
  if (a && b) return Math.max(1, Math.hypot(a.center.x - b.center.x, a.center.y - b.center.y));
  return Math.max(1, canvas.clientHeight / Math.max(1, range) * 0.82);
}

function chipImageAspect() {
  const ratios = [...state.chipImageCache.values()]
    .filter((cached) => cached.loaded && !cached.failed && cached.image.naturalHeight)
    .map((cached) => cached.image.naturalWidth / cached.image.naturalHeight)
    .sort((a, b) => a - b);
  if (!ratios.length) return 0.42;
  return ratios[Math.floor(ratios.length / 2)];
}

function chipImage(row) {
  let cached = state.chipImageCache.get(row.url);
  if (!cached) {
    const image = new Image();
    cached = { image, loaded: false, failed: false };
    image.onload = () => { cached.loaded = true; };
    image.onerror = () => { cached.failed = true; };
    image.src = row.url;
    state.chipImageCache.set(row.url, cached);
  }
  return cached.loaded && !cached.failed ? cached.image : null;
}

function drawChipImage(ctx, image, quad, cropTop = 0, cropBottom = 1) {
  const imageWidth = image.naturalWidth || image.width;
  const imageHeight = image.naturalHeight || image.height;
  let sourceX = 0;
  let sourceWidth = imageWidth;
  let sourceY = imageHeight * cropTop;
  let sourceHeight = Math.max(1, imageHeight * Math.max(0.001, cropBottom - cropTop));
  const widthAxis = { x: quad.topInner.x - quad.topOuter.x, y: quad.topInner.y - quad.topOuter.y };
  const heightAxis = { x: quad.bottomOuter.x - quad.topOuter.x, y: quad.bottomOuter.y - quad.topOuter.y };
  const destWidth = Math.max(1, Math.hypot(widthAxis.x, widthAxis.y));
  const destHeight = Math.max(1, Math.hypot(heightAxis.x, heightAxis.y));
  const destAspect = destWidth / destHeight;
  const sourceAspect = sourceWidth / sourceHeight;
  if (sourceAspect > destAspect) {
    sourceWidth = Math.max(1, sourceHeight * destAspect);
    sourceX = (imageWidth - sourceWidth) / 2;
  } else if (sourceAspect < destAspect) {
    const targetHeight = Math.max(1, sourceWidth / destAspect);
    const spareHeight = Math.max(0, sourceHeight - targetHeight);
    sourceY += spareHeight / 2;
    sourceHeight = targetHeight;
  }
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(quad.topOuter.x, quad.topOuter.y);
  ctx.lineTo(quad.topInner.x, quad.topInner.y);
  ctx.lineTo(quad.bottomInner.x, quad.bottomInner.y);
  ctx.lineTo(quad.bottomOuter.x, quad.bottomOuter.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(
    widthAxis.x / sourceWidth,
    widthAxis.y / sourceWidth,
    heightAxis.x / sourceHeight,
    heightAxis.y / sourceHeight,
    quad.topOuter.x,
    quad.topOuter.y
  );
  ctx.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight);
  ctx.restore();
}

function drawChipPlaceholderRow(ctx, quad, label) {
  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.04)";
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.beginPath();
  ctx.moveTo(quad.topOuter.x, quad.topOuter.y);
  ctx.lineTo(quad.topInner.x, quad.topInner.y);
  ctx.lineTo(quad.bottomInner.x, quad.bottomInner.y);
  ctx.lineTo(quad.bottomOuter.x, quad.bottomOuter.y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  const midX = (quad.topOuter.x + quad.topInner.x + quad.bottomInner.x + quad.bottomOuter.x) / 4;
  const midY = (quad.topOuter.y + quad.topInner.y + quad.bottomInner.y + quad.bottomOuter.y) / 4;
  ctx.fillStyle = "rgba(230, 235, 220, 0.56)";
  ctx.font = "10px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, midX, midY);
  ctx.restore();
}

function drawObservationPoints(ctx) {
  const rows = state.observations.filter((point) =>
    point.depth >= state.meter && point.depth <= state.meter + getRangeMeters().length
  );
  if (!rows.length) return;
  ctx.save();
  ctx.fillStyle = "#f2d26b";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
  for (const item of rows) {
    const projected = projectTraceDepth(item.depth);
    if (!projected) continue;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function drawPlaneDrawing(ctx) {
  const points = state.planePoints.filter((point) =>
    point.depth >= state.meter && point.depth <= state.meter + getRangeMeters().length
  );
  const visiblePlanes = state.planes.filter((plane) =>
    plane.points.some((point) => point.depth >= state.meter && point.depth <= state.meter + getRangeMeters().length)
  );
  if (!points.length && !state.planeMode && !visiblePlanes.length) return;
  ctx.save();

  for (const plane of visiblePlanes) {
    const color = planeColor(plane.category);
    const planeSelected = isPlaneSelected(plane.id);
    if (plane.fitted) {
      drawPlaneSinusoid(ctx, plane.fitted, {
        color,
        label: plane.category,
        selected: planeSelected,
        orientation: plane.orientation || orientationFromFitted(plane.fitted),
      });
    }
    if (planeSelected) {
      plane.points.forEach((point, index) => {
        drawPlanePoint(ctx, point, color, isSelectedPlaneRef({ kind: "plane-point", planeId: plane.id, index }));
      });
    }
  }

  const draftColor = planeColor(state.planeCategory);
  if (state.planeTool === "draw" || state.planeMode || state.selectedPlaneRef?.kind === "draft-point") {
    points.forEach((point, index) => {
      drawPlanePoint(ctx, point, draftColor, isSelectedPlaneRef({ kind: "draft-point", index }));
    });
  }
  if (state.planePoints.length >= 3) {
    const rawFitted = fitPlaneSinusoid(state.planePoints);
    const fitted = state.manualDraftOrientation && rawFitted
      ? fittedFromOrientation(state.manualDraftOrientation, rawFitted.center)
      : rawFitted;
    if (fitted) {
      drawPlaneSinusoid(ctx, fitted, {
        color: draftColor,
        label: `${state.planeCategory} draft`,
        selected: false,
        orientation: state.manualDraftOrientation || orientationFromFitted(fitted),
      });
    }
  }
  if (state.planeMode) {
    ctx.fillStyle = rgbaFromHex(draftColor, 0.92);
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(`${state.planeCategory} plane drawing`, 16, canvas.clientHeight - 46);
  }
  ctx.restore();
}

function drawPlanePoint(ctx, point, color, selected) {
  const projected = projectSurfacePoint(point.depth, point.angleDeg);
  if (!projected) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = selected ? "#ffffff" : "rgba(0, 0, 0, 0.65)";
  ctx.lineWidth = selected ? 2.4 : 1.4;
  ctx.beginPath();
  ctx.arc(projected.x, projected.y, selected ? 6.2 : 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function isSelectedPlaneRef(ref) {
  const selected = state.selectedPlaneRef;
  if (!selected || selected.kind !== ref.kind) return false;
  if (selected.kind === "draft-point") return selected.index === ref.index;
  if (selected.kind === "plane") return selected.planeId === ref.planeId;
  return selected.planeId === ref.planeId && selected.index === ref.index;
}

function isPlaneSelected(planeId) {
  const selected = state.selectedPlaneRef;
  return !!selected && selected.planeId === planeId;
}

function planeColor(category) {
  return PLANE_CATEGORY_COLORS[category] || PLANE_CATEGORY_COLORS.Bedding;
}

function drawStickyTrackHeaders(ctx) {
  state.trackHeaderHitboxes = [];
  ctx.save();
  ctx.font = "11px Inter, sans-serif";
  ctx.textBaseline = "middle";
  const sideCounts = { left: 0, right: 0 };
  for (const track of state.tracks.filter((item) => item.visible)) {
    const point = trackLateralPoint(
      track,
      state.meter + 0.1,
      trackOffsetPx(track) + maxTrackDrawWidthPx(track) * 0.5
    );
    if (!point) continue;
    const textWidth = ctx.measureText(track.label).width;
    const width = Math.max(76, textWidth + 18);
    const height = 20;
    const labelX = clamp(point.x, width / 2 + 8, canvas.clientWidth - width / 2 - 8);
    const x = labelX - width / 2;
    const y = 9 + sideCounts[track.side] * 24;
    sideCounts[track.side] += 1;
    ctx.fillStyle = "rgba(8, 12, 8, 0.9)";
    ctx.strokeStyle = track.id === state.selectedTrackId ? "rgba(140, 207, 157, 0.92)" : "rgba(230, 235, 220, 0.2)";
    ctx.lineWidth = 1;
    ctx.fillRect(x, y, width, height);
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = track.flatColor;
    ctx.textAlign = "center";
    ctx.fillText(track.label, labelX, y + height / 2);
    state.trackHeaderHitboxes.push({ id: track.id, x, y, width, height });
  }
  ctx.restore();
}

function trackHeaderAtEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return state.trackHeaderHitboxes
    .slice()
    .reverse()
    .find((box) => x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height);
}

function dragTrackHeader(dx) {
  const track = getTrack(state.dragTrackId);
  if (!track || !Number.isFinite(dx)) return;
  const sign = track.side === "right" ? 1 : -1;
  track.offset = clamp(track.offset + (dx * sign) / 105, 0, 4);
  updateStyleEditor();
  updateMetrics();
}

function drawOverview() {
  const ctx = overviewCtx;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = ui.overview.clientWidth;
  const height = ui.overview.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!state.data?.trace?.length) return;

  const projector = traceProfileProjector(width, height);

  ctx.save();
  ctx.strokeStyle = "rgba(230, 235, 220, 0.28)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(projector.centerX, projector.pad);
  ctx.lineTo(projector.centerX, height - projector.pad);
  ctx.stroke();

  ctx.strokeStyle = "rgba(230, 235, 220, 0.78)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  state.data.trace.forEach((row, index) => {
    const p = projector.project(row);
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  const startTrace = tracePointAtDepth(start);
  const endTrace = tracePointAtDepth(end);
  const a = projector.project(startTrace);
  const b = projector.project(endTrace);
  ctx.strokeStyle = "#8ccf9d";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();

  ctx.fillStyle = "#8ccf9d";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(`${start.toFixed(0)}-${end.toFixed(0)} m`, Math.min(width - 64, Math.max(8, (a.x + b.x) / 2)), Math.min(height - 8, Math.max(14, (a.y + b.y) / 2)));
  ctx.fillStyle = "rgba(230, 235, 220, 0.72)";
  ctx.fillText(`${CONFIG.holeId} 0-${formatNumber(projector.maxDepth, 0)} m`, 10, height - 10);
  ctx.restore();
}

function overviewDepthFromEvent(event) {
  if (!state.data?.trace?.length) return NaN;
  const rect = ui.overview.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const points = [];
  const projector = traceProfileProjector(rect.width, rect.height);
  for (let depth = CONFIG.firstMeter; depth <= CONFIG.maxDepthMeter; depth += 0.5) {
    const row = tracePointAtDepth(depth);
    const projected = projector.project(row);
    points.push({
      depth,
      x: projected.x,
      y: projected.y,
    });
  }
  return points.reduce((best, point) => {
    const distance = Math.hypot(point.x - x, point.y - y);
    return distance < best.distance ? { depth: point.depth, distance } : best;
  }, { depth: NaN, distance: Infinity }).depth;
}

function traceProfileProjector(width, height) {
  const trace = state.data?.trace || [];
  const collar = trace[0] || { depth_m: 0, x: 0, y: 0 };
  const end = trace[trace.length - 1] || collar;
  const depths = trace.map((row) => row.depth_m).filter(Number.isFinite);
  const maxDepth = Math.max(CONFIG.maxDepthMeter, ...depths);
  const dx = end.x - collar.x;
  const dy = end.y - collar.y;
  const length = Math.hypot(dx, dy);
  const ux = length > 0.001 ? dx / length : 1;
  const uy = length > 0.001 ? dy / length : 0;
  const pad = 18;
  const verticalScale = (height - pad * 2) / Math.max(1, maxDepth);
  const laterals = trace.map((row) => (row.x - collar.x) * ux + (row.y - collar.y) * uy);
  const maxAbsLateral = Math.max(1, ...laterals.map((value) => Math.abs(value)));
  const lateralScale = Math.min(verticalScale * 1.15, (width * 0.36) / maxAbsLateral);
  const centerX = width * 0.56;
  return {
    pad,
    centerX,
    maxDepth,
    project: (row) => {
      const lateral = (row.x - collar.x) * ux + (row.y - collar.y) * uy;
      return {
        x: centerX + lateral * lateralScale,
        y: pad + (row.depth_m / Math.max(1, maxDepth)) * (height - pad * 2),
      };
    },
  };
}

function addPlanePointFromEvent(event) {
  const hit = pickSurfacePoint(event);
  const depth = hit?.depth ?? nearestTraceDepthAtPointer(event);
  if (!Number.isFinite(depth)) return;
  recordPlaneHistory();
  state.planePoints.push({
    depth,
    angleDeg: hit?.angleDeg ?? 180,
    geometry: state.geometry,
    category: state.planeCategory,
  });
  state.selectedPlaneRef = { kind: "draft-point", index: state.planePoints.length - 1 };
  ui.hudStatus.textContent = `Plane point ${state.planePoints.length} at ${depth.toFixed(2)} m`;
  scheduleAnnotationSave();
  updatePlaneToolbar();
  updateMetrics();
}

function selectPlaneItemFromEvent(event) {
  const rect = canvas.getBoundingClientRect();
  const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const candidates = planeSelectionCandidates();
  const best = candidates.reduce((current, candidate) => {
    const distance = Math.hypot(candidate.x - pointer.x, candidate.y - pointer.y);
    return distance < current.distance ? { ...candidate, distance } : current;
  }, { distance: Infinity });
  if (best.distance > 14) {
    state.selectedPlaneRef = null;
    ui.hudStatus.textContent = "Plane selection cleared";
  } else {
    state.selectedPlaneRef = best.ref;
    ui.hudStatus.textContent = best.label;
  }
  updatePlaneToolbar();
  updateMetrics();
}

function planeSelectionCandidates() {
  const candidates = [];
  state.planePoints.forEach((point, index) => {
    const projected = projectSurfacePoint(point.depth, point.angleDeg);
    if (projected) {
      candidates.push({
        ...projected,
        ref: { kind: "draft-point", index },
        label: `Selected draft point ${index + 1}`,
      });
    }
  });

  for (const plane of state.planes) {
    plane.points.forEach((point, index) => {
      const projected = projectSurfacePoint(point.depth, point.angleDeg);
      if (projected) {
        candidates.push({
          ...projected,
          ref: { kind: "plane-point", planeId: plane.id, index },
          label: `Selected ${plane.category.toLowerCase()} point ${index + 1}`,
        });
      }
    });
    if (!plane.fitted) continue;
    for (let angle = state.geometry === "flat" ? -12 : 0; angle <= (state.geometry === "flat" ? 372 : 360); angle += 12) {
      const theta = degToRad(angle);
      const depth = plane.fitted.center + plane.fitted.cosCoeff * Math.cos(theta) + plane.fitted.sinCoeff * Math.sin(theta);
      if (depth < state.meter || depth > state.meter + getRangeMeters().length) continue;
      const projected = projectSurfacePoint(depth, angle, { extendFlat: state.geometry === "flat" });
      if (projected) {
        candidates.push({
          ...projected,
          ref: { kind: "plane", planeId: plane.id },
          label: `Selected ${plane.category.toLowerCase()} plane`,
        });
      }
    }
  }
  return candidates;
}

function pickSurfacePoint(event) {
  const ray = pointerRay(event);
  if (!ray) return null;
  let best = null;
  for (const mesh of state.meshes) {
    const hit = state.geometry === "flat"
      ? intersectFlatMesh(mesh, ray.near, ray.direction)
      : intersectCylinder(mesh, ray.near, ray.direction);
    if (hit && (!best || hit.t < best.t)) best = hit;
  }
  return best;
}

function intersectFlatMesh(mesh, origin, direction) {
  if (!mesh.flatWidth) return null;
  const normal = mesh.n2;
  const denom = dot(direction, normal);
  if (Math.abs(denom) < 1e-7) return null;
  const t = dot(sub(mesh.center, origin), normal) / denom;
  if (t <= 0.0001) return null;
  const point = add(origin, scale(direction, t));
  const rel = sub(point, mesh.pTop);
  const axial = dot(rel, mesh.tangent);
  const lateral = dot(rel, mesh.n1);
  if (axial < 0 || axial > mesh.length || lateral < -mesh.flatWidth / 2 || lateral > mesh.flatWidth / 2) return null;
  return {
    t,
    depth: mesh.startDepth + (axial / mesh.length) * (mesh.endDepth - mesh.startDepth),
    angleDeg: ((lateral / mesh.flatWidth) + 0.5) * 360,
  };
}

function projectPlanePoint(point) {
  const frame = screenFrameAtDepth(point.depth);
  if (!frame) return null;
  const angleNorm = ((point.angleDeg % 360) + 360) % 360 / 360;
  const lateral = (angleNorm - 0.5) * Math.max(frame.radiusPx * 2, 80);
  return {
    x: frame.center.x + frame.right.x * lateral,
    y: frame.center.y + frame.right.y * lateral,
  };
}

function fitPlaneSinusoid(points) {
  if (points.length < 3) return null;
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const aty = [0, 0, 0];
  for (const point of points) {
    const theta = degToRad(point.angleDeg || 0);
    const row = [1, Math.cos(theta), Math.sin(theta)];
    for (let r = 0; r < 3; r++) {
      aty[r] += row[r] * point.depth;
      for (let c = 0; c < 3; c++) {
        ata[r][c] += row[r] * row[c];
      }
    }
  }
  const coeff = solve3x3(ata, aty);
  if (!coeff) return null;
  const [center, cosCoeff, sinCoeff] = coeff;
  return {
    center,
    cosCoeff,
    sinCoeff,
    amplitude: Math.hypot(cosCoeff, sinCoeff),
    phaseDeg: Math.atan2(sinCoeff, cosCoeff) * 180 / Math.PI,
  };
}

function drawPlaneSinusoid(ctx, fitted, options = {}) {
  const color = options.color || PLANE_CATEGORY_COLORS.Bedding;
  const samples = [];
  const start = state.meter - 0.25;
  const end = state.meter + getRangeMeters().length + 0.25;
  const startAngle = state.geometry === "flat" ? -12 : 0;
  const endAngle = state.geometry === "flat" ? 372 : 360;
  for (let angle = startAngle; angle <= endAngle; angle += 4) {
    const theta = degToRad(angle);
    const depth = fitted.center + fitted.cosCoeff * Math.cos(theta) + fitted.sinCoeff * Math.sin(theta);
    if (depth < start || depth > end) continue;
    const projected = projectSurfacePoint(depth, angle, { extendFlat: state.geometry === "flat" });
    if (projected) samples.push({
      ...projected,
      front: state.geometry !== "wrapped" || isWrappedAngleFront(depth, angle),
    });
  }
  if (samples.length < 2) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  drawPlaneSampleSegments(ctx, samples, color, options.selected ? 13 : 9, options.selected ? 0.36 : 0.22);
  drawPlaneSampleSegments(ctx, samples, color, options.selected ? 3 : 2, 0.94);

  const label = samples[Math.floor(samples.length / 2)];
  ctx.fillStyle = rgbaFromHex(color, 0.96);
  ctx.font = "11px Inter, sans-serif";
  const orientation = options.orientation || orientationFromFitted(fitted);
  ctx.fillText(`${options.label || "Plane"} ${Math.round(orientation.azimuth)} / ${Math.round(orientation.dip)}`, label.x + 8, label.y - 8);
  ctx.restore();
}

function drawPlaneSampleSegments(ctx, samples, color, width, alpha) {
  ctx.lineWidth = width;
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1];
    const b = samples[i];
    const front = a.front && b.front;
    ctx.setLineDash(front ? [] : [4, 6]);
    ctx.strokeStyle = front ? rgbaFromHex(color, alpha) : `rgba(185, 205, 225, ${alpha * 0.2})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  ctx.setLineDash([]);
}

function isWrappedAngleFront(depth, angleDeg) {
  const mesh = state.meshes.find((candidate) => depth >= candidate.startDepth && depth <= candidate.endDepth);
  if (!mesh || !state.camera?.radial) return true;
  const theta = degToRad(angleDeg) - state.surfaceYaw;
  const radial = normalize(add(scale(mesh.n1, Math.cos(theta)), scale(mesh.n2, Math.sin(theta))));
  return dot(radial, state.camera.radial) > 0;
}

function projectSurfacePoint(depth, angleDeg, options = {}) {
  const mesh = state.meshes.find((candidate) => depth >= candidate.startDepth && depth <= candidate.endDepth);
  if (!mesh) return projectPlanePoint({ depth, angleDeg });
  const axial = (depth - mesh.startDepth) / Math.max(0.0001, mesh.endDepth - mesh.startDepth);
  const center = lerpVec(mesh.pTop, mesh.pBottom, axial);
  const angleNorm = options.extendFlat ? angleDeg / 360 : (((angleDeg % 360) + 360) % 360) / 360;
  let worldPoint;
  if (state.geometry === "flat" && mesh.flatWidth) {
    worldPoint = add(center, scale(mesh.n1, (angleNorm - 0.5) * mesh.flatWidth));
  } else {
    const theta = degToRad(angleDeg) - state.surfaceYaw;
    worldPoint = add(center, add(scale(mesh.n1, Math.cos(theta) * mesh.radius * 1.035), scale(mesh.n2, Math.sin(theta) * mesh.radius * 1.035)));
  }
  return projectPoint(worldPoint);
}

function solve3x3(a, b) {
  const m = a.map((row, index) => [...row, b[index]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-9) return null;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const scaleValue = m[col][col];
    for (let c = col; c < 4; c++) m[col][c] /= scaleValue;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = m[row][col];
      for (let c = col; c < 4; c++) m[row][c] -= factor * m[col][c];
    }
  }
  return [m[0][3], m[1][3], m[2][3]];
}

function drawLineTrack(ctx, track) {
  const rows = visibleLineRows(track);
  if (rows.length < 2) return;
  drawTrackBaseline(ctx, track);

  const points = rows.map((row) => trackPoint(track, row.depth_m, row.displayValue)).filter(Boolean);
  if (points.length < 2) return;

  ctx.save();
  ctx.lineWidth = track.lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  if (track.colorMode === "colormap") {
    for (let i = 1; i < points.length; i++) {
      ctx.strokeStyle = colorForValue(track, rows[i].displayValue);
      ctx.beginPath();
      ctx.moveTo(points[i - 1].x, points[i - 1].y);
      ctx.lineTo(points[i].x, points[i].y);
      ctx.stroke();
    }
  } else {
    ctx.strokeStyle = track.flatColor;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
    ctx.stroke();
  }
  ctx.restore();
  drawTrackLabel(ctx, track);
}

function visibleLineRows(track) {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  const all = processedPointRows(track)
    .filter((row) => Number.isFinite(row.depth_m) && Number.isFinite(row.displayValue))
    .sort((a, b) => a.depth_m - b.depth_m);
  const inside = all.filter((row) => row.depth_m > start && row.depth_m < end);
  const startRow = interpolatedDisplayRow(all, start);
  const endRow = interpolatedDisplayRow(all, end);
  return [startRow, ...inside, endRow]
    .filter(Boolean)
    .filter((row, index, rows) => index === 0 || Math.abs(row.depth_m - rows[index - 1].depth_m) > 0.0001);
}

function interpolatedDisplayRow(rows, depth) {
  if (!rows.length) return null;
  if (depth <= rows[0].depth_m) return { ...rows[0], depth_m: depth };
  if (depth >= rows[rows.length - 1].depth_m) return { ...rows[rows.length - 1], depth_m: depth };
  for (let i = 0; i < rows.length - 1; i++) {
    const a = rows[i];
    const b = rows[i + 1];
    if (depth >= a.depth_m && depth <= b.depth_m) {
      const t = (depth - a.depth_m) / Math.max(0.0001, b.depth_m - a.depth_m);
      return {
        ...a,
        depth_m: depth,
        displayValue: lerp(a.displayValue, b.displayValue, t),
      };
    }
  }
  return null;
}

function drawMidpointTrack(ctx, track) {
  const rows = processedPointRows(track)
    .filter((row) => row.depth_m >= state.meter && row.depth_m <= state.meter + getRangeMeters().length)
    .filter((row) => Number.isFinite(row.displayValue));
  if (!rows.length) return;
  drawTrackBaseline(ctx, track);

  ctx.save();
  for (const row of rows) {
    const point = trackPoint(track, row.depth_m, row[track.key]);
    if (!point) continue;
    ctx.fillStyle = colorForValue(track, row.displayValue);
    ctx.beginPath();
    ctx.arc(point.x, point.y, Math.max(2, track.lineWidth + 1.5), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  drawTrackLabel(ctx, track);
}

function processedPointRows(track) {
  const baseRows = track.rows
    .filter((row) => Number.isFinite(row.depth_m) && Number.isFinite(row[track.key]))
    .map((row) => ({ ...row, rawValue: row[track.key] }));
  const smoothWindow = track.smoothing === "none" ? 0 : Number(track.smoothing);
  const smoothed = smoothWindow > 0
    ? baseRows.map((row) => {
      const half = smoothWindow / 2;
      const nearby = baseRows.filter((candidate) => Math.abs(candidate.depth_m - row.depth_m) <= half);
      const value = nearby.reduce((sum, candidate) => sum + candidate.rawValue, 0) / Math.max(1, nearby.length);
      return { ...row, rawValue: value };
    })
    : baseRows;

  const values = smoothed.map((row) => row.rawValue).filter(Number.isFinite);
  const stats = valueStats(values);
  return smoothed.map((row) => ({
    ...row,
    displayValue: normalisedDisplayValue(row.rawValue, track, stats),
  }));
}

function valueStats(values) {
  if (!values.length) return { min: 0, max: 1, mean: 0, sd: 1, p5: 0, p95: 1 };
  const sorted = values.slice().sort((a, b) => a - b);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, values.length - 1);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean,
    sd: Math.sqrt(variance) || 1,
    p5: sorted[Math.floor((sorted.length - 1) * 0.05)],
    p95: sorted[Math.floor((sorted.length - 1) * 0.95)],
  };
}

function normalisedDisplayValue(value, track, stats) {
  if (track.normalise === "minmax") {
    return (value - stats.min) / Math.max(0.000001, stats.max - stats.min);
  }
  if (track.normalise === "zscore") {
    return (value - stats.mean) / stats.sd;
  }
  if (track.normalise === "percentile" || track.normalise === "gamma_api") {
    const clipped = clamp(value, stats.p5, stats.p95);
    return (clipped - stats.p5) / Math.max(0.000001, stats.p95 - stats.p5);
  }
  if (track.normalise === "magsus_log") {
    return Math.log10(Math.max(0.000001, value));
  }
  return value;
}

function drawBarTrack(ctx, track) {
  const rows = track.rows.filter((row) =>
    row.from_m < state.meter + getRangeMeters().length &&
    row.to_m > state.meter &&
    Number.isFinite(row[track.key])
  );
  if (!rows.length) return;
  drawTrackBaseline(ctx, track);

  ctx.save();
  ctx.globalAlpha = 0.96;
  for (const row of rows) {
    drawIntervalBar(ctx, track, row.from_m, row.to_m, row[track.key], colorForValue(track, row[track.key]));
  }
  ctx.restore();
  drawTrackLabel(ctx, track);
}

function drawStackedTrack(ctx, track) {
  const rows = track.rows.filter((row) =>
    row.from_m < state.meter + getRangeMeters().length &&
    row.to_m > state.meter
  );
  if (!rows.length) return;
  drawTrackBaseline(ctx, track);

  ctx.save();
  ctx.globalAlpha = 0.98;
  for (const row of rows) {
    const parts = MINERAL_KEYS.map((key) => ({
      key,
      value: toNumber(row[key]) || 0,
      color: MINERAL_COLORS[key] || MINERAL_COLORS.OTHER,
    })).filter((part) => part.value > 0.05);
    const total = Math.max(1, parts.reduce((sum, part) => sum + part.value, 0));
    let startNorm = 0;
    for (const part of parts) {
      const endNorm = startNorm + part.value / total;
      drawStackSegment(ctx, track, row.from_m, row.to_m, startNorm, endNorm, part.color);
      startNorm = endNorm;
    }
  }
  ctx.restore();
  drawTrackLabel(ctx, track);
}

function drawTrackBaseline(ctx, track) {
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  const offsetPx = trackOffsetPx(track);
  const points = [];
  for (let depth = start; depth <= end + 0.001; depth += 0.1) {
    const point = trackLateralPoint(track, depth, offsetPx);
    if (point) points.push(point);
  }
  if (points.length < 2) return;

  ctx.save();
  ctx.strokeStyle = "rgba(225, 230, 220, 0.42)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.stroke();
  ctx.restore();
}

function drawIntervalBar(ctx, track, fromDepth, toDepth, value, color) {
  const from = clamp(fromDepth, state.meter, state.meter + getRangeMeters().length);
  const to = clamp(toDepth, state.meter, state.meter + getRangeMeters().length);
  const base = trackOffsetPx(track);
  const width = normalizedValue(track, value) * trackScalePx(track);
  const a = trackLateralPoint(track, from, base);
  const b = trackLateralPoint(track, to, base);
  const c = trackLateralPoint(track, to, base + width);
  const d = trackLateralPoint(track, from, base + width);
  if (!a || !b || !c || !d) return;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.lineTo(d.x, d.y);
  ctx.closePath();
  ctx.fill();
}

function drawStackSegment(ctx, track, fromDepth, toDepth, startNorm, endNorm, color) {
  const from = clamp(fromDepth, state.meter, state.meter + getRangeMeters().length);
  const to = clamp(toDepth, state.meter, state.meter + getRangeMeters().length);
  const base = trackOffsetPx(track);
  const scalePx = trackScalePx(track);
  const a = trackLateralPoint(track, from, base + startNorm * scalePx);
  const b = trackLateralPoint(track, to, base + startNorm * scalePx);
  const c = trackLateralPoint(track, to, base + endNorm * scalePx);
  const d = trackLateralPoint(track, from, base + endNorm * scalePx);
  if (!a || !b || !c || !d) return;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineTo(c.x, c.y);
  ctx.lineTo(d.x, d.y);
  ctx.closePath();
  ctx.fill();
}

function drawTrackLabel(ctx, track) {
  void ctx;
  void track;
}

function trackPoint(track, depth, value) {
  const base = trackOffsetPx(track);
  const width = normalizedValue(track, value) * trackScalePx(track);
  return trackLateralPoint(track, depth, base + width);
}

function trackLateralPoint(track, depth, lateralPx) {
  const frame = screenFrameAtDepth(depth);
  if (!frame) return null;
  return lateralPoint(frame, track.side, frame.radiusPx + trackClearancePx(track.side) + lateralPx + trackReservePx(track.side));
}

function trackReservePx(side) {
  return side === "left" && shouldDrawChipTrack() ? chipColumnLayout().outer + 10 : 0;
}

function trackClearancePx(side) {
  void side;
  const fitDistance = Math.max(minZoomInDistance(), maxZoomOutDistance());
  const zoomRatio = fitDistance / Math.max(state.distance, 0.001);
  return Math.min(46, 8 + Math.max(0, zoomRatio - 1) * 6);
}

function lateralPoint(frame, side, lateralPx) {
  const sign = side === "right" ? 1 : -1;
  return {
    x: frame.center.x + frame.right.x * lateralPx * sign,
    y: frame.center.y + frame.right.y * lateralPx * sign,
  };
}

function offsetPoint(point, normal, distance) {
  return { x: point.x + normal.x * distance, y: point.y + normal.y * distance };
}

function trackOffsetPx(track) {
  return 12 + track.offset * 105;
}

function trackScalePx(track) {
  return track.scale * 130;
}

function normalizedValue(track, rawValue) {
  let value = rawValue;
  let min = Number.isFinite(track.displayMin) ? track.displayMin : track.dataMin;
  let max = Number.isFinite(track.displayMax) ? track.displayMax : track.dataMax;

  if (track.normalise === "minmax" || track.normalise === "percentile" || track.normalise === "gamma_api") {
    min = 0;
    max = 1;
  } else if (track.normalise === "zscore") {
    min = -2;
    max = 2;
  } else if (track.normalise === "magsus_log") {
    min = Math.log10(Math.max(0.000001, track.dataMin));
    max = Math.log10(Math.max(0.000001, track.dataMax));
  }

  if (track.log) {
    value = Math.log10(Math.max(0.000001, value));
    min = Math.log10(Math.max(0.000001, min));
    max = Math.log10(Math.max(0.000001, max));
  }

  if (!Number.isFinite(value) || !Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-9) return 0;
  const normal = (value - min) / (max - min);
  return track.clamp ? clamp(normal, 0, 1) : clamp(normal, 0, 1.25);
}

function colorForValue(track, value) {
  if (track.colorMode === "flat") return track.flatColor;
  const normal = clamp(normalizedValue(track, value), 0, 1);
  if (track.colormap === "iron") return rampColor(normal, ["#006c9c", "#42a53a", "#ffdf3a", "#ff861f", "#f0431e"]);
  if (track.colormap === "silica") return rampColor(normal, ["#233b69", "#4c8bc0", "#d7d9c9", "#f0c565"]);
  return rampColor(normal, ["#3e50b7", "#00a2d5", "#44b15d", "#f0d34b", "#f15a24"]);
}

function rampColor(t, colors) {
  const scaled = clamp(t, 0, 1) * (colors.length - 1);
  const index = Math.floor(scaled);
  const next = Math.min(colors.length - 1, index + 1);
  const local = scaled - index;
  const a = hexToRgb(colors[index]);
  const b = hexToRgb(colors[next]);
  return `rgb(${Math.round(lerp(a[0], b[0], local))}, ${Math.round(lerp(a[1], b[1], local))}, ${Math.round(lerp(a[2], b[2], local))})`;
}

function hexToRgb(hex) {
  const raw = hex.replace("#", "");
  return [
    parseInt(raw.slice(0, 2), 16),
    parseInt(raw.slice(2, 4), 16),
    parseInt(raw.slice(4, 6), 16),
  ];
}

function rgbaFromHex(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function screenFrameAtDepth(depth) {
  const center = projectTraceDepth(depth);
  const before = projectTraceDepth(Math.max(state.meter, depth - 0.08));
  const after = projectTraceDepth(Math.min(state.meter + getRangeMeters().length, depth + 0.08));
  if (!center || !before || !after) return null;
  if (state.geometry === "flat") {
    const mesh = state.meshes.find((candidate) => depth >= candidate.startDepth && depth <= candidate.endDepth);
    if (mesh?.n1) {
      const edge = projectPoint(add(localPointAtDepth(depth), scale(mesh.n1, 1)));
      if (edge) {
        let right = normalize2({ x: edge.x - center.x, y: edge.y - center.y });
        if (right.x < 0) right = { x: -right.x, y: -right.y };
        const tangent = normalize2({ x: -right.y, y: right.x });
        return { center, tangent, right, radiusPx: projectedOtvHalfWidth(depth, right) };
      }
    }
  }
  const tangent = normalize2({ x: after.x - before.x, y: after.y - before.y });
  const right = normalize2({ x: tangent.y, y: -tangent.x });
  return { center, tangent, right, radiusPx: projectedOtvHalfWidth(depth, right) };
}

function projectedOtvHalfWidth(depth, right) {
  const radius = (state.diameterM * state.sceneScale) / 2;
  const center3 = localPointAtDepth(depth);
  const axis = viewAxis();
  const radialFrame = viewRadialFrame(axis);
  const mesh = state.meshes.find((candidate) => depth >= candidate.startDepth && depth <= candidate.endDepth);
  const width = mesh?.flatWidth || Math.max(radius * 2, 0.12 * state.sceneScale);
  const lateralWorld = state.geometry === "flat" && mesh?.n1
    ? scale(mesh.n1, width / 2)
    : state.geometry === "flat"
    ? scale(radialFrame.right, width / 2)
    : scale(radialFrame.right, radius);
  const edgeA = projectPoint(add(center3, lateralWorld));
  const edgeB = projectPoint(add(center3, scale(lateralWorld, -1)));
  const center = projectPoint(center3);
  if (!edgeA || !edgeB || !center) return state.geometry === "flat" ? 64 : 18;
  const a = Math.abs((edgeA.x - center.x) * right.x + (edgeA.y - center.y) * right.y);
  const b = Math.abs((edgeB.x - center.x) * right.x + (edgeB.y - center.y) * right.y);
  return Math.max(state.geometry === "flat" ? 34 : 12, a, b);
}

function projectTraceDepth(depth) {
  return projectPoint(localPointAtDepth(depth));
}

function projectPoint(point) {
  if (!state.camera?.vp) return null;
  const clip = transformVec4(state.camera.vp, [point[0], point[1], point[2], 1]);
  if (!clip[3] || clip[3] <= 0) return null;
  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];
  return {
    x: (ndcX * 0.5 + 0.5) * canvas.clientWidth,
    y: (1 - (ndcY * 0.5 + 0.5)) * canvas.clientHeight,
  };
}

function updateHover(event) {
  if (!state.camera?.inverseVp || !state.meshes.length) return;
  const depth = nearestTraceDepthAtPointer(event);
  let best = null;

  const ray = pointerRay(event);
  if (ray) {
    for (const mesh of state.meshes) {
      const hit = intersectCylinder(mesh, ray.near, ray.direction);
      if (hit && (!best || hit.t < best.t)) best = hit;
    }
  }

  const readDepth = best?.depth ?? depth;
  if (!Number.isFinite(readDepth)) {
    ui.hoverReadout.textContent = "Hover cylinder or graph track for depth";
    return;
  }

  const trace = tracePointAtDepth(readDepth);
  const wall = best ? ` | wall ${best.angleDeg.toFixed(0)} deg` : "";
  ui.hoverReadout.textContent =
    `Depth ${readDepth.toFixed(3)} m | X ${trace.x.toFixed(2)} Y ${trace.y.toFixed(2)} RL ${trace.z.toFixed(2)}${wall}`;
}

function nearestTraceDepthAtPointer(event) {
  if (!state.camera) return NaN;
  const rect = canvas.getBoundingClientRect();
  const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
  const start = state.meter;
  const end = state.meter + getRangeMeters().length;
  let bestDepth = NaN;
  let bestDistance = Infinity;
  const step = Math.max(0.025, Math.min(0.1, (end - start) / 120));

  for (let depth = start; depth <= end + 0.0001; depth += step) {
    const projected = projectTraceDepth(depth);
    if (!projected) continue;
    const distance = Math.hypot(projected.x - point.x, projected.y - point.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestDepth = depth;
    }
  }
  return bestDepth;
}

function pointerRay(event) {
  if (!state.camera?.inverseVp) return null;
  const rect = canvas.getBoundingClientRect();
  const ndcX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  const ndcY = 1 - ((event.clientY - rect.top) / rect.height) * 2;
  const near = unproject(ndcX, ndcY, -1);
  const far = unproject(ndcX, ndcY, 1);
  return { near, direction: normalize(sub(far, near)) };
}

function unproject(ndcX, ndcY, ndcZ) {
  const point = transformVec4(state.camera.inverseVp, [ndcX, ndcY, ndcZ, 1]);
  const w = point[3] || 1;
  return [point[0] / w, point[1] / w, point[2] / w];
}

function intersectCylinder(mesh, origin, direction) {
  const oc = sub(origin, mesh.pTop);
  const dDotAxis = dot(direction, mesh.tangent);
  const ocDotAxis = dot(oc, mesh.tangent);
  const dPerp = sub(direction, scale(mesh.tangent, dDotAxis));
  const ocPerp = sub(oc, scale(mesh.tangent, ocDotAxis));
  const a = dot(dPerp, dPerp);
  if (a < 1e-8) return null;
  const b = 2 * dot(dPerp, ocPerp);
  const c = dot(ocPerp, ocPerp) - mesh.radius * mesh.radius;
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return null;

  const root = Math.sqrt(discriminant);
  const candidates = [(-b - root) / (2 * a), (-b + root) / (2 * a)].filter((t) => t > 0.0001);
  candidates.sort((aValue, bValue) => aValue - bValue);

  for (const t of candidates) {
    const point = add(origin, scale(direction, t));
    const axial = dot(sub(point, mesh.pTop), mesh.tangent);
    if (axial < 0 || axial > mesh.length) continue;
    const center = add(mesh.pTop, scale(mesh.tangent, axial));
    const radial = sub(point, center);
    let theta = Math.atan2(dot(radial, mesh.n2), dot(radial, mesh.n1));
    if (theta < 0) theta += Math.PI * 2;
    return {
      t,
      depth: mesh.startDepth + (axial / mesh.length) * (mesh.endDepth - mesh.startDepth),
      angleDeg: theta * 180 / Math.PI,
    };
  }

  return null;
}

function updateTraceSummary() {
  const collar = state.data.collar?.[0];
  const comparison = state.data.comparison || {};
  const tilt = comparison.tiltDeltaDeg?.median;
  const azimuth = comparison.azimuthDeltaDeg?.median;
  ui.traceSummary.innerHTML =
    `Trace ${formatNumber(toNumber(collar?.DEPTH), 0)} m, ${state.data.survey.length} survey stations<br>` +
    `TFD delta med: tilt ${formatNumber(tilt, 2)} deg, az ${formatNumber(azimuth, 2)} deg`;
}

function updateMetrics(options = {}) {
  if (!options.skipCameraSync) {
    resizeCanvas();
    computeMvp();
  }
  const meters = getRangeMeters();
  const start = state.meter;
  const end = state.meter + meters.length;
  const loaded = state.meshes.filter((mesh) => mesh.texture).length;
  const firstSize = state.meshes.find((mesh) => mesh.texture)?.imageSize || { width: CONFIG.textureWidthPx, height: CONFIG.resampledHeightPx };
  const circumference = Math.PI * state.diameterM;
  const comparison = state.data?.comparison || {};
  const visibleTracks = state.tracks.filter((track) => track.visible).length;
  const sideDataVisible = shouldDrawSideData();
  const renderedTracks = sideDataVisible ? visibleTracks : 0;

  ui.hudSegment.textContent = `${CONFIG.holeId} ${pad3(start)}-${pad3(end)} m`;
  ui.hudScale.textContent = `${meters.length} m, ${state.geometry === "flat" ? "flat" : "wrapped"} viewport`;
  ui.metricTexture.textContent = shouldRenderOtvTextures()
    ? `${loaded}/${state.meshes.length} loaded, ${firstSize.width} x ${firstSize.height} px`
    : "paused for summary range";
  ui.metricVertical.textContent = state.mode === "resampled" ? "500 px/m, 0.002 m/px" : "raw rows by record";
  ui.metricCircumference.textContent = `${circumference.toFixed(3)} m`;
  ui.metricRadius.textContent = `${(state.diameterM / 2).toFixed(3)} m`;
  ui.metricCoverage.textContent = `${start.toFixed(3)}-${end.toFixed(3)} m`;
  ui.metricLayers.textContent = state.data
    ? `${renderedTracks}/${state.tracks.length} tracks rendered, ${state.data.geophysics.length} geo, ${state.data.assays.length} assay`
    : "Loading";
  ui.metricAgreement.textContent = comparison.tiltDeltaDeg
    ? `tilt med ${comparison.tiltDeltaDeg.median.toFixed(2)} deg, az med ${comparison.azimuthDeltaDeg.median.toFixed(2)} deg`
    : "Loading";
  ui.metricFocus.textContent = `${state.focusDepth.toFixed(3)} m`;
  updatePlaneToolbar();

  canvas.dataset.ready = loaded === state.meshes.length && (loaded > 0 || !shouldRenderOtvTextures()) ? "true" : "false";
  canvas.dataset.segment = `${state.meter}`;
  canvas.dataset.range = `${meters.length}`;
  canvas.dataset.mode = state.mode;
  canvas.dataset.geometry = state.geometry;
  canvas.dataset.coverage = `${start}-${end}`;
  canvas.dataset.tracks = `${visibleTracks}`;
  canvas.dataset.renderedTracks = `${renderedTracks}`;
  canvas.dataset.sideDataVisible = sideDataVisible ? "true" : "false";
  canvas.dataset.sideDataVisibleDepthM = `${sideDataVisibleDepthM().toFixed(2)}`;
  canvas.dataset.sideDataThresholdM = `${sideDataZoomThresholdM().toFixed(2)}`;
  canvas.dataset.planeFocusMode = isPlaneFocusMode() ? "true" : "false";
  canvas.dataset.focusDepth = `${state.focusDepth.toFixed(3)}`;
  canvas.dataset.cameraDistance = `${state.distance.toFixed(3)}`;
  canvas.dataset.lateralPanM = `${state.lateralPanM.toFixed(4)}`;
  canvas.dataset.depthScrollCarryM = `${state.depthScrollCarryM.toFixed(4)}`;
  canvas.dataset.orbitYaw = `${state.orbitYaw.toFixed(4)}`;
  canvas.dataset.surfaceYaw = `${state.surfaceYaw.toFixed(4)}`;
  canvas.dataset.surfaceUOffset = `${surfaceUOffset().toFixed(4)}`;
  canvas.dataset.meshes = `${state.meshes.length}`;
  canvas.dataset.textures = shouldRenderOtvTextures() ? `${loaded}/${state.meshes.length}` : "paused";
  canvas.dataset.planeMode = state.planeMode ? "true" : "false";
  canvas.dataset.planeTool = state.planeTool;
  canvas.dataset.planeCategory = state.planeCategory;
  canvas.dataset.planePoints = `${state.planePoints.length}`;
  canvas.dataset.planes = `${state.planes.length}`;
  canvas.dataset.selectedPlane = state.selectedPlaneRef ? JSON.stringify(state.selectedPlaneRef) : "";
  canvas.dataset.planeOrientations = JSON.stringify(state.planes.map((plane) => ({
    id: plane.id,
    category: plane.category,
    azimuth: Number((plane.orientation?.azimuth ?? NaN).toFixed?.(2)),
    dip: Number((plane.orientation?.dip ?? NaN).toFixed?.(2)),
  })));
  canvas.dataset.visiblePlaneHandles = `${visiblePlaneHandleCount()}`;
  canvas.dataset.observations = `${state.observations.length}`;
  canvas.dataset.annotationStatus = state.annotationStatus;
  canvas.dataset.chipRows = `${state.chipTrayRows.length}`;
  canvas.dataset.chipVisibleRows = `${state.chipTrayRows.filter((row) => row.from_m < end && row.to_m > start).length}`;
  canvas.dataset.chipStatus = state.chipTrayStatus;
  canvas.dataset.chipWidthPx = shouldDrawChipTrack() ? chipColumnLayout().width.toFixed(2) : "0";
  canvas.dataset.chipTrackRendered = shouldDrawChipTrack() ? "true" : "false";
  canvas.dataset.trackHeaderHitboxes = JSON.stringify(state.trackHeaderHitboxes.map((box) => ({
    id: box.id,
    x: Number(box.x.toFixed(1)),
    y: Number(box.y.toFixed(1)),
    width: Number(box.width.toFixed(1)),
    height: Number(box.height.toFixed(1)),
  })));
  const snapshot = viewerSnapshot();
  const selected = selectedTrack();
  canvas.dataset.selectedTrack = selected?.id || "";
  canvas.dataset.selectedTrackSmoothing = selected?.smoothing || "";
  canvas.dataset.selectedTrackNormalise = selected?.normalise || "";
  const layoutSafety = trackLayoutSafety();
  canvas.dataset.trackAnchors = JSON.stringify((sideDataVisible ? snapshot.tracks : []).map((track) => ({
    id: track.id,
    side: track.side,
    edgeX: Number(track.edge?.x?.toFixed(2)),
    edgeY: Number(track.edge?.y?.toFixed(2)),
    baselineX: Number(track.baseline?.x?.toFixed(2)),
    baselineY: Number(track.baseline?.y?.toFixed(2)),
    innerClearancePx: Number(track.innerClearancePx?.toFixed(2)),
    edgeToBaselinePx: Number(track.edgeToBaselinePx?.toFixed(2)),
    baselineToOuterPx: Number(track.baselineToOuterPx?.toFixed(2)),
    overlapFree: track.overlapFree,
  })));
  canvas.dataset.trackLayoutSafety = JSON.stringify(sideDataVisible ? layoutSafety : []);
  canvas.dataset.trackOverlapFree = !sideDataVisible
    ? "hidden"
    : layoutSafety.length === visibleTracks && layoutSafety.every((track) => track.overlapFree) ? "true" : "false";
}

function visiblePlaneHandleCount() {
  if (state.planeTool === "draw" || state.planeMode || state.selectedPlaneRef?.kind === "draft-point") {
    return state.planePoints.length;
  }
  if (state.selectedPlaneRef?.planeId) {
    const plane = state.planes.find((candidate) => candidate.id === state.selectedPlaneRef.planeId);
    return plane?.points.length || 0;
  }
  return 0;
}

function viewerSnapshot() {
  const range = getRangeMeters().length;
  const probeDepth = clamp(state.focusDepth, state.meter + 0.05, state.meter + Math.max(0.05, range - 0.05));
  const frame = screenFrameAtDepth(probeDepth);
  const tracks = frame
    ? state.tracks.filter((track) => track.visible).map((track) => {
      const edge = lateralPoint(frame, track.side, frame.radiusPx);
      const baseline = trackLateralPoint(track, probeDepth, trackOffsetPx(track));
      const outer = trackLateralPoint(track, probeDepth, trackOffsetPx(track) + maxTrackDrawWidthPx(track));
      const outward = trackOutwardVector(frame, track.side);
      const innerClearancePx = edge && baseline
        ? (baseline.x - edge.x) * outward.x + (baseline.y - edge.y) * outward.y
        : null;
      return {
        id: track.id,
        label: track.label,
        side: track.side,
        displayType: track.displayType,
        edge,
        baseline,
        outer,
        innerClearancePx,
        overlapFree: Number.isFinite(innerClearancePx) ? innerClearancePx >= -0.5 : false,
        edgeToBaselinePx: edge && baseline ? Math.hypot(edge.x - baseline.x, edge.y - baseline.y) : null,
        baselineToOuterPx: baseline && outer ? Math.hypot(outer.x - baseline.x, outer.y - baseline.y) : null,
      };
    })
    : [];

  return {
    geometry: state.geometry,
    mode: state.mode,
    meter: state.meter,
    rangeCount: state.rangeCount,
    focusDepth: Number(state.focusDepth.toFixed(3)),
    cameraDistance: Number(state.distance.toFixed(3)),
    lateralPanM: Number(state.lateralPanM.toFixed(4)),
    depthScrollCarryM: Number(state.depthScrollCarryM.toFixed(4)),
    surfaceYaw: Number(state.surfaceYaw.toFixed(4)),
    surfaceUOffset: Number(surfaceUOffset().toFixed(4)),
    renderOtvTextures: shouldRenderOtvTextures(),
    meshCount: state.meshes.length,
    planeMode: state.planeMode,
    planeTool: state.planeTool,
    planeCategory: state.planeCategory,
    planePoints: state.planePoints.length,
    planes: state.planes.length,
    selectedPlaneRef: state.selectedPlaneRef,
    observations: state.observations.length,
    probeDepth: Number(probeDepth.toFixed(3)),
    frame: frame
      ? {
        center: frame.center,
        right: frame.right,
        radiusPx: frame.radiusPx,
      }
      : null,
    tracks,
  };
}

function trackLayoutSafety() {
  return state.tracks.filter((track) => track.visible).map((track) => {
    const reservePx = trackReservePx(track.side);
    const offsetPx = trackOffsetPx(track);
    const clearancePx = reservePx + offsetPx;
    const maxWidthPx = maxTrackDrawWidthPx(track);
    return {
      id: track.id,
      side: track.side,
      reservePx: Number(reservePx.toFixed(2)),
      offsetPx: Number(offsetPx.toFixed(2)),
      innerClearancePx: Number(clearancePx.toFixed(2)),
      maxDrawWidthPx: Number(maxWidthPx.toFixed(2)),
      overlapFree: clearancePx >= -0.5 && maxWidthPx >= 0,
    };
  });
}

function maxTrackDrawWidthPx(track) {
  if (track.displayType === "stacked") return trackScalePx(track);
  return trackScalePx(track) * (track.clamp ? 1 : 1.25);
}

function trackOutwardVector(frame, side) {
  const sign = side === "right" ? 1 : -1;
  return {
    x: frame.right.x * sign,
    y: frame.right.y * sign,
  };
}

function getRangeMeters() {
  const maxCount = Math.max(1, CONFIG.maxDepthMeter - state.meter);
  const count = Math.min(state.rangeCount, maxCount);
  return Array.from({ length: count }, (_, index) => state.meter + index);
}

function maxStartForRange() {
  return Math.max(CONFIG.firstMeter, CONFIG.maxDepthMeter - state.rangeCount);
}

function tracePointAtDepth(depth) {
  const trace = state.data?.trace || [];
  if (!trace.length) return { depth_m: depth, x: 0, y: 0, z: -depth };
  if (depth <= trace[0].depth_m) return interpolateTrace(trace[0], trace[1] || trace[0], depth);
  for (let i = 0; i < trace.length - 1; i++) {
    const a = trace[i];
    const b = trace[i + 1];
    if (depth >= a.depth_m && depth <= b.depth_m) return interpolateTrace(a, b, depth);
  }
  return interpolateTrace(trace[trace.length - 2] || trace[0], trace[trace.length - 1], depth);
}

function interpolateTrace(a, b, depth) {
  const span = Math.max(0.0001, b.depth_m - a.depth_m);
  const t = (depth - a.depth_m) / span;
  return {
    depth_m: depth,
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
    z: lerp(a.z, b.z, t),
  };
}

function traceAzimuthAtDepth(depth) {
  const before = tracePointAtDepth(Math.max(CONFIG.firstMeter, depth - 0.5));
  const after = tracePointAtDepth(Math.min(CONFIG.maxDepthMeter, depth + 0.5));
  const dx = after.x - before.x;
  const dy = after.y - before.y;
  if (Math.hypot(dx, dy) < 0.0001) return 0;
  return mod(radToDeg(Math.atan2(dx, dy)), 360);
}

function localPointAtDepth(depth) {
  const point = tracePointAtDepth(depth);
  const base = state.basePoint || point;
  return [
    (point.x - base.x) * state.sceneScale,
    (point.z - base.z) * state.sceneScale,
    (point.y - base.y) * state.sceneScale,
  ];
}

function viewAxis() {
  const start = localPointAtDepth(state.meter);
  const end = localPointAtDepth(state.meter + getRangeMeters().length);
  return normalize(sub(end, start));
}

function viewRadialFrame(axis) {
  const reference = Math.abs(dot(axis, [0, 1, 0])) > 0.92 ? [1, 0, 0] : [0, 1, 0];
  const right = normalize(cross(reference, axis));
  const forward = normalize(cross(axis, right));
  return { right, forward };
}

function selectedTrack() {
  return getTrack(state.selectedTrackId);
}

function getTrack(id) {
  return state.tracks.find((track) => track.id === id);
}

function valueRange(rows, key) {
  const values = rows.map((row) => row[key]).filter(Number.isFinite);
  if (!values.length) return { min: 0, max: 1 };
  return { min: Math.min(...values), max: Math.max(...values) };
}

function formatInputNumber(value) {
  return Number.isFinite(value) ? String(Number(value.toFixed(4))) : "";
}

function createProgram(vertexSource, fragmentSource) {
  const vertex = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "Program link failed");
  }
  return program;
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "Shader compile failed");
  }
  return shader;
}

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ]);
}

function mat4LookAt(eye, center, up) {
  const z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function mat4Invert(m) {
  const out = new Float32Array(16);
  const b00 = m[0] * m[5] - m[1] * m[4];
  const b01 = m[0] * m[6] - m[2] * m[4];
  const b02 = m[0] * m[7] - m[3] * m[4];
  const b03 = m[1] * m[6] - m[2] * m[5];
  const b04 = m[1] * m[7] - m[3] * m[5];
  const b05 = m[2] * m[7] - m[3] * m[6];
  const b06 = m[8] * m[13] - m[9] * m[12];
  const b07 = m[8] * m[14] - m[10] * m[12];
  const b08 = m[8] * m[15] - m[11] * m[12];
  const b09 = m[9] * m[14] - m[10] * m[13];
  const b10 = m[9] * m[15] - m[11] * m[13];
  const b11 = m[10] * m[15] - m[11] * m[14];
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1.0 / det;

  out[0] = (m[5] * b11 - m[6] * b10 + m[7] * b09) * det;
  out[1] = (m[2] * b10 - m[1] * b11 - m[3] * b09) * det;
  out[2] = (m[13] * b05 - m[14] * b04 + m[15] * b03) * det;
  out[3] = (m[10] * b04 - m[9] * b05 - m[11] * b03) * det;
  out[4] = (m[6] * b08 - m[4] * b11 - m[7] * b07) * det;
  out[5] = (m[0] * b11 - m[2] * b08 + m[3] * b07) * det;
  out[6] = (m[14] * b02 - m[12] * b05 - m[15] * b01) * det;
  out[7] = (m[8] * b05 - m[10] * b02 + m[11] * b01) * det;
  out[8] = (m[4] * b10 - m[5] * b08 + m[7] * b06) * det;
  out[9] = (m[1] * b08 - m[0] * b10 - m[3] * b06) * det;
  out[10] = (m[12] * b04 - m[13] * b02 + m[15] * b00) * det;
  out[11] = (m[9] * b02 - m[8] * b04 - m[11] * b00) * det;
  out[12] = (m[5] * b07 - m[4] * b09 - m[6] * b06) * det;
  out[13] = (m[0] * b09 - m[1] * b07 + m[2] * b06) * det;
  out[14] = (m[13] * b01 - m[12] * b03 - m[14] * b00) * det;
  out[15] = (m[8] * b03 - m[9] * b01 + m[10] * b00) * det;
  return out;
}

function transformVec4(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(v, scalar) {
  return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
}

function lerpVec(a, b, t) {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function vectorLength(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize(v) {
  const length = vectorLength(v) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function normalize2(v) {
  const length = Math.hypot(v.x, v.y) || 1;
  return { x: v.x / length, y: v.y / length };
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pad3(value) {
  return String(Math.round(value)).padStart(3, "0");
}

function degToRad(value) {
  return value * Math.PI / 180;
}

function radToDeg(value) {
  return value * 180 / Math.PI;
}

function mod(value, divisor) {
  if (!Number.isFinite(value)) return NaN;
  return ((value % divisor) + divisor) % divisor;
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const number = Number(value);
  return Number.isFinite(number) ? number : NaN;
}

function formatNumber(value, decimals) {
  return Number.isFinite(value) ? value.toFixed(decimals) : "n/a";
}
