"""Manual grain instance segmentation labeler for chip tray compartments.

Runs a small local web app for polygon annotation. It saves:
- annotations/<image_stem>.json with polygon vertices in image pixel coordinates
- masks/<image_stem>_masks.tif as a uint16 instance mask, Cellpose-style
- previews/<image_stem>_overlay.png for quick visual QA
- manifest.csv summarising annotation counts

Source images are only read. Nothing is moved, deleted, or edited.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from scipy import ndimage as ndi
    from skimage import color, exposure, feature, filters, measure, morphology, segmentation, util
    SEGMENTATION_AVAILABLE = True
except Exception:
    SEGMENTATION_AVAILABLE = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
CLASS_VALUE_BY_LABEL = {
    "grain": 1,
    "matrix": 2,
    "powder": 3,
}
CLASS_COLOR_BY_LABEL = {
    "grain": (79, 179, 255, 92),
    "matrix": (255, 191, 77, 92),
    "powder": (255, 111, 145, 92),
}
IMAGE_KEY_PATTERN = re.compile(
    r"(?P<hole>[A-Za-z]{1,4}\d{3,5})_CC_(?P<depth>\d+(?:[pP]\d+|\.\d+)?)",
    re.IGNORECASE,
)
APP_VERSION = "single-canvas-fragment-v3"


APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GeoVue Grain Labeler</title>
<style>
:root {
  color-scheme: dark;
  --bg: #111418;
  --panel: #191f27;
  --panel-2: #202834;
  --border: #344052;
  --text: #eef3f8;
  --muted: #9fb0c3;
  --accent: #4fb3ff;
  --good: #5ad18a;
  --warn: #ffbf4d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", Arial, sans-serif;
}
.app {
  height: 100vh;
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
}
.sidebar {
  border-right: 1px solid var(--border);
  background: var(--panel);
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  min-width: 0;
}
.brand {
  padding: 14px 14px 10px;
  border-bottom: 1px solid var(--border);
}
.brand h1 {
  margin: 0;
  font-size: 16px;
  line-height: 1.2;
  letter-spacing: 0;
}
.brand .meta {
  margin-top: 5px;
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.filters {
  padding: 10px;
  border-bottom: 1px solid var(--border);
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
}
input, select, button {
  font: inherit;
}
input, select {
  background: #0f1319;
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 9px;
  border-radius: 6px;
  min-width: 0;
}
button {
  background: var(--panel-2);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  min-width: 38px;
}
button:hover { border-color: var(--accent); }
button.primary {
  background: #12344f;
  border-color: #2d78ad;
}
button.good {
  background: #143923;
  border-color: #2a9d57;
}
button.warn {
  background: #402d10;
  border-color: #9f742a;
}
.image-list {
  overflow: auto;
  padding: 8px;
}
.image-row {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px;
  padding: 9px;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
}
.image-row:hover { background: #202834; }
.image-row.active {
  background: #173148;
  border-color: #347bac;
}
.image-name {
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-size: 13px;
}
.image-sub {
  color: var(--muted);
  font-size: 12px;
  margin-top: 3px;
}
.badge {
  align-self: start;
  color: var(--good);
  border: 1px solid #2a9d57;
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 11px;
  min-width: 24px;
  text-align: center;
}
.main {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-width: 0;
}
.toolbar {
  min-height: 58px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px;
  border-bottom: 1px solid var(--border);
  background: #151a21;
}
.toolbar .spacer { flex: 1; }
.toolbar .label-input {
  width: 140px;
}
.status {
  font-size: 12px;
  color: var(--muted);
  min-width: 220px;
  text-align: right;
}
.canvas-wrap {
  position: relative;
  min-width: 0;
  min-height: 0;
  background:
    linear-gradient(45deg, #151922 25%, transparent 25%),
    linear-gradient(-45deg, #151922 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #151922 75%),
    linear-gradient(-45deg, transparent 75%, #151922 75%);
  background-size: 24px 24px;
  background-position: 0 0, 0 12px, 12px -12px, -12px 0;
}
#canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  display: block;
  cursor: crosshair;
}
#mainImage {
  position: absolute;
  display: none;
  transform-origin: 0 0;
  user-select: none;
  -webkit-user-drag: none;
  pointer-events: none;
  max-width: none;
  max-height: none;
}
.footer {
  min-height: 40px;
  border-top: 1px solid var(--border);
  background: #151a21;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 12px;
  font-size: 12px;
}
.object-list {
  max-width: 46vw;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.review-panel {
  position: absolute;
  right: 14px;
  bottom: 14px;
  width: min(360px, 32vw);
  background: rgba(17, 20, 24, 0.94);
  border: 1px solid var(--border);
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(0, 0, 0, 0.45);
  padding: 10px;
  display: none;
  gap: 8px;
  z-index: 5;
}
.review-panel.active {
  display: grid;
}
.review-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  font-size: 12px;
  color: var(--muted);
}
.review-head strong {
  color: var(--text);
  font-size: 13px;
}
#reviewCanvas {
  width: 100%;
  aspect-ratio: 1;
  background: #05070a;
  border: 1px solid var(--border);
  border-radius: 6px;
}
.review-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
@media (max-width: 900px) {
  .app { grid-template-columns: 240px minmax(0, 1fr); }
  .toolbar { flex-wrap: wrap; }
  .status { text-align: left; min-width: 140px; }
}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <h1>GeoVue Grain Labeler</h1>
      <div class="meta"><span id="datasetMeta"></span> | <span id="appVersion"></span></div>
    </div>
    <div class="filters">
      <input id="search" placeholder="Filter" aria-label="Filter images">
      <select id="showMode" aria-label="Show mode">
        <option value="all">All</option>
        <option value="todo">Todo</option>
        <option value="done">Done</option>
      </select>
    </div>
    <div class="image-list" id="imageList"></div>
  </aside>
  <main class="main">
    <div class="toolbar">
      <button id="prevBtn" title="Previous image">Prev</button>
      <button id="nextBtn" title="Next image">Next</button>
      <button id="reloadBtn" title="Reload current image and clear unsaved overlays">Reload</button>
      <button id="fitBtn" title="Fit image">Fit</button>
      <button id="fitWidthBtn" title="Fit image width">Width</button>
      <button id="actualBtn" title="Actual image pixels">1:1</button>
      <button id="undoPointBtn" title="Undo current point">Point-</button>
      <button id="finishBtn" class="primary" title="Finish current polygon">Finish</button>
      <button id="undoObjectBtn" title="Remove last object">Object-</button>
      <button id="suggestBtn" class="primary" title="Suggest grain polygons">Suggest</button>
      <button id="truthBtn" class="primary" title="Load saved imagegen truth mask for this image">Truth</button>
      <button id="maskFileBtn" title="Load an imagegen mask file and convert it to polygons">Load mask</button>
      <input id="maskFileInput" type="file" accept="image/*" style="display:none">
      <button id="reviewBtn" title="Review proposed polygons">Review</button>
      <button id="clearBtn" class="warn" title="Clear annotation">Clear</button>
      <select id="labelInput" class="label-input" aria-label="Object label">
        <option value="grain">grain</option>
        <option value="matrix">matrix</option>
        <option value="powder">powder</option>
      </select>
      <button id="snapBtn" class="primary" title="Toggle edge snapping">Snap</button>
      <input id="snapRadius" class="label-input" type="number" min="0" max="40" value="10" aria-label="Snap radius">
      <button id="saveBtn" class="good" title="Save annotation and masks">Save</button>
      <div class="spacer"></div>
      <div class="status" id="status"></div>
    </div>
    <div class="canvas-wrap">
      <img id="mainImage" alt="">
      <canvas id="canvas"></canvas>
      <div class="review-panel" id="reviewPanel">
        <div class="review-head">
          <strong id="reviewTitle">Candidate</strong>
          <span id="reviewHint">Right accept | Left reject</span>
        </div>
        <canvas id="reviewCanvas" width="320" height="320"></canvas>
        <div class="review-actions">
          <button id="rejectBtn" class="warn">Reject</button>
          <button id="acceptBtn" class="good">Accept</button>
        </div>
      </div>
    </div>
    <div class="footer">
      <span id="imageMeta"></span>
      <span id="cursorMeta"></span>
      <span class="object-list" id="objectMeta"></span>
    </div>
  </main>
</div>
<script>
"use strict";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const mainImageEl = document.getElementById("mainImage");
const imageListEl = document.getElementById("imageList");
const statusEl = document.getElementById("status");
const datasetMetaEl = document.getElementById("datasetMeta");
const appVersionEl = document.getElementById("appVersion");
const imageMetaEl = document.getElementById("imageMeta");
const cursorMetaEl = document.getElementById("cursorMeta");
const objectMetaEl = document.getElementById("objectMeta");
const searchEl = document.getElementById("search");
const showModeEl = document.getElementById("showMode");
const labelInputEl = document.getElementById("labelInput");
const snapBtnEl = document.getElementById("snapBtn");
const snapRadiusEl = document.getElementById("snapRadius");
const reviewPanelEl = document.getElementById("reviewPanel");
const reviewCanvas = document.getElementById("reviewCanvas");
const reviewCtx = reviewCanvas.getContext("2d");
const reviewTitleEl = document.getElementById("reviewTitle");
const maskFileInputEl = document.getElementById("maskFileInput");

let images = [];
let currentIndex = 0;
let img = mainImageEl;
let edgeData = null;
let imgLoaded = false;
let objects = [];
let currentPoints = [];
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let isPanning = false;
let isTracing = false;
let lastPointer = null;
let lastTracePoint = null;
let dirty = false;
let snapEnabled = true;
let reviewActive = false;
let reviewIndex = 0;
const appVersion = "single-canvas-fragment-v3";
if (appVersionEl) appVersionEl.textContent = appVersion;

const colors = [
  "#4fb3ff", "#5ad18a", "#ffbf4d", "#ff6f91", "#b39cff",
  "#64d8cb", "#f78c6c", "#c3e88d", "#82aaff", "#f07178",
  "#ffd166", "#06d6a0", "#ef476f", "#118ab2", "#c77dff",
  "#70e000", "#ff9f1c", "#2ec4b6", "#e71d36", "#a8dadc"
];
const colorByLabel = {
  grain: "#4fb3ff",
  matrix: "#ffbf4d",
  powder: "#ff6f91"
};

function setStatus(text) {
  statusEl.textContent = text;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return response.json();
  return response.text();
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  draw();
}

function fitImage() {
  if (!imgLoaded) return;
  const pad = 24;
  const sx = (canvas.width - pad * 2) / img.naturalWidth;
  const sy = (canvas.height - pad * 2) / img.naturalHeight;
  scale = Math.max(0.05, Math.min(sx, sy));
  offsetX = (canvas.width - img.naturalWidth * scale) / 2;
  offsetY = (canvas.height - img.naturalHeight * scale) / 2;
  draw();
}

function fitImageWidth() {
  if (!imgLoaded) return;
  const pad = 24;
  scale = Math.max(0.05, (canvas.width - pad * 2) / img.naturalWidth);
  offsetX = (canvas.width - img.naturalWidth * scale) / 2;
  offsetY = 24;
  draw();
}

function actualSize() {
  if (!imgLoaded) return;
  scale = 1;
  offsetX = (canvas.width - img.naturalWidth) / 2;
  offsetY = 24;
  draw();
}

function imageToScreen(pt) {
  return [pt[0] * scale + offsetX, pt[1] * scale + offsetY];
}

function screenToImage(x, y) {
  return [(x - offsetX) / scale, (y - offsetY) / scale];
}

function pointerCanvasXY(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

function buildEdgeData() {
  const off = document.createElement("canvas");
  off.width = img.naturalWidth;
  off.height = img.naturalHeight;
  const offCtx = off.getContext("2d", {willReadFrequently: true});
  offCtx.drawImage(img, 0, 0);
  edgeData = offCtx.getImageData(0, 0, off.width, off.height);
}

function grayAt(x, y) {
  if (!edgeData) return 0;
  x = Math.max(0, Math.min(edgeData.width - 1, x));
  y = Math.max(0, Math.min(edgeData.height - 1, y));
  const idx = (y * edgeData.width + x) * 4;
  const d = edgeData.data;
  return 0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2];
}

function snapPoint(ix, iy) {
  if (!snapEnabled || !edgeData) return [ix, iy];
  const radius = Math.max(0, Math.min(40, Number(snapRadiusEl.value || 0)));
  if (radius < 1) return [ix, iy];
  const cx = Math.round(ix);
  const cy = Math.round(iy);
  let best = [ix, iy];
  let bestScore = -1;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy > radius * radius) continue;
      const x = cx + dx;
      const y = cy + dy;
      if (x <= 0 || y <= 0 || x >= edgeData.width - 1 || y >= edgeData.height - 1) continue;
      const gx = grayAt(x + 1, y) - grayAt(x - 1, y);
      const gy = grayAt(x, y + 1) - grayAt(x, y - 1);
      const distancePenalty = (dx * dx + dy * dy) * 0.75;
      const score = gx * gx + gy * gy - distancePenalty;
      if (score > bestScore) {
        bestScore = score;
        best = [x, y];
      }
    }
  }
  return bestScore > 400 ? best : [ix, iy];
}

function screenDistance(a, b) {
  const as = imageToScreen(a);
  const bs = imageToScreen(b);
  const dx = as[0] - bs[0];
  const dy = as[1] - bs[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function imageDistance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function styleWidth() {
  return Math.max(0.55, Math.min(2.2, 2 / Math.sqrt(Math.max(scale, 0.05))));
}

function vertexRadius() {
  return Math.max(1.25, Math.min(4, 4 / Math.sqrt(Math.max(scale, 0.05))));
}

function objectColor(obj, idx) {
  const value = obj.color_index ?? idx;
  return colors[((value % colors.length) + colors.length) % colors.length];
}

function renumberObjects() {
  objects.forEach((obj, idx) => {
    obj.id = idx + 1;
    if (obj.color_index === undefined || obj.color_index === null) obj.color_index = idx;
  });
}

function polygonBounds(points) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const pt of points) {
    minX = Math.min(minX, pt[0]);
    minY = Math.min(minY, pt[1]);
    maxX = Math.max(maxX, pt[0]);
    maxY = Math.max(maxY, pt[1]);
  }
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY)
  };
}

function drawReviewCrop(obj, idx) {
  const w = reviewCanvas.width;
  const h = reviewCanvas.height;
  reviewCtx.clearRect(0, 0, w, h);
  reviewCtx.fillStyle = "#05070a";
  reviewCtx.fillRect(0, 0, w, h);
  if (!obj || !obj.points || obj.points.length < 3 || !imgLoaded) return;

  const bounds = polygonBounds(obj.points);
  const pad = Math.max(18, Math.max(bounds.width, bounds.height) * 0.55);
  const sx = Math.max(0, Math.floor(bounds.minX - pad));
  const sy = Math.max(0, Math.floor(bounds.minY - pad));
  const ex = Math.min(img.naturalWidth, Math.ceil(bounds.maxX + pad));
  const ey = Math.min(img.naturalHeight, Math.ceil(bounds.maxY + pad));
  const sw = Math.max(1, ex - sx);
  const sh = Math.max(1, ey - sy);
  const cropScale = Math.min(w / sw, h / sh);
  const dx = (w - sw * cropScale) / 2;
  const dy = (h - sh * cropScale) / 2;
  const color = objectColor(obj, idx);

  reviewCtx.imageSmoothingEnabled = true;
  reviewCtx.drawImage(img, sx, sy, sw, sh, dx, dy, sw * cropScale, sh * cropScale);
  reviewCtx.save();
  reviewCtx.translate(dx, dy);
  reviewCtx.scale(cropScale, cropScale);
  reviewCtx.beginPath();
  obj.points.forEach((pt, pointIdx) => {
    const x = pt[0] - sx;
    const y = pt[1] - sy;
    if (pointIdx === 0) reviewCtx.moveTo(x, y);
    else reviewCtx.lineTo(x, y);
  });
  reviewCtx.closePath();
  reviewCtx.fillStyle = color + "44";
  reviewCtx.strokeStyle = "#ffffff";
  reviewCtx.lineWidth = Math.max(1.3 / cropScale, 0.25);
  reviewCtx.fill();
  reviewCtx.stroke();
  reviewCtx.strokeStyle = color;
  reviewCtx.lineWidth = Math.max(3 / cropScale, 0.45);
  reviewCtx.stroke();
  reviewCtx.restore();
}

function updateReviewPanel() {
  if (!reviewPanelEl) return;
  if (!reviewActive || !objects.length || !imgLoaded) {
    reviewPanelEl.classList.remove("active");
    return;
  }
  reviewIndex = Math.max(0, Math.min(reviewIndex, objects.length - 1));
  reviewPanelEl.classList.add("active");
  const obj = objects[reviewIndex];
  reviewTitleEl.textContent = `${reviewIndex + 1}/${objects.length} ${obj.label || "grain"}`;
  drawReviewCrop(obj, reviewIndex);
}

function startReview() {
  if (!objects.length) {
    reviewActive = false;
    updateReviewPanel();
    setStatus("No polygons to review");
    return;
  }
  reviewActive = true;
  reviewIndex = Math.max(0, Math.min(reviewIndex, objects.length - 1));
  draw();
  setStatus(`Reviewing ${reviewIndex + 1}/${objects.length}`);
}

function stopReview() {
  reviewActive = false;
  reviewIndex = 0;
  updateReviewPanel();
  draw();
}

function acceptCurrent() {
  if (!reviewActive) startReview();
  if (!reviewActive) return;
  reviewIndex += 1;
  if (reviewIndex >= objects.length) {
    stopReview();
    setStatus("Review complete");
    return;
  }
  draw();
  setStatus(`Accepted. Reviewing ${reviewIndex + 1}/${objects.length}`);
}

function rejectCurrent() {
  if (!reviewActive) startReview();
  if (!reviewActive || !objects.length) return;
  objects.splice(reviewIndex, 1);
  renumberObjects();
  dirty = true;
  if (!objects.length) {
    stopReview();
    setStatus("Review complete");
    return;
  }
  reviewIndex = Math.max(0, Math.min(reviewIndex, objects.length - 1));
  draw();
  setStatus(`Rejected. Reviewing ${reviewIndex + 1}/${objects.length}`);
}

function drawPolygon(points, fillStyle, strokeStyle, lineWidth) {
  if (points.length === 0) return;
  ctx.beginPath();
  const first = imageToScreen(points[0]);
  ctx.moveTo(first[0], first[1]);
  for (let i = 1; i < points.length; i++) {
    const pt = imageToScreen(points[i]);
    ctx.lineTo(pt[0], pt[1]);
  }
  if (points.length >= 3) ctx.closePath();
  if (fillStyle && points.length >= 3) {
    ctx.fillStyle = fillStyle;
    ctx.fill();
  }
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function drawVertices(points, color) {
  ctx.fillStyle = color;
  for (const pt of points) {
    const s = imageToScreen(pt);
    ctx.beginPath();
    ctx.arc(s[0], s[1], vertexRadius(), 0, Math.PI * 2);
    ctx.fill();
  }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!imgLoaded) {
    updateReviewPanel();
    return;
  }
  updateImageElement();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(
    img,
    0,
    0,
    img.naturalWidth,
    img.naturalHeight,
    offsetX,
    offsetY,
    img.naturalWidth * scale,
    img.naturalHeight * scale
  );
  objects.forEach((obj, idx) => {
    const color = objectColor(obj, idx);
    const activeReviewObject = reviewActive && idx === reviewIndex;
    const mutedReviewObject = reviewActive && idx !== reviewIndex;
    const fill = mutedReviewObject ? color + "18" : (activeReviewObject ? color + "66" : color + "4f");
    const lineWidth = mutedReviewObject ? styleWidth() * 0.7 : (activeReviewObject ? styleWidth() * 2.4 : styleWidth());
    drawPolygon(obj.points, fill, activeReviewObject ? "#ffffff" : color, lineWidth);
    drawPolygon(obj.points, null, color, activeReviewObject ? Math.max(1, styleWidth() * 1.4) : Math.max(0.5, styleWidth() * 0.5));
    if (!mutedReviewObject) drawVertices(obj.points, color);
  });
  if (currentPoints.length) {
    const nearStart = currentPoints.length > 2 && screenDistance(currentPoints[0], currentPoints[currentPoints.length - 1]) < 28;
    drawPolygon(currentPoints, "rgba(255,255,255,0.14)", nearStart ? "#5ad18a" : "#ffffff", styleWidth());
    drawVertices(currentPoints, "#ffffff");
    if (nearStart) {
      const s = imageToScreen(currentPoints[0]);
      ctx.strokeStyle = "#5ad18a";
      ctx.lineWidth = Math.max(1, styleWidth());
      ctx.beginPath();
      ctx.arc(s[0], s[1], 12, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  imageMetaEl.textContent = images[currentIndex] ? `${images[currentIndex].name} | ${images[currentIndex].hole_id || ""} ${images[currentIndex].depth ?? ""}` : "";
  const counts = objects.reduce((acc, obj) => {
    const label = obj.label || "grain";
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});
  const countText = Object.entries(counts).map(([k, v]) => `${k}:${v}`).join(" ");
  objectMetaEl.textContent = `${objects.length} objects ${countText} | ${currentPoints.length} active points`;
  updateReviewPanel();
}

function updateImageElement() {
  mainImageEl.style.display = "none";
}

function renderList() {
  const query = searchEl.value.trim().toLowerCase();
  const mode = showModeEl.value;
  imageListEl.innerHTML = "";
  images.forEach((entry, idx) => {
    if (query && !`${entry.name} ${entry.hole_id} ${entry.depth}`.toLowerCase().includes(query)) return;
    if (mode === "todo" && entry.object_count > 0) return;
    if (mode === "done" && entry.object_count === 0) return;
    const row = document.createElement("div");
    row.className = "image-row" + (idx === currentIndex ? " active" : "");
    row.addEventListener("click", () => loadImage(idx));
    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "image-name";
    name.textContent = entry.name;
    const sub = document.createElement("div");
    sub.className = "image-sub";
    sub.textContent = `${entry.hole_id || ""} ${entry.depth ?? ""}`;
    left.append(name, sub);
    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = entry.object_count || "";
    row.append(left, badge);
    imageListEl.append(row);
  });
}

async function loadImages() {
  const data = await api("/api/images");
  images = data.images;
  datasetMetaEl.textContent = `${images.length} images | ${data.output_dir}`;
  renderList();
  if (images.length) await loadImage(0);
}

async function loadImage(index) {
  if (dirty) await saveAnnotation(false);
  currentIndex = Math.max(0, Math.min(index, images.length - 1));
  currentPoints = [];
  objects = [];
  reviewActive = false;
  reviewIndex = 0;
  dirty = false;
  imgLoaded = false;
  updateImageElement();
  updateReviewPanel();
  setStatus("Loading");
  const anno = await api(`/api/annotation?index=${currentIndex}`);
  objects = anno.objects || [];
  img = mainImageEl;
  mainImageEl.onload = () => {
    imgLoaded = true;
    buildEdgeData();
    fitImage();
    setStatus("Ready");
    renderList();
  };
  mainImageEl.src = `/api/image?index=${currentIndex}&t=${Date.now()}`;
}

function finishPolygon() {
  if (currentPoints.length < 3) {
    currentPoints = [];
    isTracing = false;
    lastTracePoint = null;
    draw();
    return;
  }
  const last = currentPoints[currentPoints.length - 1];
  if (screenDistance(currentPoints[0], last) < 32) {
    currentPoints[currentPoints.length - 1] = currentPoints[0];
  }
  objects.push({
    id: objects.length + 1,
    label: labelInputEl.value || "grain",
    color_index: objects.length,
    points: currentPoints.map(pt => [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10])
  });
  currentPoints = [];
  isTracing = false;
  lastTracePoint = null;
  dirty = true;
  draw();
}

async function saveAnnotation(showSaved = true) {
  if (!images.length) return;
  const payload = {
    index: currentIndex,
    objects: objects,
    image: images[currentIndex]
  };
  const data = await api("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  dirty = false;
  images[currentIndex].object_count = objects.length;
  renderList();
  if (showSaved) setStatus(`Saved ${data.object_count} objects`);
}

async function suggestPolygons() {
  if (!images.length) return;
  if (objects.length > 0 && !confirm("Replace current polygons with suggested polygons?")) return;
  setStatus("Suggesting polygons");
  const label = labelInputEl.value || "grain";
  const data = await api(`/api/propose?index=${currentIndex}&label=${encodeURIComponent(label)}`);
  applyCandidateObjects(data, "Suggested");
}

function applyCandidateObjects(data, sourceName) {
  objects = data.objects || [];
  currentPoints = [];
  renumberObjects();
  reviewActive = objects.length > 0;
  reviewIndex = 0;
  dirty = true;
  draw();
  setStatus(objects.length ? `${sourceName}: ${objects.length} polygons. Reviewing 1/${objects.length}` : `${sourceName}: no polygons found`);
}

async function loadStoredTruthMask() {
  if (!images.length) return;
  if (objects.length > 0 && !confirm("Replace current polygons with imagegen truth mask polygons?")) return;
  setStatus("Loading imagegen truth mask");
  const label = labelInputEl.value || "grain";
  const data = await api(`/api/imagegen-truth?index=${currentIndex}&label=${encodeURIComponent(label)}`);
  applyCandidateObjects(data, "Imagegen truth");
}

function loadMaskFile() {
  if (!images.length) return;
  maskFileInputEl.click();
}

async function handleMaskFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    if (objects.length > 0 && !confirm("Replace current polygons with loaded mask polygons?")) return;
    setStatus("Vectorising mask file");
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("Failed to read mask file"));
      reader.readAsDataURL(file);
    });
    const label = labelInputEl.value || "grain";
    const data = await api("/api/vectorize-mask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        index: currentIndex,
        label,
        filename: file.name,
        data_url: dataUrl
      })
    });
    applyCandidateObjects(data, "Loaded mask");
  } catch (err) {
    setStatus(err.message);
  } finally {
    maskFileInputEl.value = "";
  }
}

function addPoint(event) {
  if (!imgLoaded) return;
  const [cx, cy] = pointerCanvasXY(event);
  const [ix, iy] = screenToImage(cx, cy);
  if (ix < 0 || iy < 0 || ix >= img.naturalWidth || iy >= img.naturalHeight) return;
  currentPoints.push(snapPoint(ix, iy));
  dirty = true;
  draw();
}

function addTracePoint(event, force = false) {
  if (!imgLoaded) return;
  const [cx, cy] = pointerCanvasXY(event);
  const [ix, iy] = screenToImage(cx, cy);
  if (ix < 0 || iy < 0 || ix >= img.naturalWidth || iy >= img.naturalHeight) return;
  const point = snapPoint(ix, iy);
  if (!force && lastTracePoint && screenDistance(lastTracePoint, point) < 5) return;
  if (!force && currentPoints.length > 1 && imageDistance(currentPoints[currentPoints.length - 1], point) < 0.8) return;
  currentPoints.push(point);
  lastTracePoint = point;
  dirty = true;
  draw();
}

function pointInPolygon(point, polygon) {
  let inside = false;
  const x = point[0];
  const y = point[1];
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    const intersect = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function deleteObjectAt(event) {
  if (!imgLoaded) return;
  const [cx, cy] = pointerCanvasXY(event);
  const point = screenToImage(cx, cy);
  for (let i = objects.length - 1; i >= 0; i--) {
    if (pointInPolygon(point, objects[i].points)) {
      objects.splice(i, 1);
      renumberObjects();
      if (reviewIndex >= objects.length) reviewIndex = Math.max(0, objects.length - 1);
      if (!objects.length) reviewActive = false;
      dirty = true;
      setStatus("Deleted polygon");
      draw();
      return;
    }
  }
  setStatus("No polygon under cursor");
}

canvas.addEventListener("mousedown", event => {
  const [cx, cy] = pointerCanvasXY(event);
  if (event.button === 2) {
    event.preventDefault();
    deleteObjectAt(event);
    return;
  }
  if (event.button === 1 || event.shiftKey || event.altKey) {
    isPanning = true;
    lastPointer = [cx, cy];
    return;
  }
  if (event.button !== 0) return;
  isTracing = true;
  currentPoints = [];
  lastTracePoint = null;
  addTracePoint(event, true);
});

canvas.addEventListener("mousemove", event => {
  const [cx, cy] = pointerCanvasXY(event);
  if (isTracing) {
    addTracePoint(event, false);
    return;
  }
  if (isPanning && lastPointer) {
    offsetX += cx - lastPointer[0];
    offsetY += cy - lastPointer[1];
    lastPointer = [cx, cy];
    draw();
    return;
  }
  if (imgLoaded) {
    const [ix, iy] = screenToImage(cx, cy);
    cursorMetaEl.textContent = `${Math.round(ix)}, ${Math.round(iy)}`;
  }
});

function finishActiveTrace(event) {
  if (isTracing) {
    addTracePoint(event, true);
    finishPolygon();
  }
  isTracing = false;
  isPanning = false;
  lastPointer = null;
  lastTracePoint = null;
}

canvas.addEventListener("mouseup", finishActiveTrace);
window.addEventListener("mouseup", event => {
  if (isTracing || isPanning) finishActiveTrace(event);
});

canvas.addEventListener("dblclick", event => {
  event.preventDefault();
  finishPolygon();
});

canvas.addEventListener("contextmenu", event => {
  event.preventDefault();
});

canvas.addEventListener("wheel", event => {
  if (!imgLoaded) return;
  event.preventDefault();
  const [cx, cy] = pointerCanvasXY(event);
  const before = screenToImage(cx, cy);
  const factor = event.deltaY < 0 ? 1.12 : 0.89;
  scale = Math.max(0.03, Math.min(scale * factor, 20));
  offsetX = cx - before[0] * scale;
  offsetY = cy - before[1] * scale;
  draw();
}, {passive: false});

window.addEventListener("keydown", event => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.ctrlKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveAnnotation(true).catch(err => setStatus(err.message));
    return;
  }
  if (reviewActive) {
    if (event.key === "ArrowRight" || event.key === " ") {
      event.preventDefault();
      acceptCurrent();
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      rejectCurrent();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      stopReview();
      setStatus("Review stopped");
      return;
    }
  }
  if (event.key === "Enter") finishPolygon();
  if (event.key === "Backspace") {
    currentPoints.pop();
    dirty = true;
    draw();
  }
  if (event.ctrlKey && event.key.toLowerCase() === "z") {
    objects.pop();
    dirty = true;
    draw();
  }
});

document.getElementById("prevBtn").addEventListener("click", () => loadImage(currentIndex - 1));
document.getElementById("nextBtn").addEventListener("click", () => loadImage(currentIndex + 1));
document.getElementById("reloadBtn").addEventListener("click", () => loadImage(currentIndex));
document.getElementById("fitBtn").addEventListener("click", fitImage);
document.getElementById("fitWidthBtn").addEventListener("click", fitImageWidth);
document.getElementById("actualBtn").addEventListener("click", actualSize);
document.getElementById("undoPointBtn").addEventListener("click", () => {
  currentPoints.pop();
  dirty = true;
  draw();
});
document.getElementById("finishBtn").addEventListener("click", finishPolygon);
document.getElementById("undoObjectBtn").addEventListener("click", () => {
  objects.pop();
  renumberObjects();
  if (reviewIndex >= objects.length) reviewIndex = Math.max(0, objects.length - 1);
  if (!objects.length) reviewActive = false;
  dirty = true;
  draw();
});
document.getElementById("suggestBtn").addEventListener("click", () => suggestPolygons().catch(err => setStatus(err.message)));
document.getElementById("truthBtn").addEventListener("click", () => loadStoredTruthMask().catch(err => setStatus(err.message)));
document.getElementById("maskFileBtn").addEventListener("click", loadMaskFile);
maskFileInputEl.addEventListener("change", event => handleMaskFileSelected(event));
document.getElementById("reviewBtn").addEventListener("click", startReview);
document.getElementById("acceptBtn").addEventListener("click", acceptCurrent);
document.getElementById("rejectBtn").addEventListener("click", rejectCurrent);
document.getElementById("clearBtn").addEventListener("click", () => {
  if (!confirm("Clear this annotation?")) return;
  objects = [];
  currentPoints = [];
  reviewActive = false;
  reviewIndex = 0;
  dirty = true;
  draw();
});
document.getElementById("saveBtn").addEventListener("click", () => saveAnnotation(true).catch(err => setStatus(err.message)));
snapBtnEl.addEventListener("click", () => {
  snapEnabled = !snapEnabled;
  snapBtnEl.classList.toggle("primary", snapEnabled);
  setStatus(snapEnabled ? "Snap on" : "Snap off");
});
searchEl.addEventListener("input", renderList);
showModeEl.addEventListener("change", renderList);
window.addEventListener("resize", resizeCanvas);

resizeCanvas();
loadImages().catch(err => setStatus(err.message));
</script>
</body>
</html>
"""


@dataclass
class ImageRecord:
    index: int
    path: Path
    name: str
    hole_id: str | None
    depth: float | None


def parse_image_key(path: Path) -> tuple[str | None, float | None]:
    match = IMAGE_KEY_PATTERN.search(path.name)
    if not match:
        return None, None
    hole_id = match.group("hole").upper()
    depth = float(match.group("depth").replace("p", ".").replace("P", "."))
    return hole_id, depth


def read_image_records(args: argparse.Namespace) -> list[ImageRecord]:
    paths: list[Path] = []
    if args.images_csv:
        df = pd.read_csv(args.images_csv)
        if args.path_col not in df.columns:
            raise ValueError(f"CSV path column not found: {args.path_col}")
        if args.label_col and args.target_label and args.label_col in df.columns:
            df = df[df[args.label_col].astype(str).eq(args.target_label)]
        paths = [Path(str(value)) for value in df[args.path_col].dropna().tolist()]
    elif args.image_root:
        root = args.image_root
        paths = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        raise ValueError("Pass --images-csv or --image-root")

    records: list[ImageRecord] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if args.name_contains and args.name_contains.lower() not in resolved.name.lower():
            continue
        hole_id, depth = parse_image_key(resolved)
        records.append(
            ImageRecord(
                index=len(records),
                path=resolved,
                name=resolved.name,
                hole_id=hole_id,
                depth=depth,
            )
        )
        if args.limit and len(records) >= args.limit:
            break

    if not records:
        raise ValueError("No images found for annotation.")
    return records


def safe_stem(record: ImageRecord) -> str:
    stem = record.path.stem
    if record.hole_id and record.depth is not None:
        depth_token = f"{record.depth:.3f}".replace(".", "p")
        return f"{record.hole_id}_CC_{depth_token}_{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or f"image_{record.index:05d}"


class GrainLabelerState:
    def __init__(self, records: list[ImageRecord], output_dir: Path):
        self.records = records
        self.output_dir = output_dir
        self.annotation_dir = output_dir / "annotations"
        self.image_dir = output_dir / "images"
        self.mask_dir = output_dir / "masks"
        self.class_mask_dir = output_dir / "class_masks"
        self.preview_dir = output_dir / "previews"
        self.clip_dir = output_dir / "grain_clips"
        self.imagegen_truth_dir = output_dir / "imagegen_truth_masks"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.class_mask_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.imagegen_truth_dir.mkdir(parents=True, exist_ok=True)

    def record(self, index: int) -> ImageRecord:
        if index < 0 or index >= len(self.records):
            raise IndexError(f"Image index out of range: {index}")
        return self.records[index]

    def annotation_path(self, record: ImageRecord) -> Path:
        return self.annotation_dir / f"{safe_stem(record)}.json"

    def image_copy_path(self, record: ImageRecord) -> Path:
        return self.image_dir / f"{safe_stem(record)}.png"

    def mask_path(self, record: ImageRecord) -> Path:
        return self.mask_dir / f"{safe_stem(record)}_masks.tif"

    def class_mask_path(self, record: ImageRecord) -> Path:
        return self.class_mask_dir / f"{safe_stem(record)}_classes.png"

    def preview_path(self, record: ImageRecord) -> Path:
        return self.preview_dir / f"{safe_stem(record)}_overlay.png"

    def clip_record_dir(self, record: ImageRecord) -> Path:
        return self.clip_dir / safe_stem(record)

    def imagegen_truth_path(self, record: ImageRecord) -> Path | None:
        candidates = [
            self.imagegen_truth_dir / f"{safe_stem(record)}.png",
            self.imagegen_truth_dir / f"{record.path.stem}.png",
            self.imagegen_truth_dir / f"{safe_stem(record)}_mask.png",
            self.imagegen_truth_dir / f"{record.path.stem}_mask.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_annotation(self, record: ImageRecord) -> dict:
        path = self.annotation_path(record)
        if not path.exists():
            return {"image": self.record_payload(record), "objects": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def record_payload(self, record: ImageRecord) -> dict:
        annotation = self.annotation_path(record)
        object_count = 0
        if annotation.exists():
            try:
                object_count = len(json.loads(annotation.read_text(encoding="utf-8")).get("objects", []))
            except Exception:
                object_count = 0
        return {
            "index": record.index,
            "name": record.name,
            "path": str(record.path),
            "hole_id": record.hole_id,
            "depth": record.depth,
            "object_count": object_count,
        }

    def save_annotation(self, index: int, objects: list[dict]) -> dict:
        record = self.record(index)
        image = Image.open(record.path).convert("RGB")
        width, height = image.size

        clean_objects: list[dict] = []
        for obj_index, obj in enumerate(objects, start=1):
            points = obj.get("points", [])
            if len(points) < 3:
                continue
            clean_points: list[list[float]] = []
            for point in points:
                if len(point) != 2:
                    continue
                x = min(max(float(point[0]), 0.0), float(width - 1))
                y = min(max(float(point[1]), 0.0), float(height - 1))
                clean_points.append([round(x, 2), round(y, 2)])
            if len(clean_points) < 3:
                continue
            clean_objects.append(
                {
                    "id": obj_index,
                    "label": str(obj.get("label") or "grain"),
                    "points": clean_points,
                }
            )

        payload = {
            "image": self.record_payload(record),
            "image_width": width,
            "image_height": height,
            "objects": clean_objects,
            "image_copy_path": str(self.image_copy_path(record)),
            "mask_path": str(self.mask_path(record)),
            "class_mask_path": str(self.class_mask_path(record)),
            "preview_path": str(self.preview_path(record)),
            "grain_clip_dir": str(self.clip_record_dir(record)),
        }
        self.annotation_path(record).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.write_mask_and_preview(record, image, clean_objects)
        self.write_manifest()
        return {
            "ok": True,
            "object_count": len(clean_objects),
            "annotation": str(self.annotation_path(record)),
            "image": str(self.image_copy_path(record)),
            "mask": str(self.mask_path(record)),
            "class_mask": str(self.class_mask_path(record)),
            "preview": str(self.preview_path(record)),
            "grain_clip_dir": str(self.clip_record_dir(record)),
        }

    def propose_objects(
        self,
        index: int,
        label: str = "grain",
        max_objects: int = 80,
    ) -> dict:
        if not SEGMENTATION_AVAILABLE:
            raise RuntimeError("Segmentation proposal dependencies are not available.")

        record = self.record(index)
        image = Image.open(record.path).convert("RGB")
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        gray = color.rgb2gray(rgb)
        gray = exposure.equalize_adapthist(gray, clip_limit=0.015)
        gray = filters.gaussian(gray, sigma=1.0, preserve_range=True)

        block = max(35, int(min(width, height) // 7) | 1)
        local_threshold = filters.threshold_local(gray, block_size=block, offset=-0.025)
        foreground = gray > local_threshold
        foreground = morphology.remove_small_objects(foreground, min_size=max(90, int(width * height * 0.00045)))
        foreground = morphology.remove_small_holes(foreground, area_threshold=max(120, int(width * height * 0.0006)))
        foreground = morphology.binary_opening(foreground, morphology.disk(1))
        foreground = morphology.binary_closing(foreground, morphology.disk(2))

        # Keep the tray interior dominant component(s), but do not force one blob;
        # dark cracks between grains are exactly what the watershed needs.
        distance = ndi.distance_transform_edt(foreground)
        min_distance = max(7, int(min(width, height) * 0.025))
        peaks = feature.peak_local_max(
            distance,
            min_distance=min_distance,
            labels=foreground,
            exclude_border=False,
        )
        markers = np.zeros_like(distance, dtype=np.int32)
        for marker_id, (row, col) in enumerate(peaks, start=1):
            markers[row, col] = marker_id
        markers = measure.label(markers > 0)
        if markers.max() == 0:
            markers = measure.label(foreground)

        gradient = filters.sobel(gray)
        labels = segmentation.watershed(gradient, markers=markers, mask=foreground)

        min_area = max(140, int(width * height * 0.00075))
        max_area = int(width * height * 0.22)
        regions = []
        for region in measure.regionprops(labels):
            if region.area < min_area or region.area > max_area:
                continue
            minr, minc, maxr, maxc = region.bbox
            if (maxr - minr) < 8 or (maxc - minc) < 8:
                continue
            regions.append(region)

        # Prefer larger, more useful grains, but keep enough mid-sized grains for training.
        regions.sort(key=lambda region: region.area, reverse=True)
        objects: list[dict] = []
        for region in regions[:max_objects]:
            mask = labels == region.label
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            if len(contour) < 12:
                continue
            tolerance = max(1.5, min(width, height) * 0.006)
            simplified = measure.approximate_polygon(contour, tolerance=tolerance)
            if len(simplified) < 3:
                continue
            # skimage contours are row,col; UI uses x,y.
            points = [
                [
                    round(float(np.clip(col, 0, width - 1)), 2),
                    round(float(np.clip(row, 0, height - 1)), 2),
                ]
                for row, col in simplified
            ]
            if len(points) >= 3:
                objects.append(
                    {
                        "id": len(objects) + 1,
                        "label": label,
                        "color_index": len(objects),
                        "points": points,
                    }
                )

        return {
            "ok": True,
            "image": self.record_payload(record),
            "objects": objects,
            "object_count": len(objects),
            "method": "adaptive_threshold_watershed",
            "note": "Proposal only: review/delete/edit before saving as training truth.",
        }

    def vectorize_truth_mask(
        self,
        index: int,
        mask_image: Image.Image,
        label: str = "grain",
        max_objects: int = 160,
        source: str = "imagegen_truth_mask",
    ) -> dict:
        if not SEGMENTATION_AVAILABLE:
            raise RuntimeError("Segmentation proposal dependencies are not available.")

        record = self.record(index)
        with Image.open(record.path) as source_image:
            width, height = source_image.size

        mask = mask_image.convert("RGBA")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        rgba = np.asarray(mask)
        rgb = rgba[:, :, :3].astype(np.uint8)
        alpha = rgba[:, :, 3]
        max_channel = rgb.max(axis=2)
        min_channel = rgb.min(axis=2)
        saturation = max_channel.astype(np.int16) - min_channel.astype(np.int16)

        # Contract: imagegen truth masks should be black background with filled
        # non-black grain islands. Use an adaptive bright-island threshold so
        # faint generated background texture does not join separate grains.
        valid = alpha > 32
        valid_values = max_channel[valid]
        if valid_values.size:
            try:
                threshold = float(filters.threshold_otsu(valid_values))
            except Exception:
                threshold = 96.0
        else:
            threshold = 96.0
        threshold = max(245.0, min(252.0, threshold))
        foreground = valid & (
            (max_channel.astype(np.float32) >= threshold)
            | ((max_channel > 100) & (saturation > 40))
        )
        foreground = morphology.remove_small_objects(foreground, min_size=max(80, int(width * height * 0.00035)))
        foreground = morphology.remove_small_holes(foreground, area_threshold=max(80, int(width * height * 0.00035)))
        foreground = morphology.binary_opening(foreground, morphology.disk(1))
        foreground = morphology.binary_closing(foreground, morphology.disk(1))

        connected = measure.label(foreground)
        regions = list(measure.regionprops(connected))
        min_area = max(80, int(width * height * 0.00035))
        max_area = int(width * height * 0.32)
        regions = [
            region for region in regions
            if min_area <= region.area <= max_area
            and (region.bbox[2] - region.bbox[0]) >= 5
            and (region.bbox[3] - region.bbox[1]) >= 5
        ]
        regions.sort(key=lambda region: region.area, reverse=True)

        objects: list[dict] = []
        for region in regions[:max_objects]:
            component = connected == region.label
            contours = measure.find_contours(component.astype(np.uint8), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            if len(contour) < 8:
                continue
            tolerance = max(0.75, min(width, height) * 0.0025)
            simplified = measure.approximate_polygon(contour, tolerance=tolerance)
            if len(simplified) < 3:
                continue
            points = [
                [
                    round(float(np.clip(col, 0, width - 1)), 2),
                    round(float(np.clip(row, 0, height - 1)), 2),
                ]
                for row, col in simplified
            ]
            if len(points) >= 3:
                objects.append(
                    {
                        "id": len(objects) + 1,
                        "label": label,
                        "color_index": len(objects),
                        "points": points,
                    }
                )

        return {
            "ok": True,
            "image": self.record_payload(record),
            "objects": objects,
            "object_count": len(objects),
            "method": source,
            "truth_mask_contract": "black background; filled non-black grain islands; one connected island per grain",
            "note": "Imagegen-derived mask vectorised to polygons. Review/accept before saving as training truth.",
        }

    def load_imagegen_truth(self, index: int, label: str = "grain") -> dict:
        record = self.record(index)
        path = self.imagegen_truth_path(record)
        if path is None:
            raise FileNotFoundError(
                f"No imagegen truth mask found in {self.imagegen_truth_dir}. "
                f"Expected {safe_stem(record)}.png or {record.path.stem}.png."
            )
        return self.vectorize_truth_mask(
            index,
            Image.open(path),
            label=label,
            source=f"imagegen_truth:{path.name}",
        )

    def write_mask_and_preview(self, record: ImageRecord, image: Image.Image, objects: list[dict]) -> None:
        width, height = image.size
        image.convert("RGB").save(self.image_copy_path(record))
        instance_mask = np.zeros((height, width), dtype=np.uint16)
        class_mask = np.zeros((height, width), dtype=np.uint8)
        clip_dir = self.clip_record_dir(record)
        clip_dir.mkdir(parents=True, exist_ok=True)
        for stale_clip in clip_dir.glob("*.png"):
            stale_clip.unlink()
        clip_rows: list[dict] = []
        texture_source = image.convert("RGBA")
        next_grain_instance_id = 1
        for obj in objects:
            layer = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(layer)
            polygon = [(float(x), float(y)) for x, y in obj["points"]]
            draw.polygon(polygon, fill=1)
            layer_array = np.array(layer, dtype=bool)
            label = str(obj.get("label") or "grain").lower()
            class_mask[layer_array] = CLASS_VALUE_BY_LABEL.get(label, 4)
            if label == "grain":
                instance_id = next_grain_instance_id
                instance_mask[layer_array] = instance_id
                bbox = layer.getbbox()
                if bbox:
                    texture_crop = texture_source.crop(bbox)
                    alpha_crop = layer.crop(bbox).point(lambda value: 255 if value else 0)
                    texture_crop.putalpha(alpha_crop)
                    clip_path = clip_dir / f"grain_{instance_id:04d}.png"
                    texture_crop.save(clip_path)
                    clip_rows.append(
                        {
                            "instance_id": instance_id,
                            "label": label,
                            "clip_path": str(clip_path),
                            "bbox_left": bbox[0],
                            "bbox_top": bbox[1],
                            "bbox_right": bbox[2],
                            "bbox_bottom": bbox[3],
                            "source_image": str(record.path),
                        }
                    )
                next_grain_instance_id += 1
        Image.fromarray(instance_mask).save(self.mask_path(record))
        Image.fromarray(class_mask).save(self.class_mask_path(record))
        clip_manifest = clip_dir / "clips.csv"
        with clip_manifest.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "instance_id",
                "label",
                "clip_path",
                "bbox_left",
                "bbox_top",
                "bbox_right",
                "bbox_bottom",
                "source_image",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clip_rows)

        overlay = image.convert("RGBA")
        overlay_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_layer)
        for obj in objects:
            label = str(obj.get("label") or "grain").lower()
            color = CLASS_COLOR_BY_LABEL.get(label, (179, 156, 255, 92))
            polygon = [(float(x), float(y)) for x, y in obj["points"]]
            draw.polygon(polygon, fill=color, outline=color[:3] + (230,))
        Image.alpha_composite(overlay, overlay_layer).convert("RGB").save(self.preview_path(record))

    def write_manifest(self) -> None:
        path = self.output_dir / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "image_path",
                    "annotation_path",
                    "mask_path",
                    "class_mask_path",
                    "preview_path",
                    "hole_id",
                    "depth",
                    "object_count",
                ],
            )
            writer.writeheader()
            for record in self.records:
                annotation = self.load_annotation(record)
                writer.writerow(
                    {
                        "index": record.index,
                        "image_path": str(record.path),
                        "annotation_path": str(self.annotation_path(record)),
                        "mask_path": str(self.mask_path(record)),
                        "class_mask_path": str(self.class_mask_path(record)),
                        "preview_path": str(self.preview_path(record)),
                        "hole_id": record.hole_id or "",
                        "depth": "" if record.depth is None else record.depth,
                        "object_count": len(annotation.get("objects", [])),
                    }
                )


def make_handler(state: GrainLabelerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GeoVueGrainLabeler/1.0"

        def log_message(self, fmt: str, *args) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_text(APP_HTML, content_type="text/html; charset=utf-8")
                    return
                if parsed.path == "/api/images":
                    self.send_json(
                        {
                            "images": [state.record_payload(record) for record in state.records],
                            "output_dir": str(state.output_dir),
                        }
                    )
                    return
                if parsed.path == "/api/annotation":
                    query = parse_qs(parsed.query)
                    record = state.record(int(query.get("index", ["0"])[0]))
                    self.send_json(state.load_annotation(record))
                    return
                if parsed.path == "/api/image":
                    query = parse_qs(parsed.query)
                    record = state.record(int(query.get("index", ["0"])[0]))
                    data = record.path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mimetypes.guess_type(record.path.name)[0] or "application/octet-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if parsed.path == "/api/propose":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    label = query.get("label", ["grain"])[0] or "grain"
                    max_objects = int(query.get("max_objects", ["80"])[0])
                    self.send_json(state.propose_objects(index, label=label, max_objects=max_objects))
                    return
                if parsed.path == "/api/imagegen-truth":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    label = query.get("label", ["grain"])[0] or "grain"
                    self.send_json(state.load_imagegen_truth(index, label=label))
                    return
                self.send_json({"error": "not found"}, status=404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if parsed.path == "/api/save":
                    result = state.save_annotation(int(payload["index"]), payload.get("objects", []))
                    self.send_json(result)
                    return
                if parsed.path == "/api/vectorize-mask":
                    data_url = str(payload.get("data_url", ""))
                    if "," not in data_url:
                        raise ValueError("Mask payload must be a data URL.")
                    encoded = data_url.split(",", 1)[1]
                    mask_bytes = base64.b64decode(encoded)
                    mask_image = Image.open(io.BytesIO(mask_bytes))
                    result = state.vectorize_truth_mask(
                        int(payload["index"]),
                        mask_image,
                        label=str(payload.get("label") or "grain"),
                        source=f"uploaded_imagegen_mask:{payload.get('filename') or 'mask'}",
                    )
                    self.send_json(result)
                    return
                self.send_json({"error": "not found"}, status=404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--path-col", default="source_path")
    parser.add_argument("--label-col", default="class_label")
    parser.add_argument("--target-label")
    parser.add_argument("--name-contains")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("ml_detection/grain_segmentation_labels"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    records = read_image_records(args)
    state = GrainLabelerState(records, args.output_dir)
    state.write_manifest()
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"GeoVue grain labeler: {url}")
    print(f"Images: {len(records)}")
    print(f"Output: {state.output_dir.resolve()}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
