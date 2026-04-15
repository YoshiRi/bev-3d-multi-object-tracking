/**
 * BEV Visualizer
 * Renders ground-truth and tracked objects on an HTML5 canvas in Bird's-Eye View.
 *
 * Coordinate system:
 *   BEV  : x = forward (right on screen), y = left (up on screen)
 *   Canvas: x = right, y = down  →  canvas_y = f(bounds.y_max - bev_y)
 */

const Visualizer = (() => {
  // Colour palette derived from track-id hash
  function trackHue(id) {
    let h = 0;
    for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
    return ((h >>> 0) % 360);
  }

  function trackColor(id, confirmed) {
    const hue = trackHue(id);
    return confirmed
      ? `hsl(${hue},75%,62%)`
      : `hsl(${hue},40%,48%)`;
  }

  // -----------------------------------------------------------------------
  // Viewport helpers (computed once per render call)
  // -----------------------------------------------------------------------
  let _scale = 1;
  let _ox = 0;    // canvas x-offset (letterbox)
  let _oy = 0;    // canvas y-offset (letterbox)
  let _bounds = null;

  function setViewport(canvas, bounds) {
    _bounds = bounds;
    const sceneW = bounds.x_max - bounds.x_min;
    const sceneH = bounds.y_max - bounds.y_min;
    const sx = canvas.width  / sceneW;
    const sy = canvas.height / sceneH;
    _scale = Math.min(sx, sy);
    _ox = (canvas.width  - sceneW * _scale) / 2;
    _oy = (canvas.height - sceneH * _scale) / 2;
  }

  function toCanvas(bx, by) {
    return [
      _ox + (bx - _bounds.x_min) * _scale,
      _oy + (_bounds.y_max - by) * _scale,
    ];
  }

  // -----------------------------------------------------------------------
  // Drawing primitives
  // -----------------------------------------------------------------------

  /** Draw a yaw-oriented bounding box centred at (bx, by). */
  function drawBox(ctx, bx, by, yaw, length, width, color, dashed) {
    const [px, py] = toCanvas(bx, by);
    const l = length * _scale;
    const w = width  * _scale;

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-yaw);          // flip because canvas y is inverted

    ctx.beginPath();
    ctx.rect(-l / 2, -w / 2, l, w);
    ctx.strokeStyle = color;
    ctx.lineWidth = dashed ? 1.5 : 2;
    if (dashed) ctx.setLineDash([5, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Direction arrow (forward = +x in BEV = +l direction in local frame)
    const arrowLen = Math.min(l * 0.45, 14);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(arrowLen, 0);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Arrow head
    ctx.beginPath();
    ctx.moveTo(arrowLen, 0);
    ctx.lineTo(arrowLen - 4, -3);
    ctx.lineTo(arrowLen - 4,  3);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    ctx.restore();
  }

  /** Draw a grid in 10-m intervals. */
  function drawGrid(ctx, canvas) {
    const step = 10;
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;

    const xStart = Math.ceil(_bounds.x_min / step) * step;
    const xEnd   = Math.floor(_bounds.x_max / step) * step;
    for (let x = xStart; x <= xEnd; x += step) {
      const [px] = toCanvas(x, 0);
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, canvas.height);
      ctx.stroke();
    }

    const yStart = Math.ceil(_bounds.y_min / step) * step;
    const yEnd   = Math.floor(_bounds.y_max / step) * step;
    for (let y = yStart; y <= yEnd; y += step) {
      const [, py] = toCanvas(0, y);
      ctx.beginPath();
      ctx.moveTo(0, py);
      ctx.lineTo(canvas.width, py);
      ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '10px monospace';
    for (let x = xStart; x <= xEnd; x += step * 2) {
      const [px, py] = toCanvas(x, _bounds.y_min);
      ctx.fillText(`${x}m`, px + 2, py - 3);
    }
  }

  /** Draw a scale bar. */
  function drawScaleBar(ctx, canvas) {
    const barM = 20;       // 20 m scale bar
    const barPx = barM * _scale;
    const margin = 14;
    const y = canvas.height - margin;
    const x = canvas.width  - margin - barPx;

    ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + barPx, y);
    ctx.moveTo(x,        y - 4);
    ctx.lineTo(x,        y + 4);
    ctx.moveTo(x + barPx, y - 4);
    ctx.lineTo(x + barPx, y + 4);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${barM} m`, x + barPx / 2, y - 6);
    ctx.textAlign = 'left';
  }

  /** Draw trails for all tracked objects. */
  function drawTrails(ctx, trails) {
    for (const [id, pts] of Object.entries(trails)) {
      if (pts.length < 2) continue;
      const hue = trackHue(id);
      for (let i = 1; i < pts.length; i++) {
        const alpha = (i / pts.length) * 0.5;
        const [x1, y1] = toCanvas(pts[i - 1].x, pts[i - 1].y);
        const [x2, y2] = toCanvas(pts[i].x,     pts[i].y);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `hsla(${hue},70%,60%,${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  /** Draw an ID label above the box. */
  function drawLabel(ctx, bx, by, length, text, color) {
    const [px, py] = toCanvas(bx, by);
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    // Offset above the box top
    ctx.fillText(text, px, py - length * _scale / 2 - 4);
    ctx.textAlign = 'left';
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Render one frame onto the canvas.
   *
   * @param {HTMLCanvasElement} canvas
   * @param {object}  bounds   - {x_min, x_max, y_min, y_max}
   * @param {object}  frame    - {gt: [...], tracks: [...]}
   * @param {object}  trails   - mutable {track_id: [{x,y}, ...]}
   * @param {number}  frameIdx
   * @param {number}  totalFrames
   */
  function render(canvas, bounds, frame, trails, frameIdx, totalFrames) {
    const ctx = canvas.getContext('2d');
    setViewport(canvas, bounds);

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Letterbox fill
    ctx.fillStyle = '#050709';
    if (_ox > 0) {
      ctx.fillRect(0, 0, _ox, canvas.height);
      ctx.fillRect(canvas.width - _ox, 0, _ox, canvas.height);
    }
    if (_oy > 0) {
      ctx.fillRect(0, 0, canvas.width, _oy);
      ctx.fillRect(0, canvas.height - _oy, canvas.width, _oy);
    }

    drawGrid(ctx, canvas);

    // Update trails with current track positions
    for (const t of frame.tracks) {
      if (!trails[t.id]) trails[t.id] = [];
      trails[t.id].push({ x: t.x, y: t.y });
      if (trails[t.id].length > 40) trails[t.id].shift();
    }
    drawTrails(ctx, trails);

    // Ground truth (green dashed)
    for (const g of frame.gt) {
      drawBox(ctx, g.x, g.y, g.yaw, g.l, g.w, '#00e676', true);
    }

    // Tracks
    for (const t of frame.tracks) {
      const color = trackColor(t.id, t.confirmed);
      drawBox(ctx, t.x, t.y, t.yaw, t.l, t.w, color, !t.confirmed);
      drawLabel(ctx, t.x, t.y, t.l, t.id.slice(0, 5), color);
    }

    // Frame counter overlay
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '12px monospace';
    ctx.fillText(`Frame ${frameIdx + 1} / ${totalFrames}`, _ox + 6, _oy + 16);

    drawScaleBar(ctx, canvas);
  }

  return { render };
})();
