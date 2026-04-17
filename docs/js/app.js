/**
 * app.js – Pyodide bootstrap + playback controller
 *
 * Flow:
 *   1. Load Pyodide + numpy/scipy
 *   2. Fetch and exec docs/py/demo_tracker.py
 *   3. User adjusts sliders → clicks "Run"
 *   4. Call Python run_demo(params_json) → JSON result
 *   5. Parse result, start animation loop
 */

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------
let pyodide      = null;
let simData      = null;   // parsed JSON from run_demo / run_demo_kitti
let frameIdx     = 0;
let playing      = false;
let playTimer    = null;
let playFps      = 10;
let trails       = {};
let kittiContent = null;   // raw text of the loaded KITTI label file
let dataSource   = 'synthetic';

const canvas     = document.getElementById('bev-canvas');

// -------------------------------------------------------------------------
// Pyodide loading
// -------------------------------------------------------------------------
async function initPyodide() {
  setStatus('Pyodide を読み込んでいます…');
  try {
    pyodide = await loadPyodide();
    setStatus('numpy / scipy をインストール中…');
    await pyodide.loadPackage(['numpy', 'scipy']);

    setStatus('トラッカーコードを読み込み中…');
    const resp = await fetch('py/demo_tracker.py');
    if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
    const code = await resp.text();
    await pyodide.runPythonAsync(code);

    setStatus('準備完了');
    document.getElementById('run-btn').disabled = false;
  } catch (err) {
    setStatus('初期化失敗: ' + err.message);
    console.error(err);
  }
}

// -------------------------------------------------------------------------
// Parameter collection
// -------------------------------------------------------------------------
function getParams() {
  function val(id) { return document.getElementById(id).value; }
  return {
    scenario:           val('p-scenario'),
    n_frames:           parseInt(val('p-n-frames')),
    min_hits_to_confirm: parseInt(val('p-min-hits')),
    max_misses_to_lose:  parseInt(val('p-max-misses')),
    max_distance:        parseFloat(val('p-max-dist')),
    pos_noise_std:       parseFloat(val('p-noise')),
    detection_rate:      parseFloat(val('p-det-rate')),
    seed:                parseInt(val('p-seed')),
  };
}

// -------------------------------------------------------------------------
// Run simulation
// -------------------------------------------------------------------------
async function runSimulation() {
  if (!pyodide) return;

  if (dataSource === 'kitti' && !kittiContent) {
    setStatus('KITTIファイルを選択してください');
    return;
  }

  pausePlayback();
  document.getElementById('run-btn').disabled = true;
  setStatus('シミュレーション実行中…');

  try {
    const params = getParams();
    pyodide.globals.set('_demo_params', JSON.stringify(params));

    let resultJson;
    if (dataSource === 'kitti') {
      pyodide.globals.set('_kitti_content', kittiContent);
      resultJson = await pyodide.runPythonAsync(
        'run_demo_kitti(_kitti_content, _demo_params)'
      );
    } else {
      resultJson = await pyodide.runPythonAsync('run_demo(_demo_params)');
    }
    simData = JSON.parse(resultJson);

    if (simData.error) {
      setStatus('エラー: ' + simData.error);
      document.getElementById('run-btn').disabled = false;
      return;
    }

    frameIdx = 0;
    trails = {};

    // Setup slider range
    const slider = document.getElementById('frame-slider');
    slider.max   = simData.frames.length - 1;
    slider.value = 0;

    updateMetricsPanel(simData.metrics);
    renderCurrentFrame();
    setStatus(`完了 — ${simData.frames.length} フレーム`);
  } catch (err) {
    setStatus('エラー: ' + err.message);
    console.error(err);
  } finally {
    document.getElementById('run-btn').disabled = false;
  }
}

// -------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------
function renderCurrentFrame() {
  if (!simData) return;
  const frame = simData.frames[frameIdx];
  Visualizer.render(canvas, simData.bounds, frame, trails, frameIdx, simData.frames.length);
  document.getElementById('frame-slider').value = frameIdx;
  document.getElementById('frame-label').textContent =
    `${frameIdx + 1} / ${simData.frames.length}`;
}

// -------------------------------------------------------------------------
// Playback controls
// -------------------------------------------------------------------------
function startPlayback() {
  if (playing || !simData) return;
  playing = true;
  document.getElementById('play-btn').textContent = '⏸ 一時停止';
  playTimer = setInterval(() => {
    if (frameIdx >= simData.frames.length - 1) {
      pausePlayback();
      return;
    }
    frameIdx++;
    renderCurrentFrame();
  }, 1000 / playFps);
}

function pausePlayback() {
  clearInterval(playTimer);
  playTimer = null;
  playing = false;
  document.getElementById('play-btn').textContent = '▶ 再生';
}

function togglePlayback() {
  playing ? pausePlayback() : startPlayback();
}

function stepBack() {
  pausePlayback();
  if (frameIdx > 0) { frameIdx--; renderCurrentFrame(); }
}

function stepForward() {
  pausePlayback();
  if (simData && frameIdx < simData.frames.length - 1) {
    frameIdx++;
    renderCurrentFrame();
  }
}

function onSliderInput(e) {
  pausePlayback();
  frameIdx = parseInt(e.target.value);
  renderCurrentFrame();
}

function onSpeedChange(e) {
  playFps = parseFloat(e.target.value);
  if (playing) {
    pausePlayback();
    startPlayback();
  }
}

// -------------------------------------------------------------------------
// Metrics panel
// -------------------------------------------------------------------------
function updateMetricsPanel(m) {
  function set(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  set('m-mota',     (m.mota  * 100).toFixed(1) + '%');
  set('m-motp',     m.motp.toFixed(3)  + ' m');
  set('m-idsw',     m.id_switches);
  set('m-prec',     (m.precision * 100).toFixed(1) + '%');
  set('m-recall',   (m.recall   * 100).toFixed(1) + '%');
  set('m-tp',       m.tp);
  set('m-fp',       m.fp);
  set('m-fn',       m.fn);
  set('m-ngt',      m.total_gt);
  set('m-nframes',  m.n_frames);
}

// -------------------------------------------------------------------------
// Slider live-value display
// -------------------------------------------------------------------------
function bindSliders() {
  const bindings = [
    ['p-n-frames',  'v-n-frames',   v => v + ' f'],
    ['p-min-hits',  'v-min-hits',   v => v],
    ['p-max-misses','v-max-misses', v => v],
    ['p-max-dist',  'v-max-dist',   v => v + ' m'],
    ['p-noise',     'v-noise',      v => v + ' m'],
    ['p-det-rate',  'v-det-rate',   v => (v * 100).toFixed(0) + '%'],
    ['p-seed',      'v-seed',       v => v],
  ];
  for (const [inputId, labelId, fmt] of bindings) {
    const el = document.getElementById(inputId);
    const lbl = document.getElementById(labelId);
    if (!el || !lbl) continue;
    lbl.textContent = fmt(el.value);
    el.addEventListener('input', () => { lbl.textContent = fmt(el.value); });
  }
}

// -------------------------------------------------------------------------
// Status bar
// -------------------------------------------------------------------------
function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

// -------------------------------------------------------------------------
// Data source toggle & KITTI file loading
// -------------------------------------------------------------------------
function onDataSourceChange(e) {
  dataSource = e.target.value;
  document.getElementById('synthetic-controls').style.display =
    dataSource === 'synthetic' ? '' : 'none';
  document.getElementById('kitti-controls').style.display =
    dataSource === 'kitti' ? '' : 'none';
}

function onKittiFileChange(e) {
  const file = e.target.files[0];
  if (!file) return;

  document.getElementById('kitti-filename').textContent = file.name;
  setStatus('ファイルを読み込み中…');

  const reader = new FileReader();
  reader.onload = (ev) => {
    kittiContent = ev.target.result;
    const lines = kittiContent.split('\n').filter(l => l.trim()).length;
    setStatus(`読み込み完了 — ${lines} 行`);
  };
  reader.onerror = () => setStatus('ファイル読み込みエラー');
  reader.readAsText(file);
}

// -------------------------------------------------------------------------
// Boot
// -------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  bindSliders();

  document.getElementById('run-btn')      .addEventListener('click', runSimulation);
  document.getElementById('play-btn')     .addEventListener('click', togglePlayback);
  document.getElementById('step-back-btn').addEventListener('click', stepBack);
  document.getElementById('step-fwd-btn') .addEventListener('click', stepForward);
  document.getElementById('frame-slider') .addEventListener('input', onSliderInput);
  document.getElementById('speed-select') .addEventListener('change', onSpeedChange);

  document.querySelectorAll('input[name="datasource"]').forEach(
    el => el.addEventListener('change', onDataSourceChange)
  );
  document.getElementById('kitti-file').addEventListener('change', onKittiFileChange);

  initPyodide();
});
