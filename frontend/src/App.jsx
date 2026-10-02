import { useEffect, useRef, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, BarChart, Bar,
} from 'recharts';

const API = 'http://localhost:8000';

async function request(path, options) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || 'Request failed');
  return body;
}

function App() {
  const [health, setHealth] = useState(null);
  const [fitResult, setFitResult] = useState(null);
  const [threshold, setThreshold] = useState(1);
  const [thresholdBounds, setThresholdBounds] = useState({ min: 0, max: 2 });
  const [source, setSource] = useState('webcam');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [rawFrame, setRawFrame] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [heatGrid, setHeatGrid] = useState([]);
  const [question, setQuestion] = useState('');
  const [chatItems, setChatItems] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [videoUrl, setVideoUrl] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const pendingRef = useRef(false);
  const timerRef = useRef(null);
  const thresholdRef = useRef(threshold);

  useEffect(() => { thresholdRef.current = threshold; }, [threshold]);

  useEffect(() => {
    const refresh = async () => {
      try {
        const [healthData, statsData, historyData, alertData, gridData, thresholdData] = await Promise.all([
          request('/health'), request('/stats'), request('/history?limit=20'),
          request('/alerts'), request('/cumulative_heatmap'), request('/threshold'),
        ]);
        setHealth(healthData);
        setStats(statsData);
        setHistory(historyData);
        setAlerts(alertData);
        setHeatGrid(gridData.grid);
        setThreshold(thresholdData.threshold);
        setThresholdBounds({ min: thresholdData.min, max: Math.max(thresholdData.max, thresholdData.threshold * 1.1, 0.01) });
      } catch (e) { setError(e.message); }
    };
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => () => {
    runningRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    if (videoUrl) URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  async function fitFromDataset() {
    setBusy(true); setError('');
    try {
      const response = await request('/fit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ use_dataset: true }) });
      applyCalibration(response);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function fitUploads(event) {
    const files = [...event.target.files];
    if (!files.length) return;
    setBusy(true); setError('');
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    try { applyCalibration(await request('/fit', { method: 'POST', body: form })); }
    catch (e) { setError(e.message); }
    finally { setBusy(false); event.target.value = ''; }
  }

  function applyCalibration(value) {
    setFitResult(value);
    setThreshold(value.threshold);
    setThresholdBounds({ min: value.calibration.min, max: Math.max(value.calibration.max * 1.25, value.threshold * 1.1, 0.01) });
  }

  async function chooseSource(next) {
    stopInspecting();
    setSource(next);
    if (next === 'webcam') {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        streamRef.current = stream;
        if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      } catch (e) { setError(`Camera unavailable: ${e.message}`); }
    } else {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  function chooseVideo(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
  }

  function stopInspecting() {
    runningRef.current = false;
    setRunning(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
  }

  async function sendFrame() {
    if (!runningRef.current || pendingRef.current) return scheduleNext();
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return scheduleNext();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    setRawFrame(dataUrl);
    const blob = await (await fetch(dataUrl)).blob();
    const form = new FormData();
    form.append('frame', blob, 'frame.jpg');
    form.append('source', source);
    pendingRef.current = true;
    try { setResult(await request('/inspect', { method: 'POST', body: form })); setError(''); }
    catch (e) { setError(e.message); stopInspecting(); }
    finally { pendingRef.current = false; }
    scheduleNext();
  }

  function scheduleNext() {
    if (runningRef.current) timerRef.current = setTimeout(sendFrame, 333);
  }

  async function toggleInspecting() {
    if (runningRef.current) return stopInspecting();
    if (source === 'webcam' && !streamRef.current) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        streamRef.current = stream;
        if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      } catch (e) { setError(`Camera unavailable: ${e.message}`); return; }
    }
    if (source === 'video' && videoRef.current?.paused) {
      try { await videoRef.current.play(); }
      catch (e) { setError(`Video could not start: ${e.message}`); return; }
    }
    runningRef.current = true;
    setRunning(true);
    sendFrame();
  }

  async function saveThreshold() {
    try { await request('/threshold', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ threshold: Number(thresholdRef.current) }) }); }
    catch (e) { setError(e.message); }
  }

  async function confirmFalseAlarm() {
    if (!result?.id || !window.confirm('Confirm this FAIL was a false alarm? Its frame will be added to the normal memory bank.')) return;
    try {
      const response = await request(`/feedback/${result.id}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirm: true }) });
      setError(`False alarm recorded. Memory bank now has ${response.bank_size} patches.`);
    } catch (e) { setError(e.message); }
  }

  async function askQuestion(event) {
    event.preventDefault();
    if (!question.trim()) return;
    const asked = question.trim();
    setQuestion('');
    setChatItems((items) => [...items, { question: asked, answer: 'Thinking...' }]);
    try {
      const response = await request('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: asked }) });
      setChatItems((items) => items.map((item, index) => index === items.length - 1 ? { ...item, answer: response.answer } : item));
    } catch (e) { setError(e.message); }
  }

  const chartData = (stats?.score_trend || []).map((point) => ({ ...point, shortTime: new Date(point.ts).toLocaleTimeString() }));
  const displayImage = result?.heatmap_png_base64 ? `data:image/png;base64,${result.heatmap_png_base64}` : rawFrame;
  const verdictClass = result?.verdict?.toLowerCase() || '';

  return (
    <main>
      <h1>VisionQC</h1>
      <p>Local visual defect inspection</p>
      {error && <p role="alert">{error}</p>}
      <p>Backend: {health?.fitted ? 'Model fitted' : health?.model_loaded ? 'Model loaded, fit required' : 'Not fitted'}{health?.model_error ? ` (${health.model_error})` : ''}</p>

      <section>
        <h2>Setup</h2>
        <button disabled={busy} onClick={fitFromDataset}>Fit from dataset</button>{' '}
        <label>Upload good photos <input type="file" accept="image/*" multiple onChange={fitUploads} /></label>
        {fitResult && <p>Fit {fitResult.fit.images} images; {fitResult.fit.patches} patches. Calibration mean {fitResult.calibration.mean.toFixed(4)}, std {fitResult.calibration.std.toFixed(4)}. Suggested threshold {fitResult.suggested_threshold.toFixed(4)}.</p>}
      </section>

      <section>
        <h2>Source</h2>
        <label><input type="radio" name="source" checked={source === 'webcam'} onChange={() => chooseSource('webcam')} /> Webcam</label>{' '}
        <label><input type="radio" name="source" checked={source === 'video'} onChange={() => chooseSource('video')} /> Video file</label>{' '}
        {source === 'video' && <input type="file" accept="video/*" onChange={chooseVideo} />}
        <div><video ref={videoRef} src={source === 'video' ? videoUrl : undefined} muted playsInline controls={source === 'video'} /></div>
        <button onClick={toggleInspecting}>{running ? 'Stop inspecting' : 'Start inspecting'}</button>
        <canvas ref={canvasRef} hidden />
      </section>

      <section>
        <h2>Result</h2>
        {result && <>
          <strong className={`verdict ${verdictClass}`}>{result.verdict}</strong>
          <p>Confidence: {result.confidence_pct.toFixed(1)}% | Score: {result.score.toFixed(4)}</p>
          <p>{result.reason}</p>
          {displayImage && <img className="result-image" src={displayImage} alt={result.heatmap_png_base64 ? 'Anomaly heatmap overlay' : 'Most recent camera frame'} />}
          {result.verdict === 'FAIL' && <button onClick={confirmFalseAlarm}>False alarm</button>}
        </>}
      </section>

      <section>
        <h2>Threshold</h2>
        <label>Current: {Number(threshold).toFixed(4)}{' '}
          <input type="range" min={thresholdBounds.min} max={thresholdBounds.max} step={Math.max((thresholdBounds.max - thresholdBounds.min) / 500, 0.0001)} value={Math.min(threshold, thresholdBounds.max)} onChange={(event) => setThreshold(Number(event.target.value))} onPointerUp={saveThreshold} onKeyUp={saveThreshold} />
        </label>
      </section>

      <section>
        <h2>Dashboard</h2>
        {stats && <p>Today rejection rate: {(stats.rejection_rate_today * 100).toFixed(1)}% | Pass: {stats.pass} | Fail: {stats.fail} | Recapture: {stats.recapture} | Recapture rate: {(stats.recapture_rate * 100).toFixed(1)}%</p>}
        {stats?.drift?.warn && <p role="alert">Score drift is above calibration baseline.</p>}
        <div className="chart"><ResponsiveContainer width="100%" height={260}><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="shortTime" /><YAxis /><Tooltip /><ReferenceLine y={threshold} stroke="red" label="Threshold" /><Line type="monotone" dataKey="score" stroke="#225c35" dot={false} /></LineChart></ResponsiveContainer></div>
        <h3>Recapture reasons</h3>
        <div className="chart"><ResponsiveContainer width="100%" height={200}><BarChart data={stats?.recapture_reasons || []}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="reason" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" fill="#c9871a" /></BarChart></ResponsiveContainer></div>
      </section>

      <section>
        <h2>Repeat defect</h2>
        {alerts.map((alert) => <p className="alert" key={alert.id}>{alert.message}</p>)}
        <table className="grid"><tbody>{heatGrid.map((row, y) => <tr key={y}>{row.map((count, x) => <td key={x} title={`${count} fails`} style={{ backgroundColor: count ? `rgba(210, 35, 35, ${Math.min(0.2 + count * 0.15, 0.9)})` : '#f2f2f2' }}>{count || ''}</td>)}</tr>)}</tbody></table>
      </section>

      <section>
        <h2>Chat</h2>
        <form onSubmit={askQuestion}><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about inspection data" /><button type="submit">Send</button></form>
        {chatItems.map((item, index) => <div key={index}><p><b>Q:</b> {item.question}</p><p><b>A:</b> {item.answer}</p></div>)}
      </section>

      <section>
        <h2>History</h2>
        <table><thead><tr><th>Time</th><th>Verdict</th><th>Score</th><th>Reason</th></tr></thead><tbody>{history.map((row) => <tr key={row.id}><td>{new Date(row.ts).toLocaleString()}</td><td>{row.verdict}</td><td>{Number(row.score).toFixed(4)}</td><td>{row.reason}</td></tr>)}</tbody></table>
      </section>
    </main>
  );
}

export default App;
