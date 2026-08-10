import { uploadDocument, streamReview, runMockReview } from './api.js';

// Application state
const state = {
  activeView: 'submission',
  isMockMode: false,
  currentFile: null,
  isReviewing: false,
  reviewData: null,
};

// DOM Element references
const navSubmission = document.getElementById('nav-submission');
const navConsole = document.getElementById('nav-console');
const navReport = document.getElementById('nav-report');
const btnNewSubmission = document.getElementById('btn-new-submission');

const viewSubmission = document.getElementById('view-submission');
const viewConsole = document.getElementById('view-console');
const viewReport = document.getElementById('view-report');
const pageTitle = document.getElementById('page-title');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileNameEl = document.getElementById('file-name');
const fileMetaEl = document.getElementById('file-meta');
const btnRemoveFile = document.getElementById('btn-remove-file');
const btnStartReview = document.getElementById('btn-start-review');

const btnToggleMock = document.getElementById('btn-toggle-mock');
const mockIndicator = document.getElementById('mock-indicator');
const mockLabel = document.getElementById('mock-label');
const backendStatusBadge = document.getElementById('backend-status-badge');

const terminalOutput = document.getElementById('terminal-output');
const currentPaperTitle = document.getElementById('current-paper-title');
const currentPaperFilename = document.getElementById('current-paper-filename');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');

const statusMethodology = document.getElementById('status-methodology');
const statusNovelty = document.getElementById('status-novelty');
const statusClarity = document.getElementById('status-clarity');
const statusLimitations = document.getElementById('status-limitations');
const statusJudge = document.getElementById('status-judge');

// Report elements
const reportVerdictBadge = document.getElementById('report-verdict-badge');
const reportConfidence = document.getElementById('report-confidence');
const reportPaperTitle = document.getElementById('report-paper-title');
const reportVenue = document.getElementById('report-venue');
const reportFinalScore = document.getElementById('report-final-score');
const scoreNovelty = document.getElementById('score-novelty');
const scoreMethodology = document.getElementById('score-methodology');
const scoreClarity = document.getElementById('score-clarity');
const scoreLimitations = document.getElementById('score-limitations');
const reportSummary = document.getElementById('report-summary');
const reportStrengths = document.getElementById('report-strengths');
const reportWeaknesses = document.getElementById('report-weaknesses');
const reportRevisions = document.getElementById('report-revisions');
const btnExportJson = document.getElementById('btn-export-json');
const btnNewReviewFromReport = document.getElementById('btn-new-review-from-report');

// View Switcher
function switchView(target) {
  state.activeView = target;

  viewSubmission.classList.add('hidden');
  viewConsole.classList.add('hidden');
  viewReport.classList.add('hidden');

  navSubmission.classList.remove('bg-surface-container/70', 'text-primary', 'font-semibold');
  navConsole.classList.remove('bg-surface-container/70', 'text-primary', 'font-semibold');
  navReport.classList.remove('bg-surface-container/70', 'text-primary', 'font-semibold');

  navSubmission.classList.add('text-on-surface-variant');
  navConsole.classList.add('text-on-surface-variant');
  navReport.classList.add('text-on-surface-variant');

  if (target === 'submission') {
    viewSubmission.classList.remove('hidden');
    navSubmission.classList.add('bg-surface-container/70', 'text-primary', 'font-semibold');
    navSubmission.classList.remove('text-on-surface-variant');
    pageTitle.textContent = 'Submission Center';
  } else if (target === 'console') {
    viewConsole.classList.remove('hidden');
    navConsole.classList.add('bg-surface-container/70', 'text-primary', 'font-semibold');
    navConsole.classList.remove('text-on-surface-variant');
    pageTitle.textContent = 'Live Review Console';
  } else if (target === 'report') {
    viewReport.classList.remove('hidden');
    navReport.classList.add('bg-surface-container/70', 'text-primary', 'font-semibold');
    navReport.classList.remove('text-on-surface-variant');
    pageTitle.textContent = 'Final Synthesis Report';
  }
}

// Event Listeners for Navigation
navSubmission.addEventListener('click', () => switchView('submission'));
navConsole.addEventListener('click', () => switchView('console'));
navReport.addEventListener('click', () => switchView('report'));
btnNewSubmission.addEventListener('click', () => switchView('submission'));
btnNewReviewFromReport.addEventListener('click', () => switchView('submission'));

// Toggle Mock Mode
btnToggleMock.addEventListener('click', () => {
  state.isMockMode = !state.isMockMode;
  if (state.isMockMode) {
    mockIndicator.className = 'w-2 h-2 rounded-full bg-amber-500';
    mockLabel.textContent = 'Standalone Mode (Mock)';
    backendStatusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-100 text-amber-800';
    backendStatusBadge.textContent = 'Mock Mode';
  } else {
    mockIndicator.className = 'w-2 h-2 rounded-full bg-emerald-500';
    mockLabel.textContent = 'Backend Active (Live)';
    backendStatusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800';
    backendStatusBadge.textContent = 'Live Fast API';
  }
});

// File Dropzone Handling
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('border-primary', 'bg-surface-container/70');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('border-primary', 'bg-surface-container/70');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('border-primary', 'bg-surface-container/70');
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileSelect(e.target.files[0]);
  }
});

function handleFileSelect(file) {
  state.currentFile = file;
  fileNameEl.textContent = file.name;
  fileMetaEl.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • ${file.type || 'Document'}`;
  fileInfo.classList.remove('hidden');
}

btnRemoveFile.addEventListener('click', () => {
  state.currentFile = null;
  fileInput.value = '';
  fileInfo.classList.add('hidden');
});

// Click handler for Example PDF to auto-select and auto-run AI Agent review
const btnExamplePdf = document.getElementById('btn-example-pdf');
if (btnExamplePdf) {
  btnExamplePdf.addEventListener('click', async () => {
    try {
      btnExamplePdf.classList.add('pointer-events-none', 'opacity-70');
      
      let file;
      try {
        const response = await fetch('/example.pdf');
        if (!response.ok) throw new Error('Example PDF not found on server');
        const blob = await response.blob();
        file = new File([blob], 'example_paper.pdf', { type: 'application/pdf' });
      } catch (err) {
        console.warn('Could not fetch example.pdf from server, using in-memory fallback', err);
        // Fallback dummy PDF file so that mock/standalone mode works perfectly offline
        const dummyPdf = "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 40 >>\nstream\nBT /F1 12 Tf 72 712 Td (Example PDF) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000213 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n304\n%%EOF\n";
        const blob = new Blob([dummyPdf], { type: 'application/pdf' });
        file = new File([blob], 'example_paper.pdf', { type: 'application/pdf' });
      }

      handleFileSelect(file);
      
      // Directly start the review process
      btnStartReview.click();
      
    } catch (err) {
      alert(`Error loading example PDF: ${err.message}`);
    } finally {
      btnExamplePdf.classList.remove('pointer-events-none', 'opacity-70');
    }
  });
}


// Terminal Logging Helper
function appendLog(message, type = 'info', agent = null) {
  const line = document.createElement('div');
  line.className = 'mb-1 leading-relaxed';

  const timeStr = new Date().toLocaleTimeString();

  let icon = 'ℹ';
  let iconColor = 'text-terminal-accent';

  if (type === 'success') {
    icon = '✔';
    iconColor = 'text-emerald-400';
  } else if (type === 'warn') {
    icon = '⚡';
    iconColor = 'text-amber-400';
  } else if (type === 'error') {
    icon = '✖';
    iconColor = 'text-rose-400';
  }

  const agentPrefix = agent ? `[${agent.toUpperCase()}] ` : '';
  line.innerHTML = `<span class="${iconColor}">${icon}</span> <span class="text-[#484f58]">[${timeStr}]</span> <span class="text-slate-300 font-bold">${agentPrefix}</span>${message}`;

  terminalOutput.appendChild(line);
  terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// Reset Agent Status Pipeline
function resetPipeline() {
  terminalOutput.innerHTML = '<div class="text-[#484f58]"># Session initialized...</div>';
  progressBar.style.width = '0%';
  progressPercent.textContent = '0%';

  const setStatus = (el, text, colorClass) => {
    el.textContent = text;
    el.className = `font-mono px-2 py-0.5 rounded text-[10px] ${colorClass}`;
  };

  setStatus(statusMethodology, 'Queued', 'bg-amber-100 text-amber-800');
  setStatus(statusNovelty, 'Queued', 'bg-amber-100 text-amber-800');
  setStatus(statusClarity, 'Queued', 'bg-amber-100 text-amber-800');
  setStatus(statusLimitations, 'Queued', 'bg-amber-100 text-amber-800');
  setStatus(statusJudge, 'Waiting', 'bg-slate-200 text-slate-700');
}

// Start Peer Review Process
btnStartReview.addEventListener('click', async () => {
  if (!state.currentFile && !state.isMockMode) {
    alert('Please select a file to upload first.');
    return;
  }

  const filename = state.currentFile ? state.currentFile.name : 'Attention_Is_All_You_Need.pdf';
  currentPaperTitle.textContent = filename.replace(/\.[^/.]+$/, "").replace(/_/g, " ");
  currentPaperFilename.textContent = filename;

  switchView('console');
  resetPipeline();
  state.isReviewing = true;

  if (state.isMockMode) {
    appendLog('Running in Standalone Mock mode (FastAPI backend toggle available in top bar).');
    runMockReview(filename, handleReviewEvent, handleReviewComplete);
  } else {
    try {
      appendLog(`Uploading file "${filename}" to /upload-single/ endpoint...`);
      const uploadRes = await uploadDocument(state.currentFile);
      appendLog(`Upload success: ${uploadRes.message}`, 'success');
      appendLog('Initiating CrewAI Event Stream from /review/...');

      await streamReview(handleReviewEvent, (err) => {
        appendLog(`Execution error: ${err.message}`, 'error');
        alert(`Review error: ${err.message}`);
      }, handleReviewComplete);
    } catch (err) {
      appendLog(`Upload or Review failed: ${err.message}`, 'error');
      alert(`Backend connection failed: ${err.message}`);
    }
  }
});

// Process SSE & Log Events
let eventCount = 0;
function handleReviewEvent(data) {
  eventCount++;
  const progressVal = Math.min(Math.round((eventCount / 12) * 100), 95);
  progressBar.style.width = `${progressVal}%`;
  progressPercent.textContent = `${progressVal}%`;

  if (data.event === 'started') {
    appendLog(data.message || 'Multi-agent crew initialized...', 'info');
  } else if (data.event === 'log') {
    const isDone = data.status === 'completed' || (data.message && (data.message.includes('complete') || data.message.includes('finished') || data.message.includes('✔')));
    appendLog(data.message, isDone ? 'success' : 'info', data.agent);

    // Dynamic sidebar badge updater per agent
    const updateBadge = (el, isCompleted, activeText = 'Analyzing') => {
      if (isCompleted) {
        el.textContent = 'Complete';
        el.className = 'font-mono px-2 py-0.5 rounded text-[10px] bg-emerald-100 text-emerald-800 font-bold';
      } else {
        el.textContent = activeText;
        el.className = 'font-mono px-2 py-0.5 rounded text-[10px] bg-blue-100 text-blue-800 font-semibold animate-pulse';
      }
    };

    if (data.agent) {
      const agentKey = data.agent.toUpperCase();
      if (agentKey.includes('METHOD')) {
        updateBadge(statusMethodology, isDone);
      } else if (agentKey.includes('NOVEL')) {
        updateBadge(statusNovelty, isDone);
      } else if (agentKey.includes('CLARIT')) {
        updateBadge(statusClarity, isDone);
      } else if (agentKey.includes('LIMIT')) {
        updateBadge(statusLimitations, isDone);
      } else if (agentKey.includes('JUDGE')) {
        updateBadge(statusJudge, isDone, 'Synthesizing');
      }
    }
  } else if (data.event === 'completed') {
    state.reviewData = data.review || data;
    appendLog('Synthesis complete! Generating final report...', 'success', 'JUDGE');
  }
}

// Review Complete Handler
function handleReviewComplete() {
  progressBar.style.width = '100%';
  progressPercent.textContent = '100%';
  state.isReviewing = false;

  const setDone = (el) => {
    el.textContent = 'Complete';
    el.className = 'font-mono px-2 py-0.5 rounded text-[10px] bg-emerald-100 text-emerald-800 font-bold';
  };
  setDone(statusMethodology);
  setDone(statusNovelty);
  setDone(statusClarity);
  setDone(statusLimitations);
  setDone(statusJudge);

  if (state.reviewData) {
    populateReportView(state.reviewData);
    setTimeout(() => {
      switchView('report');
    }, 800);
  }
}

// Populate Final Synthesis Report View
function populateReportView(review) {
  reportPaperTitle.textContent = review.paper_title || 'Attention Is All You Need';
  reportVenue.textContent = `Suggested Venue: ${review.suggested_venue || 'N/A'}`;
  reportConfidence.textContent = `${review.confidence || 90}%`;

  const verdict = review.final_verdict || 'ACCEPT';
  reportVerdictBadge.textContent = verdict;

  if (verdict === 'ACCEPT' || verdict === 'WEAK_ACCEPT') {
    reportVerdictBadge.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-100 text-emerald-800 border border-emerald-300';
  } else {
    reportVerdictBadge.className = 'px-3 py-1 rounded-full text-xs font-mono font-bold bg-rose-100 text-rose-800 border border-rose-300';
  }

  reportFinalScore.innerHTML = `${(review.weighted_final_score || 8.0).toFixed(2)}<span class="text-base font-normal text-on-surface-variant">/10</span>`;

  const scores = review.individual_scores || {};
  scoreNovelty.textContent = `${(scores.novelty || 8.0).toFixed(1)}/10`;
  scoreMethodology.textContent = `${(scores.methodology || 8.0).toFixed(1)}/10`;
  scoreClarity.textContent = `${(scores.clarity || 8.0).toFixed(1)}/10`;
  scoreLimitations.textContent = `${(scores.limitations || 7.5).toFixed(1)}/10`;

  reportSummary.textContent = review.summary_for_authors || 'No summary provided.';

  reportStrengths.innerHTML = (review.top_strengths || ['High novelty score']).map(s => `<li>${s}</li>`).join('');
  reportWeaknesses.innerHTML = (review.top_weaknesses || ['Minor clarity issues']).map(w => `<li>${w}</li>`).join('');
  reportRevisions.innerHTML = (review.mandatory_revisions || ['Clarify dataset splits']).map(r => `<li>${r}</li>`).join('');
}

// JSON Export
btnExportJson.addEventListener('click', () => {
  if (!state.reviewData) return;
  const jsonStr = JSON.stringify(state.reviewData, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `review-synthesis-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// Set the backend target hostname dynamically
const backendTargetLabel = document.getElementById('backend-target-label');
if (backendTargetLabel) {
  backendTargetLabel.textContent = window.location.host || 'localhost:8005';
}
