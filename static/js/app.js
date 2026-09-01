// DocuMindRAG Frontend
const API_BASE = '';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const uploadStatus = document.getElementById('uploadStatus');
const docList = document.getElementById('docList');
const refreshBtn = document.getElementById('refreshBtn');
const queryInput = document.getElementById('queryInput');
const queryBtn = document.getElementById('queryBtn');
const loadingState = document.getElementById('loadingState');
const answerSection = document.getElementById('answerSection');
const answerContent = document.getElementById('answerContent');
const citationsList = document.getElementById('citationsList');
const timingFooter = document.getElementById('timingFooter');
const emptyState = document.getElementById('emptyState');
const statusIndicator = document.getElementById('statusIndicator');
const mobileToggle = document.getElementById('mobileToggle');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const feedbackSection = document.getElementById('feedbackSection');
const analyticsBtn = document.getElementById('analyticsBtn');
const analyticsModal = document.getElementById('analyticsModal');
const analyticsClose = document.getElementById('analyticsClose');
const analyticsContent = document.getElementById('analyticsContent');
const evalBtn = document.getElementById('evalBtn');
const evalModal = document.getElementById('evalModal');
const evalClose = document.getElementById('evalClose');
const evalContent = document.getElementById('evalContent');
const evalRunBtn = document.getElementById('evalRunBtn');
const viewerModal = document.getElementById('viewerModal');
const viewerClose = document.getElementById('viewerClose');
const viewerContent = document.getElementById('viewerContent');

let currentQueryLogId = null;
let selectedDocIds = new Set();

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDocuments();
    setupEventListeners();
});

function setupEventListeners() {
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });

    refreshBtn.addEventListener('click', loadDocuments);

    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(); }
    });
    queryBtn.addEventListener('click', runQuery);

    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => { queryInput.value = chip.dataset.query; runQuery(); });
    });

    mobileToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        sidebarOverlay.classList.toggle('active');
    });
    sidebarOverlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    });

    // Analytics modal
    analyticsBtn.addEventListener('click', showAnalytics);
    analyticsClose.addEventListener('click', () => analyticsModal.style.display = 'none');
    analyticsModal.addEventListener('click', (e) => { if (e.target === analyticsModal) analyticsModal.style.display = 'none'; });

    // Eval modal
    evalBtn.addEventListener('click', () => { evalModal.style.display = 'flex'; evalContent.innerHTML = '<p style="color:var(--text-muted)">Select a document, then click Run.</p>'; });
    evalClose.addEventListener('click', () => evalModal.style.display = 'none');
    evalModal.addEventListener('click', (e) => { if (e.target === evalModal) evalModal.style.display = 'none'; });
    evalRunBtn.addEventListener('click', runEvaluation);

    // Citation viewer modal
    viewerClose.addEventListener('click', () => { viewerModal.style.display = 'none'; viewerContent.innerHTML = ''; });
    viewerModal.addEventListener('click', (e) => { if (e.target === viewerModal) { viewerModal.style.display = 'none'; viewerContent.innerHTML = ''; } });
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        if (data.status === 'ok' && data.qdrant === 'connected') {
            statusIndicator.innerHTML = '<span class="status-dot"></span><span>Connected</span>';
        } else {
            statusIndicator.innerHTML = '<span class="status-dot disconnected"></span><span>Partial</span>';
        }
    } catch {
        statusIndicator.innerHTML = '<span class="status-dot disconnected"></span><span>Disconnected</span>';
    }
}

function handleFileSelect(e) { if (e.target.files[0]) uploadFile(e.target.files[0]); }

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        uploadStatus.className = 'upload-status error';
        uploadStatus.textContent = 'Only PDF files are supported';
        return;
    }
    uploadStatus.className = 'upload-status';
    uploadStatus.textContent = 'Uploading...';
    uploadZone.style.pointerEvents = 'none';

    const formData = new FormData();
    formData.append('file', file);
    try {
        const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (res.ok) {
            const data = await res.json();
            uploadStatus.className = 'upload-status success';
            uploadStatus.textContent = `Uploaded: ${data.name} (${data.page_count} pages, ${data.chunk_count} chunks)`;
            loadDocuments();
        } else {
            const err = await res.json();
            uploadStatus.className = 'upload-status error';
            uploadStatus.textContent = err.detail || 'Upload failed';
        }
    } catch {
        uploadStatus.className = 'upload-status error';
        uploadStatus.textContent = 'Connection error';
    } finally {
        uploadZone.style.pointerEvents = 'auto';
        fileInput.value = '';
    }
}

async function loadDocuments() {
    try {
        const res = await fetch(`${API_BASE}/documents`);
        const docs = await res.json();
        if (docs.length === 0) {
            docList.innerHTML = '<p class="empty-state">No documents uploaded</p>';
            return;
        }
        docList.innerHTML = docs.map(doc => `
            <div class="doc-item" data-id="${escapeHtml(doc.id)}">
                <div class="doc-info">
                    <input type="checkbox" class="doc-checkbox" data-id="${escapeHtml(doc.id)}"
                        ${selectedDocIds.has(doc.id) ? 'checked' : ''}
                        onchange="toggleDocSelection('${escapeHtml(doc.id)}')" aria-label="Select ${escapeHtml(doc.name)}">
                    <span class="doc-dot ${doc.status === 'ready' ? 'ok' : 'err'}"></span>
                    <span class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span>
                </div>
                <div style="display:flex;align-items:center;gap:8px">
                    <span class="doc-meta">${doc.page_count}p / ${doc.chunk_count}c</span>
                    <button class="doc-delete" onclick="deleteDocument('${escapeHtml(doc.id)}')" title="Delete">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 18L18 6M6 6l12 12"/></svg>
                    </button>
                </div>
            </div>
        `).join('');
    } catch {
        docList.innerHTML = '<p class="empty-state">Failed to load documents</p>';
    }
}

function toggleDocSelection(docId) {
    if (selectedDocIds.has(docId)) selectedDocIds.delete(docId);
    else selectedDocIds.add(docId);
}

async function deleteDocument(docId) {
    if (!confirm('Delete this document and all its chunks?')) return;
    try {
        await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
        selectedDocIds.delete(docId);
        loadDocuments();
    } catch { alert('Failed to delete document'); }
}

async function runQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    emptyState.style.display = 'none';
    answerSection.style.display = 'none';
    feedbackSection.style.display = 'none';
    loadingState.style.display = 'flex';
    queryBtn.disabled = true;
    answerContent.textContent = '';
    citationsList.innerHTML = '';
    timingFooter.innerHTML = '';
    currentQueryLogId = null;

    const docIds = selectedDocIds.size > 0 ? Array.from(selectedDocIds) : [];

    try {
        const response = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, doc_ids: docIds, top_k: 5 }),
        });

        if (!response.ok) {
            const err = await response.json();
            answerContent.textContent = `Error: ${err.detail || 'Query failed'}`;
            answerSection.style.display = 'block';
            return;
        }

        const data = await response.json();
        currentQueryLogId = data.query_log_id || null;
        displayAnswer(data);
    } catch {
        answerContent.textContent = 'Connection error';
        answerSection.style.display = 'block';
    } finally {
        loadingState.style.display = 'none';
        queryBtn.disabled = false;
    }
}

function displayAnswer(data) {
    answerContent.textContent = data.answer;

    if (data.citations && data.citations.length > 0) {
        citationsList.innerHTML = data.citations.map(c => `
            <div class="citation-item">
                <div class="citation-top">
                    <span class="citation-page">p.${c.page_number}</span>
                    <span class="citation-score">${c.retrieval_score.toFixed(3)}</span>
                    <button class="citation-view-btn" onclick="openCitationViewer('${escapeHtml(c.doc_id)}', ${c.page_number}, '${escapeHtml(c.chunk_text.substring(0, 200).replace(/'/g, "\\'"))}')">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
                        View
                    </button>
                </div>
                <div class="citation-text">${escapeHtml(c.chunk_text.substring(0, 300))}...</div>
            </div>
        `).join('');
    } else {
        citationsList.innerHTML = '<p class="empty-state">No sources found</p>';
    }

    timingFooter.innerHTML = `${data.timing_ms.toFixed(0)}ms | ${data.model_used} | ${data.citations.length} source${data.citations.length !== 1 ? 's' : ''}`;
    answerSection.style.display = 'block';
    feedbackSection.style.display = 'flex';
}

async function submitFeedback(rating) {
    if (!currentQueryLogId) return;
    const btns = feedbackSection.querySelectorAll('.feedback-btn');
    btns.forEach(b => b.disabled = true);
    try {
        await fetch(`${API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_log_id: currentQueryLogId, rating }),
        });
        feedbackSection.innerHTML = `<span class="feedback-thanks">Thanks for your feedback!</span>`;
    } catch {
        btns.forEach(b => b.disabled = false);
    }
}

async function showAnalytics() {
    analyticsModal.style.display = 'flex';
    analyticsContent.innerHTML = '<div class="loading-state"><div class="spinner"></div><span>Loading analytics...</span></div>';
    try {
        const res = await fetch(`${API_BASE}/analytics`);
        const data = await res.json();
        analyticsContent.innerHTML = `
            <div class="analytics-grid">
                <div class="analytics-card">
                    <div class="analytics-value">${data.total_queries}</div>
                    <div class="analytics-label">Total Queries</div>
                </div>
                <div class="analytics-card">
                    <div class="analytics-value">${data.avg_latency_ms.toFixed(0)}ms</div>
                    <div class="analytics-label">Avg Latency</div>
                </div>
                <div class="analytics-card">
                    <div class="analytics-value">${(data.avg_groundedness * 100).toFixed(0)}%</div>
                    <div class="analytics-label">Avg Groundedness</div>
                </div>
                <div class="analytics-card">
                    <div class="analytics-value">${data.thumbs_up} / ${data.thumbs_down}</div>
                    <div class="analytics-label">Thumbs Up / Down</div>
                </div>
            </div>
            <h4 style="margin:16px 0 8px;color:var(--text)">Recent Queries</h4>
            <div class="analytics-queries">
                ${data.recent_queries.length === 0 ? '<p class="empty-state">No queries yet</p>' :
                    data.recent_queries.map(q => `
                        <div class="analytics-query-item">
                            <span class="analytics-query-text">${escapeHtml(q.query)}</span>
                            <span class="analytics-query-meta">${q.timing_ms ? q.timing_ms.toFixed(0) + 'ms' : '-'} | ${q.groundedness_verdict || '-'}</span>
                        </div>
                    `).join('')}
            </div>
        `;
    } catch {
        analyticsContent.innerHTML = '<p class="empty-state">Failed to load analytics</p>';
    }
}

async function runEvaluation() {
    const selected = Array.from(selectedDocIds);
    if (selected.length === 0) {
        evalContent.innerHTML = '<p class="empty-state">Select a document first</p>';
        return;
    }
    evalRunBtn.disabled = true;
    evalContent.innerHTML = '<div class="loading-state"><div class="spinner"></div><span>Running evaluation...</span></div>';
    try {
        const res = await fetch(`${API_BASE}/eval/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: selected[0] }),
        });
        if (!res.ok) throw new Error('Eval failed');
        const data = await res.json();
        evalContent.innerHTML = `
            <div class="analytics-grid">
                <div class="analytics-card">
                    <div class="analytics-value">${data.total_questions}</div>
                    <div class="analytics-label">Questions</div>
                </div>
                <div class="analytics-card">
                    <div class="analytics-value">${(data.avg_groundedness * 100).toFixed(0)}%</div>
                    <div class="analytics-label">Avg Groundedness</div>
                </div>
                <div class="analytics-card">
                    <div class="analytics-value">${data.avg_citations.toFixed(1)}</div>
                    <div class="analytics-label">Avg Citations</div>
                </div>
            </div>
            <div class="eval-results">
                ${data.results.map(r => `
                    <div class="eval-item">
                        <div class="eval-query">${escapeHtml(r.query)}</div>
                        <div class="eval-meta">${(r.groundedness_score * 100).toFixed(0)}% grounded | ${r.citation_count} citations</div>
                        <div class="eval-verdict eval-${r.groundedness_verdict}">${r.groundedness_verdict}</div>
                    </div>
                `).join('')}
            </div>
        `;
    } catch {
        evalContent.innerHTML = '<p class="empty-state">Evaluation failed</p>';
    } finally {
        evalRunBtn.disabled = false;
    }
}

function openCitationViewer(docId, pageNumber, highlight) {
    viewerModal.style.display = 'flex';
    viewerContent.innerHTML = `
        <div class="viewer-placeholder">
            <div class="viewer-header-info">
                <span>Document: ${docId.substring(0, 8)}...</span>
                <span>Page ${pageNumber}</span>
            </div>
            <div class="viewer-highlighted-text">
                <h4>Citation from page ${pageNumber}:</h4>
                <blockquote>${escapeHtml(highlight)}...</blockquote>
                <p class="viewer-hint">Full PDF viewer requires the original file.</p>
            </div>
        </div>
    `;
}
