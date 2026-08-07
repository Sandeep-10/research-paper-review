// ponytail: clean API client wrapper supporting both real backend endpoints (/upload-single/ and /review/) and a mock mode fallback for offline UI testing

export const API_BASE_URL = '';

/**
 * Upload single PDF or CSV manuscript to backend
 * @param {File} file 
 * @returns {Promise<{status: string, message: string, filename: string, chunks_created: number}>}
 */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload-single/`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(errorData.detail || `Upload error (${response.status})`);
  }

  return await response.json();
}

/**
 * Trigger review process and stream SSE events
 * @param {function(object): void} onEvent 
 * @param {function(Error): void} onError 
 * @param {function(): void} onComplete 
 */
export async function streamReview(onEvent, onError, onComplete) {
  try {
    const response = await fetch(`${API_BASE_URL}/review/`, {
      method: 'POST',
      headers: {
        'Accept': 'text/event-stream',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to start review process' }));
      throw new Error(errorData.detail || `Server error (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep last incomplete line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;

        const dataStr = trimmed.replace(/^data:\s*/, '');
        if (dataStr === '[DONE]') {
          onComplete();
          return;
        }

        try {
          const parsed = JSON.parse(dataStr);
          onEvent(parsed);
        } catch (err) {
          console.warn('Failed to parse SSE payload line:', dataStr, err);
        }
      }
    }

    onComplete();
  } catch (err) {
    onError(err);
  }
}

/**
 * Standalone mock simulator for local frontend testing without active backend server
 */
export function runMockReview(filename, onEvent, onComplete) {
  const mockSteps = [
    { event: 'started', message: `Multi-agent crew initialized for "${filename}"...` },
    { event: 'log', agent: 'COORDINATOR', message: `Parsing document: "${filename}"` },
    { event: 'log', agent: 'COORDINATOR', message: `Splitting document into 38 text chunks for vector indexing.` },
    { event: 'log', agent: 'METHODOLOGY', status: 'analyzing', message: `Analyzing experimental design, baseline comparisons, & statistical power...` },
    { event: 'log', agent: 'METHODOLOGY', status: 'completed', message: `✔ Methodology evaluation complete. Experimental design score: 8.5/10.` },
    { event: 'log', agent: 'NOVELTY', status: 'analyzing', message: `Querying literature cross-references for prior work and originality metrics...` },
    { event: 'log', agent: 'NOVELTY', status: 'completed', message: `✔ Novelty evaluation complete. Originality score: 9.0/10.` },
    { event: 'log', agent: 'CLARITY', status: 'analyzing', message: `Evaluating prose clarity, figure captions, mathematical notation consistency...` },
    { event: 'log', agent: 'CLARITY', status: 'completed', message: `✔ Clarity evaluation complete. Communication score: 8.0/10.` },
    { event: 'log', agent: 'LIMITATIONS', status: 'analyzing', message: `Scrutinizing ethical risks, unstated bounds, and negative results disclosure...` },
    { event: 'log', agent: 'LIMITATIONS', status: 'completed', message: `✔ Limitations evaluation complete. Ethics & scope score: 7.5/10.` },
    { event: 'log', agent: 'JUDGE', status: 'analyzing', message: `Synthesizing specialist findings and calculating weighted final recommendation...` },
    { event: 'log', agent: 'JUDGE', status: 'completed', message: `✔ Program Chair meta-review complete. Generating report...` },
  ];

  let stepIdx = 0;
  const interval = setInterval(() => {
    if (stepIdx < mockSteps.length) {
      onEvent(mockSteps[stepIdx]);
      stepIdx++;
    } else {
      clearInterval(interval);
      // Final payload matching app.py judge_task expected schema
      const mockResult = {
        event: 'completed',
        review: {
          paper_title: filename.replace(/\.[^/.]+$/, "") || "Attention Is All You Need",
          individual_scores: {
            novelty: 9.0,
            methodology: 8.5,
            clarity: 8.0,
            limitations: 7.5
          },
          weighted_final_score: 8.45,
          hard_rules_triggered: false,
          final_verdict: "ACCEPT",
          confidence: 92,
          top_strengths: [
            "Groundbreaking self-attention mechanism eliminating recurrent bottlenecks",
            "Extensive empirical validation on WMT 2014 English-to-German and English-to-French translation",
            "Superior training efficiency enabling fast multi-GPU scaling"
          ],
          top_weaknesses: [
            "Quadratic memory consumption complexity relative to sequence length O(N²)",
            "Limited discussion of hardware memory bandwidth bounds during long-context decoding"
          ],
          mandatory_revisions: [
            "Clarify positional encoding hyperparameter sensitivity in Section 3.5",
            "Add ablation breakdown for multi-head attention vs single-head dimension scaling"
          ],
          summary_for_authors: "The manuscript presents a transformative architecture replacing recurrence entirely with parallelizable self-attention. The empirical performance and execution speed set a new benchmark for sequence modeling.",
          suggested_venue: "NeurIPS / ICML Oral Presentation"
        }
      };
      onEvent(mockResult);
      onComplete();
    }
  }, 1200);
}
