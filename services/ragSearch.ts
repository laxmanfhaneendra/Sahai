/**
 * RAG Search: BM25 retrieval over document chunks
 * Fully offline, pure JS — no embedding API, no native modules
 */

export type RagChunk = {
  id: string;       // `chunk-${docId}-${index}`
  docId: string;
  docName: string;
  text: string;
  index: number;
};

/**
 * Split document text into overlapping ~500-char chunks.
 * Prefers paragraph/sentence boundaries for cleaner splits.
 */
export function chunkDocument(
  text: string,
  docId: string,
  docName: string,
  chunkSize: number = 500,
  overlap: number = 50
): RagChunk[] {
  const chunks: RagChunk[] = [];
  let start = 0;
  let index = 0;

  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);

    // Prefer natural break points
    if (end < text.length) {
      const paraBreak = text.lastIndexOf('\n\n', end);
      const lineBreak = text.lastIndexOf('\n', end);
      const sentBreak = text.lastIndexOf('. ', end);
      const minStart = start + chunkSize * 0.4;

      const boundary = Math.max(
        paraBreak > minStart ? paraBreak : -1,
        lineBreak > minStart ? lineBreak : -1,
        sentBreak > minStart ? sentBreak + 1 : -1,
      );
      if (boundary > start) end = boundary;
    }

    const chunkText = text.slice(start, end).trim();
    if (chunkText.length > 30) {
      chunks.push({ id: `chunk-${docId}-${index}`, docId, docName, text: chunkText, index });
      index++;
    }

    if (end >= text.length) break;
    start = end - overlap;
  }

  return chunks;
}

/**
 * Tokenize: lowercase, strip punctuation, remove tokens ≤ 2 chars
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2);
}

/**
 * Build IDF map from all chunks (higher score = rarer term)
 */
function buildIDF(chunks: RagChunk[]): Map<string, number> {
  const N = chunks.length;
  const df = new Map<string, number>();

  for (const chunk of chunks) {
    const terms = new Set(tokenize(chunk.text));
    for (const term of terms) {
      df.set(term, (df.get(term) || 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [term, freq] of df) {
    idf.set(term, Math.log((N - freq + 0.5) / (freq + 0.5) + 1));
  }
  return idf;
}

/**
 * BM25 score for a single chunk (k1=1.5, b=0.75 are standard defaults)
 */
function scoreBM25(
  queryTerms: string[],
  chunkTokens: string[],
  avgLen: number,
  idf: Map<string, number>,
  k1 = 1.5,
  b = 0.75,
): number {
  const tf = new Map<string, number>();
  for (const t of chunkTokens) tf.set(t, (tf.get(t) || 0) + 1);

  const docLen = chunkTokens.length;
  let score = 0;

  for (const term of queryTerms) {
    const termIdf = idf.get(term) || 0;
    const termTf = tf.get(term) || 0;
    if (termTf === 0) continue;
    const numerator = termTf * (k1 + 1);
    const denominator = termTf + k1 * (1 - b + b * (docLen / avgLen));
    score += termIdf * (numerator / denominator);
  }
  return score;
}

/**
 * Retrieve top K most relevant chunks for a query using BM25.
 * Scores all chunks in < 5ms even for 500 chunks (50 docs × 10 chunks).
 */
export function retrieveTopChunks(
  query: string,
  chunks: RagChunk[],
  topK: number = 5,
): RagChunk[] {
  if (chunks.length === 0) return [];

  const queryTerms = tokenize(query);
  // No meaningful query terms — fall back to recency order
  if (queryTerms.length === 0) return chunks.slice(0, topK);

  const idf = buildIDF(chunks);
  const tokenizedChunks = chunks.map(c => tokenize(c.text));
  const avgLen = tokenizedChunks.reduce((sum, t) => sum + t.length, 0) / tokenizedChunks.length;

  const scored = chunks.map((chunk, i) => ({
    chunk,
    score: scoreBM25(queryTerms, tokenizedChunks[i], avgLen, idf),
  }));

  return scored
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(item => item.chunk);
}

/**
 * Format retrieved chunks into a [KNOWLEDGE BASE] block for the LLM prompt.
 * Groups consecutive chunks from the same document together.
 */
export function formatRagContext(chunks: RagChunk[]): string {
  if (chunks.length === 0) return '';

  let context = '[KNOWLEDGE BASE]\n';
  let currentDoc = '';

  for (const chunk of chunks) {
    if (chunk.docName !== currentDoc) {
      context += `\n--- ${chunk.docName} ---\n`;
      currentDoc = chunk.docName;
    }
    context += chunk.text + '\n';
  }

  return context;
}
