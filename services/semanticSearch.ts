/**
 * Semantic Search: Find similar images/context by meaning, not keywords
 * Uses simple cosine similarity for fast, lightweight matching
 */

export type EmbeddedItem = {
  id: string;
  text: string;
  embedding: number[];
  timestamp: number;
};

/**
 * Simple word-level embedding (fast, works offline)
 * Converts text to a sparse vector based on word frequency + position
 */
function simpleEmbed(text: string): number[] {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  const embedding = new Array(100).fill(0);

  // Hash-based word embedding (deterministic)
  words.forEach((word, index) => {
    let hash = 0;
    for (let i = 0; i < word.length; i++) {
      hash = ((hash << 5) - hash) + word.charCodeAt(i);
    }
    // Convert to 32-bit signed integer
    hash = hash | 0;
    // Map hash to embedding dimension
    const dim = Math.abs(hash) % 100;
    embedding[dim] += 1 / (index + 1); // Position-weighted
  });

  // Normalize
  const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
  return magnitude > 0 ? embedding.map(v => v / magnitude) : embedding;
}

/**
 * Cosine similarity between two vectors (0 = different, 1 = identical)
 */
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * (vecB[i] || 0), 0);
  const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));

  if (magA === 0 || magB === 0) return 0;
  return dotProduct / (magA * magB);
}

/**
 * Find K most similar items to a query
 */
export function findSimilarItems(
  query: string,
  items: EmbeddedItem[],
  topK: number = 2,
  minSimilarity: number = 0.3
): EmbeddedItem[] {
  const queryEmbedding = simpleEmbed(query);

  const scored = items.map(item => ({
    ...item,
    similarity: cosineSimilarity(queryEmbedding, item.embedding),
  }));

  return scored
    .filter(item => item.similarity >= minSimilarity)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, topK)
    .map(({ similarity, ...item }) => item);
}

/**
 * Embed and store an item
 */
export function embedItem(id: string, text: string): EmbeddedItem {
  return {
    id,
    text,
    embedding: simpleEmbed(text),
    timestamp: Date.now(),
  };
}

/**
 * Remove old items (older than maxAgeMs)
 */
export function pruneOldItems(items: EmbeddedItem[], maxAgeMs: number = 3600000): EmbeddedItem[] {
  const now = Date.now();
  return items.filter(item => now - item.timestamp < maxAgeMs);
}
