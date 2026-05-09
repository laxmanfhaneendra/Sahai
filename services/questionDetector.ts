/**
 * Two-stage question detection:
 * 1. Rule-based gate: punctuation + question starters (fast)
 * 2. LLM confirmation: only for gated utterances (selective)
 */

import { chatWithGroqText } from '@/services/groqVision';

export type QuestionDetectionResult = {
  isQuestion: boolean;
  confidence: number;
  question: string;
};

const QUESTION_STARTERS = [
  'what ',
  'why ',
  'how ',
  'when ',
  'where ',
  'who ',
  'which ',
  'can ',
  'could ',
  'should ',
  'would ',
  'will ',
  'is ',
  'are ',
  'do ',
  'does ',
  'did ',
  'have ',
  'has ',
];

/**
 * Stage 1: Fast rule-based gate.
 * Returns true if text shows question-like patterns.
 */
function ruleBasedGate(text: string): boolean {
  if (!text || text.length < 3) return false;

  const lower = text.toLowerCase().trim();

  if (lower.includes('?')) return true;

  if (QUESTION_STARTERS.some((prefix) => lower.startsWith(prefix))) {
    return lower.length > 4;
  }

  return false;
}

/**
 * Stage 2: LLM-based confirmation (only called for gated utterances).
 * Fast because it only processes ~5-10% of utterances.
 */
async function llmConfirmation(text: string): Promise<boolean> {
  try {
    const response = await chatWithGroqText(
      `Determine if this is a genuine question (not rhetorical, not a statement, not noise). Respond with ONLY "yes" or "no".\n\nText: "${text}"`,
      [],
      undefined
    );

    const answer = response.content?.toLowerCase().trim();
    return answer === 'yes';
  } catch (err) {
    return true;
  }
}

/**
 * Two-stage question detection.
 * Stage 1 is fast (rule-based).
 * Stage 2 is selective (LLM only for promising candidates).
 */
export async function detectQuestion(utterance: string): Promise<QuestionDetectionResult> {
  if (!utterance || utterance.trim().length < 3) {
    return { isQuestion: false, confidence: 0, question: '' };
  }

  const cleaned = utterance.trim();

  const gatesPassed = ruleBasedGate(cleaned);
  if (!gatesPassed) {
    return { isQuestion: false, confidence: 0, question: '' };
  }

  const isConfirmed = await llmConfirmation(cleaned);

  return {
    isQuestion: isConfirmed,
    confidence: isConfirmed ? 0.95 : 0.1,
    question: isConfirmed ? cleaned : '',
  };
}

/**
 * Batch-detect questions from a list of utterances.
 * Used for processing finalized segments.
 */
export async function detectQuestionsInBatch(utterances: string[]): Promise<string[]> {
  const results: string[] = [];

  for (const utterance of utterances) {
    const detected = await detectQuestion(utterance);
    if (detected.isQuestion && detected.question) {
      results.push(detected.question);
    }
  }

  return results;
}
