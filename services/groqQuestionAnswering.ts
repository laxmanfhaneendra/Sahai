/**
 * Single-API workflow: Groq Whisper transcription + question detection + answering
 * Combines all three steps into one smart prompt for latency reduction
 */

import { transcribeAudioWithGroq } from '@/services/groqSpeech';
import { chatWithGroqText } from '@/services/groqVision';

export type QuestionAnswerResult = {
  hasQuestion: boolean;
  question: string | null;
  answer: string | null;
  confidence: number;
};

/**
 * Single Groq call: transcribe + detect question + generate answer
 * Returns answer directly if question detected, null otherwise
 */
export async function transcribeAndAnswerWithGroq(
  audioUri: string
): Promise<QuestionAnswerResult> {
  try {
    // Step 1: Transcribe with Groq Whisper
    const transcript = await transcribeAudioWithGroq(audioUri);

    if (!transcript || !transcript.trim()) {
      return {
        hasQuestion: false,
        question: null,
        answer: null,
        confidence: 0,
      };
    }

    // Step 2: Smart prompt to detect question and answer in one pass
    const smartPrompt = `You are an AI meeting assistant. Analyze this transcript and:
1. Decide if it contains a QUESTION (genuine question, not rhetorical)
2. If YES: Answer it in 1-2 sentences
3. If NO: Say "NO_QUESTION"

Format your response EXACTLY like this:
DETECTED: yes/no
ANSWER: [your answer here or "NO_QUESTION"]

Transcript: "${transcript}"`;

    const response = await chatWithGroqText(smartPrompt, [], undefined);
    const content = response.content || '';



    // More flexible parsing - handle various formats
    const detectedMatch = content.match(/DETECTED:\s*(yes|no)/i);
    const answerMatch = content.match(/ANSWER:\s*(.+?)(?:\n|$)/i);

    const hasQuestion = detectedMatch?.[1]?.toLowerCase() === 'yes';
    const answerText = (answerMatch?.[1] || '').trim();

    if (!hasQuestion || !answerText || answerText.toLowerCase() === 'no_question') {
      return {
        hasQuestion: false,
        question: transcript,
        answer: null,
        confidence: 0,
      };
    }

    return {
      hasQuestion: true,
      question: transcript,
      answer: answerText,
      confidence: 0.95,
    };
  } catch (err) {
    return {
      hasQuestion: false,
      question: null,
      answer: null,
      confidence: 0,
    };
  }
}
