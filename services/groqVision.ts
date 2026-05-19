// ─── Groq Vision Service ────────────────────────────────────────────────────
// Uses Groq's chat completions API with Llama Scout (vision model)

const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_API_KEY = process.env.EXPO_PUBLIC_GROQ_API_KEY ?? '';

// Llama 4 Scout — Groq's multimodal vision model
const VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct';
const CHAT_MODEL = 'llama-3.3-70b-versatile';

export interface LiveFrameResult {
  description: string;
  model: string;
}

export interface GroqVisionResult {
  description: string;
  model: string;
  tokens: {
    prompt: number;
    completion: number;
    total: number;
  };
}

type GroqChatResult = {
  content: string;
  model: string;
  extractedText: string;
};

type GroqTextChatResult = {
  content: string;
  model: string;
  tokens?: {
    prompt: number;
    completion: number;
    total: number;
  };
};

export type GroqChatContextMessage = {
  role: 'user' | 'assistant';
  text: string;
  hadImage?: boolean;
};

/**
 * Sends a base64-encoded image to Groq's vision API and returns
 * an AI description of the image content.
 *
 * @param base64Image - The raw base64 string (no prefix)
 * @param mimeType    - Image MIME type (default: image/jpeg)
 * @param prompt      - The question/instruction to send with the image
 */
export async function analyzeImageWithGroq(
  base64Image: string,
  mimeType: 'image/jpeg' | 'image/png' | 'image/webp' = 'image/jpeg',
  prompt: string = 'Describe this image clearly in 4-6 short bullet points. Include key objects, visible text, and setting.',
): Promise<GroqVisionResult> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured. Add EXPO_PUBLIC_GROQ_API_KEY to your .env file.');
  }

  const dataUrl = `data:${mimeType};base64,${base64Image}`;

  const body = {
    model: VISION_MODEL,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: dataUrl },
          },
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
    max_tokens: 256,
    temperature: 0.5,
  };

  const response = await fetch(GROQ_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const message = data.choices?.[0]?.message?.content ?? '';

  return {
    description: message,
    model: data.model ?? VISION_MODEL,
    tokens: {
      prompt: data.usage?.prompt_tokens ?? 0,
      completion: data.usage?.completion_tokens ?? 0,
      total: data.usage?.total_tokens ?? 0,
    },
  };
}

export async function chatWithGroq(
  prompt: string,
  image?: { base64: string; mimeType?: 'image/jpeg' | 'image/png' | 'image/webp' },
  history: GroqChatContextMessage[] = [],
): Promise<GroqChatResult> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured. Add EXPO_PUBLIC_GROQ_API_KEY to your .env file.');
  }

  const content: ({ type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } })[] = [
    {
      type: 'text',
      text: prompt.trim() || 'Describe this image clearly.',
    },
  ];

  if (image?.base64) {
    const mimeType = image.mimeType ?? 'image/jpeg';
    content.unshift({
      type: 'image_url',
      image_url: { url: `data:${mimeType};base64,${image.base64}` },
    });
  }

  const priorMessages = history.map((m) => ({
    role: m.role,
    content: m.hadImage && m.role === 'user'
      ? `[User attached an image in this turn]\n${m.text}`
      : m.text,
  }));

  // Limit context to last 2 messages only for vision (to keep tokens down)
  const limitedPriorMessages = priorMessages.slice(-2);

  const response = await fetch(GROQ_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: VISION_MODEL,
      messages: [
        {
          role: 'system',
          content: 'You are a meeting assistant. If a current image is attached, extract all readable text. Respond in JSON: {"answer":"string","extracted_text":"string"}.',
        },
        ...limitedPriorMessages,
        { role: 'user', content },
      ],
      max_tokens: 256,
      temperature: 0.4,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const raw = data.choices?.[0]?.message?.content ?? '';
  let parsedAnswer = raw;
  let parsedExtractedText = '';

  try {
    const direct = JSON.parse(raw);
    if (typeof direct?.answer === 'string') parsedAnswer = direct.answer;
    if (typeof direct?.extracted_text === 'string') parsedExtractedText = direct.extracted_text;
  } catch {
    const fencedJson = raw.match(/```json\s*([\s\S]*?)```/i)?.[1];
    if (fencedJson) {
      try {
        const parsed = JSON.parse(fencedJson);
        if (typeof parsed?.answer === 'string') parsedAnswer = parsed.answer;
        if (typeof parsed?.extracted_text === 'string') parsedExtractedText = parsed.extracted_text;
      } catch {}
    }
  }

  return {
    content: parsedAnswer,
    model: data.model ?? VISION_MODEL,
    extractedText: parsedExtractedText,
  };
}

export async function chatWithGroqText(
  prompt: string,
  history: GroqChatContextMessage[] = [],
  imageContext?: string,
): Promise<GroqTextChatResult> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured. Add EXPO_PUBLIC_GROQ_API_KEY to your .env file.');
  }

  const priorMessages = history.map((m) => ({
    role: m.role,
    content: m.text,
  }));

  const contextBlock = imageContext
    ? `\n\nImage analysis context from Llama Vision:\n${imageContext}`
    : '';

  const userPrompt = `${prompt.trim() || 'Respond to the user based on context.'}${contextBlock}`;

  const response = await fetch(GROQ_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      messages: [
        {
          role: 'system',
          content: 'You are a concise meeting assistant. Use provided conversation history and image context when present.',
        },
        ...priorMessages,
        { role: 'user', content: userPrompt },
      ],
      max_tokens: 256,
      temperature: 0.4,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return {
    content: data.choices?.[0]?.message?.content ?? '',
    model: data.model ?? CHAT_MODEL,
    tokens: {
      prompt: data.usage?.prompt_tokens ?? 0,
      completion: data.usage?.completion_tokens ?? 0,
      total: data.usage?.total_tokens ?? 0,
    },
  };
}

/**
 * Analyzes a single camera frame captured during a live meeting.
 * Optimized for low latency — short response, meeting-aware prompt.
 * Runs independently of the audio/chat pipeline so it never blocks it.
 *
 * @param base64Frame - Compressed base64 JPEG frame (no data prefix)
 * @param frameIndex  - Frame sequence number for display
 */
export async function analyzeLiveFrame(
  base64Frame: string,
  frameIndex: number = 1,
): Promise<LiveFrameResult> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured. Add EXPO_PUBLIC_GROQ_API_KEY to your .env file.');
  }

  const dataUrl = `data:image/jpeg;base64,${base64Frame}`;

  const body = {
    model: VISION_MODEL,
    messages: [
      {
        role: 'system',
        content:
          'You are a silent live meeting observer. When given a camera frame, respond with 1-3 concise bullet points describing ONLY what is relevant to a meeting context: visible text, slides, whiteboards, documents, or actions. ' +
          'If the frame shows a person with nothing noteworthy, just say "👤 Person in frame." ' +
          'If the frame is blurry or unclear, just say "🔍 Frame unclear." ' +
          'Never add preamble. Never say "I see" or "The image shows". Be extremely terse.',
      },
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: dataUrl },
          },
          {
            type: 'text',
            text: `Meeting frame #${frameIndex}. What is relevant here?`,
          },
        ],
      },
    ],
    max_tokens: 100,
    temperature: 0.3,
  };

  const response = await fetch(GROQ_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return {
    description: data.choices?.[0]?.message?.content ?? '',
    model: data.model ?? VISION_MODEL,
  };
}
