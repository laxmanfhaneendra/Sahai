const GEMINI_API_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY ?? '';
const GEMINI_MODEL = 'gemini-2.0-flash';

export type GeminiChatContextMessage = {
  role: 'user' | 'assistant';
  text: string;
};

export type GeminiChatResult = {
  content: string;
  model: string;
};

export async function chatWithGemini(
  prompt: string,
  history: GeminiChatContextMessage[] = [],
  imageContext?: string,
): Promise<GeminiChatResult> {
  if (!GEMINI_API_KEY) {
    throw new Error('Gemini API key is not configured. Add EXPO_PUBLIC_GEMINI_API_KEY to your .env file.');
  }

  const contextBlock = imageContext
    ? `\n\nImage analysis context from vision model:\n${imageContext}`
    : '';

  const finalPrompt = `${prompt.trim() || 'Respond to the user based on context.'}${contextBlock}`;

  const contents = [
    ...history.map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.text }],
    })),
    { role: 'user', parts: [{ text: finalPrompt }] },
  ];

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        generationConfig: {
          temperature: 0.4,
          maxOutputTokens: 512,
        },
      }),
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const content = (data.candidates?.[0]?.content?.parts ?? [])
    .map((part: { text?: string }) => part.text ?? '')
    .join('\n')
    .trim();

  return {
    content,
    model: data.modelVersion ?? GEMINI_MODEL,
  };
}
