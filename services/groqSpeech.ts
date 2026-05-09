const GROQ_API_KEY = process.env.EXPO_PUBLIC_GROQ_API_KEY ?? '';
const GROQ_TRANSCRIBE_URL = 'https://api.groq.com/openai/v1/audio/transcriptions';

export async function transcribeAudioWithGroq(audioUri: string): Promise<string> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured. Add EXPO_PUBLIC_GROQ_API_KEY to your .env file.');
  }

  const form = new FormData();
  form.append('model', 'whisper-large-v3-turbo');
  form.append('response_format', 'verbose_json');
  form.append('file', {
    uri: audioUri,
    name: 'live-listen.m4a',
    type: 'audio/m4a',
  } as any);

  const response = await fetch(GROQ_TRANSCRIBE_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: form,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq transcription error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  // Whisper API charges per minute of audio, not tokens
  // But log usage if available
  if (data.usage) {

  }
  return (data.text ?? '').trim();
}

export type GroqWhisperSegment = {
  id: number;
  start: number;
  end: number;
  text: string;
};

export async function transcribeAudioWithGroqSegments(audioUri: string): Promise<GroqWhisperSegment[]> {
  if (!GROQ_API_KEY) {
    throw new Error('Groq API key is not configured.');
  }

  const form = new FormData();
  form.append('model', 'whisper-large-v3-turbo');
  form.append('response_format', 'verbose_json');
  form.append('timestamp_granularities[]', 'segment');
  form.append('file', {
    uri: audioUri,
    name: 'diarize-test.m4a',
    type: 'audio/m4a',
  } as any);

  const response = await fetch(GROQ_TRANSCRIBE_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${GROQ_API_KEY}`,
    },
    body: form,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Groq transcription error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  if (!data.segments || data.segments.length === 0) {
    if (data.text) {
      return [{ id: 0, start: 0, end: 99999, text: data.text }];
    }
    return [];
  }
  return data.segments;
}
