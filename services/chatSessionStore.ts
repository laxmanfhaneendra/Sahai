import * as FileSystem from 'expo-file-system/legacy';

export type SessionChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  imageUri?: string;
  extractedText?: string;
  hadImage?: boolean;
  createdAt: number;
};

type SessionPayload = {
  sessionId: string;
  createdAt: number;
  updatedAt: number;
  messages: SessionChatMessage[];
};

export type SessionChatThread = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: SessionChatMessage[];
};

type ThreadStorePayload = {
  version: 2;
  currentThreadId: string;
  threads: SessionChatThread[];
};

const STORE_PATH = `${FileSystem.documentDirectory ?? ''}chat-session-store.json`;

const createThreadTitle = (messages: SessionChatMessage[]): string => {
  const firstUser = messages.find((m) => m.role === 'user' && m.text.trim().length > 0)?.text.trim();
  if (!firstUser) return 'New thread';
  return firstUser.length > 40 ? `${firstUser.slice(0, 40)}...` : firstUser;
};

const normalizePayload = (raw: string): ThreadStorePayload => {
  const parsed = JSON.parse(raw) as Partial<ThreadStorePayload & SessionPayload>;
  const now = Date.now();

  if (Array.isArray(parsed?.threads)) {
    const threads = parsed.threads
      .filter((thread): thread is SessionChatThread => !!thread && Array.isArray(thread.messages))
      .map((thread) => ({
        id: thread.id || `thread-${now}`,
        title: thread.title || createThreadTitle(thread.messages),
        createdAt: thread.createdAt || now,
        updatedAt: thread.updatedAt || now,
        messages: thread.messages,
      }));

    const fallbackThreadId = threads[0]?.id ?? `thread-${now}`;
    return {
      version: 2,
      currentThreadId: parsed.currentThreadId && threads.some((thread) => thread.id === parsed.currentThreadId)
        ? parsed.currentThreadId
        : fallbackThreadId,
      threads,
    };
  }

  if (!Array.isArray(parsed?.messages)) {
    throw new Error('Chat store payload is invalid.');
  }

  const threadId = parsed.sessionId ?? `thread-${now}`;
  return {
    version: 2,
    currentThreadId: threadId,
    threads: [{
      id: threadId,
      title: createThreadTitle(parsed.messages),
      createdAt: parsed.createdAt ?? now,
      updatedAt: parsed.updatedAt ?? now,
      messages: parsed.messages,
    }],
  };
};

export async function loadChatThreads(): Promise<{ threads: SessionChatThread[]; currentThreadId: string | null }> {
  if (!STORE_PATH || STORE_PATH === 'chat-session-store.json') {
    throw new Error('Chat store path is unavailable.');
  }

  const info = await FileSystem.getInfoAsync(STORE_PATH);
  if (!info.exists) return { threads: [], currentThreadId: null };

  const raw = await FileSystem.readAsStringAsync(STORE_PATH);
  const parsed = normalizePayload(raw);
  const sortedThreads = [...parsed.threads].sort((a, b) => b.updatedAt - a.updatedAt);
  const currentThreadId = sortedThreads.some((thread) => thread.id === parsed.currentThreadId)
    ? parsed.currentThreadId
    : sortedThreads[0]?.id ?? null;

  return { threads: sortedThreads, currentThreadId };
}

export async function saveChatThreads(threads: SessionChatThread[], currentThreadId: string): Promise<void> {
  if (!STORE_PATH || STORE_PATH === 'chat-session-store.json') {
    throw new Error('Chat store path is unavailable.');
  }

  const payload: ThreadStorePayload = {
    version: 2,
    currentThreadId,
    threads,
  };

  await FileSystem.writeAsStringAsync(STORE_PATH, JSON.stringify(payload));
}

export async function loadSessionChatMessages(): Promise<SessionChatMessage[]> {
  const { threads, currentThreadId } = await loadChatThreads();
  if (!threads.length || !currentThreadId) return [];
  return threads.find((thread) => thread.id === currentThreadId)?.messages ?? [];
}

export async function saveSessionChatMessages(messages: SessionChatMessage[]): Promise<void> {
  const now = Date.now();
  const { threads, currentThreadId } = await loadChatThreads();

  if (!threads.length || !currentThreadId) {
    const newThread: SessionChatThread = {
      id: `thread-${now}`,
      title: createThreadTitle(messages),
      createdAt: now,
      updatedAt: now,
      messages,
    };
    await saveChatThreads([newThread], newThread.id);
    return;
  }

  const updatedThreads = threads.map((thread) =>
    thread.id === currentThreadId
      ? {
          ...thread,
          title: createThreadTitle(messages),
          updatedAt: now,
          messages,
        }
      : thread
  );

  await saveChatThreads(updatedThreads, currentThreadId);
}
