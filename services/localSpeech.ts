import * as FileSystem from 'expo-file-system/legacy';
import { NativeModules, Platform } from 'react-native';
import { initWhisper, releaseAllWhisper, WhisperContext } from 'whisper.rn';

export type LocalSpeechModelId = 'base.en';

export const SPEECH_MODELS: Record<LocalSpeechModelId, {
  label: string;
  sizeLabel: string;
  filename: string;
  url: string;
}> = {
  'base.en': {
    label: 'Whisper Base (English)',
    sizeLabel: '142 MB',
    filename: 'ggml-base.en.bin',
    url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin',
  },
};

let activeContext: WhisperContext | null = null;
let currentDownload: FileSystem.DownloadResumable | null = null;

export function getSpeechModelPath(modelId: LocalSpeechModelId = 'base.en'): string {
  const dir = FileSystem.documentDirectory || '';
  return `${dir}${SPEECH_MODELS[modelId].filename}`;
}

export async function isSpeechModelDownloaded(modelId: LocalSpeechModelId = 'base.en'): Promise<boolean> {
  if (Platform.OS === 'web') return false;
  try {
    const path = getSpeechModelPath(modelId);
    const info = await FileSystem.getInfoAsync(path);
    return info.exists && (info.size ?? 0) > 50 * 1024 * 1024;
  } catch (error) {
    console.error('Error checking speech model:', error);
    return false;
  }
}

export async function downloadSpeechModel(
  modelId: LocalSpeechModelId,
  onProgress: (progress: number) => void
): Promise<string> {
  if (Platform.OS === 'web') {
    throw new Error('Local speech model download is only supported on mobile devices.');
  }

  const model = SPEECH_MODELS[modelId];
  const localUri = getSpeechModelPath(modelId);

  if (currentDownload) {
    try {
      await currentDownload.cancelAsync();
    } catch {}
    currentDownload = null;
  }

  if (FileSystem.documentDirectory) {
    const dirInfo = await FileSystem.getInfoAsync(FileSystem.documentDirectory);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(FileSystem.documentDirectory, { intermediates: true });
    }
  }

  currentDownload = FileSystem.createDownloadResumable(
    model.url,
    localUri,
    {},
    (downloadProgress) => {
      const progress = downloadProgress.totalBytesWritten / downloadProgress.totalBytesExpectedToWrite;
      onProgress(isNaN(progress) ? 0 : progress);
    }
  );

  try {
    const result = await currentDownload.downloadAsync();
    currentDownload = null;
    if (!result?.uri) {
      throw new Error('Speech model download failed: no URI returned');
    }
    return result.uri;
  } catch (error: any) {
    currentDownload = null;
    throw error;
  }
}

export async function cancelSpeechModelDownload(): Promise<void> {
  if (Platform.OS === 'web') return;
  if (currentDownload) {
    try {
      await currentDownload.cancelAsync();
    } catch (e) {
      console.warn('Failed to cancel speech model download:', e);
    }
    currentDownload = null;
  }
}

async function initLocalWhisper(modelId: LocalSpeechModelId = 'base.en'): Promise<WhisperContext> {
  if (Platform.OS === 'web') {
    throw new Error('Local speech recognition is not supported in web environments.');
  }
  const hasNativeModule = !!(NativeModules as any)?.RNWhisper;
  const hasJsiGlobal = typeof (globalThis as any)?.whisperInitContext === 'function';
  if (!hasNativeModule && !hasJsiGlobal) {
    throw new Error(
      'Whisper native module is not installed. Use a custom dev build after adding whisper.rn.'
    );
  }

  if (activeContext) return activeContext;

  const downloaded = await isSpeechModelDownloaded(modelId);
  if (!downloaded) {
    throw new Error('Local speech model is not downloaded. Please download it first.');
  }

  const context = await initWhisper({
    filePath: getSpeechModelPath(modelId),
    useGpu: Platform.OS === 'ios',
    useCoreMLIos: Platform.OS === 'ios',
    useFlashAttn: false,
  });
  activeContext = context;
  return context;
}

export async function transcribeAudioWithLocalWhisper(
  audioPath: string,
  modelId: LocalSpeechModelId = 'base.en'
): Promise<string> {
  const context = await initLocalWhisper(modelId);
  const { promise } = context.transcribe(audioPath, {
    language: 'en',
    maxThreads: Platform.OS === 'android' ? 2 : 4,
  });
  const result = await promise;
  return (result.result ?? '').trim();
}

export async function releaseLocalWhisper(): Promise<void> {
  if (!activeContext) return;
  try {
    await activeContext.release();
  } catch (e) {
    console.warn('Error releasing whisper context:', e);
  } finally {
    activeContext = null;
  }
  try {
    await releaseAllWhisper();
  } catch {}
}
