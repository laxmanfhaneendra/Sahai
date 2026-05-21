import * as FileSystem from 'expo-file-system/legacy';
import { NativeModules, Platform } from 'react-native';

export type LocalModelId = 'qwen' | 'phi3_vision';

export const MODELS: Record<LocalModelId, {
  name: string;
  label: string;
  sizeLabel: string;
  modelFilename: string;
  modelUrl: string;
  mmprojFilename?: string;
  mmprojUrl?: string;
}> = {
  qwen: {
    name: 'qwen',
    label: 'Qwen 2.5 1.5B (Text)',
    sizeLabel: '980 MB',
    modelFilename: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    modelUrl: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
  },
  phi3_vision: {
    name: 'phi3_vision',
    label: 'Phi-3 Vision 3.8B (Vision)',
    sizeLabel: '2.5 GB',
    modelFilename: 'llava-phi-3-mini-int4.gguf',
    modelUrl: 'https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-int4.gguf',
    mmprojFilename: 'llava-phi-3-mini-mmproj-f16.gguf',
    mmprojUrl: 'https://huggingface.co/xtuner/llava-phi-3-mini-gguf/resolve/main/llava-phi-3-mini-mmproj-f16.gguf',
  }
};

let initLlama: any = null;
let releaseAllLlama: any = null;

// Dynamically load native llama.rn JSI bindings only on iOS/Android
if (Platform.OS !== 'web') {
  try {
    const llama = require('llama.rn');
    initLlama = llama.initLlama;
    releaseAllLlama = llama.releaseAllLlama;
  } catch (error) {
    console.warn('llama.rn native modules could not be loaded:', error);
  }
}

const activeContexts: Partial<Record<LocalModelId, any>> = {};
let currentDownload: FileSystem.DownloadResumable | null = null;

async function releaseOtherContexts(keepModelId: LocalModelId): Promise<void> {
  const entries = Object.entries(activeContexts) as Array<[LocalModelId, any]>;
  for (const [modelId, context] of entries) {
    if (modelId === keepModelId) continue;
    try {
      await context.release();
    } catch (e) {
      console.warn('Error releasing local llama context:', e);
    }
    delete activeContexts[modelId];
  }
}

/**
 * Returns the local URI where the GGUF model should be stored.
 */
export function getModelFilePath(modelId: LocalModelId): string {
  const dir = FileSystem.documentDirectory || '';
  return `${dir}${MODELS[modelId].modelFilename}`;
}

/**
 * Returns the local URI where the GGUF visual projector should be stored (if applicable).
 */
export function getMmprojFilePath(modelId: LocalModelId): string | null {
  const model = MODELS[modelId];
  if (!model.mmprojFilename) return null;
  const dir = FileSystem.documentDirectory || '';
  return `${dir}${model.mmprojFilename}`;
}

/**
 * Check if the model has already been fully downloaded to the device.
 */
export async function isModelDownloaded(modelId: LocalModelId): Promise<boolean> {
  if (Platform.OS === 'web') {
    return false;
  }
  if (!MODELS[modelId]) {
    console.warn(`Unknown local model id "${String(modelId)}"`);
    return false;
  }
  try {
    const model = MODELS[modelId];
    const path = getModelFilePath(modelId);
    const fileInfo = await FileSystem.getInfoAsync(path);
    const mainOk = fileInfo.exists && (fileInfo.size ?? 0) > 100 * 1024 * 1024;
    
    if (!mainOk) return false;
    
    if (model.mmprojFilename) {
      const mmprojPath = getMmprojFilePath(modelId)!;
      const mmprojInfo = await FileSystem.getInfoAsync(mmprojPath);
      return mmprojInfo.exists && (mmprojInfo.size ?? 0) > 10 * 1024 * 1024;
    }
    
    return true;
  } catch (error) {
    console.error('Error checking if model is downloaded:', error);
    return false;
  }
}

/**
 * Starts downloading the model from Hugging Face with progress callbacks.
 * Supports cancellation and sequential downloading for multi-file models.
 * 
 * @param modelId The ID of the model to download.
 * @param onProgress Callback receiving a number between 0 and 1 representing overall progress
 */
export async function downloadModel(
  modelId: LocalModelId,
  onProgress: (progress: number) => void
): Promise<string> {
  if (Platform.OS === 'web') {
    throw new Error('Model downloading is only supported on mobile devices.');
  }

  const model = MODELS[modelId];
  const localModelUri = getModelFilePath(modelId);
  
  // Ensure any prior download is cancelled to avoid conflicts
  if (currentDownload) {
    try {
      await currentDownload.cancelAsync();
    } catch {}
    currentDownload = null;
  }

  // Create document directory if it doesn't exist
  if (FileSystem.documentDirectory) {
    const dirInfo = await FileSystem.getInfoAsync(FileSystem.documentDirectory);
    if (!dirInfo.exists) {
      await FileSystem.makeDirectoryAsync(FileSystem.documentDirectory, { intermediates: true });
    }
  }

  // If there's no projector file, perform a simple single-file download
  if (!model.mmprojFilename || !model.mmprojUrl) {
    currentDownload = FileSystem.createDownloadResumable(
      model.modelUrl,
      localModelUri,
      {},
      (downloadProgress) => {
        const progress = downloadProgress.totalBytesWritten / downloadProgress.totalBytesExpectedToWrite;
        onProgress(isNaN(progress) ? 0 : progress);
      }
    );

    try {
      const result = await currentDownload.downloadAsync();
      currentDownload = null;
      if (!result || !result.uri) {
        throw new Error('Download failed: No URI returned from downloader');
      }
      return result.uri;
    } catch (error: any) {
      currentDownload = null;
      throw error;
    }
  }

  // Otherwise, do a sequential download for Phi-3 Vision (Main model + projector mmproj)
  const localMmprojUri = getMmprojFilePath(modelId)!;
  const mainWeight = 0.88; // Main model is ~2.2GB of ~2.5GB total (88%)
  const mmprojWeight = 0.12; // Projector is ~300MB of ~2.5GB total (12%)

  // Phase 1: Download Main GGUF model
  currentDownload = FileSystem.createDownloadResumable(
    model.modelUrl,
    localModelUri,
    {},
    (downloadProgress) => {
      const subProgress = downloadProgress.totalBytesWritten / downloadProgress.totalBytesExpectedToWrite;
      const progress = isNaN(subProgress) ? 0 : subProgress * mainWeight;
      onProgress(progress);
    }
  );

  try {
    const mainResult = await currentDownload.downloadAsync();
    currentDownload = null;
    if (!mainResult || !mainResult.uri) {
      throw new Error('Download failed: Main model download failed');
    }
  } catch (error: any) {
    currentDownload = null;
    throw error;
  }

  // Phase 2: Download Multimodal Projector mmproj file
  currentDownload = FileSystem.createDownloadResumable(
    model.mmprojUrl,
    localMmprojUri,
    {},
    (downloadProgress) => {
      const subProgress = downloadProgress.totalBytesWritten / downloadProgress.totalBytesExpectedToWrite;
      const progress = isNaN(subProgress) ? mainWeight : mainWeight + (subProgress * mmprojWeight);
      onProgress(progress);
    }
  );

  try {
    const mmprojResult = await currentDownload.downloadAsync();
    currentDownload = null;
    if (!mmprojResult || !mmprojResult.uri) {
      throw new Error('Download failed: Projector model download failed');
    }
    return localModelUri;
  } catch (error: any) {
    currentDownload = null;
    throw error;
  }
}

/**
 * Cancels any active model download.
 */
export async function cancelModelDownload(): Promise<void> {
  if (Platform.OS === 'web') return;
  if (currentDownload) {
    try {
      await currentDownload.cancelAsync();
    } catch (e) {
      console.warn('Failed to cancel download:', e);
    }
    currentDownload = null;
  }
}

/**
 * Deletes the local GGUF model files to free up storage space.
 */
export async function deleteModelFile(modelId: LocalModelId): Promise<void> {
  if (Platform.OS === 'web') return;
  
  // If the active context is utilizing the deleted model, release it first
  if (activeContexts[modelId]) {
    await releaseLocalLlama(modelId);
  }
  
  const model = MODELS[modelId];
  const modelPath = getModelFilePath(modelId);
  const modelInfo = await FileSystem.getInfoAsync(modelPath);
  if (modelInfo.exists) {
    await FileSystem.deleteAsync(modelPath, { idempotent: true });
  }
  
  if (model.mmprojFilename) {
    const mmprojPath = getMmprojFilePath(modelId)!;
    const mmprojInfo = await FileSystem.getInfoAsync(mmprojPath);
    if (mmprojInfo.exists) {
      await FileSystem.deleteAsync(mmprojPath, { idempotent: true });
    }
  }
}

/**
 * Initializes the Llama context using the downloaded model.
 * Uses a singleton pattern to reuse the context for the active model type.
 */
export async function initLocalLlama(modelId: LocalModelId): Promise<any> {
  if (Platform.OS === 'web') {
    throw new Error('Local reasoning is not supported in web environments.');
  }
  const hasNativeModule = !!(NativeModules as any)?.RNLlama;
  const hasJsiGlobal = typeof (globalThis as any)?.llamaInitContext === 'function';
  if (!hasNativeModule && !hasJsiGlobal) {
    throw new Error(
      'Llama native module is not installed. Use a custom dev build (EAS dev client) after adding the llama.rn Expo plugin.'
    );
  }
  if (!initLlama) {
    throw new Error('Llama native bindings are not available on this platform.');
  }

  // Reuse existing context if it is already loaded for the target model
  if (activeContexts[modelId]) {
    await releaseOtherContexts(modelId);
    return activeContexts[modelId];
  }
  await releaseOtherContexts(modelId);

  const isDownloaded = await isModelDownloaded(modelId);
  if (!isDownloaded) {
    throw new Error('Model is not downloaded. Please download the model first.');
  }

  const modelPath = getModelFilePath(modelId);
  const mmprojPath = getMmprojFilePath(modelId);

  try {
    const isVision = modelId === 'phi3_vision';
    const isAndroid = Platform.OS === 'android';
    const options: any = {
      model: modelPath,
      use_mlock: true,       // Prevent page swapping for higher performance
      n_ctx: isVision ? (isAndroid ? 768 : 1024) : 2048, // Lower vision context on Android to reduce RAM spikes
      n_gpu_layers: isVision && isAndroid ? 0 : 1,       // Avoid GPU memory spikes on Android vision
    };

    if (mmprojPath) {
      options.mmproj = mmprojPath;
    }

    const context = await initLlama(options);
    activeContexts[modelId] = context;
    if (mmprojPath && typeof context.initMultimodal === 'function') {
      const isEnabled = typeof context.isMultimodalEnabled === 'function'
        ? await context.isMultimodalEnabled()
        : false;
      if (!isEnabled) {
        await context.initMultimodal({
          path: mmprojPath,
          use_gpu: !isAndroid,
          image_max_tokens: isAndroid ? 256 : 512,
          image_min_tokens: 64,
        });
      }
    }
    return context;
  } catch (error) {
    delete activeContexts[modelId];
    throw error;
  }
}

/**
 * Releases the active Llama context to free up memory.
 */
export async function releaseLocalLlama(modelId?: LocalModelId): Promise<void> {
  if (Platform.OS === 'web') return;
  if (modelId) {
    const context = activeContexts[modelId];
    if (context) {
      try {
        await context.release();
      } catch (e) {
        console.warn('Error releasing local llama context:', e);
      }
      delete activeContexts[modelId];
    }
    return;
  }
  const entries = Object.entries(activeContexts) as Array<[LocalModelId, any]>;
  for (const [, context] of entries) {
    try {
      await context.release();
    } catch (e) {
      console.warn('Error releasing local llama context:', e);
    }
  }
  for (const key of Object.keys(activeContexts) as LocalModelId[]) {
    delete activeContexts[key];
  }
  try {
    if (releaseAllLlama) {
      await releaseAllLlama();
    }
  } catch {}
}

export type LocalChatMessage = {
  role: 'user' | 'assistant' | 'system';
  text: string;
};

/**
 * Formats local message history into a ChatML string for Qwen/Gemma.
 */
function formatChatML(history: LocalChatMessage[], userMessage: string, systemMessage?: string): string {
  let prompt = '';
  
  // 1. System Prompt
  const sys = systemMessage || 'You are a concise, helpful on-device assistant. Answer directly and keep your replies brief and clear.';
  prompt += `<|im_start|>system\n${sys}<|im_end|>\n`;

  // 2. Chat History
  for (const msg of history) {
    prompt += `<|im_start|>${msg.role}\n${msg.text}<|im_end|>\n`;
  }

  // 3. User Message
  prompt += `<|im_start|>user\n${userMessage}<|im_end|>\n`;
  
  // 4. Assistant Start
  prompt += `<|im_start|>assistant\n`;

  return prompt;
}

/**
 * Runs on-device inference using the local text or multimodal model.
 * Streams generated tokens back to the UI in real-time.
 * 
 * @param modelId The ID of the model to use (qwen or phi3_vision)
 * @param userMessage Current prompt/query from user
 * @param history Prior message sequence
 * @param onToken Streaming callback for real-time text accumulation
 * @param imageUri Optional local image path for multimodal vision tasks
 * @param systemMessage Custom system prompt instructions
 */
export async function chatWithLocalLlama(
  modelId: LocalModelId,
  userMessage: string,
  history: LocalChatMessage[],
  onToken: (token: string, accumulated: string) => void,
  imageUri?: string,
  systemMessage?: string
): Promise<string> {
  if (Platform.OS === 'web') {
    throw new Error('On-device reasoning is not supported in the web browser.');
  }

  const context = await initLocalLlama(modelId);
  if (!context) {
    throw new Error('Failed to initialize local model context');
  }

  const stopTokens = ['<|im_end|>', '<|im_start|>', 'user:', 'assistant:', 'system:', '</s>', '<|end|>'];

  try {
    let result: any;

    const isVision = modelId === 'phi3_vision' && imageUri;
    const nPredict = isVision ? 256 : 512;
    if (isVision) {
      // Build JSI multimodal messages payload
      const messages: any[] = [];
      
      const sys = systemMessage || 'You are a concise, helpful on-device assistant. Analyze the image and prompt directly.';
      messages.push({
        role: 'system',
        content: sys,
      });

      for (const msg of history) {
        messages.push({
          role: msg.role,
          content: msg.text,
        });
      }

      messages.push({
        role: 'user',
        content: [
          { type: 'text', text: userMessage },
          { type: 'image_url', image_url: { url: imageUri } }
        ]
      });

      result = await context.completion(
        {
          messages,
          n_predict: nPredict,
          stop: stopTokens,
          temperature: 0.7,
          top_k: 40,
          top_p: 0.9,
        },
        (data: { token: string; accumulated_text: string }) => {
          onToken(data.token, data.accumulated_text);
        }
      );
    } else {
      // Regular ChatML/text-only completion
      const prompt = formatChatML(history, userMessage, systemMessage);
      result = await context.completion(
        {
          prompt,
          n_predict: nPredict,
          stop: stopTokens,
          temperature: 0.7,
          top_k: 40,
          top_p: 0.9,
        },
        (data: { token: string; accumulated_text: string }) => {
          onToken(data.token, data.accumulated_text);
        }
      );
    }

    return result.text;
  } catch (error) {
    console.error('Error running local llama completion:', error);
    throw error;
  }
}
