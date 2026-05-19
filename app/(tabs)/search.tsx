import { loadChatThreads, saveChatThreads, SessionChatThread } from '@/services/chatSessionStore';
import { transcribeAudioWithGroq } from '@/services/groqSpeech';
import { analyzeImageWithGroq, analyzeLiveFrame, chatWithGroqText, GroqChatContextMessage } from '@/services/groqVision';
import { transcribeAndAnswerWithGroq } from '@/services/groqQuestionAnswering';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { findSimilarItems, embedItem, EmbeddedItem, pruneOldItems } from '@/services/semanticSearch';
import { retrieveTopChunks, formatRagContext } from '@/services/ragSearch';
import { Ionicons } from '@expo/vector-icons';
import MaskedView from '@react-native-masked-view/masked-view';
import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
} from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';
import { loadRagDocuments, addRagDocument, deleteRagDocument, replaceRagDocument, RagDocument, RagGroup, loadRagGroups, addRagGroup, deleteRagGroup, migrateToFileStorage, cleanupOrphanedFiles, loadChunkIndex } from '@/services/ragContextStore';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Animated,
  Dimensions,
  Easing,
  FlatList,
  Image,
  Keyboard,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  imageUri?: string;
  extractedText?: string;
  hadImage?: boolean;
  createdAt: number;
};

type SelectedImage = {
  uri: string;
  base64: string;
  mimeType: 'image/jpeg' | 'image/png' | 'image/webp';
};

type ChatMode = 'chat' | 'listen';

const LIVE_CHUNK_MS = 9000;
const SCREEN_WIDTH = Dimensions.get('window').width;

const stripWelcomeMessages = (threadMessages: ChatMessage[]): ChatMessage[] =>
  threadMessages.filter((message) => {
    if (message.role !== 'assistant') return true;
    if (!/^welcome/.test(message.id)) return true;
    return !message.text.toLowerCase().includes('groq');
  });

const buildThreadTitle = (threadMessages: ChatMessage[]): string => {
  const firstUserText = threadMessages.find((message) => message.role === 'user' && message.text.trim())?.text.trim();
  if (!firstUserText) return 'New thread';
  return firstUserText.length > 40 ? `${firstUserText.slice(0, 40)}...` : firstUserText;
};

const formatThreadTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
};

const extractFirstQuestion = (text: string): string | null => {
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.find((line) => line.includes('?')) ?? null;
};

// Vision extraction prompt used for image analysis
const VISION_EXTRACTION_PROMPT = 'Extract all readable text and describe the visual scene in detail. Output EXACTLY in this format:\nEXTRACTED_TEXT_START\n<all readable text exactly as seen>\nEXTRACTED_TEXT_END\nVISUAL_CONTEXT_START\n<brief scene/object summary>\nVISUAL_CONTEXT_END';

const parseVisionPayload = (raw: string): { extractedText: string; visualContext: string; primaryQuestion: string } => {
  const defaultValue = { extractedText: raw.trim(), visualContext: '', primaryQuestion: '' };

  const sectionBetween = (source: string, start: string, end: string): string => {
    const startIdx = source.indexOf(start);
    const endIdx = source.indexOf(end);
    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) return '';
    return source.slice(startIdx + start.length, endIdx).trim();
  };

  const markerExtractedText = sectionBetween(raw, 'EXTRACTED_TEXT_START', 'EXTRACTED_TEXT_END');
  const markerVisualContext = sectionBetween(raw, 'VISUAL_CONTEXT_START', 'VISUAL_CONTEXT_END');
  const markerQuestion = (raw.match(/PRIMARY_QUESTION:\s*(.*)/)?.[1] ?? '').trim();
  if (markerExtractedText || markerVisualContext || markerQuestion) {
    return {
      extractedText: markerExtractedText,
      visualContext: markerVisualContext,
      primaryQuestion: markerQuestion === 'NONE' ? '' : markerQuestion,
    };
  }

  const readParsed = (value: unknown): { extractedText: string; visualContext: string; primaryQuestion: string } => {
    if (!value || typeof value !== 'object') return defaultValue;
    const record = value as { extracted_text?: unknown; visual_context?: unknown; primary_question?: unknown };
    return {
      extractedText: typeof record.extracted_text === 'string' ? record.extracted_text.trim() : '',
      visualContext: typeof record.visual_context === 'string' ? record.visual_context.trim() : '',
      primaryQuestion: typeof record.primary_question === 'string' ? record.primary_question.trim() : '',
    };
  };

  try {
    const parsed = JSON.parse(raw);
    return readParsed(parsed);
  } catch {
    const fenced = raw.match(/```json\s*([\s\S]*?)```/i)?.[1];
    if (!fenced) return defaultValue;
    try {
      return readParsed(JSON.parse(fenced));
    } catch {
      return defaultValue;
    }
  }
};

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

const COOLDOWN_MS = 12000; // 12 second cooldown between triggers
const CONTEXT_BUFFER_SIZE = 10; // Keep last 10 utterances for context

export default function SearchScreen() {
  const insets = useSafeAreaInsets();
  const listRef = useRef<FlatList<ChatMessage>>(null);
  const listenLoopActiveRef = useRef(false);
  const threadsRef = useRef<SessionChatThread[]>([]);
  const latestRecordingUrlRef = useRef<string | null>(null);
  const lastQuestionTimeRef = useRef<number>(0); // Track cooldown
  const imageMemoryRef = useRef<EmbeddedItem[]>([]); // Semantic search for images
  const recorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY, (status) => {
    if (status.url) {
      latestRecordingUrlRef.current = status.url;
    }
  });

  const [mode, setMode] = useState<ChatMode>('chat');
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [liveListening, setLiveListening] = useState(false);
  const [liveStatus, setLiveStatus] = useState('Live listen is off');
  const [keyboardVisible, setKeyboardVisible] = useState(false);
  const [historyVisible, setHistoryVisible] = useState(false);
  const [threadMenuVisible, setThreadMenuVisible] = useState<string | null>(null);
  const [holdRecording, setHoldRecording] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threads, setThreads] = useState<SessionChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [storeReady, setStoreReady] = useState(false);
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null);

  // â”€â”€ Live Camera PiP State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Completely isolated from audio/chat state â€” uses its own interval ref
  // and its own analyzing flag so it never blocks or races with the mic loop.
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraAnalyzing, setCameraAnalyzing] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const cameraRef = useRef<CameraView>(null);
  const cameraIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameCountRef = useRef(0); // shadow ref so interval closure always has fresh value
  const cameraAnalyzingRef = useRef(false); // shadow ref to avoid stale closure in interval
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const [contextVisible, setContextVisible] = useState(false);
  const [ragDocs, setRagDocs] = useState<RagDocument[]>([]);
  const [ragGroups, setRagGroups] = useState<RagGroup[]>([]);
  const [currentGroupId, setCurrentGroupId] = useState<string | null>(null);
  const [isCreatingGroup, setIsCreatingGroup] = useState(false);
  const [newGroupName, setNewGroupName] = useState('');
  const [isProcessingDocs, setIsProcessingDocs] = useState(false);
  const [processingProgress, setProcessingProgress] = useState('');
  const [fileMenuVisible, setFileMenuVisible] = useState<string | null>(null);
  const contextDrawerAnim = useRef(new Animated.Value(0)).current;

  const openContextTab = () => {
    setContextVisible(true);
    Animated.timing(contextDrawerAnim, {
      toValue: 1,
      duration: 300,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: true,
    }).start();
  };

  const closeContextTab = () => {
    Animated.timing(contextDrawerAnim, {
      toValue: 0,
      duration: 250,
      easing: Easing.in(Easing.cubic),
      useNativeDriver: true,
    }).start(() => {
      setContextVisible(false);
      setIsCreatingGroup(false);
      setCurrentGroupId(null);
    });
  };

  const handleAddDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['*/*'],
        copyToCacheDirectory: true,
      });

      if (result.canceled) return;
      
      const file = result.assets[0];
      let content = '';
      if (file.mimeType?.startsWith('image/')) {
        Alert.alert('Processing Image', 'Extracting visual context with AI...');
        const base64 = await FileSystem.readAsStringAsync(file.uri, { encoding: FileSystem.EncodingType.Base64 });
        const vision = await analyzeImageWithGroq(
          base64, 
          file.mimeType as any, 
          VISION_EXTRACTION_PROMPT
        );
        const parsed = parseVisionPayload(vision.description);
        content = `[Image Content - ${file.name}]\n`;
        if (parsed.extractedText) content += `Extracted Text:\n${parsed.extractedText}\n\n`;
        if (parsed.visualContext) content += `Visual Description:\n${parsed.visualContext}\n`;
        if (!parsed.extractedText && !parsed.visualContext) content += vision.description;
      } else {
        content = await FileSystem.readAsStringAsync(file.uri);
      }
      const updatedDocs = await addRagDocument(file.name, content, currentGroupId || undefined);
      setRagDocs(updatedDocs);
    } catch (e) {
      Alert.alert('Error', 'Could not read document');
      console.warn(e);
    }
  };

  const handleBatchImageProcess = async (assets: ImagePicker.ImagePickerAsset[]) => {
    setIsProcessingDocs(true);
    let updatedDocs = ragDocs;
    for (let i = 0; i < assets.length; i++) {
      setProcessingProgress(`Processing image ${i + 1} of ${assets.length}...`);
      const file = assets[i];
      try {
        const base64 = await FileSystem.readAsStringAsync(file.uri, { encoding: FileSystem.EncodingType.Base64 });
        const vision = await analyzeImageWithGroq(
          base64, 
          file.mimeType as any || 'image/jpeg', 
          VISION_EXTRACTION_PROMPT
        );
        const parsed = parseVisionPayload(vision.description);
        let content = `[Image Content - ${file.fileName || 'Photo'}]\n`;
        if (parsed.extractedText) content += `Extracted Text:\n${parsed.extractedText}\n\n`;
        if (parsed.visualContext) content += `Visual Description:\n${parsed.visualContext}\n`;
        if (!parsed.extractedText && !parsed.visualContext) content += vision.description;
        
        updatedDocs = await addRagDocument(file.fileName || `Photo-${Date.now()}`, content, currentGroupId || undefined);
        setRagDocs(updatedDocs);
      } catch (e) {
        console.warn('Failed to process image', i, e);
      }
    }
    setIsProcessingDocs(false);
    setProcessingProgress('');
  };

  const handleAddPhotos = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow photo library access.');
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsMultipleSelection: true,
      selectionLimit: 15,
      quality: 0.6,
    });
    if (!result.canceled && result.assets?.length > 0) {
      handleBatchImageProcess(result.assets);
    }
  };

  const handleTakePhotoContext = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow camera access.');
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.6,
    });
    if (!result.canceled && result.assets?.length > 0) {
      handleBatchImageProcess(result.assets);
    }
  };

  const handleRemoveDocument = async (id: string) => {
    const updatedDocs = await deleteRagDocument(id);
    setRagDocs(updatedDocs);
    setFileMenuVisible(null);
  };

  const handleReplaceDocument = async (docId: string) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['*/*'],
        copyToCacheDirectory: true,
      });

      if (result.canceled) return;

      const file = result.assets[0];
      let content = '';
      if (file.mimeType?.startsWith('image/')) {
        Alert.alert('Processing Image', 'Extracting visual context with AI...');
        const base64 = await FileSystem.readAsStringAsync(file.uri, { encoding: FileSystem.EncodingType.Base64 });
        const vision = await analyzeImageWithGroq(
          base64,
          file.mimeType as any,
          VISION_EXTRACTION_PROMPT
        );
        const parsed = parseVisionPayload(vision.description);
        content = `[Image Content - ${file.name}]\n`;
        if (parsed.extractedText) content += `Extracted Text:\n${parsed.extractedText}\n\n`;
        if (parsed.visualContext) content += `Visual Description:\n${parsed.visualContext}\n`;
        if (!parsed.extractedText && !parsed.visualContext) content += vision.description;
      } else {
        content = await FileSystem.readAsStringAsync(file.uri);
      }

      // Replace in-place: updates file, metadata, and chunk index atomically
      const updatedDocs = await replaceRagDocument(docId, file.name, content);
      setRagDocs(updatedDocs);
      setFileMenuVisible(null);
    } catch (e) {
      Alert.alert('Error', 'Could not replace document');
      console.warn(e);
    }
  };

  const handleRemoveGroup = async (id: string) => {
    const { updatedGroups, updatedDocs } = await deleteRagGroup(id);
    setRagGroups(updatedGroups);
    setRagDocs(updatedDocs);
    if (currentGroupId === id) setCurrentGroupId(null);
  };

  const handleCreateGroup = async () => {
    if (!newGroupName.trim()) return;
    const updated = await addRagGroup(newGroupName.trim());
    setRagGroups(updated);
    setNewGroupName('');
    setIsCreatingGroup(false);
  };

  const shineAnim = useRef(new Animated.Value(-1)).current;
  const heroOffsetAnim = useRef(new Animated.Value(0)).current;
  const drawerAnim = useRef(new Animated.Value(0)).current;
  const baseComposerYRef = useRef<number | null>(null);
  const lastHeroTargetRef = useRef(0);
  const [composerY, setComposerY] = useState<number | null>(null);

  const openHistory = () => { setHistoryVisible(true); Animated.timing(drawerAnim, { toValue: 1, duration: 300, easing: Easing.out(Easing.cubic), useNativeDriver: true }).start(); };
  const closeHistory = () => { Animated.timing(drawerAnim, { toValue: 0, duration: 250, easing: Easing.in(Easing.cubic), useNativeDriver: true }).start(() => setHistoryVisible(false)); };

  useEffect(() => {
    const loop = Animated.loop(Animated.timing(shineAnim, { toValue: 1, duration: 3500, easing: Easing.inOut(Easing.quad), useNativeDriver: true }));
    loop.start(); return () => loop.stop();
  }, [shineAnim]);

  useEffect(() => {
    if (composerY == null) return;
    if (baseComposerYRef.current == null) baseComposerYRef.current = composerY;
    const delta = composerY - baseComposerYRef.current;
    const targetOffset = Math.max(-140, Math.min(60, delta * 0.55));
    if (Math.abs(targetOffset - lastHeroTargetRef.current) < 1) return;
    lastHeroTargetRef.current = targetOffset;
    Animated.timing(heroOffsetAnim, { toValue: targetOffset, duration: 280, easing: Easing.inOut(Easing.cubic), useNativeDriver: true }).start();
  }, [composerY, heroOffsetAnim]);

  const canSend = useMemo(() => !sending && (!!input.trim() || !!selectedImage), [input, selectedImage, sending]);
  const showHero = mode === 'chat' && messages.length === 0;

  useEffect(() => {
    const loadThreadStore = async () => {
      try {
        await migrateToFileStorage(); await cleanupOrphanedFiles();
        const { threads: storedThreads } = await loadChatThreads();
        setRagDocs(await loadRagDocuments()); setRagGroups(await loadRagGroups());
        const now = Date.now(); const initialMessages: ChatMessage[] = [];
        const newThread: SessionChatThread = { id: `thread-${Date.now()}`, title: 'New thread', createdAt: now, updatedAt: now, messages: initialMessages };
        if (storedThreads.length > 0) {
          const normalized = storedThreads.map((t) => ({ ...t, messages: stripWelcomeMessages(t.messages as ChatMessage[]) }));
          const all = [newThread, ...normalized]; threadsRef.current = all; setThreads(all); setActiveThreadId(newThread.id); setMessages(initialMessages);
        } else {
          threadsRef.current = [newThread]; setThreads([newThread]); setActiveThreadId(newThread.id); setMessages(initialMessages);
          await saveChatThreads([newThread], newThread.id);
        }
      } catch (err) { console.warn('Failed to load session chat store', err); } finally { setStoreReady(true); }
    };
    void loadThreadStore();
  }, []);

  useEffect(() => {
    if (!storeReady || !activeThreadId) return;
    const now = Date.now(); const existing = threadsRef.current;
    const updatedThreads = (existing.some((t) => t.id === activeThreadId)
      ? existing.map((t) => t.id === activeThreadId ? { ...t, title: buildThreadTitle(messages), updatedAt: now, messages } : t)
      : [{ id: activeThreadId, title: buildThreadTitle(messages), createdAt: now, updatedAt: now, messages }, ...existing]
    ).sort((a, b) => b.updatedAt - a.updatedAt);
    const ft = updatedThreads.filter(t => t.id === activeThreadId || t.messages.length > 0);
    threadsRef.current = ft; setThreads(ft);
    void saveChatThreads(ft.filter(t => t.messages.length > 0), activeThreadId).catch((e) => console.warn('save failed', e));
  }, [messages, activeThreadId, storeReady]);

  useEffect(() => { setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 40); }, [messages, sending]);
  useEffect(() => { return () => { listenLoopActiveRef.current = false; recorder.stop().catch(() => {}); }; }, [recorder]);

  // Cleanup camera interval on unmount
  useEffect(() => {
    return () => {
      if (cameraIntervalRef.current) clearInterval(cameraIntervalRef.current);
    };
  }, []);

  // Pulse animation for the live indicator dot
  useEffect(() => {
    if (!cameraActive) return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 0.3, duration: 600, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [cameraActive, pulseAnim]);
  useEffect(() => {
    const se = Platform.OS === 'ios' ? 'keyboardWillShow' : 'keyboardDidShow';
    const he = Platform.OS === 'ios' ? 'keyboardWillHide' : 'keyboardDidHide';
    const s = Keyboard.addListener(se, () => setKeyboardVisible(true));
    const h = Keyboard.addListener(he, () => setKeyboardVisible(false));
    return () => { s.remove(); h.remove(); };
  }, []);

  const prepareImage = async (uri: string): Promise<SelectedImage> => {
    const m = await ImageManipulator.manipulateAsync(uri, [{ resize: { width: 640 } }], { compress: 0.5, format: ImageManipulator.SaveFormat.JPEG });
    return { uri: m.uri, base64: await FileSystem.readAsStringAsync(m.uri, { encoding: 'base64' as any }), mimeType: 'image/jpeg' };
  };

  // â”€â”€ Live Camera Frame Capture â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Captures one JPEG frame from the CameraView, compresses it, and sends it
  // to Groq Llama 4 Scout vision. Uses its own analyzing flag so it never
  // races with the audio transcription pipeline.
  const captureAndAnalyzeFrame = useCallback(async () => {
    if (!cameraRef.current || cameraAnalyzingRef.current) return;
    cameraAnalyzingRef.current = true;
    setCameraAnalyzing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: false, quality: 0.35, skipProcessing: true });
      if (!photo?.uri) return;
      const compressed = await ImageManipulator.manipulateAsync(
        photo.uri,
        [{ resize: { width: 480 } }],
        { compress: 0.4, format: ImageManipulator.SaveFormat.JPEG },
      );
      const b64 = await FileSystem.readAsStringAsync(compressed.uri, { encoding: 'base64' as any });
      await FileSystem.deleteAsync(photo.uri, { idempotent: true }).catch(() => {});
      await FileSystem.deleteAsync(compressed.uri, { idempotent: true }).catch(() => {});
      frameCountRef.current += 1;
      const idx = frameCountRef.current;
      setFrameCount(idx);
      const result = await analyzeLiveFrame(b64, idx);
      if (result.description?.trim()) {
        // Inject as a visually distinct camera message â€” never touches sending state
        setMessages((prev) => [
          ...prev,
          {
            id: `cam-${Date.now()}`,
            role: 'assistant' as const,
            text: `\uD83D\uDCF9 **Frame ${idx}** \u2022 Live Camera\n${result.description}`,
            createdAt: Date.now(),
          },
        ]);
      }
    } catch (err: any) {
      console.warn('[LiveCamera] Frame analysis error:', err?.message ?? err);
    } finally {
      cameraAnalyzingRef.current = false;
      setCameraAnalyzing(false);
    }
  }, []);

  const startLiveCamera = useCallback(async () => {
    const granted = cameraPermission?.granted ?? false;
    if (!granted) {
      const result = await requestCameraPermission();
      if (!result.granted) {
        Alert.alert('Camera permission needed', 'Please allow camera access for live analysis.');
        return;
      }
    }
    frameCountRef.current = 0;
    setFrameCount(0);
    setCameraActive(true);
    // Start frame loop â€” 4 second interval gives Groq enough time to respond
    // without creating a backlog of pending requests
    cameraIntervalRef.current = setInterval(() => {
      void captureAndAnalyzeFrame();
    }, 4000);
    // Capture the first frame immediately so the user sees instant feedback
    setTimeout(() => { void captureAndAnalyzeFrame(); }, 800);
  }, [cameraPermission, requestCameraPermission, captureAndAnalyzeFrame]);

  const stopLiveCamera = useCallback(() => {
    if (cameraIntervalRef.current) {
      clearInterval(cameraIntervalRef.current);
      cameraIntervalRef.current = null;
    }
    cameraAnalyzingRef.current = false;
    setCameraActive(false);
    setCameraAnalyzing(false);
    setFrameCount(0);
    frameCountRef.current = 0;
  }, []);

  const pickImage = async () => {
    if (sending) return;
    const p = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!p.granted) { Alert.alert('Permission needed', 'Please allow photo library access.'); return; }
    const r = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], allowsEditing: false, quality: 0.6, base64: false, exif: false });
    if (r.canceled || !r.assets?.length || !r.assets[0].uri) return;
    setSelectedImage(await prepareImage(r.assets[0].uri));
  };

  const submitMessage = async (messageText: string, imageForRequest: SelectedImage | null, clearComposer: boolean = true) => {
    const userMessage: ChatMessage = { id: `u-${Date.now()}`, role: 'user', text: messageText, imageUri: imageForRequest?.uri, hadImage: !!imageForRequest, createdAt: Date.now() };
    const contextHistory: GroqChatContextMessage[] = messages.filter((m) => m.id !== 'welcome').slice(-12).map((m) => ({ role: m.role, text: m.text }));
    setMessages((prev) => [...prev, userMessage]);
    if (clearComposer) { setInput(''); setSelectedImage(null); }
    setSending(true);
    try {
      let extractedText = ''; let imageContext = ''; let promptForChat = userMessage.text;
      if (imageForRequest) {
        const vision = await analyzeImageWithGroq(imageForRequest.base64, imageForRequest.mimeType,
          'Extract image text and visual context. Output EXACTLY in this format:\nEXTRACTED_TEXT_START\n<all readable text exactly as seen>\nEXTRACTED_TEXT_END\nVISUAL_CONTEXT_START\n<brief scene/object summary>\nVISUAL_CONTEXT_END\nPRIMARY_QUESTION: <most important question found in extracted text, or NONE>');
        const parsed = parseVisionPayload(vision.description);
        extractedText = parsed.extractedText;
        if (extractedText) { imageMemoryRef.current.push(embedItem(`img-${Date.now()}`, extractedText)); imageMemoryRef.current = pruneOldItems(imageMemoryRef.current, 3600000); }
        imageContext = [extractedText ? `Extracted text:\n${extractedText}` : '', parsed.visualContext ? `Visual context:\n${parsed.visualContext}` : ''].filter(Boolean).join('\n\n');
        const pq = parsed.primaryQuestion || extractFirstQuestion(extractedText) || '';
        promptForChat = pq ? `Question detected in image text: "${pq}". Answer only this question directly.` : 'No question was found in image text. Give a concise answer about what you think this image is conveying.';
      }
      const pastImages = imageMemoryRef.current.length > 0 ? findSimilarItems(promptForChat, imageMemoryRef.current, 2, 0.4) : [];
      const pastImageContext = pastImages.length > 0 ? '\n\n[Relevant past images for context]:\n' + pastImages.map(i => `â€¢ ${i.text.substring(0, 150)}`).join('\n') : '';
      const ragContextWithContent = ragDocs.length > 0 ? await (async () => formatRagContext(retrieveTopChunks(promptForChat, await loadChunkIndex(), 5)))() : '';
      const res = await chatWithGroqText(promptForChat + pastImageContext + (ragContextWithContent ? '\n\n' + ragContextWithContent : ''), contextHistory, imageContext || undefined);
      setMessages((prev) => [...prev, { id: `a-${Date.now()}`, role: 'assistant', text: res.content || 'No response from model.', extractedText: extractedText || imageContext, createdAt: Date.now() }]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { id: `aerr-${Date.now()}`, role: 'assistant', text: `Error: ${err?.message ?? 'Request failed'}`, createdAt: Date.now() }]);
    } finally { setSending(false); }
  };

  const addAssistantMessage = (answer: string) => { setMessages((prev) => [...prev, { id: `a-${Date.now()}`, role: 'assistant', text: answer, createdAt: Date.now() }]); };
  const handleSend = async () => { if (!canSend) return; await submitMessage(input.trim() || 'Analyze this image.', selectedImage); };

  const startHoldRecording = async () => {
    if (sending || liveListening || holdRecording) return;
    const p = await requestRecordingPermissionsAsync();
    if (!p.granted) { Alert.alert('Permission needed', 'Please allow microphone access.'); return; }
    try { latestRecordingUrlRef.current = null; await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true }); await recorder.prepareToRecordAsync(); recorder.record(); setHoldRecording(true); }
    catch { setHoldRecording(false); }
  };

  const stopHoldRecording = async () => {
    if (!holdRecording) return; setHoldRecording(false);
    try {
      await recorder.stop(); const uri = latestRecordingUrlRef.current; if (!uri) return;
      const transcript = await transcribeAudioWithGroq(uri);
      await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
      const fp = transcript.trim(); if (!fp) return;
      await submitMessage(fp, null, false);
    } finally { await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true }).catch(() => {}); }
  };

  const captureAudioChunk = async (): Promise<string | null> => {
    try {
      latestRecordingUrlRef.current = null;
      await recorder.prepareToRecordAsync();
      recorder.record();
      await new Promise((resolve) => setTimeout(resolve, LIVE_CHUNK_MS));
      if (!listenLoopActiveRef.current) { await recorder.stop().catch(() => {}); return null; }
      await recorder.stop();
      return recorder.uri || latestRecordingUrlRef.current;
    } catch (err: any) {
      setLiveStatus(`Capture error: ${err?.message ?? 'Unknown'}`);
      return null;
    }
  };

  const stopLiveListening = async () => {
    listenLoopActiveRef.current = false;
    setLiveListening(false);
    setLiveStatus('Live listen is off');
    await recorder.stop().catch(() => {});
    await setAudioModeAsync({ allowsRecording: false, playsInSilentMode: true }).catch(() => {});
  };

  // NEW: Groq-only workflow (combined transcription + detection + answering)
  const runLiveLoopGroqOnly = async () => {
    while (listenLoopActiveRef.current) {
      try {
        setLiveStatus('Listening...');
        const audioUri = await captureAudioChunk();
        if (!listenLoopActiveRef.current) break;
        if (!audioUri) continue;

        setLiveStatus('Processing with Groq...');
        const result = await transcribeAndAnswerWithGroq(audioUri);

        await FileSystem.deleteAsync(audioUri, { idempotent: true }).catch(() => { });

        // Debug: show what was transcribed
        if (result.question) {
          setLiveStatus(`Heard: "${result.question.substring(0, 50)}..."`);
        }

        if (!result.hasQuestion || !result.answer) {
          setLiveStatus('No question detected. Continuing...');
          continue;
        }

        // Check cooldown
        const now = Date.now();
        const timeSinceLastQuestion = now - lastQuestionTimeRef.current;
        if (timeSinceLastQuestion < COOLDOWN_MS) {
          setLiveStatus(`Cooldown: ${Math.ceil((COOLDOWN_MS - timeSinceLastQuestion) / 1000)}s remaining...`);
          continue;
        }

        // Trigger response — store the transcribed question as a user message
        // so conversations are readable when reopened from history.
        lastQuestionTimeRef.current = now;
        setLiveStatus('Question detected! Answering...');
        setMessages((prev) => [
          ...prev,
          {
            id: `live-q-${Date.now()}`,
            role: 'user' as const,
            text: `[Live] ${result.question}`,
            createdAt: Date.now(),
          },
        ]);
        addAssistantMessage(result.answer);
        setLiveStatus('Listening...');
      } catch (err: any) {
        setLiveStatus(`Error: ${err?.message ?? 'Unknown error'}`);
      }
    }
  };

  const startLiveListening = async () => {
    if (liveListening || sending) return;

    const permission = await requestRecordingPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow microphone access for live listening mode.');
      return;
    }

    await setAudioModeAsync({
      allowsRecording: true,
      playsInSilentMode: true,
    });

    // Reset state for new session
    lastQuestionTimeRef.current = 0;
    listenLoopActiveRef.current = true;
    setLiveListening(true);
    setLiveStatus('Starting Groq Live Listen...');

    // Start the loop with a small delay
    setTimeout(() => {
      if (listenLoopActiveRef.current) {
        setLiveStatus('Listening...');
        void runLiveLoopGroqOnly();
      }
    }, 500);
  };

  const handleCaptureToChat = async () => {
    if (sending) return;

    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert('Permission needed', 'Please allow camera access to capture images.');
      return;
    }

    const captured = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      allowsEditing: false,
      quality: 0.6,
      base64: false,
      exif: false,
    });

    if (captured.canceled || !captured.assets?.length) return;
    const asset = captured.assets[0];
    if (!asset.uri) return;

    const prepared = await prepareImage(asset.uri);
    await submitMessage(input.trim() || 'Analyze this image.', prepared);
  };

  const openThread = (threadId: string) => {
    const thread = threadsRef.current.find((item) => item.id === threadId);
    if (!thread) return;
    
    const validExisting = threadsRef.current.filter(t => t.id === threadId || t.messages.length > 0);
    threadsRef.current = validExisting;
    setThreads(validExisting);
    
    setActiveThreadId(thread.id);
    setMessages(thread.messages as ChatMessage[]);
    closeHistory();
    void saveChatThreads(validExisting.filter(t => t.messages.length > 0), thread.id).catch(() => { });
  };

  const createNewThread = () => {
    const now = Date.now();
    const initialMessages: ChatMessage[] = [];
    const newThread: SessionChatThread = {
      id: `thread-${now}`,
      title: 'New thread',
      createdAt: now,
      updatedAt: now,
      messages: initialMessages,
    };
    const validExisting = threadsRef.current.filter(t => t.messages.length > 0);
    const updatedThreads = [newThread, ...validExisting].sort((a, b) => b.updatedAt - a.updatedAt);
    threadsRef.current = updatedThreads;
    setThreads(updatedThreads);
    setActiveThreadId(newThread.id);
    setMessages(initialMessages);
    setSelectedImage(null);
    setInput('');
    closeHistory();
    void saveChatThreads(validExisting, newThread.id).catch(() => { });
  };

  const deleteThread = (threadId: string) => {
    const remaining = threadsRef.current.filter((t) => t.id !== threadId);
    threadsRef.current = remaining;
    setThreads(remaining);
    setThreadMenuVisible(null);
    if (activeThreadId === threadId) {
      const next = remaining.find((t) => t.messages.length > 0);
      if (next) {
        setActiveThreadId(next.id);
        setMessages(next.messages as ChatMessage[]);
      } else {
        const now = Date.now();
        const fresh: SessionChatThread = {
          id: `thread-${now}`,
          title: 'New thread',
          createdAt: now,
          updatedAt: now,
          messages: [],
        };
        threadsRef.current = [fresh];
        setThreads([fresh]);
        setActiveThreadId(fresh.id);
        setMessages([]);
      }
    }
    void saveChatThreads(
      remaining.filter((t) => t.messages.length > 0),
      activeThreadId ?? ''
    ).catch(() => {});
  };

  const typingMessage: ChatMessage | null = sending
    ? { id: 'typing', role: 'assistant', text: 'Thinking...', createdAt: Date.now() }
    : null;

  const listData = typingMessage ? [...messages, typingMessage] : messages;

  return (
    <View style={styles.root}>
      <LinearGradient colors={['#FFFFFF08', '#00000000']} style={styles.ambient} />

      <View style={[styles.header, { paddingTop: insets.top + 8 }]}>
        <Pressable style={styles.historyBtn} onPress={openHistory}>
          <Ionicons name="time-outline" size={20} color="#E5E5E5" />
        </Pressable>
        <View style={styles.modeSwitch}>
          <Pressable style={[styles.modeBtn, mode === 'chat' && styles.modeBtnActive]} onPress={() => setMode('chat')}>
            <Ionicons name="sparkles-outline" size={18} color={mode === 'chat' ? '#F5F5F5' : '#8A8A8A'} />
          </Pressable>
          <Pressable style={[styles.modeBtn, mode === 'listen' && styles.modeBtnActive]} onPress={() => setMode('listen')}>
            <Ionicons name="mic-outline" size={18} color={mode === 'listen' ? '#F5F5F5' : '#8A8A8A'} />
          </Pressable>
        </View>
        <Pressable
          style={styles.avatarSquare}
          onPress={openContextTab}
          accessibilityRole="button"
          accessibilityLabel="Meeting Context"
        >
          <Ionicons name="document-text-outline" size={18} color="#E5E5E5" />
        </Pressable>
      </View>

      <KeyboardAvoidingView
        style={styles.chatArea}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={0}
      >
        <FlatList
          ref={listRef}
          data={listData}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          keyboardDismissMode={Platform.OS === 'ios' ? 'interactive' : 'on-drag'}
          keyboardShouldPersistTaps="handled"
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          renderItem={({ item }) => {
            const isUser = item.role === 'user';
            const isTyping = item.id === 'typing';
            return (
              <View style={[styles.row, isUser ? styles.rowRight : styles.rowLeft]}>
                <BlurView intensity={20} tint="dark" style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
                  {item.imageUri ? <Image source={{ uri: item.imageUri }} style={styles.imagePreview} /> : null}
                  {isTyping ? (
                    <View style={styles.typingRow}>
                      <ActivityIndicator size="small" color="#D4D4D4" />
                      <Text style={styles.typingText}>ThinkingÃ¢â‚¬Â¦</Text>
                    </View>
                  ) : (
                    <Text style={styles.messageText}>{item.text}</Text>
                  )}
                </BlurView>
              </View>
            );
          }}
        />

        {mode === 'listen' ? (
          <View style={styles.livePanel}>
            <View style={{ flex: 1 }}>
              <Text style={styles.liveTitle}>Live Listen Mode</Text>
              <Text style={styles.liveStatus}>{liveStatus}</Text>
            </View>
            {/* Camera toggle - fully independent of the mic loop */}
            <Pressable
              style={[styles.liveCameraBtn, cameraActive && styles.liveCameraBtnActive]}
              onPress={cameraActive ? stopLiveCamera : startLiveCamera}
              accessibilityLabel={cameraActive ? 'Stop live camera' : 'Start live camera'}
            >
              <Ionicons
                name={cameraActive ? 'videocam' : 'videocam-outline'}
                size={18}
                color={cameraActive ? '#000' : '#CFCFCF'}
              />
            </Pressable>
            <Pressable
              style={[styles.liveBtn, liveListening && styles.liveBtnStop]}
              onPress={liveListening ? stopLiveListening : startLiveListening}
              disabled={sending}
            >
              <Ionicons name={liveListening ? 'stop' : 'mic'} size={20} color="#000" />
              <Text style={styles.liveBtnText}>{liveListening ? 'Stop' : 'Start'}</Text>
            </Pressable>
          </View>
        ) : null}

        {mode === 'chat' ? (
          <>
            {selectedImage ? (
              <View style={styles.attachBar}>
                <Image source={{ uri: selectedImage.uri }} style={styles.attachImage} />
                <Text style={styles.attachText}>Image attached</Text>
                <Pressable onPress={() => setSelectedImage(null)} style={styles.attachClose}>
                  <Ionicons name="close" size={18} color="#F5F5F5" />
                </Pressable>
              </View>
            ) : null}

            <BlurView
              intensity={70}
              tint="dark"
              style={[styles.composer, { marginBottom: keyboardVisible ? 12 : insets.bottom + 20 }]}
              onLayout={(event) => setComposerY(event.nativeEvent.layout.y)}
            >
              <TextInput
                value={input}
                onChangeText={setInput}
                placeholder="Ask anything..."
                placeholderTextColor="#8A8A8A"
                style={styles.input}
                editable={!sending}
                multiline
              />
              <View style={styles.composerActions}>
                <Pressable style={styles.plusBtn} onPress={pickImage} disabled={sending}>
                  <Ionicons name="add" size={20} color="#CFCFCF" />
                </Pressable>
                <View style={styles.modelChip}>
                  <Text style={styles.modelChipText}>Model</Text>
                </View>
                <View style={{ flex: 1 }} />
                <Pressable style={styles.iconBtn} onPress={handleCaptureToChat} disabled={sending}>
                  <Ionicons name="camera-outline" size={20} color="#CFCFCF" />
                </Pressable>
                <Pressable
                  style={[styles.iconBtn, holdRecording && styles.iconBtnActive]}
                  onPressIn={startHoldRecording}
                  onPressOut={stopHoldRecording}
                  disabled={sending || liveListening}
                >
                  <Ionicons name="mic" size={20} color={holdRecording ? '#111111' : '#CFCFCF'} />
                </Pressable>
                <Pressable style={[styles.sendBtn, !canSend && styles.sendBtnDisabled]} onPress={handleSend} disabled={!canSend}>
                  <Ionicons name="arrow-up" size={18} color="#000000" />
                </Pressable>
              </View>
            </BlurView>
          </>
        ) : (
          <View style={{ height: insets.bottom + 8 }} />
        )}
      </KeyboardAvoidingView>

      {showHero ? (
        <Animated.View style={[styles.heroWrap, { transform: [{ translateY: heroOffsetAnim }] }]} pointerEvents="none">
          <View style={styles.brandStack}>
            <Text style={styles.brandText}>SahAi</Text>
            <MaskedView
              pointerEvents="none"
              style={styles.brandMask}
              maskElement={<Text style={styles.brandMaskText}>SahAi</Text>}
            >
              <Animated.View
                style={[
                  styles.brandShine,
                  {
                    transform: [
                      {
                        translateX: shineAnim.interpolate({
                          inputRange: [-1, 1],
                          outputRange: [-180, 180],
                        }),
                      },
                      { rotate: '18deg' },
                    ],
                  },
                ]}
              >
                <LinearGradient
                  colors={['#FFFFFF00', '#FFFFFFE6', '#FFFFFF00']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={StyleSheet.absoluteFill}
                />
              </Animated.View>
            </MaskedView>
          </View>
        </Animated.View>
      ) : null}

      <Modal visible={historyVisible} transparent animationType="none" onRequestClose={closeHistory}>
        <View style={styles.historyOverlay}>
          <Animated.View style={[StyleSheet.absoluteFill, { backgroundColor: '#000', opacity: drawerAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 0.6] }) }]}>
            <Pressable style={StyleSheet.absoluteFill} onPress={closeHistory} />
          </Animated.View>
          <AnimatedBlurView intensity={60} tint="dark" style={[styles.historySheet, { 
            transform: [{ translateX: drawerAnim.interpolate({ inputRange: [0, 1], outputRange: [-SCREEN_WIDTH, 0] }) }],
            paddingBottom: Math.max(insets.bottom, 12),
            paddingTop: Math.max(insets.top, 12)
          }]}>
            <View style={styles.historyHeader}>
              <Pressable onPress={closeHistory} style={styles.closeDrawerBtn}>
                <Ionicons name="chevron-back" size={28} color="#F5F5F5" />
              </Pressable>
              <Text style={[styles.historyTitle, { flex: 1, marginLeft: 8 }]}>Conversations</Text>
              <Pressable style={styles.newThreadBtn} onPress={createNewThread}>
                <Ionicons name="add" size={24} color="#000000" />
              </Pressable>
            </View>
            <FlatList
              data={threads.filter(t => t.id !== activeThreadId && t.messages.length > 0)}
              keyExtractor={(item) => item.id}
              contentContainerStyle={styles.historyList}
              renderItem={({ item }) => (
                <View style={[styles.historyItem, item.id === activeThreadId && styles.historyItemActive, { flexDirection: 'row', alignItems: 'center' }]}>
                  <Pressable style={{ flex: 1 }} onPress={() => { setThreadMenuVisible(null); openThread(item.id); }}>
                    <Text style={styles.historyItemTitle} numberOfLines={1}>{item.title}</Text>
                    <Text style={styles.historyItemMeta}>
                      {formatThreadTime(item.updatedAt)}  •  {item.messages.length} messages
                    </Text>
                  </Pressable>
                  <Pressable
                    style={styles.threadMenuBtn}
                    onPress={() => setThreadMenuVisible(threadMenuVisible === item.id ? null : item.id)}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  >
                    <Ionicons name="ellipsis-vertical" size={18} color="#6A6A6A" />
                  </Pressable>
                  {threadMenuVisible === item.id && (
                    <View style={styles.threadMenuDropdown}>
                      <Pressable
                        style={styles.threadMenuOption}
                        onPress={() => deleteThread(item.id)}
                      >
                        <Ionicons name="trash-outline" size={16} color="#F87171" />
                        <Text style={styles.threadMenuOptionText}>Delete</Text>
                      </Pressable>
                    </View>
                  )}
                </View>
              )}
              ListEmptyComponent={<Text style={styles.historyEmpty}>No threads yet</Text>}
            />
          </AnimatedBlurView>
        </View>
      </Modal>

      <Modal visible={contextVisible} transparent animationType="none" onRequestClose={closeContextTab}>
        <View style={styles.historyOverlay}>
          <Animated.View style={[StyleSheet.absoluteFill, { backgroundColor: '#000', opacity: contextDrawerAnim.interpolate({ inputRange: [0, 1], outputRange: [0, 0.6] }) }]}>
            <Pressable style={StyleSheet.absoluteFill} onPress={closeContextTab} />
          </Animated.View>
          <AnimatedBlurView intensity={60} tint="dark" style={[styles.contextSheet, { 
            transform: [{ translateX: contextDrawerAnim.interpolate({ inputRange: [0, 1], outputRange: [SCREEN_WIDTH, 0] }) }],
            paddingBottom: Math.max(insets.bottom, 12),
            paddingTop: Math.max(insets.top, 12)
          }]}>
            <View style={styles.historyHeader}>
              <Pressable onPress={closeContextTab} style={styles.closeDrawerBtn}>
                <Ionicons name="close" size={28} color="#F5F5F5" />
              </Pressable>
              <Text style={[styles.historyTitle, { flex: 1, marginLeft: 8 }]}>Meeting Context</Text>
            </View>
                        {isCreatingGroup ? (
              <View style={styles.createGroupForm}>
                <TextInput
                  style={styles.createGroupInput}
                  value={newGroupName}
                  onChangeText={setNewGroupName}
                  placeholder="Folder Name"
                  placeholderTextColor="#666"
                  autoFocus
                />
                <View style={{ flexDirection: 'row', gap: 8 }}>
                  <Pressable style={styles.groupSaveBtn} onPress={handleCreateGroup}>
                    <Text style={{ color: '#000', fontWeight: 'bold' }}>Save</Text>
                  </Pressable>
                  <Pressable style={styles.groupCancelBtn} onPress={() => setIsCreatingGroup(false)}>
                    <Text style={{ color: '#FFF' }}>Cancel</Text>
                  </Pressable>
                </View>
              </View>
            ) : (
              <View style={{ marginBottom: 16 }}>
                <View style={{ flexDirection: 'row', gap: 8, marginBottom: 8 }}>
                  {!currentGroupId && (
                    <Pressable style={[styles.addDocBtn, { flex: 1, marginBottom: 0 }]} onPress={() => setIsCreatingGroup(true)}>
                      <Ionicons name="folder-outline" size={20} color="#000" />
                      <Text style={styles.addDocText}>New Folder</Text>
                    </Pressable>
                  )}
                  <Pressable style={[styles.addDocBtn, { flex: 1, marginBottom: 0 }]} onPress={handleAddDocument}>
                    <Ionicons name="document-text-outline" size={20} color="#000" />
                    <Text style={styles.addDocText}>Add File</Text>
                  </Pressable>
                </View>
                <View style={{ flexDirection: 'row', gap: 8 }}>
                  <Pressable style={[styles.addDocBtn, { flex: 1, marginBottom: 0, backgroundColor: '#333' }]} onPress={handleAddPhotos}>
                    <Ionicons name="images-outline" size={20} color="#FFF" />
                    <Text style={[styles.addDocText, { color: '#FFF' }]}>Add Photos</Text>
                  </Pressable>
                  <Pressable style={[styles.addDocBtn, { flex: 1, marginBottom: 0, backgroundColor: '#333' }]} onPress={handleTakePhotoContext}>
                    <Ionicons name="camera-outline" size={20} color="#FFF" />
                    <Text style={[styles.addDocText, { color: '#FFF' }]}>Take Photo</Text>
                  </Pressable>
                </View>
              </View>
            )}

            {isProcessingDocs && (
              <View style={styles.processingOverlay}>
                <ActivityIndicator size="large" color="#FFF" />
                <Text style={styles.processingText}>{processingProgress}</Text>
              </View>
            )}

            {currentGroupId && (
              <Pressable style={styles.backToRootBtn} onPress={() => setCurrentGroupId(null)}>
                <Ionicons name="chevron-back" size={20} color="#F5F5F5" />
                <Text style={{ color: '#F5F5F5', fontSize: 16, fontWeight: '600' }}>Back to Folders</Text>
              </Pressable>
            )}

            <FlatList
              data={[
                ...(currentGroupId === null ? ragGroups.map(g => ({ ...g, isGroup: true })) : []),
                ...(currentGroupId === null 
                    ? ragDocs.filter(d => !d.groupId).map(d => ({ ...d, isGroup: false })) 
                    : ragDocs.filter(d => d.groupId === currentGroupId).map(d => ({ ...d, isGroup: false })))
              ]}
              keyExtractor={(item) => item.id}
              contentContainerStyle={styles.historyList}
              renderItem={({ item }) => {
                if (item.isGroup) {
                  return (
                    <Pressable style={[styles.historyItem, { flexDirection: 'row', alignItems: 'center' }]} onPress={() => setCurrentGroupId(item.id)}>
                      <Ionicons name="folder" size={24} color="#8A8A8A" style={{ marginRight: 12 }} />
                      <View style={{ flex: 1 }}>
                        <Text style={styles.historyItemTitle} numberOfLines={1}>{item.name}</Text>
                        <Text style={styles.historyItemMeta}>{ragDocs.filter(d => d.groupId === item.id).length} files</Text>
                      </View>

                    </Pressable>
                  );
                } else {
                  return (
                    <View style={[styles.historyItem, { flexDirection: 'row', alignItems: 'center' }]}>
                      <Ionicons name="document-text-outline" size={24} color="#8A8A8A" style={{ marginRight: 12 }} />
                      <View style={{ flex: 1 }}>
                        <Text style={styles.historyItemTitle} numberOfLines={1}>{item.name}</Text>
                        <Text style={styles.historyItemMeta}>{formatThreadTime(item.createdAt)}</Text>
                      </View>
                      <Pressable
                        onPress={() => setFileMenuVisible(fileMenuVisible === item.id ? null : item.id)}
                        style={styles.fileMenuBtn}
                      >
                        <Ionicons name="ellipsis-vertical" size={20} color="#8A8A8A" />
                      </Pressable>
                      {fileMenuVisible === item.id && (
                        <View style={styles.fileMenuDropdown}>
                          <Pressable
                            style={styles.fileMenuOption}
                            onPress={() => handleReplaceDocument(item.id)}
                          >
                            <Ionicons name="swap-horizontal-outline" size={18} color="#F5F5F5" />
                            <Text style={styles.fileMenuOptionText}>Replace</Text>
                          </Pressable>
                          <Pressable
                            style={[styles.fileMenuOption, styles.fileMenuOptionDanger]}
                            onPress={() => handleRemoveDocument(item.id)}
                          >
                            <Ionicons name="trash-outline" size={18} color="#F87171" />
                            <Text style={[styles.fileMenuOptionText, { color: '#F87171' }]}>Delete</Text>
                          </Pressable>
                        </View>
                      )}
                    </View>
                  );
                }
              }}
              ListEmptyComponent={<Text style={styles.historyEmpty}>Nothing here yet</Text>}
            />
          </AnimatedBlurView>
        </View>
      </Modal>

      {/* Floating Camera PiP Overlay - sits above all content via zIndex 999 */}
      {cameraActive ? (
        <View style={styles.pip} pointerEvents="box-none">
          <View style={styles.pipHeader}>
            <Animated.View style={[styles.pipDot, { opacity: pulseAnim }]} />
            <Text style={styles.pipLive}>LIVE</Text>
            <Text style={styles.pipFrameCount}>
              {cameraAnalyzing ? 'Analyzing...' : 'Frame ' + frameCount}
            </Text>
            <Pressable style={styles.pipClose} onPress={stopLiveCamera} accessibilityLabel="Close camera preview">
              <Ionicons name="close" size={14} color="#E5E5E5" />
            </Pressable>
          </View>
          <CameraView ref={cameraRef} style={styles.pipCamera} facing="back" />
          {cameraAnalyzing ? (
            <View style={styles.pipAnalyzingOverlay} pointerEvents="none">
              <ActivityIndicator size="small" color="#FFFFFF" />
            </View>
          ) : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000000' },
  ambient: { position: 'absolute', top: 0, left: 0, right: 0, height: 120 },
  header: {
    paddingHorizontal: 16,
    paddingBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: 'transparent',
  },
  historyBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#ffffff15',
    borderWidth: 1,
    borderColor: '#ffffff30',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarSquare: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: '#ffffff15',
    borderWidth: 1,
    borderColor: '#ffffff30',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modeSwitch: {
    flexDirection: 'row',
    backgroundColor: '#ffffff10',
    borderWidth: 1,
    borderColor: '#ffffff30',
    borderRadius: 18,
    overflow: 'hidden',
  },
  modeBtn: { paddingHorizontal: 22, paddingVertical: 10 },
  modeBtnActive: { backgroundColor: '#ffffff25' },
  modeText: { color: '#8A8A8A', fontSize: 14, fontWeight: '600' },
  modeTextActive: { color: '#F5F5F5' },
  chatArea: { flex: 1 },
  listContent: { paddingHorizontal: 12, paddingTop: 10, paddingBottom: 180, gap: 10 },
  heroWrap: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandStack: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  brandMask: {
    ...StyleSheet.absoluteFillObject,
  },
  brandMaskText: {
    color: '#000000',
    fontSize: 44,
    letterSpacing: 0.8,
    fontFamily: Platform.select({
      ios: 'AvenirNext-DemiBold',
      android: 'sans-serif-medium',
      default: 'System',
    }),
  },
  brandText: {
    color: '#B5B5B5',
    fontSize: 44,
    letterSpacing: 0.8,
    fontFamily: Platform.select({
      ios: 'AvenirNext-DemiBold',
      android: 'sans-serif-medium',
      default: 'System',
    }),
    backgroundColor: 'transparent',
  },
  brandShine: {
    position: 'absolute',
    top: -28,
    bottom: -28,
    width: 110,
    opacity: 0.95,
  },
  row: { flexDirection: 'row' },
  rowLeft: { justifyContent: 'flex-start' },
  rowRight: { justifyContent: 'flex-end' },
  bubble: {
    maxWidth: '86%',
    borderWidth: 1,
    borderRadius: 16,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
    overflow: 'hidden',
  },
  assistantBubble: { backgroundColor: '#12121280', borderColor: '#2E2E2E' },
  userBubble: { backgroundColor: '#1A1A1A80', borderColor: '#3A3A3A' },
  messageText: { color: '#F5F5F5', fontSize: 16, lineHeight: 24 },
  typingRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  typingText: { color: '#B5B5B5', fontSize: 16 },
  imagePreview: { width: 200, height: 160, borderRadius: 10 },
  livePanel: {
    marginHorizontal: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 12,
    backgroundColor: '#111111',
    padding: 10,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  liveTitle: { color: '#F5F5F5', fontSize: 15, fontWeight: '700' },
  liveStatus: { color: '#B5B5B5', fontSize: 14, marginTop: 2 },
  liveCameraBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#1D1D1D',
    borderWidth: 1,
    borderColor: '#3A3A3A',
    alignItems: 'center',
    justifyContent: 'center',
  },
  liveCameraBtnActive: {
    backgroundColor: '#E5E5E5',
    borderColor: '#D0D0D0',
  },
  liveBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#E5E5E5',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  liveBtnStop: { backgroundColor: '#BEBEBE' },
  liveBtnText: { color: '#000', fontSize: 14, fontWeight: '700' },
  pip: {
    position: 'absolute',
    bottom: 210,
    right: 14,
    width: 168,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#3A3A3A',
    backgroundColor: '#0C0C0C',
    overflow: 'hidden',
    zIndex: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.7,
    shadowRadius: 16,
    elevation: 20,
  },
  pipHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 8,
    paddingVertical: 5,
    backgroundColor: '#0C0C0CEE',
  },
  pipDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: '#EF4444',
  },
  pipLive: {
    color: '#EF4444',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  pipFrameCount: {
    flex: 1,
    color: '#8A8A8A',
    fontSize: 10,
    fontWeight: '500',
  },
  pipClose: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#2A2A2A',
    alignItems: 'center',
    justifyContent: 'center',
  },
  pipCamera: {
    width: '100%',
    height: 224,
  },
  pipAnalyzingOverlay: {
    ...StyleSheet.absoluteFillObject,
    top: 27,
    backgroundColor: '#00000066',
    alignItems: 'center',
    justifyContent: 'center',
  },
  attachBar: {
    marginHorizontal: 10,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 12,
    backgroundColor: '#111111',
    padding: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  attachImage: { width: 50, height: 50, borderRadius: 8 },
  attachText: { flex: 1, color: '#B5B5B5', fontSize: 15 },
  attachClose: {
    width: 26,
    height: 26,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: '#2E2E2E',
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickBar: {
    marginHorizontal: 10,
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  holdMicBtn: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 12,
    backgroundColor: '#111111',
    paddingHorizontal: 10,
    paddingVertical: 9,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  holdMicBtnActive: {
    backgroundColor: '#F5F5F5',
    borderColor: '#D0D0D0',
  },
  holdMicText: { color: '#B5B5B5', fontSize: 12, fontWeight: '600' },
  holdMicTextActive: { color: '#000' },
  captureNowBtn: {
    borderRadius: 12,
    backgroundColor: '#E5E5E5',
    paddingHorizontal: 10,
    paddingVertical: 9,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  captureNowText: { color: '#000', fontSize: 12, fontWeight: '700' },
  composer: {
    marginHorizontal: 12,
    borderRadius: 28,
    borderWidth: 1,
    borderColor: '#FFFFFF30',
    backgroundColor: '#88888825',
    paddingHorizontal: 10,
    paddingTop: 16,
    paddingBottom: 16,
    gap: 8,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.8,
    shadowRadius: 20,
    elevation: 12,
  },
  composerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  plusBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#1D1D1D',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modelChip: {
    height: 30,
    borderRadius: 15,
    paddingHorizontal: 12,
    backgroundColor: '#1D1D1D',
    justifyContent: 'center',
  },
  modelChipText: {
    color: '#A6A6A6',
    fontSize: 14,
    fontWeight: '600',
  },
  iconBtn: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#1D1D1D',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnActive: {
    backgroundColor: '#F5F5F5',
  },
  input: {
    minHeight: 40,
    maxHeight: 120,
    color: '#F5F5F5',
    paddingHorizontal: 4,
    paddingVertical: 8,
    fontSize: 16,
  },
  sendBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnDisabled: { opacity: 0.45 },
  historyOverlay: {
    flex: 1,
  },
  historySheet: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    width: '100%',
    borderRightWidth: 1,
    borderColor: '#2E2E2E',
    backgroundColor: '#0C0C0C80',
    paddingHorizontal: 12,
    shadowColor: '#000',
    shadowOffset: { width: 4, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 10,
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  closeDrawerBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: -8,
  },
  historyTitle: {
    color: '#F5F5F5',
    fontSize: 18,
    fontWeight: '700',
  },
  newThreadBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  historyList: {
    gap: 8,
    paddingBottom: 8,
  },
  historyItem: {
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 12,
    backgroundColor: '#111111',
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  historyItemActive: {
    borderColor: '#5A5A5A',
    backgroundColor: '#181818',
  },
  historyItemTitle: {
    color: '#F5F5F5',
    fontSize: 15,
    fontWeight: '600',
  },
  historyItemMeta: {
    marginTop: 3,
    color: '#A8A8A8',
    fontSize: 13,
  },
  historyEmpty: {
    color: '#A8A8A8',
    textAlign: 'center',
    paddingVertical: 12,
  },
  threadMenuBtn: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  threadMenuDropdown: {
    position: 'absolute',
    top: 0,
    right: 28,
    backgroundColor: '#1A1A1A',
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 8,
    paddingVertical: 4,
    minWidth: 110,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
    elevation: 8,
    zIndex: 1000,
  },
  threadMenuOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
  },
  threadMenuOptionText: {
    color: '#F87171',
    fontSize: 14,
    fontWeight: '500',
  },
  fileMenuBtn: {
    padding: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  fileMenuDropdown: {
    position: 'absolute',
    top: 40,
    right: 0,
    backgroundColor: '#1A1A1A',
    borderWidth: 1,
    borderColor: '#2E2E2E',
    borderRadius: 8,
    paddingVertical: 8,
    minWidth: 140,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
    elevation: 8,
    zIndex: 1000,
  },
  fileMenuOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 10,
  },
  fileMenuOptionDanger: {
    borderTopWidth: 1,
    borderTopColor: '#2E2E2E',
  },
  fileMenuOptionText: {
    color: '#F5F5F5',
    fontSize: 14,
    fontWeight: '500',
  },
  contextSheet: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    right: 0,
    width: '85%',
    borderLeftWidth: 1,
    borderColor: '#2E2E2E',
    backgroundColor: '#0C0C0C80',
    paddingHorizontal: 12,
    shadowColor: '#000',
    shadowOffset: { width: -4, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 10,
  },
  addDocBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#F5F5F5',
    borderRadius: 16,
    paddingVertical: 12,
    marginBottom: 16,
    gap: 8,
  },
  addDocText: { color: '#000', fontSize: 16, fontWeight: '600' },
  createGroupForm: { backgroundColor: '#1A1A1A', borderRadius: 12, padding: 12, marginBottom: 16, borderWidth: 1, borderColor: '#333' },
  createGroupInput: { color: '#FFF', fontSize: 16, backgroundColor: '#000', borderRadius: 8, padding: 10, marginBottom: 12, borderWidth: 1, borderColor: '#444' },
  groupSaveBtn: { flex: 1, backgroundColor: '#F5F5F5', paddingVertical: 10, borderRadius: 8, alignItems: 'center' },
  groupCancelBtn: { flex: 1, backgroundColor: '#333', paddingVertical: 10, borderRadius: 8, alignItems: 'center' },
  backToRootBtn: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 12, paddingVertical: 8 },
  processingOverlay: { backgroundColor: '#1A1A1A', borderRadius: 12, padding: 16, alignItems: 'center', justifyContent: 'center', marginBottom: 16, borderWidth: 1, borderColor: '#333', flexDirection: 'row', gap: 12 },
  processingText: { color: '#FFF', fontSize: 14, fontWeight: '500' },
});
