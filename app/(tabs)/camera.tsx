import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Image,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system/legacy';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSequence,
  withSpring,
  withTiming,
  FadeIn,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Colors } from '@/constants/Colors';
import { analyzeImageWithGroq, GroqVisionResult } from '@/services/groqVision';
import { AIResultSheet } from '@/components/AIResultSheet';

const CAPTURE_SIZE = 78;

function ShutterButton({ onPress, disabled }: { onPress: () => void; disabled?: boolean }) {
  const scale = useSharedValue(1);
  const ringOpacity = useSharedValue(0);
  const ringScale = useSharedValue(1);

  const outerStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));
  const ringStyle = useAnimatedStyle(() => ({
    opacity: ringOpacity.value,
    transform: [{ scale: ringScale.value }],
  }));

  const handlePress = () => {
    if (disabled) return;
    scale.value = withSequence(withSpring(0.88), withSpring(1));
    ringOpacity.value = withSequence(withTiming(0.7, { duration: 80 }), withTiming(0, { duration: 350 }));
    ringScale.value = withSequence(withTiming(1), withTiming(1.5, { duration: 400 }));
    onPress();
  };

  return (
    <Pressable onPress={handlePress} style={styles.shutterHitArea}>
      <Animated.View style={[styles.shutterRipple, ringStyle]} />
      <Animated.View style={[styles.shutterOuter, outerStyle, disabled && { opacity: 0.5 }]}>
        <LinearGradient
          colors={disabled ? ['#444', '#333'] : ['#87CEEB', '#38BDF8', '#0EA5E9']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFill}
        />
        <View style={styles.shutterInner} />
      </Animated.View>
    </Pressable>
  );
}

export default function CameraScreen() {
  const [cameraPermission, requestCameraPermission] = ImagePicker.useCameraPermissions();

  const [analyzing, setAnalyzing] = useState(false);
  const [sheetVisible, setSheetVisible] = useState(false);
  const [lastPhotoUri, setLastPhotoUri] = useState<string | null>(null);
  const [lastBase64, setLastBase64] = useState<string | null>(null);
  const [aiResult, setAiResult] = useState<GroqVisionResult | null>(null);
  const [aiError, setAiError] = useState<string | null>(null);

  const insets = useSafeAreaInsets();

  const sendToGroq = useCallback(async (base64: string) => {
    setAiResult(null);
    setAiError(null);
    setAnalyzing(true);
    setSheetVisible(true);

    try {
      const result = await analyzeImageWithGroq(base64);
      setAiResult(result);
    } catch (err: any) {
      setAiError(err?.message ?? 'Unknown error from Groq API');
    } finally {
      setAnalyzing(false);
    }
  }, []);

  const handleCapture = useCallback(async () => {
    if (analyzing) return;

    const permission = cameraPermission?.granted
      ? cameraPermission
      : await requestCameraPermission();

    if (!permission?.granted) {
      Alert.alert('Camera permission needed', 'Please allow camera access to capture an image.');
      return;
    }

    try {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 0.4,
        base64: false,
        exif: false,
      });

      if (result.canceled || !result.assets?.length) return;

      const asset = result.assets[0];
      if (!asset.uri) {
        Alert.alert('Capture failed', 'Could not process the image. Please try again.');
        return;
      }

      setLastPhotoUri(asset.uri);

      const compressed = await ImageManipulator.manipulateAsync(
        asset.uri,
        [{ resize: { width: 640 } }],
        { compress: 0.45, format: ImageManipulator.SaveFormat.JPEG },
      );

      const base64 = await FileSystem.readAsStringAsync(compressed.uri, {
        encoding: 'base64' as any,
      });

      setLastBase64(base64);
      await sendToGroq(base64);
    } catch {
      setAnalyzing(false);
      setSheetVisible(false);
      Alert.alert('Capture failed', 'Could not open the camera. Please try again.');
    }
  }, [analyzing, cameraPermission, requestCameraPermission, sendToGroq]);

  return (
    <View style={styles.root}>
      <LinearGradient
        colors={['#38BDF812', '#38BDF806', '#000000']}
        style={StyleSheet.absoluteFill}
      />

      <View style={[styles.topBar, { paddingTop: insets.top + 12 }]}>
        <View style={styles.badge}>
          <Ionicons name={analyzing ? 'hourglass-outline' : 'sparkles'} size={12} color={analyzing ? '#F59E0B' : Colors.accent} />
          <Text style={[styles.badgeText, analyzing && { color: '#F59E0B' }]}>
            {analyzing ? 'Analyzing...' : 'Vision Ready'}
          </Text>
        </View>
      </View>

      <Animated.View entering={FadeIn.duration(350)} style={styles.centerCard}>
        <LinearGradient colors={['#87CEEB', '#38BDF8']} style={styles.centerIcon}>
          <Ionicons name="camera-outline" size={28} color="#000" />
        </LinearGradient>
        <Text style={styles.title}>System Camera Pipeline</Text>
        <Text style={styles.subtitle}>
          Capture any meeting whiteboard, slide, or document and get instant visual intelligence.
        </Text>
      </Animated.View>

      <View style={[styles.bottomBar, { paddingBottom: insets.bottom + 16 }]}>
        <Pressable
          style={styles.thumbnailBtn}
          onPress={() => {
            if (analyzing || aiResult || aiError) setSheetVisible(true);
          }}
        >
          {lastPhotoUri ? (
            <Image source={{ uri: lastPhotoUri }} style={styles.thumbnail} />
          ) : (
            <View style={styles.thumbnailPlaceholder}>
              <Ionicons name="images-outline" size={20} color={Colors.textMuted} />
            </View>
          )}
        </Pressable>

        <ShutterButton onPress={handleCapture} disabled={analyzing} />

        <View style={styles.rightPlaceholder} />
      </View>

      <AIResultSheet
        visible={sheetVisible}
        imageUri={lastPhotoUri}
        loading={analyzing}
        error={aiError}
        result={aiResult}
        onClose={() => setSheetVisible(false)}
        onRetry={() => {
          if (lastBase64) sendToGroq(lastBase64);
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 100,
    borderWidth: 1,
    borderColor: Colors.accentDim,
    backgroundColor: '#00000066',
  },
  badgeText: { color: Colors.accent, fontSize: 12, fontWeight: '600' },
  centerCard: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    gap: 10,
  },
  centerIcon: {
    width: 70,
    height: 70,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: { color: Colors.textPrimary, fontSize: 22, fontWeight: '800' },
  subtitle: {
    color: Colors.textSecondary,
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 21,
    maxWidth: 320,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingTop: 30,
    paddingHorizontal: 24,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#00000088',
  },
  thumbnailBtn: {
    width: 52,
    height: 52,
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: Colors.border,
  },
  thumbnail: { width: 52, height: 52 },
  thumbnailPlaceholder: {
    width: 52,
    height: 52,
    backgroundColor: Colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rightPlaceholder: { width: 52, height: 52 },
  shutterHitArea: {
    width: CAPTURE_SIZE + 20,
    height: CAPTURE_SIZE + 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  shutterRipple: {
    position: 'absolute',
    width: CAPTURE_SIZE + 20,
    height: CAPTURE_SIZE + 20,
    borderRadius: (CAPTURE_SIZE + 20) / 2,
    borderWidth: 2,
    borderColor: Colors.accent,
  },
  shutterOuter: {
    width: CAPTURE_SIZE,
    height: CAPTURE_SIZE,
    borderRadius: CAPTURE_SIZE / 2,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 3,
    borderColor: '#FFFFFF',
    overflow: 'hidden',
  },
  shutterInner: {
    width: CAPTURE_SIZE - 16,
    height: CAPTURE_SIZE - 16,
    borderRadius: (CAPTURE_SIZE - 16) / 2,
  },
});
