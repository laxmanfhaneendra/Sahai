import React, { useEffect } from 'react';
import {
  View, Text, StyleSheet, ScrollView, Pressable, Dimensions,
} from 'react-native';
import Animated, {
  cancelAnimation,
  useSharedValue,
  useAnimatedStyle,
  withDelay,
  withRepeat,
  withTiming,
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { Colors } from '@/constants/Colors';
import { GroqVisionResult } from '@/services/groqVision';

const { height } = Dimensions.get('window');
const SHEET_MAX_HEIGHT = height * 0.72;

type Props = {
  visible: boolean;
  imageUri: string | null;
  loading: boolean;
  error: string | null;
  result: GroqVisionResult | null;
  onClose: () => void;
  onRetry: () => void;
};

// ─── Pulsing thinking dots ────────────────────────────────────────────────────
function ThinkingDots() {
  const dot1 = useSharedValue(0.3);
  const dot2 = useSharedValue(0.3);
  const dot3 = useSharedValue(0.3);

  useEffect(() => {
    const pulse = (v: typeof dot1, delay: number) => {
      v.value = withDelay(
        delay,
        withRepeat(withTiming(1, { duration: 420 }), -1, true),
      );
    };

    pulse(dot1, 0);
    pulse(dot2, 180);
    pulse(dot3, 360);

    return () => {
      cancelAnimation(dot1);
      cancelAnimation(dot2);
      cancelAnimation(dot3);
    };
  }, [dot1, dot2, dot3]);

  const d1 = useAnimatedStyle(() => ({ opacity: dot1.value }));
  const d2 = useAnimatedStyle(() => ({ opacity: dot2.value }));
  const d3 = useAnimatedStyle(() => ({ opacity: dot3.value }));

  return (
    <View style={dots.row}>
      <Animated.View style={[dots.dot, d1]} />
      <Animated.View style={[dots.dot, d2]} />
      <Animated.View style={[dots.dot, d3]} />
    </View>
  );
}

const dots = StyleSheet.create({
  row: { flexDirection: 'row', gap: 7, alignItems: 'center' },
  dot: { width: 9, height: 9, borderRadius: 5, backgroundColor: Colors.accent },
});

// ─── Token badge ──────────────────────────────────────────────────────────────
function TokenBadge({ label, value }: { label: string; value: number }) {
  return (
    <View style={badge.wrap}>
      <Text style={badge.label}>{label}</Text>
      <Text style={badge.value}>{value}</Text>
    </View>
  );
}
const badge = StyleSheet.create({
  wrap: {
    flex: 1, alignItems: 'center',
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 12, borderWidth: 1, borderColor: Colors.border,
    paddingVertical: 10, paddingHorizontal: 10,
  },
  label: { color: Colors.textMuted, fontSize: 10, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 0.8 },
  value: { color: Colors.textPrimary, fontSize: 18, fontWeight: '700', marginTop: 2 },
});

// ─── AI Result Bottom Sheet ───────────────────────────────────────────────────
export function AIResultSheet({ visible, loading, error, result, onClose, onRetry }: Props) {
  if (!visible) return null;

  return (
    // Outer wrapper — plain View so Reanimated can manage children independently
    <View style={StyleSheet.absoluteFill} pointerEvents="box-none">

      {/* ── Backdrop ── */}
      <Animated.View
        entering={FadeIn.duration(200)}
        exiting={FadeOut.duration(200)}
        style={styles.backdrop}
      >
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
      </Animated.View>

      {/* ── Sheet — wrapped in Animated.View for slide animation ── */}
      <Animated.View
        entering={SlideInDown.springify().damping(20).stiffness(130)}
        exiting={SlideOutDown.duration(260)}
        style={styles.sheet}
      >
        {/* Shimmer top line */}
        <LinearGradient
          colors={['#38BDF800', '#38BDF855', '#38BDF800']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.shimmerLine}
        />

        {/* Drag handle */}
        <View style={styles.handle} />

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <LinearGradient colors={['#87CEEB', '#38BDF8']} style={styles.headerIcon}>
              <Ionicons name="sparkles" size={14} color="#000" />
            </LinearGradient>
            <View>
              <Text style={styles.headerTitle}>AI Analysis</Text>
              <Text style={styles.headerSubtitle}>
                {loading ? 'Processing...' : result ? 'Llama 4 Scout · Groq' : 'Sahai Vision'}
              </Text>
            </View>
          </View>
          <Pressable onPress={onClose} style={styles.closeBtn}>
            <Ionicons name="close" size={18} color={Colors.textMuted} />
          </Pressable>
        </View>

        {/* Body */}
        <ScrollView
          style={styles.body}
          contentContainerStyle={styles.bodyContent}
          showsVerticalScrollIndicator={false}
        >
          {/* ── Loading ── */}
          {loading && (
            <View style={styles.loadingSection}>
              <ThinkingDots />
              <Text style={styles.loadingTitle}>Analyzing with Llama Scout</Text>
              <Text style={styles.loadingSubtitle}>Reading objects, text, context...</Text>
            </View>
          )}

          {/* ── Error ── */}
          {!loading && !!error && (
            <View style={styles.errorSection}>
              <View style={styles.errorIconWrap}>
                <Ionicons name="warning-outline" size={28} color="#F87171" />
              </View>
              <Text style={styles.errorTitle}>Analysis Failed</Text>
              <Text style={styles.errorMsg}>{error}</Text>
              <Pressable onPress={onRetry}>
                <LinearGradient
                  colors={['#87CEEB', '#38BDF8']}
                  start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }}
                  style={styles.retryBtn}
                >
                  <Ionicons name="refresh-outline" size={15} color="#000" style={{ marginRight: 6 }} />
                  <Text style={styles.retryText}>Try Again</Text>
                </LinearGradient>
              </Pressable>
            </View>
          )}

          {/* ── Result ── */}
          {!loading && !error && !!result && (
            <View style={{ gap: 16 }}>
              {/* Response text card */}
              <View style={styles.responseCard}>
                {/* card shimmer top */}
                <LinearGradient
                  colors={['#38BDF820', '#38BDF800']}
                  style={styles.responseCardShimmer}
                />
                <Text style={styles.responseText}>{result.description}</Text>
              </View>

              {/* Token usage */}
              <Text style={styles.sectionLabel}>Token Usage</Text>
              <View style={styles.tokenRow}>
                <TokenBadge label="Prompt" value={result.tokens.prompt} />
                <TokenBadge label="Output" value={result.tokens.completion} />
                <TokenBadge label="Total" value={result.tokens.total} />
              </View>

              {/* Model info */}
              <View style={styles.modelRow}>
                <Ionicons name="hardware-chip-outline" size={12} color={Colors.textMuted} />
                <Text style={styles.modelText}>{result.model}</Text>
                <View style={styles.modelDot} />
                <Text style={styles.modelText}>Groq Cloud</Text>
              </View>
            </View>
          )}
        </ScrollView>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#000000A0',
    zIndex: 50,
  },
  sheet: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    maxHeight: SHEET_MAX_HEIGHT,
    backgroundColor: Colors.surface,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    borderWidth: 1,
    borderColor: Colors.border,
    zIndex: 60,
    overflow: 'hidden',
  },
  shimmerLine: { height: 1 },
  handle: {
    width: 40, height: 4, borderRadius: 2,
    backgroundColor: Colors.border,
    alignSelf: 'center',
    marginTop: 12, marginBottom: 2,
  },

  // Header
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: Colors.borderSubtle,
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  headerIcon: { width: 30, height: 30, borderRadius: 9, alignItems: 'center', justifyContent: 'center' },
  headerTitle: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700' },
  headerSubtitle: { color: Colors.textMuted, fontSize: 11, marginTop: 1 },
  closeBtn: {
    width: 32, height: 32, borderRadius: 10,
    backgroundColor: Colors.surfaceElevated,
    alignItems: 'center', justifyContent: 'center',
  },

  // Body
  body: { flex: 1 },
  bodyContent: { padding: 20, paddingBottom: 36 },

  // Loading
  loadingSection: { alignItems: 'center', gap: 12, paddingVertical: 36 },
  loadingTitle: { color: Colors.textPrimary, fontSize: 16, fontWeight: '600' },
  loadingSubtitle: { color: Colors.textMuted, fontSize: 13 },

  // Error
  errorSection: { alignItems: 'center', gap: 12, paddingVertical: 28 },
  errorIconWrap: {
    width: 60, height: 60, borderRadius: 20,
    backgroundColor: '#F8717115', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: '#F8717130',
  },
  errorTitle: { color: '#F87171', fontSize: 17, fontWeight: '700' },
  errorMsg: { color: Colors.textSecondary, fontSize: 13, textAlign: 'center', lineHeight: 20 },
  retryBtn: {
    flexDirection: 'row', alignItems: 'center',
    borderRadius: 12, paddingVertical: 11, paddingHorizontal: 24, marginTop: 4,
  },
  retryText: { color: '#000', fontSize: 14, fontWeight: '700' },

  // Result
  responseCard: {
    backgroundColor: Colors.surfaceCard,
    borderRadius: 18, borderWidth: 1, borderColor: Colors.border,
    padding: 18, overflow: 'hidden',
  },
  responseCardShimmer: { position: 'absolute', top: 0, left: 0, right: 0, height: 40 },
  responseText: { color: Colors.textPrimary, fontSize: 15, lineHeight: 25 },

  sectionLabel: {
    color: Colors.textMuted, fontSize: 11, fontWeight: '700',
    letterSpacing: 1.1, textTransform: 'uppercase',
  },
  tokenRow: { flexDirection: 'row', gap: 10 },

  modelRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6,
    paddingVertical: 4,
  },
  modelText: { color: Colors.textMuted, fontSize: 11 },
  modelDot: { width: 3, height: 3, borderRadius: 2, backgroundColor: Colors.textMuted },
});
