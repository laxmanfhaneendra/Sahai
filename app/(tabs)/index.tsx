import React from 'react';
import { View, Text, StyleSheet, Pressable, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { Colors } from '@/constants/Colors';

function QuickAction({
  icon,
  title,
  subtitle,
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  subtitle: string;
  onPress: () => void;
}) {
  return (
    <Pressable onPress={onPress} style={styles.cardWrap}>
      <View style={styles.card}>
        <LinearGradient colors={['#87CEEB', '#38BDF8']} style={styles.cardIcon}>
          <Ionicons name={icon} size={18} color="#000" />
        </LinearGradient>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardTitle}>{title}</Text>
          <Text style={styles.cardSubtitle}>{subtitle}</Text>
        </View>
        <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
      </View>
    </Pressable>
  );
}

export default function HomeScreen() {
  const insets = useSafeAreaInsets();

  return (
    <View style={styles.root}>
      <LinearGradient
        colors={['#38BDF81A', '#0EA5E911', '#000000']}
        style={StyleSheet.absoluteFill}
      />

      <ScrollView
        style={{ flex: 1 }}
        contentContainerStyle={[styles.content, { paddingTop: insets.top + 10, paddingBottom: 120 }]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.hero}>
          <Text style={styles.heroKicker}>SAHAI ASSISTANT</Text>
          <Text style={styles.heroTitle}>Premium meeting{'\n'}copilot</Text>
          <Text style={styles.heroSubtitle}>
            Capture questions instantly, listen live, and get concise answers with image intelligence.
          </Text>
        </View>

        <View style={{ gap: 12 }}>
          <QuickAction
            icon="chatbubbles-outline"
            title="Open Chat Copilot"
            subtitle="Live listen + hold-to-ask + camera quick actions"
            onPress={() => router.push('/(tabs)/search')}
          />
          <QuickAction
            icon="camera-outline"
            title="Instant Visual Analysis"
            subtitle="Capture images and get context-aware answers"
            onPress={() => router.push('/(tabs)/camera')}
          />
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>Realtime</Text>
            <Text style={styles.statLabel}>Question support</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>Session</Text>
            <Text style={styles.statLabel}>History memory</Text>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.bg },
  content: { paddingHorizontal: 16, gap: 16 },
  hero: {
    borderRadius: 24,
    borderWidth: 1,
    borderColor: '#38BDF83A',
    backgroundColor: '#0A0F15C0',
    padding: 18,
  },
  heroKicker: {
    color: Colors.accentBright,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.1,
    marginBottom: 8,
  },
  heroTitle: {
    color: Colors.textPrimary,
    fontSize: 30,
    fontWeight: '800',
    lineHeight: 36,
    letterSpacing: -0.8,
  },
  heroSubtitle: {
    color: Colors.textSecondary,
    fontSize: 14,
    lineHeight: 20,
    marginTop: 10,
  },
  cardWrap: { borderRadius: 16, overflow: 'hidden' },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderWidth: 1,
    borderColor: '#1a2535',
    padding: 14,
    backgroundColor: '#0F141A80',
  },
  cardIcon: {
    width: 36,
    height: 36,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTitle: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700' },
  cardSubtitle: { color: Colors.textSecondary, fontSize: 12, marginTop: 2 },
  statsRow: { flexDirection: 'row', gap: 10, marginTop: 6 },
  statCard: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#1a2535',
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 12,
    backgroundColor: '#0F141A80',
  },
  statValue: { color: Colors.textPrimary, fontSize: 15, fontWeight: '700' },
  statLabel: { color: Colors.textMuted, fontSize: 12, marginTop: 4 },
});
