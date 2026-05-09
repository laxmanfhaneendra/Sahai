import * as FileSystem from 'expo-file-system/legacy';
import { RagChunk, chunkDocument } from './ragSearch';

export type RagGroup = {
  id: string;
  name: string;
  createdAt: number;
};

export type RagDocument = {
  id: string;
  name: string;
  content: string;
  createdAt: number;
  groupId?: string;
};

type RagDocumentMetadata = {
  id: string;
  name: string;
  createdAt: number;
  groupId?: string;
  fileUri?: string;
};

const RAG_STORE_PATH = `${FileSystem.documentDirectory}rag_documents.json`;
const RAG_GROUPS_PATH = `${FileSystem.documentDirectory}rag_groups.json`;
const RAG_FILES_DIR = `${FileSystem.documentDirectory}rag_files/`;
const RAG_CHUNKS_PATH = `${FileSystem.documentDirectory}rag_chunks.json`;

// ─── In-memory caches ────────────────────────────────────────────────────────
let metadataCache: RagDocumentMetadata[] | null = null;
let contentCache = new Map<string, string>();
let cacheTimestamp = 0;
let chunkCache: RagChunk[] | null = null;
const CACHE_VALIDITY_MS = 60000;

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function ensureRagFilesDir(): Promise<void> {
  const dirInfo = await FileSystem.getInfoAsync(RAG_FILES_DIR);
  if (!dirInfo.exists) {
    await FileSystem.makeDirectoryAsync(RAG_FILES_DIR, { intermediates: true });
  }
}

function getDocumentFilePath(docId: string): string {
  return `${RAG_FILES_DIR}${docId}.txt`;
}

function metaToDoc(meta: RagDocumentMetadata): RagDocument {
  return {
    id: meta.id,
    name: meta.name,
    content: '',
    createdAt: meta.createdAt,
    ...(meta.groupId && { groupId: meta.groupId }),
  };
}

// ─── Chunk Index ─────────────────────────────────────────────────────────────

export async function loadChunkIndex(): Promise<RagChunk[]> {
  if (chunkCache !== null) return chunkCache;
  try {
    const info = await FileSystem.getInfoAsync(RAG_CHUNKS_PATH);
    if (!info.exists) return [];
    const raw = await FileSystem.readAsStringAsync(RAG_CHUNKS_PATH);
    const parsed = JSON.parse(raw);
    chunkCache = Array.isArray(parsed) ? parsed : [];
    return chunkCache;
  } catch {
    return [];
  }
}

async function saveChunkIndex(chunks: RagChunk[]): Promise<void> {
  await FileSystem.writeAsStringAsync(RAG_CHUNKS_PATH, JSON.stringify(chunks));
  chunkCache = chunks;
}

// ─── Metadata ────────────────────────────────────────────────────────────────

async function loadMetadata(): Promise<RagDocumentMetadata[]> {
  const now = Date.now();
  if (metadataCache && (now - cacheTimestamp) < CACHE_VALIDITY_MS) {
    return metadataCache;
  }
  try {
    const info = await FileSystem.getInfoAsync(RAG_STORE_PATH);
    if (!info.exists) return [];
    const raw = await FileSystem.readAsStringAsync(RAG_STORE_PATH);
    const parsed = JSON.parse(raw);
    metadataCache = Array.isArray(parsed) ? parsed : [];
    cacheTimestamp = now;
    return metadataCache!;
  } catch {
    return [];
  }
}

async function persistMetadata(metadata: RagDocumentMetadata[]): Promise<void> {
  await FileSystem.writeAsStringAsync(RAG_STORE_PATH, JSON.stringify(metadata));
  metadataCache = metadata;
  cacheTimestamp = Date.now();
}

// ─── Documents (public API) ───────────────────────────────────────────────────

export async function loadRagDocuments(): Promise<RagDocument[]> {
  const metadata = await loadMetadata();
  return metadata.map(metaToDoc);
}

/**
 * Get document content — lazy-loaded and cached.
 */
export async function getDocumentContent(docId: string): Promise<string> {
  if (contentCache.has(docId)) return contentCache.get(docId)!;

  const metadata = await loadMetadata();
  const meta = metadata.find(m => m.id === docId);
  if (!meta?.fileUri) return '';

  try {
    const text = await FileSystem.readAsStringAsync(meta.fileUri);
    contentCache.set(docId, text);
    return text;
  } catch {
    return '';
  }
}

/**
 * Add a new document.
 * Writes content to its own file, updates metadata, and rebuilds chunk index.
 */
export async function addRagDocument(
  name: string,
  content: string,
  groupId?: string,
): Promise<RagDocument[]> {
  await ensureRagFilesDir();

  const id = `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const fileUri = getDocumentFilePath(id);

  // Write content to its own file
  await FileSystem.writeAsStringAsync(fileUri, content);
  contentCache.set(id, content);

  // Update metadata
  const metadata = await loadMetadata();
  const newMeta: RagDocumentMetadata = {
    id,
    name,
    createdAt: Date.now(),
    fileUri,
    ...(groupId && { groupId }),
  };
  const updatedMeta = [newMeta, ...metadata];
  await persistMetadata(updatedMeta);

  // Append new chunks to index
  const existingChunks = await loadChunkIndex();
  const newChunks = chunkDocument(content, id, name);
  await saveChunkIndex([...existingChunks, ...newChunks]);

  return updatedMeta.map(metaToDoc);
}

/**
 * Delete a document.
 * Removes file, targeted cache entries, and its chunks from the index.
 */
export async function deleteRagDocument(id: string): Promise<RagDocument[]> {
  // Remove content file
  const fileUri = getDocumentFilePath(id);
  await FileSystem.deleteAsync(fileUri, { idempotent: true }).catch(() => {});

  // Remove from content cache (targeted — don't clear everyone else)
  contentCache.delete(id);

  // Update metadata
  const metadata = await loadMetadata();
  const updatedMeta = metadata.filter(m => m.id !== id);
  await persistMetadata(updatedMeta);

  // Remove this doc's chunks from the index
  const existingChunks = await loadChunkIndex();
  const updatedChunks = existingChunks.filter(c => c.docId !== id);
  await saveChunkIndex(updatedChunks);

  return updatedMeta.map(metaToDoc);
}

/**
 * Replace a document's content in-place (keeps same id, same group).
 */
export async function replaceRagDocument(
  id: string,
  newName: string,
  newContent: string,
): Promise<RagDocument[]> {
  await ensureRagFilesDir();

  const fileUri = getDocumentFilePath(id);
  await FileSystem.writeAsStringAsync(fileUri, newContent);
  contentCache.set(id, newContent);

  // Update name in metadata
  const metadata = await loadMetadata();
  const updatedMeta = metadata.map(m =>
    m.id === id ? { ...m, name: newName, fileUri } : m
  );
  await persistMetadata(updatedMeta);

  // Rebuild chunks for this doc
  const existingChunks = await loadChunkIndex();
  const otherChunks = existingChunks.filter(c => c.docId !== id);
  const newChunks = chunkDocument(newContent, id, newName);
  await saveChunkIndex([...otherChunks, ...newChunks]);

  return updatedMeta.map(metaToDoc);
}

/**
 * @deprecated Use addRagDocument / deleteRagDocument directly.
 * Kept for migration compatibility. Does NOT overwrite existing content files.
 */
export async function saveRagDocuments(documents: RagDocument[]): Promise<void> {
  try {
    await ensureRagFilesDir();
    const metadata: RagDocumentMetadata[] = [];

    for (const doc of documents) {
      const fileUri = getDocumentFilePath(doc.id);
      // CRITICAL FIX: Only write content if non-empty.
      // Empty content = lazy-loaded placeholder; existing file should be preserved.
      if (doc.content) {
        await FileSystem.writeAsStringAsync(fileUri, doc.content);
        contentCache.set(doc.id, doc.content);
      }
      metadata.push({
        id: doc.id,
        name: doc.name,
        createdAt: doc.createdAt,
        ...(doc.groupId && { groupId: doc.groupId }),
        fileUri,
      });
    }

    await persistMetadata(metadata);
  } catch (error) {
    console.warn('Failed to save RAG documents:', error);
  }
}

// ─── Cleanup ──────────────────────────────────────────────────────────────────

export async function cleanupOrphanedFiles(): Promise<number> {
  try {
    await ensureRagFilesDir();
    const metadata = await loadMetadata();
    const validUris = new Set(metadata.map(m => getDocumentFilePath(m.id)));

    const files = await FileSystem.readDirectoryAsync(RAG_FILES_DIR);
    let deletedCount = 0;

    for (const file of files) {
      const uri = `${RAG_FILES_DIR}${file}`;
      if (!validUris.has(uri)) {
        await FileSystem.deleteAsync(uri, { idempotent: true }).catch(() => {});
        deletedCount++;
      }
    }

    // Sync chunk index — remove chunks whose doc no longer exists
    const validIds = new Set(metadata.map(m => m.id));
    const chunks = await loadChunkIndex();
    const syncedChunks = chunks.filter(c => validIds.has(c.docId));
    if (syncedChunks.length !== chunks.length) {
      await saveChunkIndex(syncedChunks);
    }

    return deletedCount;
  } catch (error) {
    console.warn('Failed to cleanup orphaned files:', error);
    return 0;
  }
}

// ─── Groups ───────────────────────────────────────────────────────────────────

export async function loadRagGroups(): Promise<RagGroup[]> {
  try {
    const info = await FileSystem.getInfoAsync(RAG_GROUPS_PATH);
    if (!info.exists) return [];
    const raw = await FileSystem.readAsStringAsync(RAG_GROUPS_PATH);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export async function saveRagGroups(groups: RagGroup[]): Promise<void> {
  await FileSystem.writeAsStringAsync(RAG_GROUPS_PATH, JSON.stringify(groups));
}

export async function addRagGroup(name: string): Promise<RagGroup[]> {
  const existing = await loadRagGroups();
  const newGroup: RagGroup = {
    id: `grp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    name,
    createdAt: Date.now(),
  };
  const updated = [newGroup, ...existing];
  await saveRagGroups(updated);
  return updated;
}

export async function deleteRagGroup(
  id: string,
): Promise<{ updatedGroups: RagGroup[]; updatedDocs: RagDocument[] }> {
  // Delete group
  const groups = await loadRagGroups();
  const updatedGroups = groups.filter(g => g.id !== id);
  await saveRagGroups(updatedGroups);

  // Delete all docs in this group
  const metadata = await loadMetadata();
  const toDelete = metadata.filter(m => m.groupId === id);
  const updatedMeta = metadata.filter(m => m.groupId !== id);

  for (const m of toDelete) {
    const fileUri = getDocumentFilePath(m.id);
    await FileSystem.deleteAsync(fileUri, { idempotent: true }).catch(() => {});
    contentCache.delete(m.id);
  }

  await persistMetadata(updatedMeta);

  // Remove chunks for all deleted docs
  const deletedIds = new Set(toDelete.map(m => m.id));
  const chunks = await loadChunkIndex();
  await saveChunkIndex(chunks.filter(c => !deletedIds.has(c.docId)));

  return { updatedGroups, updatedDocs: updatedMeta.map(metaToDoc) };
}

// ─── Migration ────────────────────────────────────────────────────────────────

/**
 * One-time migration: convert old base64-embedded docs to file storage.
 */
export async function migrateToFileStorage(): Promise<{ migratedCount: number; error?: string }> {
  try {
    const info = await FileSystem.getInfoAsync(RAG_STORE_PATH);
    if (!info.exists) return { migratedCount: 0 };

    const raw = await FileSystem.readAsStringAsync(RAG_STORE_PATH);
    const parsed = JSON.parse(raw);

    // Already migrated
    if (Array.isArray(parsed) && parsed[0]?.fileUri) return { migratedCount: 0 };

    // Old format: RagDocument[] with inline content
    if (Array.isArray(parsed) && parsed[0]?.content) {
      const oldDocs = parsed as RagDocument[];
      await ensureRagFilesDir();

      const metadata: RagDocumentMetadata[] = [];
      const allChunks: RagChunk[] = [];

      for (const doc of oldDocs) {
        const fileUri = getDocumentFilePath(doc.id);
        await FileSystem.writeAsStringAsync(fileUri, doc.content);
        contentCache.set(doc.id, doc.content);
        metadata.push({ id: doc.id, name: doc.name, createdAt: doc.createdAt, fileUri, ...(doc.groupId && { groupId: doc.groupId }) });
        allChunks.push(...chunkDocument(doc.content, doc.id, doc.name));
      }

      await persistMetadata(metadata);
      await saveChunkIndex(allChunks);
      return { migratedCount: oldDocs.length };
    }

    return { migratedCount: 0 };
  } catch (error) {
    return { migratedCount: 0, error: String(error) };
  }
}
