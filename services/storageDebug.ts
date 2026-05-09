/**
 * Debug utility to check storage usage and clean up
 */

import * as FileSystem from 'expo-file-system/legacy';
import { loadRagDocuments } from './ragContextStore';

const RAG_STORE_PATH = `${FileSystem.documentDirectory}rag_documents.json`;

/**
 * Get total size of stored documents
 */
export async function getStorageUsage(): Promise<{
  fileSize: number;
  documentCount: number;
  documentsDetail: Array<{ name: string; sizeKB: number }>;
  totalSizeMB: number;
}> {
  try {
    const docs = await loadRagDocuments();

    // Calculate individual document sizes
    const documentsDetail = docs.map(doc => ({
      name: doc.name,
      sizeKB: Math.round(Buffer.byteLength(JSON.stringify(doc), 'utf8') / 1024),
    }));

    const totalKB = documentsDetail.reduce((sum, doc) => sum + doc.sizeKB, 0);
    const totalSizeMB = Math.round(totalKB / 1024 * 100) / 100;

    return {
      fileSize: totalKB * 1024,
      documentCount: docs.length,
      documentsDetail,
      totalSizeMB,
    };
  } catch (err) {
    console.error('Failed to get storage usage:', err);
    return {
      fileSize: 0,
      documentCount: 0,
      documentsDetail: [],
      totalSizeMB: 0,
    };
  }
}

/**
 * Force cleanup: remove orphaned/temp files
 */
export async function cleanupStorage(): Promise<string> {
  try {
    // List all files in document directory
    const docDir = FileSystem.documentDirectory;
    if (!docDir) return 'Document directory not available';

    const files = await FileSystem.readDirectoryAsync(docDir);

    // Keep only essential files
    const keepFiles = ['rag_documents.json', 'rag_groups.json', 'chat_sessions.json'];
    let deletedCount = 0;

    for (const file of files) {
      if (!keepFiles.includes(file) && (file.endsWith('.tmp') || file.endsWith('.bak'))) {
        await FileSystem.deleteAsync(`${docDir}${file}`).catch(() => {});
        deletedCount++;
      }
    }

    return `Cleaned up ${deletedCount} temporary files`;
  } catch (err) {
    console.error('Cleanup failed:', err);
    return 'Cleanup failed';
  }
}

/**
 * Log storage details to console
 */
export async function logStorageDetails(): Promise<void> {
  const usage = await getStorageUsage();
  console.log('=== Storage Usage ===');
  console.log(`Total: ${usage.totalSizeMB}MB`);
  console.log(`Documents: ${usage.documentCount}`);
  console.log('Details:');
  usage.documentsDetail.forEach(doc => {
    console.log(`  • ${doc.name}: ${doc.sizeKB}KB`);
  });
}
