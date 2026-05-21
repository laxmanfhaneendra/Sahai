import * as FileSystem from 'expo-file-system/legacy';
import { Buffer } from 'buffer';

export type WavFormat = {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
};

function createWavHeader(dataLength: number, format: WavFormat): Buffer {
  const blockAlign = (format.channels * format.bitsPerSample) / 8;
  const byteRate = format.sampleRate * blockAlign;
  const buffer = Buffer.alloc(44);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(format.channels, 22);
  buffer.writeUInt32LE(format.sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(format.bitsPerSample, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataLength, 40);

  return buffer;
}

export async function writeWavFileFromChunks(
  chunks: string[],
  format: WavFormat
): Promise<string | null> {
  if (!chunks.length) return null;

  const pcmBuffers = chunks.map((chunk) => Buffer.from(chunk, 'base64'));
  const pcmData = Buffer.concat(pcmBuffers);
  if (!pcmData.length) return null;

  const wavHeader = createWavHeader(pcmData.length, format);
  const wavBuffer = Buffer.concat([wavHeader, pcmData]);

  const dir = FileSystem.documentDirectory || '';
  if (!dir) return null;

  const outputPath = `${dir}local-stt-${Date.now()}.wav`;
  await FileSystem.writeAsStringAsync(outputPath, wavBuffer.toString('base64'), { encoding: 'base64' as any });
  return outputPath;
}
