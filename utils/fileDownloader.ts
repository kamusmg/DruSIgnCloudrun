/**
 * Triggers a browser download for a file from a base64 data URL.
 * @param dataUrl The base64 data URL (e.g., "data:image/png;base64,...").
 * @param filename The desired filename for the download.
 */
export const downloadFromDataUrl = (dataUrl: string, filename: string): void => {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

/**
 * Triggers a browser download for file content.
 * @param content The file content as a string.
 * @param filename The desired filename for the download.
 * @param mimeType The MIME type of the file.
 */
export const downloadTextFile = (content: string, filename: string, mimeType: string = 'text/plain'): void => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};


/**
 * Triggers a browser download for a Blob object.
 * @param blob The Blob object to download.
 * @param filename The desired filename for the download.
 */
export const downloadBlob = (blob: Blob, filename: string): void => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

/**
 * Converts a base64 data URL into a File object.
 * @param dataUrl The base64 data URL (e.g., "data:image/png;base64,...").
 * @param filename The desired filename for the new File object.
 * @returns A promise that resolves with the created File object.
 */
export async function dataUrlToFile(dataUrl: string, filename: string): Promise<File> {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type });
}