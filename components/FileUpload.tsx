import React, { useState, useCallback, useRef, useEffect } from 'react';
// Fix: Correct the import path for Icons.
import { UploadCloudIcon, XIcon } from './Icons.tsx';

interface FileUploadProps {
  onFileUpload: (file: File | null, base64: string | null) => void;
  accept?: string;
  capture?: boolean;
}

const MAX_FILE_SIZE_MB = 15;
const MAX_IMAGE_DIMENSION = 2000;

export const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, accept = "image/jpeg, image/png, image/webp", capture = false }) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const processFile = useCallback(async (file: File | null) => {
    setError(null);
    if (!file) return;

    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`Arquivo muito grande. O limite é de ${MAX_FILE_SIZE_MB} MB.`);
      onFileUpload(null, null);
      setPreview(null);
      setFileName(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    const supportedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!supportedTypes.includes(file.type)) {
      setError(`Tipo de arquivo não suportado. Por favor, use JPEG, PNG ou WEBP.`);
      onFileUpload(null, null);
      setPreview(null);
      setFileName(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      return;
    }

    try {
      // Step 1: Use createImageBitmap to handle EXIF orientation automatically.
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });

      // Step 2: Calculate new dimensions to fit within MAX_IMAGE_DIMENSION.
      let { width, height } = bitmap;
      if (width > height) {
        if (width > MAX_IMAGE_DIMENSION) {
          height = Math.round((height * MAX_IMAGE_DIMENSION) / width);
          width = MAX_IMAGE_DIMENSION;
        }
      } else {
        if (height > MAX_IMAGE_DIMENSION) {
          width = Math.round((width * MAX_IMAGE_DIMENSION) / height);
          height = MAX_IMAGE_DIMENSION;
        }
      }
      
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        throw new Error('Não foi possível obter o contexto do canvas para otimizar a imagem.');
      }
      
      // Step 3: Apply light brightness/contrast adjustment.
      ctx.filter = 'brightness(1.05) contrast(1.05)';

      // Step 4: Draw the correctly oriented and resized image onto the canvas.
      ctx.drawImage(bitmap, 0, 0, width, height);
      
      // Step 5: Generate a compressed JPEG image.
      // The quality parameter (0.0 to 1.0) achieves the compression. 0.9 provides a good balance.
      const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.9);

      // Clean up the bitmap to free memory
      bitmap.close();
      
      setPreview(compressedDataUrl);
      setFileName(file.name);
      onFileUpload(file, compressedDataUrl);

    } catch (err) {
      const errorMessage = "Ocorreu um erro ao otimizar a imagem. Tente uma imagem diferente.";
      console.error("Error processing file", err);
      setError(errorMessage);
      onFileUpload(null, null);
    }
  }, [onFileUpload]);

  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    processFile(file || null);
  }, [processFile]);

  const handlePaste = useCallback((event: ClipboardEvent) => {
    const items = event.clipboardData?.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          event.preventDefault(); // Prevent pasting into other inputs
          const file = items[i].getAsFile();
          if (file) {
            processFile(file);
          }
          break;
        }
      }
    }
  }, [processFile]);
  
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
        container.addEventListener('paste', handlePaste);
    }
    return () => {
      if (container) {
          container.removeEventListener('paste', handlePaste);
      }
    };
  }, [handlePaste]);

  const handleClear = () => {
    setPreview(null);
    setFileName(null);
    setError(null);
    onFileUpload(null, null);
    if(fileInputRef.current) {
        fileInputRef.current.value = "";
    }
    containerRef.current?.focus();
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };
  
  const handleFocus = () => {
      containerRef.current?.focus();
  }

  return (
    <div 
      ref={containerRef}
      onClick={handleFocus} // Focus the container on click to enable paste
      tabIndex={0} 
      className="focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 dark:focus:ring-offset-black rounded-lg"
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept={accept}
        capture={capture ? "environment" : undefined}
      />
      {!preview ? (
        <div 
          onClick={handleClick}
          className="flex flex-col items-center justify-center w-full p-8 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition"
        >
          <UploadCloudIcon />
          <p className="mt-2 text-sm font-semibold text-gray-700 dark:text-gray-300">Clique, arraste e solte ou cole (Ctrl+V)</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">Use a câmera do celular ou envie um arquivo</p>
          {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
        </div>
      ) : (
        <div className="relative p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <div className="flex items-center space-x-4">
            <img src={preview} alt="Preview" className="w-20 h-20 object-cover rounded-md" />
            <div className="flex-1">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{fileName}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Arquivo otimizado e pronto</p>
            </div>
            <button
              onClick={handleClear}
              className="p-2 text-gray-500 dark:text-gray-400 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 hover:text-gray-800 dark:hover:text-gray-100 transition"
              aria-label="Remove file"
            >
              <XIcon />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};