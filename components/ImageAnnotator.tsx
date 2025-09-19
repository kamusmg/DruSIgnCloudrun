import React, { useState, useRef, MouseEvent } from 'react';
// Fix: Correct import path for types.
import { Annotation } from '../types.ts';
// Fix: Correct the import path for Icons.
import { XIcon } from './Icons.tsx';

interface ImageAnnotatorProps {
  src: string;
  annotations: Annotation[];
  onAnnotationsChange: (annotations: Annotation[]) => void;
}

interface NewAnnotationState {
  x: number;
  y: number;
  instruction: string;
}

export const ImageAnnotator: React.FC<ImageAnnotatorProps> = ({ src, annotations, onAnnotationsChange }) => {
  const [newAnnotation, setNewAnnotation] = useState<NewAnnotationState | null>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);

  const handleImageClick = (e: MouseEvent<HTMLDivElement>) => {
    if (newAnnotation) return; // Only one new annotation at a time

    const rect = imageContainerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    setNewAnnotation({ x, y, instruction: '' });
  };

  const handleSaveAnnotation = () => {
    if (!newAnnotation || !newAnnotation.instruction.trim()) {
      setNewAnnotation(null); // Cancel if empty
      return;
    };
    
    const annotationToAdd: Annotation = {
      ...newAnnotation,
      id: Date.now(),
    };
    onAnnotationsChange([...annotations, annotationToAdd]);
    setNewAnnotation(null);
  };

  const handleRemoveAnnotation = (id: number) => {
    onAnnotationsChange(annotations.filter((ann) => ann.id !== id));
  };

  const handleInstructionChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      if (newAnnotation) {
          setNewAnnotation({ ...newAnnotation, instruction: e.target.value });
      }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSaveAnnotation();
    }
  };


  return (
    <div className="relative w-full" ref={imageContainerRef} onClick={handleImageClick}>
      <img src={src} alt="Fachada para anotar" className="w-full h-auto rounded-lg" />

      {/* Display existing annotations */}
      {annotations.map((ann, index) => (
        <div
          key={ann.id}
          className="absolute group"
          style={{ left: `${ann.x}%`, top: `${ann.y}%`, transform: 'translate(-50%, -50%)' }}
        >
          <div className="flex items-center justify-center w-8 h-8 bg-cyan-500 text-black rounded-full font-bold shadow-lg cursor-default">
            {index + 1}
          </div>
          <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 p-2 bg-black text-white text-xs rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
            {ann.instruction}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRemoveAnnotation(ann.id);
              }}
              className="absolute -top-2 -right-2 p-0.5 bg-red-500 text-white rounded-full hover:bg-red-600"
            >
              <XIcon className="w-3 h-3"/>
            </button>
          </div>
        </div>
      ))}

      {/* Form for new annotation */}
      {newAnnotation && (
        <div
          className="absolute p-3 bg-white dark:bg-gray-800 rounded-lg shadow-2xl border border-gray-200 dark:border-gray-700 z-20"
          style={{ left: `${newAnnotation.x}%`, top: `${newAnnotation.y}%` }}
          onClick={(e) => e.stopPropagation()}
        >
          <p className="text-sm font-semibold mb-2 text-gray-800 dark:text-gray-200">Adicionar instrução</p>
          <textarea
            autoFocus
            value={newAnnotation.instruction}
            onChange={handleInstructionChange}
            onKeyDown={handleKeyDown}
            rows={3}
            placeholder="Ex: 'Adicionar uma placa aqui'"
            className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 focus:ring-cyan-500 focus:border-cyan-500"
          />
          <div className="mt-2 flex justify-end gap-2">
            <button 
                onClick={() => setNewAnnotation(null)} 
                className="px-3 py-1 text-xs text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-md"
            >
                Cancelar
            </button>
            <button 
                onClick={handleSaveAnnotation} 
                className="px-3 py-1 text-xs bg-cyan-600 text-white font-semibold rounded-md hover:bg-cyan-700 transition"
            >
                Salvar
            </button>
          </div>
        </div>
      )}
    </div>
  );
};