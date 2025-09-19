import React from 'react';
import { XIcon } from './Icons.tsx';

interface ErrorDisplayProps {
  error: Error | null;
  onReset: () => void;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ error, onReset }) => {
  if (!error) {
    return null;
  }
  
  // A specific check for the handled image generation failure.
  // This error has a user-friendly message that is sufficient.
  const isGenerationFailure = error.name === 'Falha na Geração de Imagem';

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[100] p-4 animate-fade-in" role="alertdialog" aria-modal="true" aria-labelledby="error-title">
      <div className="bg-red-50 dark:bg-gray-900 border-2 border-red-300 dark:border-red-800 rounded-xl shadow-2xl max-w-2xl w-full p-8 transform transition-all animate-slide-up">
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-4">
            <div className="flex-shrink-0 w-12 h-12 flex items-center justify-center bg-red-100 dark:bg-red-900/50 rounded-full">
              <XIcon className="w-7 h-7 text-red-600 dark:text-red-400" />
            </div>
            <div>
                <h2 id="error-title" className="text-2xl font-bold text-red-900 dark:text-red-200">
                  Oops! Algo deu errado.
                </h2>
                <p className="text-red-700 dark:text-red-300 mt-1">
                    Encontramos um erro inesperado. Por favor, reinicie o aplicativo.
                </p>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-red-100 dark:bg-gray-800 p-4 rounded-lg text-sm">
            <details open>
                <summary className="font-semibold text-red-800 dark:text-red-300 cursor-pointer">Detalhes Técnicos</summary>
                <div className="mt-2 text-red-700 dark:text-red-400 font-mono text-xs whitespace-pre-wrap break-words">
                  {/* For controlled generation failures, show only the helpful message. */}
                  {isGenerationFailure ? (
                    <>
                      <p><strong>Tipo:</strong> {error.name}</p>
                      <p><strong>Mensagem:</strong> {error.message}</p>
                    </>
                  ) : (
                    /* For all other unexpected errors, show full details. */
                    <>
                      <p><strong>Tipo:</strong> {error.name}</p>
                      <p><strong>Mensagem:</strong> {error.message}</p>
                      {error.stack && <p className="mt-2"><strong>Stack Trace:</strong><br/>{error.stack}</p>}
                    </>
                  )}
                </div>
            </details>
        </div>

        <div className="mt-6 flex justify-end">
          <button 
            onClick={onReset} 
            className="px-8 py-3 bg-red-600 text-white font-semibold rounded-lg hover:bg-red-700 transition shadow-lg"
          >
            Recomeçar Projeto
          </button>
        </div>
      </div>
    </div>
  );
};
