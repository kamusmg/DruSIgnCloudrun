import React from 'react';
// Fix: Correct the import path for Icons.
import { XIcon } from './Icons.tsx';

interface EnhanceConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirmEnhance: () => void;
  onConfirmOriginal: () => void;
  isLoading: boolean;
}

export const EnhanceConfirmModal: React.FC<EnhanceConfirmModalProps> = ({ 
    isOpen, 
    onClose, 
    onConfirmEnhance,
    onConfirmOriginal,
    isLoading 
}) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-md w-full p-8 border border-gray-200 dark:border-gray-800 transform transition-all animate-slide-up">
        <div className="flex justify-between items-start">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
              Aprimorar Descrição com IA?
            </h2>
             <button onClick={onClose} className="p-1 -mt-2 -mr-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
                <XIcon className="w-6 h-6" />
            </button>
        </div>
        
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Nossa IA pode refinar sua descrição com termos técnicos para obter um resultado visual mais detalhado e profissional. Deseja fazer isso ou usar seu texto original?
        </p>

        <div className="flex flex-col sm:flex-row justify-end gap-3">
          <button 
            onClick={onConfirmOriginal}
            disabled={isLoading}
            className="px-6 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition disabled:opacity-50"
          >
            Não, Usar Meu Texto
          </button>
          <button 
            onClick={onConfirmEnhance}
            disabled={isLoading}
            className="px-6 py-2.5 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Gerando...' : 'Sim, Aprimorar'}
          </button>
        </div>
      </div>
    </div>
  );
};