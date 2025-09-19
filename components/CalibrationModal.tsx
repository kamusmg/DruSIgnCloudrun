

import React, { useState } from 'react';
import { RulerIcon, XIcon } from './Icons.tsx';

interface CalibrationModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: (distanceCm: number) => void;
}

export const CalibrationModal: React.FC<CalibrationModalProps> = ({ isOpen, onClose, onConfirm }) => {
    const [distance, setDistance] = useState<string>('');
    const [error, setError] = useState<string>('');

    if (!isOpen) {
        return null;
    }

    const handleConfirm = () => {
        const numDistance = parseFloat(distance.replace(',', '.'));
        if (isNaN(numDistance) || numDistance <= 0) {
            setError('Por favor, insira um número válido e positivo.');
            return;
        }
        setError('');
        onConfirm(numDistance);
        setDistance('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleConfirm();
        }
    };

    return (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4 animate-fade-in">
            <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-lg w-full p-8 border border-gray-200 dark:border-gray-800 transform transition-all animate-slide-up">
                <div className="flex justify-between items-start">
                    <div className="flex items-center gap-3">
                        <RulerIcon className="w-8 h-8 text-cyan-500" />
                        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                            Calibrar Medidas
                        </h2>
                    </div>
                    <button onClick={onClose} className="p-1 -mt-2 -mr-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
                        <XIcon className="w-6 h-6" />
                    </button>
                </div>

                <p className="text-gray-600 dark:text-gray-400 mt-4 mb-6">
                    Você desenhou uma linha de referência na imagem. Agora, informe a medida real correspondente a essa linha em **centímetros** para calibrar as dimensões do projeto.
                </p>

                <div className="space-y-2">
                    <label htmlFor="distance-input" className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        Distância Real (em centímetros)
                    </label>
                    <input
                        id="distance-input"
                        type="text"
                        value={distance}
                        onChange={(e) => setDistance(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ex: 210"
                        className="w-full p-3 text-lg border-2 border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 focus:ring-cyan-500 focus:border-cyan-500"
                        autoFocus
                    />
                    {error && <p className="text-sm text-red-500 mt-1">{error}</p>}
                </div>

                <div className="mt-6 flex justify-end">
                    <button
                        onClick={handleConfirm}
                        className="px-8 py-3 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md"
                    >
                        Calibrar
                    </button>
                </div>
            </div>
        </div>
    );
};
