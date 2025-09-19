
import React, { useState } from 'react';
import { Annotation, VisualEditorState } from '../types.ts';
import { ImageAnnotator } from './ImageAnnotator.tsx';
import { XIcon } from './Icons.tsx';

interface VisualEditorProps {
    initialState: VisualEditorState;
    redesignedImage: string;
    onCancel: () => void;
    onRegenerate: (prompt: string) => Promise<void>;
}

export const VisualEditor: React.FC<VisualEditorProps> = ({
    initialState,
    redesignedImage,
    onCancel,
    onRegenerate,
}) => {
    const [annotations, setAnnotations] = useState<Annotation[]>(initialState.annotations);
    const [generalPrompt, setGeneralPrompt] = useState(initialState.generalPrompt);
    const [isRegenerating, setIsRegenerating] = useState(false);

    const handleAnnotationsChange = (newAnnotations: Annotation[]) => {
        setAnnotations(newAnnotations);
    };

    const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setGeneralPrompt(e.target.value);
    };

    const handleRegenerateClick = async () => {
        setIsRegenerating(true);
        
        // Construct the final prompt from annotations and general instructions
        const annotationLines = annotations.map((ann, index) => 
            `${index + 1}. (Na anotação ${index + 1}) ${ann.instruction}`
        );
        
        const finalEditPrompt = [
            "Aplique as seguintes edições à imagem fornecida:",
            ...annotationLines,
            generalPrompt
        ].filter(line => line.trim() !== '').join('\n');
        
        await onRegenerate(finalEditPrompt);
        setIsRegenerating(false);
    };

    return (
        <div className="fixed inset-0 bg-black/80 flex flex-col z-50 p-4 animate-fade-in">
            <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl w-full h-full flex flex-col p-6 border border-gray-200 dark:border-gray-800">
                <div className="flex justify-between items-center mb-4 flex-shrink-0">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Editor Visual</h2>
                        <p className="text-gray-600 dark:text-gray-400">Clique na imagem para adicionar anotações ou descreva as mudanças abaixo.</p>
                    </div>
                    <button onClick={onCancel} className="p-2 -mt-2 -mr-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
                        <XIcon className="w-6 h-6" />
                    </button>
                </div>

                <div className="flex-grow grid grid-cols-1 lg:grid-cols-3 gap-6 overflow-hidden">
                    <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto pr-2">
                        <div className="relative">
                            <ImageAnnotator
                                src={redesignedImage}
                                annotations={annotations}
                                onAnnotationsChange={handleAnnotationsChange}
                            />
                        </div>
                        
                        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Anotações</h3>
                        <div className="space-y-2 max-h-48 overflow-y-auto pr-2">
                             {annotations.length === 0 ? (
                                <p className="text-sm text-gray-500">Nenhuma anotação adicionada.</p>
                             ) : (
                                annotations.map((ann, index) => (
                                <div key={ann.id} className="flex items-center gap-3 p-2 bg-gray-100 dark:bg-gray-800 rounded-md">
                                    <span className="flex-shrink-0 flex items-center justify-center w-6 h-6 bg-cyan-500 text-black text-xs font-bold rounded-full">{index + 1}</span>
                                    <p className="flex-grow text-sm text-gray-800 dark:text-gray-200">{ann.instruction}</p>
                                    <button onClick={() => handleAnnotationsChange(annotations.filter(a => a.id !== ann.id))} className="p-1 text-gray-400 hover:text-red-500"><XIcon className="w-4 h-4" /></button>
                                </div>
                            )))}
                        </div>
                    </div>

                    <div className="flex flex-col space-y-4">
                        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Instruções de Edição</h3>
                        <textarea
                            value={generalPrompt}
                            onChange={handlePromptChange}
                            rows={15}
                            placeholder="Descreva aqui as mudanças gerais que você deseja. Para mudanças em locais específicos, clique na imagem."
                            className="w-full h-full p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 focus:ring-cyan-500 focus:border-cyan-500"
                        />
                         <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                            <p className="font-semibold">Dicas para Boas Edições:</p>
                            <ul className="list-disc list-inside">
                                <li><b>Seja específico:</b> "Mude a cor do ACM para Aço Escovado" é melhor que "mude a cor".</li>
                                <li><b>Descreva o resultado:</b> "Adicione uma rampa de acesso para cadeirantes" é melhor que "rampa".</li>
                                <li><b>Use as anotações:</b> Para mudanças em locais específicos, clique na imagem.</li>
                            </ul>
                        </div>
                    </div>
                </div>
                 <div className="flex-shrink-0 pt-4 mt-4 border-t border-gray-200 dark:border-gray-800 flex justify-end gap-3">
                    <button onClick={onCancel} className="px-6 py-3 text-gray-800 dark:text-gray-200 bg-gray-200 dark:bg-gray-700 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600">Cancelar</button>
                    <button
                        onClick={handleRegenerateClick}
                        disabled={isRegenerating}
                        className="px-6 py-3 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 shadow-md disabled:bg-gray-500"
                    >
                        {isRegenerating ? 'Gerando...' : 'Gerar Edição'}
                    </button>
                </div>
            </div>
        </div>
    );
};
