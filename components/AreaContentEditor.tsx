import React, { useState, useEffect } from 'react';
// Fix: Correct import path for types.
import { Area } from '../types.ts';

interface AreaContentEditorProps {
    area: Area;
    onAreaChange: (updatedArea: Area) => void;
}

const BackgroundEditor: React.FC<{
    value: Area['content']['background'];
    onChange: (bg: Area['content']['background']) => void;
}> = ({ value, onChange }) => {
    const type = value?.type ?? 'none';
    const color = value?.color ?? '#555555';
    
    return (
        <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Fundo</label>
            <div className="flex gap-2">
                <select value={type} onChange={e => onChange({ ...value, type: e.target.value as any, color })}
                    className="p-2 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500 flex-grow"
                >
                    <option value="none">Nenhum</option>
                    <option value="paint">Pintura</option>
                    <option value="acm">ACM</option>
                    <option value="vinyl">Adesivo Vinil</option>
                </select>
                {type !== 'none' && (
                    <input type="color" value={color} onChange={e => onChange({ ...value, type, color: e.target.value })}
                        className="p-1 h-10 w-12 rounded-md border-2 border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 cursor-pointer"
                    />
                )}
            </div>
        </div>
    );
};

export const AreaContentEditor: React.FC<AreaContentEditorProps> = ({ area, onAreaChange }) => {
    const [texts, setTexts] = useState(area.content?.texts ?? {});
    const [background, setBackground] = useState(area.content?.background);

    useEffect(() => {
        setTexts(area.content?.texts ?? {});
        setBackground(area.content?.background);
    }, [area.id, area.content]);

    useEffect(() => {
        // Debounce or direct update
        const handler = setTimeout(() => {
            onAreaChange({
                ...area,
                content: { ...area.content, texts, background }
            });
        }, 300); // Debounce changes to avoid excessive re-renders
        return () => clearTimeout(handler);
    }, [texts, background, area, onAreaChange]);
    
    const handleTextChange = (field: 'title' | 'subtitle' | 'phone', value: string) => {
        setTexts(prev => ({ ...prev, [field]: value }));
    };

    return (
        <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Conteúdo para "{area.type}"</h3>
            
            <div className="space-y-3">
                <input type="text" placeholder="Título (Ex: Nome da Empresa)" value={texts.title ?? ''} onChange={e => handleTextChange('title', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                <input type="text" placeholder="Slogan ou serviço" value={texts.subtitle ?? ''} onChange={e => handleTextChange('subtitle', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                <input type="text" placeholder="Telefone / WhatsApp" value={texts.phone ?? ''} onChange={e => handleTextChange('phone', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
            </div>

            <BackgroundEditor value={background} onChange={setBackground} />
        </div>
    );
};