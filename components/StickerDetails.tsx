import React, { useState } from 'react';
import { StickerDetail, StickerFinish, Logo } from '../types.ts';
import { StickerIcon, TrashIcon, SparkleIcon, PatternIcon, UploadCloudIcon, XIcon } from './Icons.tsx';
import { generatePattern } from '../services/geminiService.ts';
import { removeSimpleBackground } from '../utils/imageProcessing.ts';

interface StickerDetailsProps {
    value: StickerDetail[];
    onChange: (value: StickerDetail[]) => void;
    onFocus: () => void;
    onBlur: () => void;
}

const FINISH_OPTIONS: StickerFinish[] = [
    'Brilhante', 'Fosco', 'Jateado', 'Metálico', 'Aço Escovado', 
    'Cromado / Espelhado', 'Refletivo', 'Fibra de Carbono', 'Holográfico'
];

// --- Sub-editors for each sticker type ---

const PatternStickerEditor: React.FC<{
    details: StickerDetail & { type: 'pattern' };
    onChange: (updatedDetails: StickerDetail) => void;
    onRemove: () => void;
    onFocus: () => void;
    onBlur: () => void;
}> = ({ details, onChange, onRemove, onFocus, onBlur }) => {
    const [isGenerating, setIsGenerating] = useState(false);
    const data = details.data;

    const handleChange = (field: keyof typeof data, value: any) => {
        onChange({ ...details, data: { ...data, [field]: value } });
    };
    
    const handleGenerate = async () => {
        if (!data.theme) return;
        setIsGenerating(true);
        try {
            const newPatternLogo = await generatePattern(data.theme);
            handleChange('generatedPattern', newPatternLogo);
        } catch (error) {
            console.error("Failed to generate pattern", error);
            alert("Não foi possível gerar o padrão. Tente novamente.");
        } finally {
            setIsGenerating(false);
        }
    };
    
    const removePattern = () => {
        handleChange('generatedPattern', undefined);
    };

    return (
        <div onFocus={onFocus} onBlur={onBlur} className="p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-3 border-l-4 border-cyan-500">
             <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                    <PatternIcon className="w-5 h-5 text-gray-700 dark:text-gray-300"/>
                    <p className="font-semibold text-sm">Padrão Decorativo (IA)</p>
                </div>
                <button onClick={onRemove} className="p-1 text-gray-400 hover:text-red-500"><TrashIcon className="w-4 h-4" /></button>
            </div>
             <input type="text" placeholder="Local de aplicação (Ex: Parede principal)" value={data.placement} onChange={e => handleChange('placement', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
            <div className="flex gap-2">
                <input type="text" placeholder="Tema do padrão (ex: 'folhas tropicais')" value={data.theme} onChange={e => handleChange('theme', e.target.value)} className="flex-grow p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                 <button onClick={handleGenerate} disabled={isGenerating || !data.theme} className="px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md disabled:bg-gray-500">
                    {isGenerating ? '...' : <SparkleIcon className="w-5 h-5"/>}
                </button>
            </div>
            {data.generatedPattern && (
                 <div className="p-2 bg-gray-200 dark:bg-gray-700 rounded-md relative">
                    <img src={data.generatedPattern.base64} alt="Padrão gerado" className="w-full h-16 object-cover rounded"/>
                     <button onClick={removePattern} className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600" title="Remover Padrão">
                        <XIcon className="w-3 h-3"/>
                    </button>
                </div>
            )}
        </div>
    );
};

const CutStickerEditor: React.FC<{
    details: StickerDetail & { type: 'cut' };
    onChange: (updatedDetails: StickerDetail) => void;
    onRemove: () => void;
    onFocus: () => void;
    onBlur: () => void;
}> = ({ details, onChange, onRemove, onFocus, onBlur }) => {
    const data = details.data;
    const handleChange = (field: keyof typeof data, value: any) => {
        onChange({ ...details, data: { ...data, [field]: value } });
    };

    const showColorPicker = data.finish === 'Brilhante' || data.finish === 'Fosco';
    const showColorTextInput = ['Metálico', 'Cromado / Espelhado', 'Refletivo', 'Holográfico'].includes(data.finish);

    return (
        <div onFocus={onFocus} onBlur={onBlur} className="p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-3 border-l-4 border-purple-500">
             <div className="flex justify-between items-center">
                <p className="font-semibold text-sm">Adesivo de Recorte</p>
                <button onClick={onRemove} className="p-1 text-gray-400 hover:text-red-500"><TrashIcon className="w-4 h-4" /></button>
            </div>
             <input type="text" placeholder="Local de aplicação (Ex: Vitrine da direita)" value={data.placement} onChange={e => handleChange('placement', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
             <input type="text" placeholder="Descrição (texto ou forma)" value={data.description} onChange={e => handleChange('description', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
             <div className="grid grid-cols-2 gap-3">
                 <div>
                    <label className="text-xs font-medium text-gray-500">Material / Acabamento</label>
                    <select value={data.finish} onChange={e => handleChange('finish', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500">
                        {FINISH_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                </div>
                 <div>
                    <label className="text-xs font-medium text-gray-500">Cor / Material</label>
                    {showColorPicker && (
                         <div className="flex items-center gap-2">
                            <input type="color" value={data.color.startsWith('#') ? data.color : '#ffffff'} onChange={e => handleChange('color', e.target.value)} className="p-0 h-10 w-12 rounded-md border-2 border-transparent bg-transparent cursor-pointer" />
                            <input type="text" value={data.color} onChange={e => handleChange('color', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                        </div>
                    )}
                    {showColorTextInput && (
                        <input type="text" placeholder="Ex: Dourado" value={data.color} onChange={e => handleChange('color', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                    )}
                 </div>
             </div>
        </div>
    );
};

const PrintStickerEditor: React.FC<{
    details: StickerDetail & { type: 'print' };
    onChange: (updatedDetails: StickerDetail) => void;
    onRemove: () => void;
    onFocus: () => void;
    onBlur: () => void;
}> = ({ details, onChange, onRemove, onFocus, onBlur }) => {
     const data = details.data;
     const handleChange = (field: keyof typeof data, value: any) => {
        onChange({ ...details, data: { ...data, [field]: value } });
    };
    
    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            const base64 = event.target?.result as string;
            handleChange('printFile', { base64, prompt: file.name });
        };
        reader.readAsDataURL(file);
    };

    return (
         <div onFocus={onFocus} onBlur={onBlur} className="p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-3 border-l-4 border-green-500">
             <div className="flex justify-between items-center">
                <p className="font-semibold text-sm">Adesivo de Impressão Digital</p>
                <button onClick={onRemove} className="p-1 text-gray-400 hover:text-red-500"><TrashIcon className="w-4 h-4" /></button>
            </div>
             <input type="text" placeholder="Local de aplicação (Ex: Porta de vidro)" value={data.placement} onChange={e => handleChange('placement', e.target.value)} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
             
             {data.printFile ? (
                 <div className="p-2 bg-gray-200 dark:bg-gray-700 rounded-md relative text-center">
                    <img src={data.printFile.base64} alt="Preview" className="max-h-20 mx-auto object-contain"/>
                    <p className="text-xs text-gray-500 mt-1 truncate">{data.printFile.prompt}</p>
                    <button onClick={() => handleChange('printFile', null)} className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600" title="Remover Imagem">
                        <XIcon className="w-3 h-3"/>
                    </button>
                </div>
             ) : (
                <label htmlFor={`print-upload-${details.id}`} className="w-full flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition">
                    <UploadCloudIcon />
                    <span className="text-sm">Carregar arquivo</span>
                    <input id={`print-upload-${details.id}`} type="file" className="hidden" accept="image/png, image/jpeg" onChange={handleUpload}/>
                </label>
             )}

             <div className="flex items-center gap-4">
                 <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={data.hasContourCut} onChange={e => handleChange('hasContourCut', e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-cyan-600 focus:ring-cyan-500 bg-transparent" />
                    Recorte de Contorno
                </label>
                 <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={data.isPerforated} onChange={e => handleChange('isPerforated', e.target.checked)} className="h-4 w-4 rounded border-gray-300 text-cyan-600 focus:ring-cyan-500 bg-transparent" />
                    Adesivo Perfurado
                </label>
             </div>
        </div>
    );
};


export const StickerDetailsComponent: React.FC<StickerDetailsProps> = ({ value, onChange, onFocus, onBlur }) => {
    
    const addSticker = (type: StickerDetail['type']) => {
        const newSticker: StickerDetail = {
            id: `sticker_${Date.now()}`,
            type: type,
            data: {} as any // Initial empty data
        };
        
        switch (type) {
            case 'pattern':
                newSticker.data = { placement: '', theme: '' };
                break;
            case 'cut':
                newSticker.data = { placement: '', description: '', finish: 'Brilhante', color: '#ffffff' };
                break;
            case 'print':
                newSticker.data = { placement: '', printFile: null, hasContourCut: false, isPerforated: false };
                break;
        }
        
        onChange([...value, newSticker]);
    };
    
    const updateSticker = (updatedSticker: StickerDetail) => {
        onChange(value.map(s => s.id === updatedSticker.id ? updatedSticker : s));
    };

    const removeSticker = (id: string) => {
        onChange(value.filter(s => s.id !== id));
    };

    return (
        <div className="space-y-4">
            <div>
                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">Adesivos Decorativos</h3>
                <div className="space-y-3">
                    {value.map(sticker => {
                        switch (sticker.type) {
                            case 'pattern':
                                return <PatternStickerEditor key={sticker.id} details={sticker} onChange={updateSticker} onRemove={() => removeSticker(sticker.id)} onFocus={onFocus} onBlur={onBlur} />;
                            case 'cut':
                                return <CutStickerEditor key={sticker.id} details={sticker} onChange={updateSticker} onRemove={() => removeSticker(sticker.id)} onFocus={onFocus} onBlur={onBlur} />;
                            case 'print':
                                return <PrintStickerEditor key={sticker.id} details={sticker} onChange={updateSticker} onRemove={() => removeSticker(sticker.id)} onFocus={onFocus} onBlur={onBlur} />;
                            default:
                                return null;
                        }
                    })}
                     {value.length === 0 && <p className="text-sm text-gray-500 text-center py-2">Nenhum adesivo adicionado.</p>}
                </div>
            </div>
             <div className="flex flex-col gap-2">
                <div className="flex flex-wrap gap-2">
                    <button type="button" onClick={() => addSticker('pattern')} className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 dark:bg-gray-700/80 dark:text-gray-200 dark:hover:bg-gray-700/100 transition">
                        + Padrão (IA)
                    </button>
                    <button type="button" onClick={() => addSticker('cut')} className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 dark:bg-gray-700/80 dark:text-gray-200 dark:hover:bg-gray-700/100 transition">
                        + Recorte (Cor Única)
                    </button>
                    <button type="button" onClick={() => addSticker('print')} className="flex items-center gap-2 px-3 py-2 text-xs font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 dark:bg-gray-700/80 dark:text-gray-200 dark:hover:bg-gray-700/100 transition">
                        + Impressão Digital
                    </button>
                </div>
            </div>
        </div>
    );
};