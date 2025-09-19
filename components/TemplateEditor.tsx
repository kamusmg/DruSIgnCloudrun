import React, { useState, useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
// Fix: Correct import paths with extensions.
import { TEMPLATES } from '../templates/index.ts';
import { TemplateSpec } from '../templates/types.ts';
import { renderTemplate } from '../templates/renderer.ts';
import { useDebounce } from '../hooks/useDebounce.ts';
import { CheckIcon, FullscreenIcon, PhoneIcon } from './Icons.tsx';

type Texts = { title: string; phone: string; subtitle: string };
type PaletteOption = 'auto' | 'light' | 'dark';
type QuickAdjustments = {
    isUpper: boolean;
    hasShadow: boolean;
    hasStroke: boolean;
    palette: PaletteOption;
};

interface TemplateEditorProps {
    imageSrc: string;
    texts: Texts;
    onTextsChange: (texts: Texts) => void;
    onBack: () => void;
}

// Helper components are defined outside the main component to prevent re-definition on re-render,
// which was causing the input focus loss issue.
const FormInput: React.FC<{
    name: keyof Texts;
    label: string;
    placeholder: string;
    value: string;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}> = ({ name, label, placeholder, value, onChange }) => (
    <div>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">{label}</label>
        <input
            type="text"
            id={name}
            name={name}
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500"
            aria-label={label}
        />
    </div>
);

const AdjustmentButton: React.FC<{
    label: string;
    isActive: boolean;
    onClick: () => void;
}> = ({ label, isActive, onClick }) => (
     <button
        onClick={onClick}
        aria-pressed={!!isActive}
        className={`flex-1 p-2 text-sm rounded-md transition border-2 ${isActive ? 'bg-cyan-500/10 border-cyan-500 text-cyan-500' : 'bg-gray-200 dark:bg-gray-700 border-transparent hover:border-gray-400'}`}>
        {label}
     </button>
);


export const TemplateEditor = forwardRef<{ getFinalCanvas: () => Promise<string | null> }, TemplateEditorProps>(({ imageSrc, texts, onTextsChange, onBack }, ref) => {
    const [selectedTemplate, setSelectedTemplate] = useState<TemplateSpec>(TEMPLATES[0]);
    const [adjustments, setAdjustments] = useState<QuickAdjustments>({
        isUpper: true,
        hasShadow: true,
        hasStroke: false,
        palette: 'auto'
    });
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const debouncedTexts = useDebounce(texts, 200);

    useImperativeHandle(ref, () => ({
        getFinalCanvas: async () => {
            if (!canvasRef.current || !imageRef.current) return null;
            await renderTemplate(canvasRef.current, imageRef.current, selectedTemplate, texts, adjustments);
            return canvasRef.current.toDataURL('image/jpeg', 0.9);
        }
    }));
    
    const drawCanvas = useCallback(async () => {
        if (!canvasRef.current || !imageRef.current) return;
        await renderTemplate(canvasRef.current, imageRef.current, selectedTemplate, debouncedTexts, adjustments);
    }, [selectedTemplate, debouncedTexts, adjustments]);

    useEffect(() => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = imageSrc;
        img.onload = () => {
            imageRef.current = img;
            drawCanvas();
        };
    }, [imageSrc, drawCanvas]);

    useEffect(() => {
        drawCanvas();
    }, [drawCanvas]);

    const handleFieldChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        onTextsChange({ ...texts, [e.target.name]: e.target.value });
    };

    const handleAdjustmentChange = (key: keyof QuickAdjustments, value: boolean | PaletteOption) => {
        setAdjustments(prev => ({ ...prev, [key]: value }));
    };

    const handleFullScreen = () => {
        canvasRef.current?.requestFullscreen().catch(err => console.error(err));
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Preview */}
            <div className="lg:col-span-2 space-y-4">
                <div ref={containerRef} className="relative w-full aspect-[16/10] bg-gray-900 rounded-lg shadow-lg overflow-hidden">
                    <canvas ref={canvasRef} className="w-full h-full" />
                </div>
                <div className="flex gap-2">
                    <button onClick={onBack} className="flex-1 px-4 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Trocar Foto</button>
                    <button onClick={handleFullScreen} className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition"><FullscreenIcon className="w-5 h-5"/> Tela Cheia</button>
                </div>
            </div>

            {/* Right Column: Controls */}
            <div className="space-y-6">
                 {/* Fields */}
                <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-4">
                     <h4 className="text-md font-semibold">Informações da Placa</h4>
                     <FormInput name="title" label="Nome da Empresa" placeholder="Sua Empresa Incrível" value={texts.title} onChange={handleFieldChange} />
                     <FormInput name="phone" label="Telefone / WhatsApp" placeholder="(11) 99999-8888" value={texts.phone} onChange={handleFieldChange} />
                     <FormInput name="subtitle" label="Slogan ou Serviço" placeholder="Qualidade e Confiança" value={texts.subtitle} onChange={handleFieldChange} />
                </div>
                
                 {/* Models */}
                <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg">
                    <h4 className="text-md font-semibold mb-3">Modelos</h4>
                    <div className="grid grid-cols-3 gap-2">
                        {TEMPLATES.map(template => (
                            <button key={template.id} onClick={() => setSelectedTemplate(template)}
                                className={`relative aspect-video rounded-md text-xs transition border-2 flex items-center justify-center p-1 text-center font-semibold
                                ${selectedTemplate.id === template.id ? 'border-cyan-500 ring-2 ring-cyan-500' : 'border-gray-400 dark:border-gray-600 hover:border-cyan-400'}`}
                                aria-pressed={selectedTemplate.id === template.id}
                            >
                               {template.id === 'telefone-destaque' && <PhoneIcon className="w-4 h-4 text-gray-500" />}
                               <span>{template.name}</span>
                               {selectedTemplate.id === template.id && <div className="absolute -top-2 -right-2 w-5 h-5 bg-cyan-500 rounded-full flex items-center justify-center"><CheckIcon className="w-3 h-3 text-white"/></div>}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Quick Adjustments */}
                <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg">
                    <h4 className="text-md font-semibold mb-3">Ajustes Rápidos</h4>
                    <div className="space-y-3">
                         <div className="flex gap-2">
                            <AdjustmentButton label="MAIÚSCULAS" isActive={adjustments.isUpper} onClick={() => handleAdjustmentChange('isUpper', !adjustments.isUpper)} />
                            <AdjustmentButton label="Sombra" isActive={adjustments.hasShadow} onClick={() => handleAdjustmentChange('hasShadow', !adjustments.hasShadow)} />
                            <AdjustmentButton label="Contorno" isActive={adjustments.hasStroke} onClick={() => handleAdjustmentChange('hasStroke', !adjustments.hasStroke)} />
                         </div>
                         <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">Paleta:</span>
                            {['auto', 'light', 'dark'].map((p) => (
                               <button key={p} onClick={() => handleAdjustmentChange('palette', p as PaletteOption)}
                                className={`flex-1 p-2 text-sm rounded-md transition border-2 capitalize ${adjustments.palette === p ? 'bg-cyan-500/10 border-cyan-500 text-cyan-500' : 'bg-gray-200 dark:bg-gray-700 border-transparent hover:border-gray-400'}`}>
                                {p}
                               </button>
                            ))}
                         </div>
                    </div>
                </div>
            </div>
        </div>
    );
});