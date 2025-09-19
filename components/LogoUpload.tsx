import React, { useState } from 'react';
// Fix: Correct import path for types.
import { Logo } from '../types.ts';
// Fix: Correct import paths with extensions.
import { generateLogo } from '../services/geminiService.ts';
import { removeSimpleBackground } from '../utils/imageProcessing.ts';
import { DownloadIcon, UploadCloudIcon, XIcon } from './Icons.tsx';

interface LogoUploadProps {
    logo: Logo | null;
    onLogoChange: (logo: Logo | null) => void;
    logoPrompt: string;
    onLogoPromptChange: (prompt: string) => void;
}

export const LogoUpload: React.FC<LogoUploadProps> = ({ logo, onLogoChange, logoPrompt, onLogoPromptChange }) => {
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    const handleGenerate = async () => {
        if (!logoPrompt) return;
        setIsGenerating(true);
        setError(null);
        try {
            // FIX: The `generateLogo` function returns a complete `Logo` object.
            // The previous code was incorrectly treating the result as a raw base64 string.
            const newLogo = await generateLogo(logoPrompt);
            onLogoChange(newLogo);
        } catch (err) {
            console.error("Failed to generate logo:", err);
            setError("Falha ao gerar o logo. Tente novamente.");
        } finally {
            setIsGenerating(false);
        }
    };

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            let base64 = event.target?.result as string;
            try {
                // Try to remove white/light backgrounds from uploaded logos
                base64 = await removeSimpleBackground(base64, 30);
            } catch (err) {
                console.warn("Could not remove background, using original.", err);
            } finally {
                onLogoChange({ base64, prompt: file.name });
            }
        };
        reader.readAsDataURL(file);
    };

    const handleDownload = () => {
        if (!logo) return;
        const link = document.createElement('a');
        link.href = logo.base64;
        link.download = `logo_${logo.prompt.substring(0, 15).replace(/\s/g, '_') || 'gerado'}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    if (logo) {
        return (
            <div>
                 <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Logo</h3>
                 <div className="relative p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800/50 text-center">
                    <img src={logo.base64} alt="Logo preview" className="max-h-24 mx-auto object-contain"/>
                    <p className="text-xs text-gray-500 mt-2 truncate">{logo.prompt}</p>
                    <div className="absolute -top-2 -right-2 flex space-x-1">
                        <button onClick={handleDownload} className="p-1 bg-blue-500 text-white rounded-full hover:bg-blue-600" title="Baixar Logo">
                           <DownloadIcon className="w-4 h-4" />
                        </button>
                        <button onClick={() => onLogoChange(null)} className="p-1 bg-red-500 text-white rounded-full hover:bg-red-600" title="Remover Logo">
                            <XIcon className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">Logo (Opcional)</h3>
             <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-3">
                 <div className="flex gap-2">
                     <input type="text" value={logoPrompt} onChange={e => onLogoPromptChange(e.target.value)} placeholder="Descreva um logo para a IA criar" className="flex-grow p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                     <button onClick={handleGenerate} disabled={isGenerating || !logoPrompt} className="px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md disabled:bg-gray-500 disabled:cursor-not-allowed">
                         {isGenerating ? 'Gerando...' : 'Gerar'}
                     </button>
                </div>
                <div className="text-center text-xs text-gray-500">ou</div>
                <label htmlFor="logo-upload-simple" className="w-full flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition">
                    <UploadCloudIcon />
                    <span className="text-sm">Carregar arquivo de logo</span>
                    <input id="logo-upload-simple" type="file" className="hidden" accept="image/png, image/jpeg" onChange={handleUpload}/>
                </label>
                {error && <p className="text-sm text-red-500 text-center">{error}</p>}
            </div>
        </div>
    );
};
