

import React, { useState } from 'react';
import { DetailedRequestData, Logo, LogoOption } from '../types.ts';
import { DownloadIcon, UploadCloudIcon, XIcon } from './Icons.tsx';
import { removeSimpleBackground } from '../utils/imageProcessing.ts';

interface LogoOptionsProps {
    value: DetailedRequestData;
    onChange: (changes: Partial<DetailedRequestData>) => void;
    onGenerateLogo: () => Promise<void>;
    onReinventLogo: () => Promise<void>;
    isGeneratingLogo: boolean;
    isReinventingLogo: boolean;
    logoError: string | null;
}

const options: { value: LogoOption, label: string, description: string }[] = [
    { value: 'use_from_photo', label: 'Usar Logo da Foto', description: 'A IA vai identificar e usar o logo já existente na sua fachada.' },
    { value: 'reinvent_from_photo', label: 'Reinventar Logo da Foto', description: 'A IA vai modernizar o logo existente na sua fachada.' },
    { value: 'generate', label: 'Gerar Logo com IA', description: 'Descreva uma ideia e a IA criará um novo logo para você.' },
    { value: 'upload', label: 'Carregar meu Logo', description: 'Envie um arquivo de imagem do seu logo (PNG, JPG).' },
];

export const LogoOptions: React.FC<LogoOptionsProps> = ({ 
    value, 
    onChange,
    onGenerateLogo,
    onReinventLogo,
    isGeneratingLogo,
    isReinventingLogo,
    logoError,
}) => {
    
    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const base64 = event.target?.result as string;
            // Removed slow background removal step. Upload is now instant.
            onChange({ logoFile: { base64, prompt: file.name } });
        };
        reader.readAsDataURL(file);
    };
    
    const handleDownload = () => {
        if (!value.logoFile) return;
        const link = document.createElement('a');
        link.href = value.logoFile.base64;
        link.download = `logo_${value.logoFile.prompt.substring(0, 15).replace(/\s/g, '_') || 'gerado'}.png`;
        link.click();
    }
    
    const showCompanyNameInput = value.logoOption === 'use_from_photo' || value.logoOption === 'reinvent_from_photo';

    return (
        <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">Logo</h3>
            <div className="grid sm:grid-cols-2 gap-3 mb-4">
                {options.map(option => (
                    <label key={option.value} className={`flex items-start p-3 rounded-lg border-2 cursor-pointer transition ${value.logoOption === option.value ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500'}`}>
                        <input
                            type="radio"
                            name="logoOption"
                            value={option.value}
                            checked={value.logoOption === option.value}
                            onChange={() => onChange({ logoOption: option.value, logoFile: null })} // Reset logo file on option change
                            className="h-5 w-5 mt-0.5 text-cyan-600 focus:ring-cyan-500 border-gray-300 dark:border-gray-600 bg-transparent"
                        />
                        <div className="ml-3 text-sm">
                            <span className="font-bold text-gray-900 dark:text-gray-100">{option.label}</span>
                            <p className="text-gray-500 dark:text-gray-400">{option.description}</p>
                        </div>
                    </label>
                ))}
            </div>
            
             {showCompanyNameInput && (
                <div className="mb-4 animate-fade-in flex gap-2">
                    <input 
                        type="text" 
                        value={value.companyName ?? ''} 
                        onChange={e => onChange({ companyName: e.target.value })} 
                        placeholder="Digite o nome da empresa aqui" 
                        className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" 
                    />
                    {value.logoOption === 'reinvent_from_photo' && (
                         <button onClick={onReinventLogo} disabled={isReinventingLogo || !value.companyName} className="px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md disabled:bg-gray-500">
                            {isReinventingLogo ? '...' : 'Reinventar'}
                        </button>
                    )}
                </div>
            )}

            {value.logoFile && (
                <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-800/50 text-center relative">
                    <img src={value.logoFile.base64} alt="Logo preview" className="max-h-24 mx-auto object-contain"/>
                    <p className="text-xs text-gray-500 mt-2 truncate">{value.logoFile.prompt}</p>
                    <div className="absolute top-2 right-2 flex space-x-1">
                        <button onClick={handleDownload} className="p-1.5 bg-blue-500 text-white rounded-full hover:bg-blue-600" title="Baixar Logo">
                           <DownloadIcon className="w-4 h-4" />
                        </button>
                        <button onClick={() => onChange({ logoFile: null })} className="p-1.5 bg-red-500 text-white rounded-full hover:bg-red-600" title="Remover Logo">
                            <XIcon className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            )}

            {!value.logoFile && value.logoOption === 'generate' && (
                <div className="flex gap-2 animate-fade-in">
                    <input type="text" value={value.logoPrompt} onChange={e => onChange({ logoPrompt: e.target.value })} placeholder="Descreva o logo a ser criado" className="flex-grow p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                    <button onClick={onGenerateLogo} disabled={isGeneratingLogo || !value.logoPrompt} className="px-4 py-2 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-md disabled:bg-gray-500">
                        {isGeneratingLogo ? '...' : 'Gerar'}
                    </button>
                </div>
            )}
            
            {!value.logoFile && value.logoOption === 'upload' && (
                <label htmlFor="logo-upload-main" className="w-full flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition animate-fade-in">
                    <UploadCloudIcon />
                    <span className="text-sm">Carregar arquivo de logo</span>
                    <input id="logo-upload-main" type="file" className="hidden" accept="image/png, image/jpeg, image/svg+xml" onChange={handleUpload}/>
                </label>
            )}

            {logoError && <p className="text-sm text-red-500 text-center mt-2">{logoError}</p>}
        </div>
    );
};

export default LogoOptions;