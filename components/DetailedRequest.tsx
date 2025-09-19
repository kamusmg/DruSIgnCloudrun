

import React, { useState, useEffect, useCallback } from 'react';
// Fix: Added StickerDetail import to use in type guards.
import { DetailedRequestData, StickerDetail, LightingOption, TotemSize, TotemFeature, BannerFaixaDetails, VehicleDetails } from '../types.ts';
import { generateInspirationText } from '../utils/textHelper.ts';
import { extractDominantColors } from '../utils/colors.ts';
import { generateLogo, reinventLogo, generatePattern } from '../services/geminiService.ts';

// Component Imports
import { StepCard } from './StepCard.tsx';
import { ElementSelector, elementColors } from './ElementSelector.tsx';
import MainSignDetails from './MainSignDetails.tsx';
import LogoOptions from './LogoOptions.tsx';
import { InspirationGallery } from './InspirationGallery.tsx';
// Fix: Updated the component import to reflect the renamed component, resolving a name collision.
import { StickerDetailsComponent } from './StickerDetails.tsx';
import { SunIcon, MoonIcon, UploadCloudIcon, XIcon } from './Icons.tsx';

// --- ACM Color Palettes Data ---
const acmPalettes = [
  {
    category: 'Sólidas / Foscas (Matte)',
    colors: [
      { name: 'Preto Fosco', value: '#2d2d2d' },
      { name: 'Branco Fosco', value: '#f5f5f5' },
      { name: 'Cinza Escuro Fosco', value: '#555555' },
    ],
  },
  {
    category: 'Brilhantes (Glossy)',
    colors: [
      { name: 'Preto Brilhante', value: '#1C1C1C' },
      { name: 'Branco Brilhante', value: '#FFFFFF' },
      { name: 'Vermelho Vivo Brilhante', value: '#d90429' },
      { name: 'Amarelo Ouro Brilhante', value: '#ffc300' },
      { name: 'Azul Royal Brilhante', value: '#00509d' },
      { name: 'Verde Bandeira Brilhante', value: '#008000' },
    ],
  },
  {
    category: 'Metálicas / Especiais',
    colors: [
      { name: 'Prata Metálico', value: '#c0c0c0' },
      { name: 'Champanhe Metálico', value: '#f7e7ce' },
      { name: 'Dourado Metálico', value: '#daa520' },
      { name: 'Cobre Metálico', value: '#b87333' },
      { name: 'Bronze Metálico', value: '#cd7f32' },
      { name: 'Aço Escovado', value: '#a9a9a9' },
    ],
  },
  {
    category: 'Amadeiradas (Wood Finish)',
    colors: [
      { name: 'ACM Amadeirado Imbuia', value: '#5c3a21' },
      { name: 'ACM Amadeirado Carvalho', value: '#a07a58' },
      { name: 'ACM Amadeirado Nogueira', value: '#6f4f28' },
    ],
  },
];

const acmColorMap = acmPalettes.flatMap(p => p.colors).reduce((acc, color) => {
  acc[color.name] = color.value;
  return acc;
}, {} as Record<string, string>);

const getAcmColorValue = (colorNameOrHex?: string): string => {
  if (!colorNameOrHex) return '#888888';
  if (colorNameOrHex.startsWith('#')) return colorNameOrHex;
  return acmColorMap[colorNameOrHex] || colorNameOrHex;
};

// --- Sub-components ---
const ColorSwatch: React.FC<{ name: string; color: string; isSelected: boolean; onClick: () => void; }> = ({ name, color, isSelected, onClick }) => (
    <div className="flex flex-col items-center gap-1.5 w-24 text-center">
        <button
            type="button"
            onClick={onClick}
            className={`w-10 h-10 rounded-full border-2 transition ${isSelected ? 'border-cyan-500 ring-2 ring-offset-2 ring-offset-gray-100 dark:ring-offset-gray-800 ring-cyan-500' : 'border-white/20 hover:border-white'}`}
            style={{ 
                backgroundColor: color,
                backgroundImage: name.toLowerCase().includes('metálico') || name.toLowerCase().includes('escovado') 
                    ? `linear-gradient(45deg, rgba(255,255,255,0.4), rgba(0,0,0,0.1))`
                    : 'none'
            }}
            aria-label={`Select color ${name}`}
        />
        <span className="text-xs text-gray-600 dark:text-gray-400">{name}</span>
    </div>
);

const AcmColorSelector: React.FC<{
    logoColors: string[];
    value: string | undefined;
    onChange: (color: string, colorName?: string) => void;
    onClose: () => void;
}> = ({ logoColors, value, onChange, onClose }) => {
    
    const handleSwatchClick = (colorValue: string, colorName: string) => {
        onChange(colorValue, colorName);
        onClose();
    };
    
    const handleCustomColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newColor = e.target.value;
        onChange(newColor, newColor);
    };

    return (
        <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-6 mt-2">
            {logoColors.length > 0 && (
                <div>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block">Cores da Logo</label>
                    <div className="flex flex-wrap gap-3">
                        {logoColors.map(color => (
                            <ColorSwatch key={`logo-${color}`} name={color} color={color} isSelected={value === color} onClick={() => handleSwatchClick(color, color)} />
                        ))}
                    </div>
                </div>
            )}

            {acmPalettes.map(palette => (
                <div key={palette.category}>
                    <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 block">{palette.category}</label>
                    <div className="flex flex-wrap gap-3">
                         {palette.colors.map(colorInfo => (
                            <ColorSwatch key={colorInfo.name} name={colorInfo.name} color={colorInfo.value} isSelected={value === colorInfo.name} onClick={() => handleSwatchClick(colorInfo.value, colorInfo.name)} />
                        ))}
                    </div>
                </div>
            ))}
            
             <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 block">Cor Personalizada</label>
                <div className="flex items-center gap-2">
                     <input
                        type="color"
                        value={value?.startsWith('#') ? value : '#555555'}
                        onChange={handleCustomColorChange}
                        className="p-0 h-10 w-12 rounded-md border-2 border-transparent bg-transparent cursor-pointer"
                     />
                    <input
                        type="text"
                        value={value || ''}
                        onChange={e => onChange(e.target.value, e.target.value)}
                        placeholder="#C0C0C0"
                        className="p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500 w-48"
                     />
                </div>
            </div>
        </div>
    );
};

const SimpleColorSelector: React.FC<{ value: string | undefined; onChange: (color: string) => void; onClose: () => void; }> = ({ value, onChange, onClose }) => {
    const handleColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onChange(e.target.value);
    };
    
    const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onChange(e.target.value);
    };

    return (
        <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg animate-fade-in">
            <div className="flex items-center gap-2">
                <input
                    type="color"
                    value={value || '#FFFFFF'}
                    onChange={handleColorChange}
                    className="p-0 h-10 w-12 rounded-md border-2 border-transparent bg-transparent cursor-pointer"
                />
                <input
                    type="text"
                    value={value || ''}
                    onChange={handleTextChange}
                    placeholder="#FFFFFF"
                    className="p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500 w-48"
                />
            </div>
        </div>
    );
};

const RadioOption: React.FC<{ name: string; value: string; label: string | React.ReactNode; checked: boolean; onChange: () => void; }> = ({ name, value, label, checked, onChange }) => (
    <label className={`flex-1 flex items-center p-3 rounded-lg border-2 cursor-pointer transition ${checked ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500'}`}>
        <input
            type="radio"
            name={name}
            value={value}
            checked={checked}
            onChange={onChange}
            className="h-5 w-5 text-cyan-600 focus:ring-cyan-500 border-gray-300 dark:border-gray-600 bg-transparent"
        />
        <span className="ml-3 text-sm font-bold text-gray-900 dark:text-gray-100">{label}</span>
    </label>
);

const CheckboxOption: React.FC<{ label: string; checked: boolean; onChange: () => void; onFocus: () => void; onBlur: () => void; }> = ({ label, checked, onChange, onFocus, onBlur }) => (
    <label onFocus={onFocus} onBlur={onBlur} className="flex items-center space-x-2 px-3 py-1.5 bg-gray-200 dark:bg-gray-700/60 rounded-lg cursor-pointer transition hover:bg-gray-300 dark:hover:bg-gray-600 has-[:checked]:bg-cyan-500/10 has-[:checked]:ring-1 has-[:checked]:ring-cyan-500">
        <input
            type="checkbox"
            checked={checked}
            onChange={onChange}
            className="h-4 w-4 rounded border-gray-300 text-cyan-600 focus:ring-cyan-500 bg-transparent"
        />
        <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{label}</span>
    </label>
);

interface DetailedRequestProps {
    requestData: DetailedRequestData;
    onDataChange: (changes: Partial<DetailedRequestData>) => void;
    onSubmit: (useEnhancedPrompt: boolean, data: DetailedRequestData) => Promise<void>;
    onBack: () => void;
    originalImage: string;
    isEditingGeneratedImage: boolean;
}

const acmPlacementOptions = [
    'Fachada Completa (Tudo)',
    'Fachada (Abaixo da Placa)',
    'Moldura da Placa',
    'Detalhes / Faixas',
    'Colunas / Pilares',
];

const lightingOptionsList: { id: LightingOption, label: string }[] = [
    { id: 'spots', label: 'Spots de Destaque' },
    { id: 'uplighting', label: 'Iluminação de Piso' },
    { id: 'contour_led', label: 'Fita de LED de Contorno' }
];

const totemSizeOptions: { id: TotemSize, label: string }[] = [
    { id: 'small', label: 'Pequeno (~1.0m)' },
    { id: 'medium', label: 'Médio (~2.2m)' },
    { id: 'large', label: 'Grande (+3m)' },
];

const totemFeatureOptions: { id: TotemFeature, label: string }[] = [
    { id: 'luminous', label: 'Luminoso' },
    { id: 'channel_letter', label: 'Letra Caixa' },
];

const bannerPresetSizes = ['80x120cm', '60x90cm', '90x120cm', '100x150cm'];
const faixaPresetSizes = ['60x200cm', '60x300cm', '100x50cm', '200x100cm'];
const vehicleTypeOptions = ['Van de Carga', 'Carro Popular', 'Moto de Entrega', 'Caminhonete / SUV'];


export const DetailedRequest: React.FC<DetailedRequestProps> = ({ requestData, onDataChange, onSubmit, onBack, originalImage, isEditingGeneratedImage }) => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [logoColors, setLogoColors] = useState<string[]>([]);
    const [isAcmSelectorOpen, setIsAcmSelectorOpen] = useState(true);
    const [isPaintSelectorOpen, setIsPaintSelectorOpen] = useState(false);
    const [isLetterSelectorOpen, setIsLetterSelectorOpen] = useState(false);
    const [isGeneratingLogo, setIsGeneratingLogo] = useState(false);
    const [isReinventingLogo, setIsReinventingLogo] = useState(false);
    const [logoError, setLogoError] = useState<string | null>(null);
    const [activeDetailElement, setActiveDetailElement] = useState<string | null>(null);

    const extractColorsFromImage = useCallback(async (imgSrc: string) => {
        return new Promise<string[]>((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = async () => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                if (!ctx) return reject([]);
                ctx.drawImage(img, 0, 0);
                const colors = await extractDominantColors(ctx, 3);
                resolve(colors);
            };
            img.onerror = () => reject([]);
            img.src = imgSrc;
        });
    }, []);

    useEffect(() => {
        if (requestData.logoFile?.base64) {
            extractColorsFromImage(requestData.logoFile.base64)
                .then(setLogoColors)
                .catch(() => setLogoColors([]));
        } else {
            setLogoColors([]);
        }
    }, [requestData.logoFile, extractColorsFromImage]);

    
    useEffect(() => {
        if (requestData.mainSignType === 'placa_iluminada' && requestData.ambiance !== 'night') {
            onDataChange({ ambiance: 'night' });
        }
    }, [requestData.mainSignType, requestData.ambiance, onDataChange]);

    useEffect(() => {
        const updateInspiration = () => {
            const generatedText = generateInspirationText(requestData, isEditingGeneratedImage);
            if (requestData.inspiration !== generatedText) {
                onDataChange({ inspiration: generatedText });
            }
        };
        
        updateInspiration();

    }, [requestData, onDataChange, isEditingGeneratedImage]);

    const handleElementsChange = (newElements: string[]) => {
        const oldElements = requestData.elements;
        const justAddedAcm = !oldElements.includes('Revestimento em ACM') && newElements.includes('Revestimento em ACM');
        
        const changes: Partial<DetailedRequestData> = { elements: newElements };

        if (justAddedAcm) {
            // Default to 'Fachada Completa (Tudo)' when ACM is first selected.
            changes.acmPlacement = {
                ...requestData.acmPlacement,
                selections: ['Fachada Completa (Tudo)'],
            };
        }

        onDataChange(changes);
    };

    const handleAcmColorChange = (colorValue: string, colorName?: string) => {
        onDataChange({ acmColor: colorValue, acmColorName: colorName || colorValue });
    };

    const handleAcmPlacementChange = (placement: string) => {
        const currentSelections = requestData.acmPlacement?.selections || [];
        const newSelections = currentSelections.includes(placement)
            ? currentSelections.filter(p => p !== placement)
            : [...currentSelections, placement];
        onDataChange({ acmPlacement: { ...requestData.acmPlacement, selections: newSelections } });
    };

    const handleAcmCustomPlacementChange = (customText: string) => {
        onDataChange({ acmPlacement: { ...requestData.acmPlacement, custom: customText } });
    };
    
    const handleLightingOptionChange = (option: LightingOption) => {
        const currentOptions = requestData.lightingOptions || [];
        const newOptions = currentOptions.includes(option)
            ? currentOptions.filter(o => o !== option)
            : [...currentOptions, option];
        onDataChange({ lightingOptions: newOptions });
    };

    const handlePaintColorChange = (color: string) => {
        onDataChange({ paintColor: color });
        setIsPaintSelectorOpen(false);
    };
    
    const handleAdditionalTextChange = (field: 'text' | 'location', value: string) => {
        onDataChange({
            additionalText: {
                ...requestData.additionalText,
                [field]: value,
            },
        });
    };
    
    const handleTotemDetailsChange = (changes: Partial<typeof requestData.totemDetails>) => {
        onDataChange({
            totemDetails: {
                ...requestData.totemDetails,
                ...changes,
            },
        });
    };

    const handleBannerFaixaDetailsChange = (changes: Partial<BannerFaixaDetails>) => {
        const newDetails = { ...requestData.bannerFaixaDetails, ...changes };
        // Reset preset size if type changes
        if (changes.type) {
            newDetails.presetSize = changes.type === 'banner' ? bannerPresetSizes[0] : faixaPresetSizes[0];
        }
        onDataChange({ bannerFaixaDetails: newDetails });
    };
    
    const handleArtFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const base64 = event.target?.result as string;
            handleBannerFaixaDetailsChange({ artFile: { base64, prompt: file.name } });
        };
        reader.readAsDataURL(file);
    };

    const handleTotemFeatureChange = (feature: TotemFeature) => {
        const currentFeatures = requestData.totemDetails.features || [];
        const newFeatures = currentFeatures.includes(feature)
            ? currentFeatures.filter(f => f !== feature)
            : [...currentFeatures, feature];
        handleTotemDetailsChange({ features: newFeatures });
    };
    
    const handleVehicleTypeChange = (type: string) => {
        const currentTypes = requestData.vehicleDetails.selectedTypes || [];
        const newTypes = currentTypes.includes(type)
            ? currentTypes.filter(t => t !== type)
            : [...currentTypes, type];
        onDataChange({ vehicleDetails: { ...requestData.vehicleDetails, selectedTypes: newTypes } });
    };
    
    const handleCustomVehicleNameChange = (name: string) => {
        onDataChange({ vehicleDetails: { ...requestData.vehicleDetails, customName: name } });
    };


    const handleLetterColorChange = (color: string) => {
        onDataChange({ channelLetterColor: color });
        setIsLetterSelectorOpen(false);
    };

    const handleGenerateLogo = async () => {
        if (!requestData.logoPrompt) return;
        setIsGeneratingLogo(true);
        setLogoError(null);
        try {
            const newLogo = await generateLogo(requestData.logoPrompt);
            onDataChange({ logoFile: newLogo });
        } catch (err) {
            console.error("Failed to generate logo:", err);
            setLogoError("Falha ao gerar o logo. Tente novamente.");
        } finally {
            setIsGeneratingLogo(false);
        }
    };

    const handleReinventLogo = async () => {
        if (!requestData.companyName || !originalImage) return;
        setIsReinventingLogo(true);
        setLogoError(null);
        try {
            const newLogo = await reinventLogo(originalImage, requestData.companyName);
            onDataChange({ logoFile: newLogo });
        } catch (err) {
            console.error("Failed to reinvent logo:", err);
            setLogoError("Falha ao reinventar o logo. Verifique o nome da empresa e tente novamente.");
        } finally {
            setIsReinventingLogo(false);
        }
    };

    const handleAddCustomInstruction = (instruction: string) => {
        onDataChange({
            customInstructions: [...requestData.customInstructions, instruction],
        });
    };

    const handleRemoveCustomInstruction = (index: number) => {
        onDataChange({
            customInstructions: requestData.customInstructions.filter((_, i) => i !== index),
        });
    };


    const handleSubmit = async (useEnhancedPrompt: boolean) => {
        setIsSubmitting(true);
        setLogoError(null);
        
        try {
            let dataForSubmission = { ...requestData };

            // Explicitly ensure logoFile is null when using the photo's logo
            if (dataForSubmission.logoOption === 'use_from_photo') {
                dataForSubmission.logoFile = null;
            }
            
            // "Smart" submit: if user didn't generate logo first, do it for them.
            if (dataForSubmission.logoOption === 'generate' && !dataForSubmission.logoFile && dataForSubmission.logoPrompt) {
                const newLogo = await generateLogo(dataForSubmission.logoPrompt);
                dataForSubmission = { ...dataForSubmission, logoFile: newLogo };
            } else if (dataForSubmission.logoOption === 'reinvent_from_photo' && !dataForSubmission.logoFile && dataForSubmission.companyName) {
                const newLogo = await reinventLogo(originalImage, dataForSubmission.companyName);
                dataForSubmission = { ...dataForSubmission, logoFile: newLogo };
            }

            // Call the main submission handler from App.tsx with the final, complete data
            await onSubmit(useEnhancedPrompt, dataForSubmission);

        } catch (error) {
            console.error("Error during pre-submission or submission:", error);
            setLogoError("Houve um erro ao preparar o logo ou o padrão de adesivo. Por favor, tente gerá-lo novamente antes de continuar.");
            setIsSubmitting(false); // Only set to false on error
        }
    };

    // UI Visibility Conditions
    const showMainSignDetailsSection = requestData.elements.includes('Placa Principal');
    const showAcmColorSelector = requestData.elements.includes('Revestimento em ACM') || (showMainSignDetailsSection && (requestData.mainSignType.includes('acm') || requestData.mainSignType === 'placa_iluminada'));
    const showAcmPlacementSelector = requestData.elements.includes('Revestimento em ACM');
    const showPaintSelector = requestData.elements.includes('Pintura Nova');
    const showLightingDetails = requestData.elements.includes('Iluminação Especial');
    const showTotemDetails = requestData.elements.includes('Totem');
    const showBannerFaixaDetails = requestData.elements.includes('Banner ou Faixa');
    const showInformationalSigns = requestData.elements.includes('Placas Informativas');
    const showVehicleDetails = requestData.elements.includes('Veículos');
    const showAdditionalTextInput = requestData.elements.includes('Texto Adicional');
    const showMainSignLetterColorSelector = showMainSignDetailsSection && requestData.mainSignType === 'acm_letra_caixa';
    const showStickerDetails = requestData.elements.includes('Adesivos');


    return (
        <div className="space-y-6">
            {/* Step 1: Photo Preview */}
            <StepCard stepNumber={1} title="Foto da Fachada">
                <div className="flex items-center justify-between">
                    <img src={originalImage} alt="Fachada original" className="w-24 h-16 object-cover rounded-md" />
                    <button onClick={onBack} className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Trocar Foto</button>
                </div>
            </StepCard>

            {/* Step 2: Project Details */}
            <StepCard stepNumber={2} title="Detalhes do Projeto" highlight>
                 <div className="space-y-8 p-2">
                    <ElementSelector 
                        selectedElements={requestData.elements} 
                        onChange={handleElementsChange}
                        activeElement={activeDetailElement}
                    />
                    
                    {/* Logically Reordered Detail Sections */}
                    <div className="space-y-4">
                        {showAdditionalTextInput && (
                             <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Texto Adicional"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes do Texto Adicional</h3>
                                <input 
                                    type="text" 
                                    value={requestData.additionalText.text || ''} 
                                    onChange={(e) => handleAdditionalTextChange('text', e.target.value)} 
                                    onFocus={() => setActiveDetailElement("Texto Adicional")}
                                    onBlur={() => setActiveDetailElement(null)}
                                    placeholder="Texto (Ex: WhatsApp, Endereço)" 
                                    className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" 
                                />
                                <input 
                                    type="text" 
                                    value={requestData.additionalText.location || ''} 
                                    onChange={(e) => handleAdditionalTextChange('location', e.target.value)}
                                    onFocus={() => setActiveDetailElement("Texto Adicional")}
                                    onBlur={() => setActiveDetailElement(null)}
                                    placeholder="Local Sugerido (Opcional, ex: na porta de vidro)" 
                                    className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500"
                                />
                            </div>
                        )}

                        {showStickerDetails && (
                            <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Adesivos"].main}`}>
                                <StickerDetailsComponent
                                    value={requestData.stickerDetails}
                                    onChange={(stickerDetails) => onDataChange({ stickerDetails })}
                                    onFocus={() => setActiveDetailElement("Adesivos")}
                                    onBlur={() => setActiveDetailElement(null)}
                                />
                            </div>
                        )}

                        {showAcmPlacementSelector && (
                            <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Revestimento em ACM"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes do Revestimento de ACM</h3>
                                <p className="text-sm text-gray-600 dark:text-gray-400">Especifique onde o revestimento deve ser aplicado.</p>
                                <div className="flex flex-wrap gap-2">
                                    {acmPlacementOptions.map(option => (
                                        <CheckboxOption
                                            key={option}
                                            label={option}
                                            checked={requestData.acmPlacement?.selections.includes(option)}
                                            onChange={() => handleAcmPlacementChange(option)}
                                            onFocus={() => setActiveDetailElement("Revestimento em ACM")}
                                            onBlur={() => setActiveDetailElement(null)}
                                        />
                                    ))}
                                </div>
                                <input
                                    type="text"
                                    value={requestData.acmPlacement?.custom || ''}
                                    onChange={(e) => handleAcmCustomPlacementChange(e.target.value)}
                                    onFocus={() => setActiveDetailElement("Revestimento em ACM")}
                                    onBlur={() => setActiveDetailElement(null)}
                                    placeholder="Outras áreas (especificar)"
                                    className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500"
                                />
                            </div>
                        )}
                        
                        {showPaintSelector && (
                            <div className={`animate-fade-in space-y-2 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Pintura Nova"].main}`}>
                                <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Cor da Pintura Nova</h3>
                                 <div onFocus={() => setActiveDetailElement("Pintura Nova")} onBlur={() => setActiveDetailElement(null)}>
                                    <SimpleColorSelector value={requestData.paintColor} onChange={handlePaintColorChange} onClose={() => setIsPaintSelectorOpen(false)} />
                                </div>
                            </div>
                        )}

                        {showLightingDetails && (
                             <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Iluminação Especial"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes da Iluminação Especial</h3>
                                <p className="text-sm text-gray-600 dark:text-gray-400">Selecione os tipos de iluminação para adicionar.</p>
                                <div className="flex flex-wrap gap-2">
                                    {lightingOptionsList.map(option => (
                                        <CheckboxOption
                                            key={option.id}
                                            label={option.label}
                                            checked={requestData.lightingOptions?.includes(option.id)}
                                            onChange={() => handleLightingOptionChange(option.id)}
                                            onFocus={() => setActiveDetailElement("Iluminação Especial")}
                                            onBlur={() => setActiveDetailElement(null)}
                                        />
                                    ))}
                                </div>
                            </div>
                        )}
                        
                        {showTotemDetails && (
                            <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Totem"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes do Totem</h3>
                                <div onFocus={() => setActiveDetailElement("Totem")} onBlur={() => setActiveDetailElement(null)}>
                                    <div>
                                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Tamanho do Totem</p>
                                        <div className="flex gap-2">
                                            {totemSizeOptions.map(option => (
                                                <RadioOption
                                                    key={option.id}
                                                    name="totemSize"
                                                    value={option.id}
                                                    label={option.label}
                                                    checked={requestData.totemDetails.size === option.id}
                                                    onChange={() => handleTotemDetailsChange({ size: option.id })}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 mt-3">Recursos</p>
                                        <div className="flex flex-wrap gap-2">
                                            {totemFeatureOptions.map(option => (
                                                <CheckboxOption
                                                    key={option.id}
                                                    label={option.label}
                                                    checked={requestData.totemDetails.features?.includes(option.id)}
                                                    onChange={() => handleTotemFeatureChange(option.id)}
                                                    onFocus={() => setActiveDetailElement("Totem")}
                                                    onBlur={() => setActiveDetailElement(null)}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {showBannerFaixaDetails && (
                            <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Banner ou Faixa"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes do Banner ou Faixa</h3>
                                <div onFocus={() => setActiveDetailElement("Banner ou Faixa")} onBlur={() => setActiveDetailElement(null)} className="space-y-4">
                                    {/* Type Selection */}
                                    <div className="flex gap-2">
                                        <RadioOption name="bannerFaixaType" value="banner" label="Banner (Vertical)" checked={requestData.bannerFaixaDetails.type === 'banner'} onChange={() => handleBannerFaixaDetailsChange({ type: 'banner' })} />
                                        <RadioOption name="bannerFaixaType" value="faixa" label="Faixa (Horizontal)" checked={requestData.bannerFaixaDetails.type === 'faixa'} onChange={() => handleBannerFaixaDetailsChange({ type: 'faixa' })} />
                                    </div>
                                    
                                    {/* Size Selection */}
                                    <div className="flex gap-2">
                                        <RadioOption name="bannerFaixaSizeType" value="preset" label="Tamanho Padrão" checked={requestData.bannerFaixaDetails.sizeType === 'preset'} onChange={() => handleBannerFaixaDetailsChange({ sizeType: 'preset' })} />
                                        <RadioOption name="bannerFaixaSizeType" value="custom" label="Personalizado" checked={requestData.bannerFaixaDetails.sizeType === 'custom'} onChange={() => handleBannerFaixaDetailsChange({ sizeType: 'custom' })} />
                                    </div>
                                    
                                    {requestData.bannerFaixaDetails.sizeType === 'preset' && (
                                        <select value={requestData.bannerFaixaDetails.presetSize} onChange={e => handleBannerFaixaDetailsChange({ presetSize: e.target.value })} className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500">
                                            {(requestData.bannerFaixaDetails.type === 'banner' ? bannerPresetSizes : faixaPresetSizes).map(size => (
                                                <option key={size} value={size}>{size}</option>
                                            ))}
                                        </select>
                                    )}

                                    {requestData.bannerFaixaDetails.sizeType === 'custom' && (
                                        <div className="flex gap-2">
                                            <input type="text" value={requestData.bannerFaixaDetails.customWidth} onChange={e => handleBannerFaixaDetailsChange({ customWidth: e.target.value })} placeholder="Largura (cm)" className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                                            <input type="text" value={requestData.bannerFaixaDetails.customHeight} onChange={e => handleBannerFaixaDetailsChange({ customHeight: e.target.value })} placeholder="Altura (cm)" className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" />
                                        </div>
                                    )}

                                    {/* Art Selection */}
                                    <div className="space-y-2">
                                        <div className="flex gap-2">
                                            <RadioOption name="bannerFaixaArtType" value="ai_generic" label="Arte Genérica (IA)" checked={requestData.bannerFaixaDetails.artType === 'ai_generic'} onChange={() => handleBannerFaixaDetailsChange({ artType: 'ai_generic', artFile: null })} />
                                            <RadioOption name="bannerFaixaArtType" value="ai_prompt" label="Descrever Arte (IA)" checked={requestData.bannerFaixaDetails.artType === 'ai_prompt'} onChange={() => handleBannerFaixaDetailsChange({ artType: 'ai_prompt', artFile: null })} />
                                        </div>
                                         <RadioOption name="bannerFaixaArtType" value="upload" label="Carregar Arte Pronta" checked={requestData.bannerFaixaDetails.artType === 'upload'} onChange={() => handleBannerFaixaDetailsChange({ artType: 'upload' })} />
                                    </div>

                                    {requestData.bannerFaixaDetails.artType === 'ai_prompt' && (
                                        <textarea value={requestData.bannerFaixaDetails.artPrompt} onChange={e => handleBannerFaixaDetailsChange({ artPrompt: e.target.value })} placeholder="Descreva a arte a ser gerada..." className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500" rows={3}></textarea>
                                    )}

                                    {requestData.bannerFaixaDetails.artType === 'upload' && (
                                        <div>
                                            {requestData.bannerFaixaDetails.artFile ? (
                                                <div className="p-2 bg-gray-200 dark:bg-gray-700 rounded-md relative text-center">
                                                    <img src={requestData.bannerFaixaDetails.artFile.base64} alt="Preview da arte" className="max-h-20 mx-auto object-contain"/>
                                                    <button onClick={() => handleBannerFaixaDetailsChange({ artFile: null })} className="absolute top-1 right-1 p-1 bg-red-500 text-white rounded-full hover:bg-red-600" title="Remover Arte">
                                                        <XIcon className="w-3 h-3"/>
                                                    </button>
                                                </div>
                                            ) : (
                                                <label className="w-full flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-700 transition">
                                                    <UploadCloudIcon />
                                                    <span className="text-sm">Carregar arquivo da arte</span>
                                                    <input type="file" className="hidden" accept="image/png, image/jpeg" onChange={handleArtFileUpload}/>
                                                </label>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        
                        {showInformationalSigns && (
                             <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Placas Informativas"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes das Placas Informativas</h3>
                                <textarea
                                    value={requestData.informationalSignsText || ''}
                                    onChange={(e) => onDataChange({ informationalSignsText: e.target.value })}
                                    onFocus={() => setActiveDetailElement("Placas Informativas")}
                                    onBlur={() => setActiveDetailElement(null)}
                                    placeholder={`Exemplo 1: Horário: Seg a Sex - 08h às 18h\nExemplo 2: Wi-Fi Grátis para Clientes`}
                                    rows={4}
                                    className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500"
                                />
                            </div>
                        )}

                        {showVehicleDetails && (
                            <div className={`animate-fade-in space-y-3 p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg border-l-4 ${elementColors["Veículos"].main}`}>
                                <h3 className="text-md font-semibold text-gray-800 dark:text-gray-200">Detalhes dos Veículos</h3>
                                <div onFocus={() => setActiveDetailElement("Veículos")} onBlur={() => setActiveDetailElement(null)}>
                                    <div className="space-y-2">
                                        <p className="text-sm text-gray-600 dark:text-gray-400">Selecione tipos comuns ou especifique um modelo.</p>
                                        <div className="flex flex-wrap gap-2">
                                            {vehicleTypeOptions.map(option => (
                                                <CheckboxOption
                                                    key={option}
                                                    label={option}
                                                    checked={requestData.vehicleDetails.selectedTypes?.includes(option)}
                                                    onChange={() => handleVehicleTypeChange(option)}
                                                    onFocus={() => setActiveDetailElement("Veículos")}
                                                    onBlur={() => setActiveDetailElement(null)}
                                                />
                                            ))}
                                        </div>
                                        <input
                                            type="text"
                                            value={requestData.vehicleDetails.customName || ''}
                                            onChange={(e) => handleCustomVehicleNameChange(e.target.value)}
                                            placeholder="Nome do Veículo (Opcional, ex: Fiat Uno)"
                                            className="w-full p-2.5 rounded-md border-2 transition border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:border-cyan-500 focus:ring-cyan-500"
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                    {/* End of Reordered Sections */}

                    <div>
                        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">Ambientação</h3>
                        <div className="flex gap-3">
                            <RadioOption name="ambiance" value="day" label={<div className="flex items-center gap-2"><SunIcon/> Dia</div>} checked={requestData.ambiance === 'day'} onChange={() => onDataChange({ ambiance: 'day' })} />
                            <RadioOption name="ambiance" value="night" label={<div className="flex items-center gap-2"><MoonIcon/> Noite</div>} checked={requestData.ambiance === 'night'} onChange={() => onDataChange({ ambiance: 'night' })} />
                        </div>
                    </div>
                    
                    {showAcmColorSelector && (
                        <div className="animate-fade-in space-y-2" onFocus={() => setActiveDetailElement("Revestimento em ACM")} onBlur={() => setActiveDetailElement(null)}>
                            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Cor do Revestimento/Placa de ACM</h3>
                            <button type="button" onClick={() => setIsAcmSelectorOpen(!isAcmSelectorOpen)} className="w-full flex items-center justify-between p-3 bg-gray-100 dark:bg-gray-800/50 rounded-lg text-left transition hover:bg-gray-200 dark:hover:bg-gray-700/60" aria-expanded={isAcmSelectorOpen}>
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full border-2 border-white/20 shadow-inner" style={{ backgroundColor: getAcmColorValue(requestData.acmColor) }}></div>
                                    <span className="font-medium text-gray-800 dark:text-gray-200">{requestData.acmColorName || requestData.acmColor || 'Clique para selecionar'}</span>
                                </div>
                                <svg xmlns="http://www.w3.org/2000/svg" className={`w-5 h-5 text-gray-500 dark:text-gray-400 transition-transform ${isAcmSelectorOpen ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
                            </button>
                            {isAcmSelectorOpen && <AcmColorSelector logoColors={logoColors} value={requestData.acmColor} onChange={handleAcmColorChange} onClose={() => setIsAcmSelectorOpen(false)} />}
                        </div>
                    )}

                    {showMainSignDetailsSection && (
                        <div className="space-y-4 animate-fade-in">
                            <MainSignDetails
                                value={requestData.mainSignType}
                                onChange={(mainSignType) => onDataChange({ mainSignType })}
                                lonaInstallationType={requestData.lonaInstallationType}
                                onLonaInstallationChange={(lonaInstallationType) => onDataChange({ lonaInstallationType })}
                            />
                        
                            <div className="space-y-4">
                                {showMainSignLetterColorSelector && (
                                    <div className="animate-fade-in space-y-2" onFocus={() => setActiveDetailElement("Placa Principal")} onBlur={() => setActiveDetailElement(null)}>
                                        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Cor do Logo em Letra Caixa</h3>
                                        <div className="flex gap-3">
                                            <RadioOption name="channelLetterColorMode" value="original" label="Manter Cores Originais" checked={requestData.channelLetterColorMode === 'original'} onChange={() => onDataChange({ channelLetterColorMode: 'original' })} />
                                            <RadioOption name="channelLetterColorMode" value="monochromatic" label="Usar Cor Única" checked={requestData.channelLetterColorMode === 'monochromatic'} onChange={() => onDataChange({ channelLetterColorMode: 'monochromatic' })} />
                                        </div>
                                        {requestData.channelLetterColorMode === 'monochromatic' && <SimpleColorSelector value={requestData.channelLetterColor} onChange={handleLetterColorChange} onClose={() => setIsLetterSelectorOpen(false)} />}
                                    </div>
                                )}
                            </div>
                            <div onFocus={() => setActiveDetailElement("Placa Principal")} onBlur={() => setActiveDetailElement(null)}>
                                <LogoOptions 
                                    value={requestData} 
                                    onChange={onDataChange}
                                    onGenerateLogo={handleGenerateLogo}
                                    onReinventLogo={handleReinventLogo}
                                    isGeneratingLogo={isGeneratingLogo}
                                    isReinventingLogo={isReinventingLogo}
                                    logoError={logoError}
                                />
                            </div>
                        </div>
                    )}
                 </div>
            </StepCard>

            {/* Step 3: Generation */}
            <StepCard stepNumber={3} title="Geração do Projeto">
                 <div className="space-y-6 p-2">
                    <InspirationGallery 
                        autoGeneratedInspiration={requestData.inspiration}
                        customInstructions={requestData.customInstructions}
                        onAddCustomInstruction={handleAddCustomInstruction}
                        onRemoveCustomInstruction={handleRemoveCustomInstruction}
                    />
                    <div className="flex flex-col sm:flex-row-reverse gap-4">
                        <button onClick={() => handleSubmit(false)} disabled={isSubmitting} className="w-full px-6 py-4 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-lg text-lg disabled:bg-gray-500 disabled:cursor-not-allowed">{isSubmitting ? 'Gerando...' : 'Gerar Projeto'}</button>
                        <button onClick={() => handleSubmit(true)} disabled={isSubmitting} className="w-full sm:w-auto px-6 py-4 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition disabled:opacity-50">Otimizar e Gerar</button>
                    </div>
                 </div>
            </StepCard>
        </div>
    );
};