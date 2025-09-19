import React, { useState, useEffect } from 'react';
// Fix: Corrected import for StickerDetail to handle type checks.
import { RedesignResult, DetailedRequestData, Logo, Calibration, StickerDetail, StickerPatternData, DeliverableKey } from '../types.ts';

// Component Imports
import { ComparisonSlider } from './ComparisonSlider.tsx';
import { DeliverablesSection } from './DeliverablesSection.tsx';
import { dataUrlToFile, downloadFromDataUrl } from '../utils/fileDownloader.ts';
import { DownloadIcon, EditIcon, RedoIcon, UndoIcon, SparkleIcon, RulerIcon, PatternIcon, WhatsAppIcon } from './Icons.tsx';

interface ResultDisplayProps {
  result: RedesignResult;
  originalImage: string;
  requestData: DetailedRequestData;
  onEdit: () => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  calibration: Calibration | null;
  onGenerateDeliverableRequest: (key: DeliverableKey) => void;
  deliverableStatus: { key: DeliverableKey | null, message: string };
}

const InfoCard: React.FC<{ title: string; children: React.ReactNode; }> = ({ title, children }) => (
    <div className="bg-white dark:bg-gray-900/50 shadow-lg rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800">
        <div className="p-5">
            <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-3">{title}</h3>
            {children}
        </div>
    </div>
);

// Helper function to crop the original image to the new aspect ratio
const cropImageToMatchAspectRatio = (
  originalSrc: string,
  targetSrc: string
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const originalImg = new Image();
    const targetImg = new Image();

    originalImg.crossOrigin = "Anonymous";
    targetImg.crossOrigin = "Anonymous";

    let originalLoaded = false;
    let targetLoaded = false;

    const onBothLoaded = () => {
      if (!originalLoaded || !targetLoaded) return;
      
      const targetAspectRatio = targetImg.naturalWidth / targetImg.naturalHeight;
      const originalAspectRatio = originalImg.naturalWidth / originalImg.naturalHeight;

      if (Math.abs(targetAspectRatio - originalAspectRatio) < 0.01) {
        resolve(originalSrc); // Aspect ratios are close enough, no crop needed
        return;
      }
      
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return reject(new Error('Canvas context not available'));
      }
      
      let sx = 0, sy = 0, sWidth = originalImg.naturalWidth, sHeight = originalImg.naturalHeight;
      
      if (originalAspectRatio > targetAspectRatio) {
        // Original is wider, crop the width
        sWidth = originalImg.naturalHeight * targetAspectRatio;
        sx = (originalImg.naturalWidth - sWidth) / 2;
      } else {
        // Original is taller, crop the height
        sHeight = originalImg.naturalWidth / targetAspectRatio;
        sy = (originalImg.naturalHeight - sHeight) / 2;
      }
      
      canvas.width = sWidth;
      canvas.height = sHeight;
      
      ctx.drawImage(originalImg, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);
      
      resolve(canvas.toDataURL('image/jpeg'));
    };

    originalImg.onload = () => { originalLoaded = true; onBothLoaded(); };
    targetImg.onload = () => { targetLoaded = true; onBothLoaded(); };
    originalImg.onerror = reject;
    targetImg.onerror = reject;

    originalImg.src = originalSrc;
    targetImg.src = targetSrc;
  });
};


export const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, originalImage, requestData, onEdit, onUndo, onRedo, canUndo, canRedo, calibration, onGenerateDeliverableRequest, deliverableStatus }) => {
    const [sliderMode, setSliderMode] = useState<'slider' | 'side-by-side'>('slider');
    const [croppedOriginalImage, setCroppedOriginalImage] = useState<string | null>(null);

     useEffect(() => {
        let isMounted = true;
        cropImageToMatchAspectRatio(originalImage, result.redesignedImage)
            .then(croppedSrc => {
                if (isMounted) {
                    setCroppedOriginalImage(croppedSrc);
                }
            })
            .catch(error => {
                console.error("Failed to crop image:", error);
                if(isMounted) {
                   setCroppedOriginalImage(originalImage); // Fallback to original
                }
            });
            
        return () => { isMounted = false; };
    }, [originalImage, result.redesignedImage]);
    
    const handleDownloadImage = () => {
        downloadFromDataUrl(result.redesignedImage, `redesign_${requestData.companyName || 'projeto'}.jpg`);
    };

    const handleDownloadLogo = (logo: Logo | null) => {
        if (!logo) return;
        downloadFromDataUrl(logo.base64, `logo_${requestData.companyName || 'final'}.png`);
    }

    // Fix: Correctly access the `data` property on the StickerDetail union type instead of the non-existent `pattern` property.
    const generatedPatterns = requestData.stickerDetails
        .filter((s): s is (StickerDetail & { type: 'pattern', data: StickerPatternData & { generatedPattern: Logo } }) => 
            s.type === 'pattern' && !!s.data.generatedPattern
        )
        .map(s => ({ pattern: s.data.generatedPattern, theme: s.data.theme }));

    const handleDownloadPattern = (pattern: Logo, theme: string, index: number) => {
        if (!pattern) return;
        const safeTheme = theme.replace(/\s/g, '_').substring(0, 20);
        downloadFromDataUrl(pattern.base64, `pattern_${safeTheme || index + 1}.jpg`);
    };

    const handleShareWhatsApp = async () => {
        const text = `Olá! Veja a proposta de redesign para a fachada da "${requestData.companyName || 'sua empresa'}" que criamos com nossa ferramenta de IA. O que você achou?`;
        const filename = `redesign_${requestData.companyName || 'projeto'}.jpg`;

        try {
            const imageFile = await dataUrlToFile(result.redesignedImage, filename);

            if (navigator.share && navigator.canShare({ files: [imageFile] })) {
                await navigator.share({
                    title: 'Proposta de Redesign de Fachada',
                    text: text,
                    files: [imageFile],
                });
            } else {
                throw new Error('Web Share API for files not supported.');
            }
        } catch (error) {
            console.warn('Web Share API failed, falling back to text-only link:', error);
            const encodedText = encodeURIComponent(text);
            const whatsappUrl = `https://wa.me/?text=${encodedText}`;
            window.open(whatsappUrl, '_blank');
            alert('Para compartilhar a imagem, use o botão "Baixar Imagem" e anexe-a manualmente na conversa do WhatsApp.');
        }
    };


    return (
        <div className="space-y-8 animate-fade-in-up">
            <div className="text-center">
                <h2 className="text-3xl md:text-4xl font-extrabold text-gray-900 dark:text-gray-100 mb-2">Seu Projeto de Fachada está Pronto!</h2>
                <p className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
                    Compare o resultado, edite os detalhes e baixe os entregáveis do seu projeto.
                </p>
            </div>
            
            {/* Comparison Viewer */}
            <div className="bg-white dark:bg-gray-900/50 shadow-xl rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-800 p-4">
                <div className="flex justify-between items-center mb-4 px-2 flex-wrap gap-2">
                     <h3 className="text-xl font-bold">Antes e Depois</h3>
                     <div className="flex items-center gap-4">
                        {/* Undo/Redo Buttons */}
                        <div className="flex items-center gap-1">
                            <button onClick={onUndo} disabled={!canUndo} className="p-2 rounded-full transition bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed" aria-label="Desfazer">
                                <UndoIcon className="w-5 h-5" />
                            </button>
                             <button onClick={onRedo} disabled={!canRedo} className="p-2 rounded-full transition bg-gray-200 dark:bg-gray-800 hover:bg-gray-300 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed" aria-label="Refazer">
                                <RedoIcon className="w-5 h-5" />
                            </button>
                        </div>
                        {/* View Mode Toggle */}
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">Visualização:</span>
                            <div className="bg-gray-200 dark:bg-gray-800 p-1 rounded-full flex">
                                <button onClick={() => setSliderMode('slider')} className={`px-3 py-1 text-xs font-semibold rounded-full transition ${sliderMode === 'slider' ? 'bg-white dark:bg-gray-900 shadow' : ''}`}>Slider</button>
                                <button onClick={() => setSliderMode('side-by-side')} className={`px-3 py-1 text-xs font-semibold rounded-full transition ${sliderMode === 'side-by-side' ? 'bg-white dark:bg-gray-900 shadow' : ''}`}>Lado a Lado</button>
                            </div>
                         </div>
                     </div>
                </div>
                {croppedOriginalImage ? (
                    <ComparisonSlider before={croppedOriginalImage} after={result.redesignedImage} mode={sliderMode} />
                ) : (
                    <div className="w-full aspect-video flex items-center justify-center bg-gray-900 rounded-lg"><p>Carregando comparativo...</p></div>
                )}
            </div>
            
            {/* Project Actions */}
             <div className="flex flex-col items-center gap-6">
                <h3 className="text-xl font-bold text-center">Ações do Projeto</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 w-full max-w-4xl">
                     <button onClick={handleDownloadImage} className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition shadow-md">
                        <DownloadIcon className="w-5 h-5"/> Baixar Imagem
                    </button>
                    {result.finalLogo && (
                        <button onClick={() => handleDownloadLogo(result.finalLogo)} className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition shadow-md">
                            <DownloadIcon className="w-5 h-5"/> Baixar Logo
                        </button>
                    )}
                    {generatedPatterns.map(({ pattern, theme }, index) => (
                        <button key={index} onClick={() => handleDownloadPattern(pattern, theme, index)} className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition shadow-md">
                            <PatternIcon className="w-5 h-5"/> Baixar Padrão {generatedPatterns.length > 1 ? index + 1 : ''}
                        </button>
                    ))}
                     <button onClick={onEdit} className="w-full col-span-full flex items-center justify-center gap-2 px-6 py-3 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700 transition shadow-lg animate-pulse-highlight">
                        <EditIcon className="w-5 h-5"/> Adicionar / Modificar
                    </button>
                </div>
                 {calibration && (
                    <p className="text-sm text-green-600 dark:text-green-400 animate-fade-in">✔️ Medidas calibradas com sucesso! Os entregáveis usarão as novas dimensões.</p>
                )}
            </div>

             {/* Deliverables */}
            <InfoCard title="Entregáveis do Projeto">
                 <DeliverablesSection 
                    result={result} 
                    requestData={requestData} 
                    originalImage={originalImage} 
                    calibration={calibration}
                    onGenerateRequest={onGenerateDeliverableRequest}
                    deliverableStatus={deliverableStatus}
                    onShareRequest={handleShareWhatsApp}
                 />
            </InfoCard>
            
             {/* Prompts for transparency */}
             <div className="grid md:grid-cols-2 gap-8">
                <InfoCard title="Sua Solicitação Original">
                    <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 p-3 rounded-md whitespace-pre-wrap font-mono">{result.originalPrompt}</p>
                </InfoCard>
                <InfoCard title="Prompt Técnico Enviado à IA">
                     <p className="text-sm text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 p-3 rounded-md whitespace-pre-wrap font-mono">{result.enhancedPrompt}</p>
                </InfoCard>
             </div>

        </div>
    );
};