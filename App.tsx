

import React, { useState, useEffect, useCallback } from 'react';
import {
  RedesignResult,
  DetailedRequestData,
  Logo,
  DeliverablesStatus,
  VisualEditorState,
  Calibration,
  StickerDetail,
  LonaInstallationType,
  DeliverableKey,
} from './types.ts';
import {
  generateRedesign,
  enhancePrompt,
  refinePlacementPrompt,
  generatePdfCoverImage,
} from './services/geminiService.ts';
import { saveState, loadState, clearState } from './utils/statePersistence.ts';
import { generateStickerDescriptionText } from './utils/textHelper.ts';
import { generatePresentationPdf } from './utils/pdfGenerator.ts';
import { generateInteractiveBudgetSheetXlsx } from './utils/xlsxGenerator.ts';
import { downloadBlob } from './utils/fileDownloader.ts';


// Component Imports
import { Header } from './components/Header.tsx';
import { FileUpload } from './components/FileUpload.tsx';
import { DetailedRequest } from './components/DetailedRequest.tsx';
import { LoadingSpinner } from './components/LoadingSpinner.tsx';
import { ResultDisplay } from './components/ResultDisplay.tsx';
import { StepCard } from './components/StepCard.tsx';
import { EnhanceConfirmModal } from './components/EnhanceConfirmModal.tsx';
import { AreaEditor } from './components/AreaEditor.tsx';
import { ErrorDisplay } from './components/ErrorDisplay.tsx';


type AppStep = 'upload' | 'details' | 'loading' | 'result';

interface AppState {
  step: AppStep;
  originalImage: string | null;
  requestData: DetailedRequestData;
  resultHistory: RedesignResult[];
  historyIndex: number;
  loadingStep: string;
  isEnhanceModalOpen: boolean;
  isCalibrationModalOpen: boolean;
  isEditingGeneratedImage: boolean;
  calibration: Calibration | null;
  pendingDeliverable: DeliverableKey | null;
  deliverableStatus: {
    key: DeliverableKey | null;
    message: string;
  };
}

const initialRequestData: DetailedRequestData = {
  companyName: '',
  mainSignType: 'acm_adesivo',
  lonaInstallationType: 'grommets',
  logoOption: 'use_from_photo',
  logoPrompt: '',
  logoFile: null,
  elements: ['Placa Principal'],
  acmColor: '#1C1C1C',
  acmColorName: 'Preto Brilhante',
  acmPlacement: {
    selections: [],
    custom: '',
  },
  paintColor: '#f5f5f5',
  lightingOptions: [],
  totemDetails: {
    size: 'medium',
    features: [],
  },
  bannerFaixaDetails: {
    type: 'banner',
    sizeType: 'preset',
    presetSize: '80x120cm',
    customWidth: '',
    customHeight: '',
    artType: 'ai_generic',
    artPrompt: '',
    artFile: null,
  },
  informationalSignsText: '',
  vehicleDetails: {
    selectedTypes: [],
    customName: '',
  },
  channelLetterColor: '#FFFFFF',
  channelLetterColorMode: 'original',
  additionalText: { text: '', location: '' },
  ambiance: 'day',
  inspiration: '',
  customInstructions: [],
  stickerDetails: [], // Changed to an empty array
};

const getInitialState = (): AppState => ({
  step: 'upload',
  originalImage: null,
  requestData: initialRequestData,
  resultHistory: [],
  historyIndex: -1,
  loadingStep: 'Preparando a IA...',
  isEnhanceModalOpen: false,
  isCalibrationModalOpen: false,
  isEditingGeneratedImage: false,
  calibration: null,
  pendingDeliverable: null,
  deliverableStatus: { key: null, message: '' },
});

function App() {
  const [appState, setAppState] = useState<AppState>(() => {
    // FIX: Corrected the generic type for `loadState`. The original intersection (`&`)
    // caused the `step` property's type to be too narrow. Using `Omit` correctly
    // overrides the property, allowing the legacy 'onboarding' value to be checked.
    const savedState = loadState<Omit<AppState, 'step'> & { step: AppStep | 'onboarding' }>(); // Allow legacy type
    // Handle legacy state where the app might be stuck on the onboarding step
    if (savedState?.step === 'onboarding') {
        savedState.step = 'upload';
    }
    return (savedState as AppState | null) || getInitialState();
  });
  const [globalError, setGlobalError] = useState<Error | null>(null);

  // Effect for saving state to localStorage
  useEffect(() => {
    saveState(appState);
  }, [appState]);

  // Effect for global error handling
  useEffect(() => {
    const handleError = (event: ErrorEvent) => {
        console.error("Global error caught:", event.error);
        setGlobalError(event.error instanceof Error ? event.error : new Error(String(event.error)));
    };
    const handleRejection = (event: PromiseRejectionEvent) => {
        console.error("Global rejection caught:", event.reason);
        setGlobalError(event.reason instanceof Error ? event.reason : new Error(String(event.reason)));
    };

    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);

    return () => {
        window.removeEventListener('error', handleError);
        window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, []);

  const handleFileUpload = (file: File | null, base64: string | null) => {
    if (base64) {
      setAppState(prev => ({
        ...prev,
        step: 'details',
        originalImage: base64,
        resultHistory: [], // Clear history when a new image is uploaded
        historyIndex: -1,
        calibration: null, // Reset calibration on new image
        isEditingGeneratedImage: false,
      }));
    }
  };

  const handleBackToUpload = () => {
    setAppState(prev => ({
      ...prev,
      step: 'upload',
      requestData: {
          ...initialRequestData,
          inspiration: '', 
      },
      resultHistory: [],
      historyIndex: -1,
      isEditingGeneratedImage: false,
    }));
  };
  
  const handleReset = () => {
      if(window.confirm("Tem certeza que deseja recomeçar? Todo o progresso será perdido.")) {
          clearState();
          setAppState(getInitialState());
          setGlobalError(null);
      }
  }

  const handleDataChange = (changes: Partial<DetailedRequestData>) => {
    setAppState(prev => ({
      ...prev,
      requestData: { ...prev.requestData, ...changes },
    }));
  };

  const handleGenerateRedesign = useCallback(async (useEnhanced: boolean, dataOverride?: DetailedRequestData) => {
      let dataToUse = dataOverride || appState.requestData;
      if (!appState.originalImage) return;

      setAppState(prev => ({ ...prev, step: 'loading', isEnhanceModalOpen: false, loadingStep: 'Analisando seu projeto...', requestData: dataToUse }));
      
      const combinedOriginalPrompt = [dataToUse.inspiration, ...dataToUse.customInstructions].filter(Boolean).join(' ');
      
      let finalPrompt = combinedOriginalPrompt;
      let originalPrompt = combinedOriginalPrompt;
      
      if (useEnhanced) {
          try {
            setAppState(prev => ({ ...prev, loadingStep: 'Otimizando o prompt para a IA...' }));
            const enhanced = await enhancePrompt(finalPrompt);
            finalPrompt = enhanced;

            // Refine sticker placements if there are any
            if (dataToUse.stickerDetails.length > 0) {
                 setAppState(prev => ({ ...prev, loadingStep: 'Refinando locais dos adesivos...' }));
                 // FIX: Use a switch statement to handle the discriminated union `StickerDetail` correctly.
                 // This ensures TypeScript preserves the specific data type for each sticker type,
                 // preventing type errors when updating the `placement` property.
                 const refinedStickers: StickerDetail[] = await Promise.all(dataToUse.stickerDetails.map(async (sticker): Promise<StickerDetail> => {
                    const newPlacement = await refinePlacementPrompt(appState.originalImage!, sticker.data.placement);
                    switch (sticker.type) {
                        case 'pattern':
                            return { ...sticker, data: { ...sticker.data, placement: newPlacement }};
                        case 'cut':
                            return { ...sticker, data: { ...sticker.data, placement: newPlacement }};
                        case 'print':
                            return { ...sticker, data: { ...sticker.data, placement: newPlacement }};
                    }
                 }));
                 // Update the data used for this generation
                 dataToUse = {...dataToUse, stickerDetails: refinedStickers};
            }

          } catch (error) {
              console.error("Failed to enhance prompt, using original.", error);
              alert("Não foi possível otimizar o prompt. Usando a descrição original.");
          }
      }
      
       // Combine prompts
      const stickerText = generateStickerDescriptionText(dataToUse.stickerDetails);
      let combinedPrompt = finalPrompt; // Use the potentially enhanced main prompt
      if (stickerText) {
          combinedPrompt = (combinedPrompt + '\n\n' + stickerText).trim();
      }
      finalPrompt = combinedPrompt;

      try {
          setAppState(prev => ({ ...prev, loadingStep: 'Gerando o redesign da fachada...' }));
          const result = await generateRedesign(appState.originalImage, finalPrompt, dataToUse);
          
          const newHistory = appState.resultHistory.slice(0, appState.historyIndex + 1);
          
          setAppState(prev => ({
              ...prev,
              step: 'result',
              resultHistory: [...newHistory, {
                  redesignedImage: result.redesignedImage,
                  finalLogo: result.finalLogo,
                  technicalPlan: result.technicalPlan,
                  originalPrompt: originalPrompt,
                  enhancedPrompt: finalPrompt,
              }],
              historyIndex: newHistory.length,
          }));
      } catch (error) {
          console.error("Redesign generation failed:", error);
          let customError: Error;
          if (error instanceof Error) {
              if (error.message === "AI_IMAGE_GENERATION_FAILED") {
                  customError = new Error("A IA não conseguiu gerar uma imagem. Isso pode acontecer por alguns motivos:\n\n- O pedido pode ter sido bloqueado por filtros de segurança (ex: uso de marcas registradas, nomes de pessoas, etc.).\n- A IA pode ter tido dificuldade em interpretar o pedido.\n\nPor favor, tente simplificar ou reformular sua solicitação e tente novamente.");
                  customError.name = "Falha na Geração de Imagem";
              } else {
                 customError = error;
              }
          } else {
             customError = new Error(String(error) || "Ocorreu um erro desconhecido ao gerar o redesign.");
          }
          setGlobalError(customError);
      }
  }, [appState.originalImage, appState.requestData, appState.historyIndex]);

  const handleSubmitRequest = async (useEnhanced: boolean, data: DetailedRequestData) => {
      setAppState(prev => ({ ...prev, requestData: data }));
      if (useEnhanced) {
          setAppState(prev => ({ ...prev, isEnhanceModalOpen: true }));
      } else {
          await handleGenerateRedesign(false, data);
      }
  };

  const handleEditRequest = () => {
    const currentResult = appState.resultHistory[appState.historyIndex];
    if (!currentResult) return;
    setAppState(prev => ({
        ...prev,
        step: 'details',
        originalImage: currentResult.redesignedImage,
        isEditingGeneratedImage: true,
        requestData: {
            ...initialRequestData,
            companyName: prev.requestData.companyName,
        }
    }));
  };
  
  const handleUndo = () => {
    setAppState(prev => ({
      ...prev,
      historyIndex: Math.max(0, prev.historyIndex - 1),
    }));
  };

  const handleRedo = () => {
    setAppState(prev => ({
      ...prev,
      historyIndex: Math.min(prev.resultHistory.length - 1, prev.historyIndex + 1),
    }));
  };

    const generateAndDownloadDeliverable = useCallback(async (key: DeliverableKey, cal: Calibration | null) => {
        const currentResult = appState.resultHistory[appState.historyIndex];
        if (!currentResult || !appState.originalImage) return;

        setAppState(prev => ({ ...prev, deliverableStatus: { key, message: 'Iniciando...' }}));

        try {
            let blob: Blob;
            let filename: string;
            const companyNameSlug = (appState.requestData.companyName || 'projeto').replace(/\s+/g, '-').toLowerCase();
            const dateSlug = new Date().toISOString().slice(0, 10);

            if (key === 'presentationPdf') {
                setAppState(prev => ({ ...prev, deliverableStatus: { key, message: 'Gerando capa exclusiva...' }}));
                const coverImageBase64 = await generatePdfCoverImage(
                    currentResult.finalLogo,
                    appState.requestData.companyName,
                    currentResult.originalPrompt,
                    appState.originalImage
                );

                setAppState(prev => ({ ...prev, deliverableStatus: { key, message: 'Montando PDF...' }}));
                blob = await generatePresentationPdf(currentResult, appState.requestData, appState.originalImage, cal, coverImageBase64);
                filename = `relatorio-fachada-${companyNameSlug}-${dateSlug}.pdf`;
            } else if (key === 'budgetSheetXlsx') {
                 setAppState(prev => ({ ...prev, deliverableStatus: { key, message: 'Gerando planilha...' }}));
                blob = await generateInteractiveBudgetSheetXlsx(currentResult, appState.requestData, cal);
                filename = `orcamento-fachada-${companyNameSlug}-${dateSlug}.xlsx`;
            } else {
                throw new Error('Invalid deliverable key');
            }

            downloadBlob(blob, filename);
        } catch (error) {
            console.error(`Failed to generate ${key}:`, error);
            const customError = error instanceof Error ? error : new Error(`Ocorreu um erro ao gerar o arquivo ${key}.`);
            setGlobalError(customError);
        } finally {
            setAppState(prev => ({ ...prev, deliverableStatus: { key: null, message: '' }}));
        }
    }, [appState.resultHistory, appState.historyIndex, appState.originalImage, appState.requestData]);


  const handleGenerateDeliverableRequest = (key: DeliverableKey) => {
    if (appState.calibration) {
        generateAndDownloadDeliverable(key, appState.calibration);
    } else {
        setAppState(prev => ({
            ...prev,
            isCalibrationModalOpen: true,
            pendingDeliverable: key,
        }));
    }
  };


  const handleFinishCalibration = (cal: Calibration | null) => {
    const deliverableToGenerate = appState.pendingDeliverable;

    // Save the calibration result and return to the main result view
    setAppState(prev => ({
      ...prev,
      calibration: cal,
      isCalibrationModalOpen: false,
      pendingDeliverable: null,
    }));

    // If a deliverable was pending, generate it now with the new calibration data
    if (deliverableToGenerate) {
      generateAndDownloadDeliverable(deliverableToGenerate, cal);
    }
  };
  
  const renderContent = () => {
    const currentResult = appState.resultHistory[appState.historyIndex];

    switch (appState.step) {
      case 'upload':
        return (
          <StepCard stepNumber={1} title="Envie uma Foto da Fachada" highlight>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Para melhores resultados, tire uma foto de frente, durante o dia e sem muitos obstáculos.
            </p>
            <FileUpload onFileUpload={handleFileUpload} />
          </StepCard>
        );
      case 'details':
        if (!appState.originalImage) {
            handleReset(); 
            return null;
        }
        return (
          <DetailedRequest
            requestData={appState.requestData}
            onDataChange={handleDataChange}
            onSubmit={handleSubmitRequest}
            onBack={handleBackToUpload}
            originalImage={appState.originalImage}
            isEditingGeneratedImage={appState.isEditingGeneratedImage}
          />
        );
      case 'loading':
         return (
            <LoadingSpinner
                step={appState.loadingStep}
                originalText={currentResult?.originalPrompt}
                enhancedText={currentResult?.enhancedPrompt}
            />
         );
      case 'result':
        if (!currentResult || !appState.originalImage) {
            handleReset();
            return null;
        }
        return (
          <ResultDisplay
            result={currentResult}
            originalImage={appState.originalImage}
            requestData={appState.requestData}
            onEdit={handleEditRequest}
            onUndo={handleUndo}
            onRedo={handleRedo}
            canUndo={appState.historyIndex > 0}
            canRedo={appState.historyIndex < appState.resultHistory.length - 1}
            calibration={appState.calibration}
            onGenerateDeliverableRequest={handleGenerateDeliverableRequest}
            deliverableStatus={appState.deliverableStatus}
          />
        );
      default:
        return (
          <StepCard stepNumber={1} title="Envie uma Foto da Fachada">
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Para melhores resultados, tire uma foto de frente, durante o dia e sem muitos obstáculos.
            </p>
            <FileUpload onFileUpload={handleFileUpload} />
          </StepCard>
        );
    }
  };
  
  const currentResult = appState.resultHistory[appState.historyIndex];

  return (
    <div className="bg-gray-100 dark:bg-black min-h-screen text-gray-900 dark:text-gray-100 font-sans flex flex-col">
      {globalError && <ErrorDisplay error={globalError} onReset={handleReset} />}

      <EnhanceConfirmModal 
        isOpen={appState.isEnhanceModalOpen}
        onClose={() => setAppState(prev => ({...prev, isEnhanceModalOpen: false}))}
        onConfirmEnhance={() => handleGenerateRedesign(true)}
        onConfirmOriginal={() => handleGenerateRedesign(false)}
        isLoading={appState.step === 'loading'}
      />

      {appState.isCalibrationModalOpen && currentResult && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="w-full max-w-5xl">
            <AreaEditor
              imageSrc={currentResult.redesignedImage}
              initialAreas={[]}
              initialCalibration={appState.calibration}
              onAreasChange={() => {}}
              onCalibrationChange={handleFinishCalibration}
              onCancel={() => handleFinishCalibration(null)}
              isCalibrationMode={true}
            />
          </div>
        </div>
      )}
      
      <Header onHome={handleBackToUpload} onReset={appState.step !== 'upload' ? handleReset : undefined} />
      <main className="container mx-auto p-4 md:p-8 flex-grow">
        {renderContent()}
      </main>
      <footer className="text-center py-4 px-8 text-xs text-gray-500 dark:text-gray-600">
        <p>Feito por Kamus &amp; Gemini &nbsp;&middot;&nbsp; Powered by Google AI Studio &nbsp;&middot;&nbsp; S/N: 2024-DRU-FP-001</p>
      </footer>
    </div>
  );
}

export default App;