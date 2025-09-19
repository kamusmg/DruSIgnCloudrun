import React from 'react';
import { RedesignResult, DetailedRequestData, Calibration, DeliverableKey } from '../types.ts';
import { DocumentDownloadIcon, TableIcon, WhatsAppIcon } from './Icons.tsx';
import { calculatePriceEstimate } from '../utils/pricing.ts';


interface DeliverablesSectionProps {
    result: RedesignResult;
    requestData: DetailedRequestData;
    originalImage: string;
    calibration: Calibration | null;
    onGenerateRequest: (key: DeliverableKey) => void;
    deliverableStatus: { key: DeliverableKey | null, message: string };
    onShareRequest: () => void;
}

export const DeliverablesSection: React.FC<DeliverablesSectionProps> = ({ result, requestData, onGenerateRequest, deliverableStatus, onShareRequest }) => {
    const price = calculatePriceEstimate(result.technicalPlan);

    const DeliverableButton: React.FC<{
        dKey: DeliverableKey;
        icon: React.ReactNode;
        title: string;
        description: string;
    }> = ({ dKey, icon, title, description }) => {
        const isLoading = deliverableStatus.key === dKey;
        
        let buttonTitle = title;
        if (isLoading) {
            buttonTitle = deliverableStatus.message || "Gerando...";
        }

        return (
             <button
                onClick={() => onGenerateRequest(dKey)}
                disabled={isLoading}
                className="flex items-start text-left p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg w-full transition hover:bg-gray-200 dark:hover:bg-gray-700/60 disabled:opacity-70 disabled:cursor-not-allowed"
            >
                <div className="flex-shrink-0 w-10 h-10 flex items-center justify-center bg-cyan-100 dark:bg-cyan-900/50 text-cyan-600 dark:text-cyan-400 rounded-lg mr-4">
                    {isLoading ? (
                        <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                        icon
                    )}
                </div>
                <div>
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100">{buttonTitle}</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
                </div>
            </button>
        );
    }

    return (
        <div className="space-y-6">
            <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h4 className="text-md font-semibold text-gray-900 dark:text-gray-100">Estimativa de Orçamento Preliminar</h4>
                <p className="text-2xl font-bold text-cyan-600 dark:text-cyan-400 my-1">
                    R$ {price.min.toLocaleString('pt-BR')} - R$ {price.max.toLocaleString('pt-BR')}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                    *Valor baseado no plano técnico gerado. Um orçamento final será fornecido após visita técnica.
                </p>
            </div>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <DeliverableButton 
                    dKey="presentationPdf"
                    icon={<DocumentDownloadIcon className="w-5 h-5" />}
                    title="Baixar Apresentação (PDF)"
                    description="Proposta visual com antes/depois e plano técnico."
                />
                 <DeliverableButton 
                    dKey="budgetSheetXlsx"
                    icon={<TableIcon className="w-5 h-5" />}
                    title="Baixar Orçamento (XLSX)"
                    description="Planilha interativa com itens e custos estimados."
                />
                <button
                    onClick={onShareRequest}
                    className="flex items-start text-left p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg w-full transition hover:bg-gray-200 dark:hover:bg-gray-700/60"
                >
                    <div className="flex-shrink-0 w-10 h-10 flex items-center justify-center bg-green-100 dark:bg-green-900/50 text-green-600 dark:text-green-400 rounded-lg mr-4">
                        <WhatsAppIcon className="w-5 h-5" />
                    </div>
                    <div>
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100">Compartilhar no WhatsApp</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Envie a imagem do projeto para seus contatos.</p>
                    </div>
                </button>
            </div>
        </div>
    );
};