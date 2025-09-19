import React from 'react';
// Fix: Correct the import path for Icons.
import { CameraIcon, DocumentReportIcon, PaletteIcon } from './Icons.tsx';

interface OnboardingModalProps {
  onClose: () => void;
}

const InfoCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => (
  <div className="flex-1 bg-gray-100 dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
    <div className="w-16 h-16 mx-auto bg-cyan-500/10 text-cyan-400 rounded-full flex items-center justify-center mb-4 border-2 border-cyan-500/30">
        {icon}
    </div>
    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">{title}</h3>
    <p className="text-sm text-gray-600 dark:text-gray-400">{description}</p>
  </div>
);


export const OnboardingModal: React.FC<OnboardingModalProps> = ({ onClose }) => {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4 animate-fade-in">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-4xl w-full p-8 md:p-10 border border-gray-200 dark:border-gray-800 transform transition-all animate-slide-up">
        <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-gray-100 mb-4">
          Redesenhe sua Fachada em 3 Passos
        </h2>
        <p className="text-center text-gray-600 dark:text-gray-400 mb-8 max-w-2xl mx-auto">
          Nossa ferramenta de Inteligência Artificial transforma a foto da sua fachada em um projeto profissional com orçamento preliminar em minutos.
        </p>

        <div className="flex flex-col md:flex-row gap-6 mb-10">
            <InfoCard 
                icon={<CameraIcon className="w-8 h-8"/>}
                title="1. Envie uma Foto"
                description="Tire uma foto nítida da sua fachada ou envie um arquivo da sua galeria."
            />
            <InfoCard 
                icon={<PaletteIcon className="w-8 h-8"/>}
                title="2. Personalize"
                description="Escolha materiais, layouts e dê suas instruções para a nossa IA criar o design."
            />
            <InfoCard 
                icon={<DocumentReportIcon className="w-8 h-8"/>}
                title="3. Receba o Projeto"
                description="Compare o resultado, baixe a proposta em PDF e fale conosco para executar."
            />
        </div>

        <div className="flex justify-center">
          <button 
            onClick={onClose} 
            className="px-10 py-4 bg-cyan-600 text-white font-semibold rounded-lg hover:bg-cyan-700 transition shadow-lg text-lg"
          >
            Entendi, vamos começar!
          </button>
        </div>
      </div>
    </div>
  );
};