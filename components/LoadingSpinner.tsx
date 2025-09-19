import React, { useState, useEffect, useMemo } from 'react';
import {
    PhoneIcon,
    WhatsAppIcon,
    DruSignLogoIcon,
} from './Icons.tsx';
import { SYMBOLS } from './SymbolLibrary.ts';


// All new texts combined into one array
const marketingTexts = [
    // Institucional
    "Dru Sign: design, produção e instalação de fachadas.",
    "Especialistas em comunicação visual para negócios.",
    "Projetos sob medida, do conceito à entrega.",
    "Equipe técnica e criativa no mesmo lugar.",
    "Materiais de alta performance e acabamento fino.",
    "Sua marca visível, clara e inesquecível.",
    // Soluções
    "Fachadas em ACM, vidro e madeira.",
    "Letreiros luminosos e caixa alta em LED.",
    "Totens, painéis e outdoors profissionais.",
    "Sinalização interna, wayfinding e adesivação.",
    "Revestimentos e comunicação para pontos de venda.",
    "Padronização visual para redes e franquias.",
    // Produção e Instalação
    "Orçamento transparente e cronograma alinhado.",
    "Produção própria com controle de qualidade.",
    "Instalação segura, limpa e dentro das normas.",
    "Medição técnica e conferência no local.",
    "Detalhes que aumentam a durabilidade.",
    "Apoio em licenças e aprovações quando necessário.",
    // Atendimento e Suporte
    "Atendimento consultivo do briefing à instalação.",
    "Suporte pós-venda atento e ágil.",
    "Acompanhe seu projeto com atualizações claras.",
    "Opções de materiais com melhor custo-benefício.",
    "Manuais de manutenção para sua fachada durar.",
    "Parceria para múltiplas unidades e rollouts.",
    // Benefícios e CTA
    "Mais destaque na rua, mais gente na porta.",
    "Visual pensado para leitura rápida e impacto.",
    "Iluminação eficiente para dia e noite.",
    "Design alinhado à identidade da sua marca.",
    "Peça seu orçamento com a Dru Sign.",
    "Fale com um consultor pelo WhatsApp."
];

// Fisher-Yates shuffle algorithm to randomize arrays
const shuffleArray = <T,>(array: T[]): T[] => {
    const newArray = [...array];
    for (let i = newArray.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
    }
    return newArray;
};


interface LoadingSpinnerProps {
  step: string;
  originalText?: string | null;
  enhancedText?: string | null;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ step, originalText, enhancedText }) => {
    const shuffledTexts = useMemo(() => shuffleArray(marketingTexts), []);
    const shuffledSymbols = useMemo(() => shuffleArray(SYMBOLS), []);

    const [textIndex, setTextIndex] = useState(0);
    const [symbolIndex, setSymbolIndex] = useState(0);
    
    const showTexts = enhancedText && originalText && enhancedText !== originalText;
    
    useEffect(() => {
        // Start with random indices to ensure variety on first load
        setTextIndex(Math.floor(Math.random() * shuffledTexts.length));
        setSymbolIndex(Math.floor(Math.random() * shuffledSymbols.length));

        const interval = setInterval(() => {
            // Cycle through the shuffled arrays sequentially
            setTextIndex(prevIndex => (prevIndex + 1) % shuffledTexts.length);
            setSymbolIndex(prevIndex => (prevIndex + 1) % shuffledSymbols.length);
        }, 3000); // Change every 3 seconds

        return () => clearInterval(interval);
    }, [shuffledTexts.length, shuffledSymbols.length]);

    const currentText = shuffledTexts[textIndex];
    const currentSymbol = shuffledSymbols[symbolIndex];


    return (
        <div className="flex flex-col items-center justify-between p-8 text-center bg-black rounded-lg min-h-[80vh] relative overflow-hidden">
        
            {/* Top Section: AI Progress */}
            <div className="w-full">
                <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-6"></div>
                <h3 className="text-xl font-semibold text-gray-100 mb-2">{step}</h3>
                <p className="text-sm text-gray-400 mb-6 max-w-lg mx-auto">
                Isso pode levar de 30 a 90 segundos. Nossa IA está trabalhando para criar um redesign incrível para você!
                </p>
                
                {showTexts && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-4xl mx-auto text-left mt-4">
                    <div className="p-4 bg-gray-900 rounded-lg">
                    <p className="text-xs font-semibold text-gray-400 uppercase mb-2">Seu Projeto Ideal (Original):</p>
                    <p className="text-sm text-gray-300">{originalText}</p>
                    </div>
                    <div className="p-4 bg-gray-900 rounded-lg border-2 border-cyan-500/50">
                    <p className="text-xs font-semibold text-cyan-400 uppercase mb-2">Prompt Técnico Enviado à IA:</p>
                    <p className="text-sm text-gray-300">{enhancedText}</p>
                    </div>
                </div>
                )}
            </div>

            {/* Middle Section: Animated Services */}
            <div className="flex-grow flex items-center justify-center w-full">
                <div key={`${textIndex}-${symbolIndex}`} className="flex flex-col items-center gap-6 text-cyan-400 animate-fade-in-up">
                    <div className="text-9xl animate-neon-glow" aria-hidden="true">
                        {currentSymbol}
                    </div>
                    <p className="text-2xl font-bold bg-gradient-to-r from-white via-cyan-300 to-white text-transparent bg-clip-text animate-shine-text [background-size:200%_auto]">
                        {currentText}
                    </p>
                </div>
            </div>

            {/* Bottom Section: Dru Sign Footer */}
            <div className="w-full max-w-4xl pt-8 border-t border-gray-700/50 flex flex-col items-center gap-4">
                <div className="flex items-center gap-3">
                    <DruSignLogoIcon className="w-8 h-8 text-cyan-500" />
                    <span className="text-lg font-bold text-white">DRU SIGN</span>
                </div>
                <div className="flex items-center gap-4 text-cyan-400 font-semibold text-base">
                    <a href="tel:+553836766918" className="flex items-center gap-2 hover:text-cyan-300 transition">
                        <PhoneIcon className="w-4 h-4" />
                        <span>(38) 3676-6918</span>
                    </a>
                    <span className="text-gray-600">|</span>
                    <a href="https://wa.me/553836766918" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:text-cyan-300 transition">
                        <WhatsAppIcon className="w-4 h-4" />
                        <span>WhatsApp</span>
                    </a>
                </div>
            </div>
        </div>
    );
};