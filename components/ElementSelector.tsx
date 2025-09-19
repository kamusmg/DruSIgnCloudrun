

import React from 'react';

interface ElementSelectorProps {
    selectedElements: string[];
    onChange: (elements: string[]) => void;
    activeElement: string | null;
}

const elementsOptions = [
    "Placa Principal", "Texto Adicional", "Adesivos", "Revestimento em ACM",
    "Pintura Nova", "Iluminação Especial", "Totem", "Wind Banner (Bandeira)",
    "Banner ou Faixa", "Placas Informativas", "Adicionar Pessoas", "Veículos"
];

// Curated gradient palette
export const elementColors: Record<string, { from: string; to: string; main: string; }> = {
    "Placa Principal":      { from: 'from-cyan-500',   to: 'to-blue-500',     main: 'border-cyan-500' },
    "Texto Adicional":      { from: 'from-sky-400',    to: 'to-cyan-500',     main: 'border-sky-400' },
    "Adesivos":             { from: 'from-emerald-400',to: 'to-teal-500',     main: 'border-emerald-400' },
    "Revestimento em ACM":  { from: 'from-slate-400',  to: 'to-gray-500',     main: 'border-slate-400' },
    "Pintura Nova":         { from: 'from-indigo-400', to: 'to-purple-500',   main: 'border-indigo-400' },
    "Iluminação Especial":  { from: 'from-amber-400',  to: 'to-yellow-500',   main: 'border-amber-400' },
    "Totem":                { from: 'from-gray-500',   to: 'to-slate-600',    main: 'border-gray-500' },
    "Wind Banner (Bandeira)":{ from: 'from-cyan-400',   to: 'to-teal-400',     main: 'border-cyan-400' },
    "Banner ou Faixa":      { from: 'from-sky-500',    to: 'to-indigo-500',   main: 'border-sky-500' },
    "Placas Informativas":  { from: 'from-teal-400',   to: 'to-green-500',    main: 'border-teal-400' },
    "Adicionar Pessoas":    { from: 'from-purple-400', to: 'to-violet-500',   main: 'border-purple-400' },
    "Veículos":             { from: 'from-gray-400',   to: 'to-slate-500',    main: 'border-gray-400' },
};


export const ElementSelector: React.FC<ElementSelectorProps> = ({ selectedElements, onChange, activeElement }) => {

    const handleToggle = (element: string) => {
        const newSelection = selectedElements.includes(element)
            ? selectedElements.filter(el => el !== element)
            : [...selectedElements, element];
        onChange(newSelection);
    };

    return (
        <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Que elementos você gostaria de ver no projeto? <span className="text-sm font-normal text-gray-500">(Opcional)</span>
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Marque os itens que são importantes para você. A IA focará em criá-los.
            </p>
            <div className="flex flex-wrap gap-3">
                {elementsOptions.map(element => {
                    const color = elementColors[element] || elementColors["Placa Principal"];
                    const isChecked = selectedElements.includes(element);
                    const isActive = activeElement === element;

                    const gradientClasses = isChecked ? `bg-gradient-to-r ${color.from} ${color.to}` : '';
                    const animationClasses = isActive ? 'animate-gradient-flow [background-size:200%_200%]' : '';

                    return (
                        <div key={element} className={`p-0.5 rounded-full transition-all duration-300 ${gradientClasses} ${animationClasses}`}>
                             <label 
                                className="flex items-center space-x-2 px-4 py-2 bg-gray-100 dark:bg-gray-800/80 rounded-full cursor-pointer"
                            >
                                <input
                                    type="checkbox"
                                    checked={isChecked}
                                    onChange={() => handleToggle(element)}
                                    className="h-4 w-4 rounded border-gray-400 bg-transparent text-cyan-500 focus:ring-cyan-500"
                                />
                                <span className="text-sm font-medium text-gray-800 dark:text-gray-200 select-none">{element}</span>
                            </label>
                        </div>
                    )
                })}
            </div>
        </div>
    );
};

export default ElementSelector;