import React from 'react';
import { MainSignType, LonaInstallationType } from '../types.ts';

interface MainSignDetailsProps {
    value: MainSignType;
    onChange: (value: MainSignType) => void;
    lonaInstallationType: LonaInstallationType;
    onLonaInstallationChange: (value: LonaInstallationType) => void;
}

const options: { value: MainSignType, label: string, description: string }[] = [
    { value: 'lona', label: 'Lona', description: 'Estrutura metálica com lona impressa.' },
    { value: 'acm_adesivo', label: 'ACM com Adesivo', description: 'Placa rígida de ACM com vinil adesivo.' },
    { value: 'acm_letra_caixa', label: 'ACM com Letra Caixa', description: 'Placa de ACM com logo em relevo (letra caixa).' },
    { value: 'placa_iluminada', label: 'Placa Iluminada', description: 'Letreiro com iluminação interna (LED).' },
];

const lonaOptions: { value: LonaInstallationType, label: string }[] = [
    { value: 'grommets', label: 'Estrutura com Ilhós (cordas/abraçadeiras)' },
    { value: 'hidden_finish', label: 'Estrutura com Acabamento Embutido' },
];

const RadioSubOption: React.FC<{ name: string; value: string; label: string; checked: boolean; onChange: () => void; }> = ({ name, value, label, checked, onChange }) => (
    <label className={`flex-1 flex items-center p-2 rounded-md border-2 cursor-pointer text-xs transition ${checked ? 'border-cyan-600 bg-cyan-500/10' : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'}`}>
        <input
            type="radio"
            name={name}
            value={value}
            checked={checked}
            onChange={onChange}
            className="h-4 w-4 text-cyan-600 focus:ring-cyan-500 border-gray-300 dark:border-gray-500 bg-transparent"
        />
        <span className="ml-2 font-medium text-gray-800 dark:text-gray-200">{label}</span>
    </label>
);


export const MainSignDetails: React.FC<MainSignDetailsProps> = ({ value, onChange, lonaInstallationType, onLonaInstallationChange }) => {
    return (
        <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Detalhes da Placa Principal
            </h3>
            <div className="grid sm:grid-cols-2 gap-3">
                {options.map(option => (
                    <div key={option.value}>
                        <label 
                            className={`flex items-start p-3 rounded-lg border-2 cursor-pointer transition ${value === option.value ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500'}`}
                        >
                            <input
                                type="radio"
                                name="mainSignType"
                                value={option.value}
                                checked={value === option.value}
                                onChange={() => onChange(option.value)}
                                className="h-5 w-5 mt-0.5 text-cyan-600 focus:ring-cyan-500 border-gray-300 dark:border-gray-600 bg-transparent"
                            />
                            <div className="ml-3 text-sm">
                                <span className="font-bold text-gray-900 dark:text-gray-100">{option.label}</span>
                                <p className="text-gray-500 dark:text-gray-400">{option.description}</p>
                            </div>
                        </label>

                        {/* Conditional Lona Options */}
                        {value === 'lona' && option.value === 'lona' && (
                            <div className="mt-2 p-2 bg-gray-100 dark:bg-gray-800/50 rounded-md animate-fade-in">
                                <p className="text-xs font-semibold mb-2 text-gray-700 dark:text-gray-300">Tipo de Instalação da Lona:</p>
                                <div className="flex flex-col sm:flex-row gap-2">
                                    {lonaOptions.map(lonaOpt => (
                                        <RadioSubOption
                                            key={lonaOpt.value}
                                            name="lonaInstallationType"
                                            value={lonaOpt.value}
                                            label={lonaOpt.label}
                                            checked={lonaInstallationType === lonaOpt.value}
                                            onChange={() => onLonaInstallationChange(lonaOpt.value)}
                                        />
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MainSignDetails;