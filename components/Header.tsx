// Fix: Add onReset prop to Header component to handle project reset functionality.
import React from 'react';
import { DruSignLogoIcon, HomeIcon } from './Icons.tsx';

interface HeaderProps {
  onHome?: () => void;
  onReset?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onHome, onReset }) => {
  return (
    <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm sticky top-0 z-40 border-b border-gray-200 dark:border-gray-800">
      <div className="container mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <DruSignLogoIcon className="w-8 h-8 text-cyan-500" />
          <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            DRU SIGN <span className="font-light text-cyan-500">Redesign com Fachada Pronta AI</span>
          </h1>
        </div>
        <div className="flex items-center gap-2">
            {onHome && (
              <button
                onClick={onHome}
                className="px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 rounded-lg hover:bg-gray-300 dark:bg-gray-700/80 dark:text-gray-200 dark:hover:bg-gray-700/100 transition flex items-center gap-2"
                aria-label="Voltar ao início"
              >
                <HomeIcon className="w-4 h-4" />
                Início
              </button>
            )}
            {onReset && (
              <button
                onClick={onReset}
                className="px-4 py-2 text-sm font-semibold text-red-700 bg-red-100 rounded-lg hover:bg-red-200 dark:bg-red-900/30 dark:text-red-300 dark:hover:bg-red-800/50 transition"
                aria-label="Recomeçar projeto"
              >
                Recomeçar
              </button>
            )}
        </div>
      </div>
    </header>
  );
};