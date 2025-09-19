import React from 'react';

interface StepCardProps {
  stepNumber: number;
  title: string;
  children: React.ReactNode;
  highlight?: boolean;
}

export const StepCard: React.FC<StepCardProps> = ({ stepNumber, title, children, highlight = false }) => {
  const highlightClasses = highlight 
    ? 'animate-pulse-highlight ring-2 ring-cyan-500/80 ring-offset-4 ring-offset-black' 
    : 'border-gray-200 dark:border-gray-800';

  return (
    <div className={`bg-white dark:bg-gray-900/50 shadow-lg rounded-xl overflow-hidden border transition-all duration-500 ${highlightClasses}`}>
      <div className="p-6">
        <div className="flex items-center space-x-4 mb-4">
          <div className="flex-shrink-0 flex items-center justify-center w-10 h-10 bg-cyan-500 text-black rounded-full font-bold text-lg">
            {stepNumber}
          </div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">{title}</h2>
        </div>
        <div>
          {children}
        </div>
      </div>
    </div>
  );
};