import React, { useMemo } from 'react';
// Fix: Correct import path for types.
import { Area, Calibration, Point } from '../types.ts';
// Fix: Correct the import path for Icons.
import { RulerIcon, TrashIcon } from './Icons.tsx';

interface AreaSummaryProps {
    areas: Area[];
    calibration: Calibration | null;
    selectedAreaId: string | null;
    onAreaSelect: (id: string | null) => void;
    onAreasChange: (areas: Area[]) => void;
}

export const AreaSummary: React.FC<AreaSummaryProps> = ({ areas, calibration, selectedAreaId, onAreaSelect, onAreasChange }) => {
    
    const calculateDimensions = (area: Area) => {
        if (!calibration) return null;
        
        const worldPoints = area.points.map(p => {
            const imgWidth = 1000; // Assume a nominal width for calculation consistency
            return [p[0] * imgWidth, p[1] * (imgWidth * (1/area.points[0][0] * area.points[0][1]))] as Point; // Approximate height based on first point ratio
        });

        const dist = (p1: Point, p2: Point) => Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
        
        let widthPx, heightPx;
        if (worldPoints.length === 4) { // rect or quad
            widthPx = (dist(worldPoints[0], worldPoints[1]) + dist(worldPoints[3], worldPoints[2])) / 2;
            heightPx = (dist(worldPoints[0], worldPoints[3]) + dist(worldPoints[1], worldPoints[2])) / 2;
        } else { return null; }

        const widthM = widthPx / calibration.pixelsPerMeter;
        const heightM = heightPx / calibration.pixelsPerMeter;

        return {
            w: widthM.toFixed(2),
            h: heightM.toFixed(2),
            area: (widthM * heightM).toFixed(2)
        };
    };

    const handleDelete = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        onAreasChange(areas.filter(a => a.id !== id));
        if (selectedAreaId === id) {
            onAreaSelect(null);
        }
    }

    return (
        <div className="p-4 bg-gray-100 dark:bg-gray-800/50 rounded-lg space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Peças Marcadas</h3>
            
            {!calibration && (
                 <div className="flex items-center p-3 text-sm text-yellow-800 rounded-lg bg-yellow-50 dark:bg-gray-800 dark:text-yellow-300 border border-yellow-300 dark:border-yellow-600" role="alert">
                    <RulerIcon className="w-5 h-5 mr-2"/>
                    <span className="font-medium">Aguardando calibração.</span> Use a ferramenta Régua na imagem.
                </div>
            )}
            
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
                {areas.length > 0 ? areas.map((area, index) => {
                    const dims = calculateDimensions(area);
                    const isSelected = area.id === selectedAreaId;
                    return (
                        <div 
                            key={area.id}
                            onClick={() => onAreaSelect(area.id)}
                            className={`p-2.5 rounded-md cursor-pointer transition border-2 ${isSelected ? 'bg-cyan-500/10 border-cyan-500' : 'bg-white dark:bg-gray-700 border-transparent hover:border-gray-400'}`}
                        >
                            <div className="flex justify-between items-center">
                                <div>
                                    <p className="font-semibold text-sm text-gray-900 dark:text-gray-100">{index + 1}. {area.type}</p>
                                    <p className={`text-xs ${dims ? 'text-gray-500 dark:text-gray-400' : 'text-yellow-600 dark:text-yellow-400'}`}>
                                      {dims ? `~ ${dims.w}m (L) x ${dims.h}m (A) | ${dims.area} m²` : 'Sem calibração'}
                                    </p>
                                </div>
                                <button 
                                    onClick={(e) => handleDelete(e, area.id)}
                                    className="p-1 rounded-full text-gray-400 hover:bg-red-100 hover:text-red-500 dark:hover:bg-red-900/50"
                                    aria-label={`Excluir ${area.type}`}
                                >
                                    <TrashIcon className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    )
                }) : (
                    <p className="text-sm text-center text-gray-500 dark:text-gray-400 py-4">Nenhuma área marcada ainda.</p>
                )}
            </div>
        </div>
    );
};