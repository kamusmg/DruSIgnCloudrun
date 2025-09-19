import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
// Fix: Correct import path for types.
import { Area, Calibration, Point, RedesignResult } from '../types.ts';

interface InteractiveResultViewerProps {
    originalImage: string;
    redesignedImage: string;
    areas: Area[];
    technicalPlan: RedesignResult['technicalPlan'];
    calibration: Calibration | null;
}

const dist = (p1: Point, p2: Point) => Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);

const getAreaDimensionsText = (area: Area, calibration: Calibration | null): string => {
    if (!calibration || area.points.length !== 4) return 'Dimensões não calculadas';
    
    // Note: These calculations are simplified and assume a near-rectangular shape for area.
    // They are relative to canvas percentage and need pixel conversion for accuracy.
    const avgWidthPercent = (dist(area.points[0], area.points[1]) + dist(area.points[3], area.points[2])) / 2;
    const avgHeightPercent = (dist(area.points[0], area.points[3]) + dist(area.points[1], area.points[2])) / 2;

    // This is a rough estimation assuming canvas width is the reference for pixelsPerMeter
    // A more robust solution would pass canvas dimensions.
    const estimatedCanvasWidth = 1000;
    const widthM = (avgWidthPercent * estimatedCanvasWidth) / calibration.pixelsPerMeter;
    const heightM = (avgHeightPercent * estimatedCanvasWidth) / calibration.pixelsPerMeter;

    if (isNaN(widthM) || isNaN(heightM)) return 'Erro no cálculo';
    
    return `~ ${widthM.toFixed(2)}m x ${heightM.toFixed(2)}m`;
};


const InteractiveResultViewer: React.FC<InteractiveResultViewerProps> = ({ originalImage, redesignedImage, areas, technicalPlan, calibration }) => {
    const [hoveredAreaId, setHoveredAreaId] = useState<string | null>(null);
    const [view, setView] = useState<'after' | 'before'>('after');
    const containerRef = useRef<HTMLDivElement>(null);
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                setContainerSize({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight,
                });
            }
        };
        updateSize();
        window.addEventListener('resize', updateSize);
        return () => window.removeEventListener('resize', updateSize);
    }, []);

    const hoveredItem = useMemo(() => {
        // Fix: Add a check for technicalPlan to prevent runtime errors since it's an optional property.
        if (!hoveredAreaId || !technicalPlan) return null;
        // Find technical plan item that might correspond to the area
        // This is a simple match by index, a more robust system might use IDs
        const areaIndex = areas.findIndex(a => a.id === hoveredAreaId);
        return technicalPlan[areaIndex] || null;
    }, [hoveredAreaId, areas, technicalPlan]);

    const renderPolygon = useCallback((area: Area) => {
        const points = area.points.map(p => `${p[0] * 100}%,${p[1] * 100}%`).join(' ');
        const isHovered = area.id === hoveredAreaId;

        return (
            <polygon
                key={area.id}
                points={points}
                onMouseEnter={() => setHoveredAreaId(area.id)}
                onMouseLeave={() => setHoveredAreaId(null)}
                className={`transition-all duration-300 cursor-pointer ${isHovered ? 'fill-cyan-500/40 stroke-cyan-300' : 'fill-cyan-500/20 stroke-cyan-400'}`}
                strokeWidth="2"
                strokeDasharray="6 4"
            />
        );
    }, [hoveredAreaId]);

    return (
        <div className="bg-white dark:bg-gray-900/50 shadow-lg rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 p-2 space-y-2">
             <div className="flex justify-between items-center px-2">
                 <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    {areas.length > 0 ? "Passe o mouse sobre as peças para ver os detalhes." : "Visualização do projeto."}
                 </p>
                 <div className="bg-gray-200 dark:bg-gray-800 p-1 rounded-full flex">
                    <button onClick={() => setView('before')} className={`px-4 py-1 text-xs font-semibold rounded-full ${view === 'before' ? 'bg-white dark:bg-gray-900 shadow' : ''}`}>Antes</button>
                    <button onClick={() => setView('after')} className={`px-4 py-1 text-xs font-semibold rounded-full ${view === 'after' ? 'bg-white dark:bg-gray-900 shadow' : ''}`}>Depois</button>
                 </div>
            </div>
            <div ref={containerRef} className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
                <img src={view === 'after' ? redesignedImage : originalImage} alt={view} className="absolute inset-0 w-full h-full object-contain" />
                
                {view === 'after' && areas.length > 0 && (
                     <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0">
                         {areas.map(renderPolygon)}
                     </svg>
                )}

                {hoveredItem && (
                    <div className="absolute bottom-4 left-4 p-3 bg-black/80 text-white rounded-lg shadow-lg max-w-sm pointer-events-none animate-fade-in text-sm">
                        <h4 className="font-bold text-cyan-400">{hoveredItem.item}</h4>
                        <p><span className="font-semibold">Material:</span> {hoveredItem.material}</p>
                        <p><span className="font-semibold">Dimensões:</span> {hoveredItem.dimensions}</p>
                        <p><span className="font-semibold">Detalhes:</span> {hoveredItem.details}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default InteractiveResultViewer;