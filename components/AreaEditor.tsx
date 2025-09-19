import React, { useState, useRef, useEffect, useCallback, MouseEvent } from 'react';
import { Area, Calibration, Point, Logo } from '../types.ts';
import { AreaSummary } from './AreaSummary.tsx';
import { AreaContentEditor } from './AreaContentEditor.tsx';
import { RulerIcon, TrashIcon, RectangleIcon, QuadIcon, MousePointerIcon, XIcon } from './Icons.tsx';
import { CalibrationModal } from './CalibrationModal.tsx';

interface AreaEditorProps {
  imageSrc: string;
  initialAreas: Area[];
  initialCalibration: Calibration | null;
  onAreasChange: (areas: Area[]) => void;
  onCalibrationChange: (cal: Calibration | null) => void;
  onCancel: () => void;
  isCalibrationMode?: boolean;
}

type Tool = 'select' | 'rect' | 'quad' | 'ruler';

interface DraggingState {
  type: 'point' | 'area';
  areaId: string;
  pointIndex?: number;
  startX: number;
  startY: number;
}


export const AreaEditor: React.FC<AreaEditorProps> = ({ 
  imageSrc, 
  initialAreas,
  initialCalibration,
  onAreasChange,
  onCalibrationChange,
  onCancel,
  isCalibrationMode = false,
}) => {
  const [areas, setAreas] = useState<Area[]>(initialAreas);
  const [selectedAreaId, setSelectedAreaId] = useState<string | null>(null);
  const [activeTool, setActiveTool] = useState<Tool>(isCalibrationMode ? 'ruler' : 'rect');
  const [tempPoints, setTempPoints] = useState<Point[]>([]);
  const [draggingState, setDraggingState] = useState<DraggingState | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // No need for these useEffects in calibration-only mode
  // useEffect(() => { onAreasChange(areas) }, [areas, onAreasChange]);
  // useEffect(() => { onCalibrationChange(calibration) }, [calibration, onCalibrationChange]);

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    const image = imageRef.current;

    if (!canvas || !ctx || !image || image.naturalWidth === 0) {
      ctx?.clearRect(0, 0, canvas?.width || 0, canvas?.height || 0);
      return;
    }
    
    const containerWidth = containerRef.current?.clientWidth ?? canvas.width;
    const aspectRatio = image.naturalWidth / image.naturalHeight;
    canvas.width = containerWidth;
    canvas.height = containerWidth / aspectRatio;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    
    // Draw temporary ruler line if it exists
    if (activeTool === 'ruler' && tempPoints.length > 0) {
        ctx.save();
        ctx.strokeStyle = '#06b6d4';
        ctx.fillStyle = '#06b6d4';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 5;

        // Draw points
        tempPoints.forEach(p => {
            ctx.beginPath();
            ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 6, 0, 2 * Math.PI);
            ctx.fill();
        });

        // Draw line
        if (tempPoints.length === 2) {
            ctx.beginPath();
            ctx.moveTo(tempPoints[0][0] * canvas.width, tempPoints[0][1] * canvas.height);
            ctx.lineTo(tempPoints[1][0] * canvas.width, tempPoints[1][1] * canvas.height);
            ctx.lineWidth = 3;
            ctx.stroke();
        }
        ctx.restore();
    }

  }, [tempPoints, activeTool]);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;
    img.onload = () => {
      imageRef.current = img;
      redrawCanvas();
    };
     const handleResize = () => redrawCanvas();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [imageSrc, redrawCanvas]);
  
  useEffect(() => {
      redrawCanvas();
  }, [redrawCanvas]);


  const getRelativeCoords = (e: MouseEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [x, y];
  };

  const handleCanvasClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const point = getRelativeCoords(e);
     if (activeTool === 'ruler') {
        const newPoints = [...tempPoints, point];
        if (newPoints.length === 2) {
            setTempPoints(newPoints);
            setIsModalOpen(true); // Open modal to ask for distance
        } else {
            setTempPoints(newPoints.slice(0, 1)); // Start over or keep first point
        }
     }
  };

   const handleCalibrationConfirm = (distanceCm: number) => {
    setIsModalOpen(false);
    if (tempPoints.length !== 2) return;

    const canvas = canvasRef.current;
    if(!canvas) return;

    const p1 = tempPoints[0];
    const p2 = tempPoints[1];

    const distPx = Math.hypot(
        (p2[0] - p1[0]) * canvas.width,
        (p2[1] - p1[1]) * canvas.height
    );

    const pixelsPerCm = distPx / distanceCm;
    
    const newCalibration: Calibration = {
        points: tempPoints,
        realDistanceMeters: distanceCm / 100, // Store as meters
        pixelsPerMeter: pixelsPerCm * 100,
    };
    setTempPoints([]);
    onCalibrationChange(newCalibration);
  };
  
  if (isCalibrationMode) {
      return (
           <div className="bg-white dark:bg-gray-900/50 shadow-lg rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800 p-4 space-y-4">
                <CalibrationModal 
                    isOpen={isModalOpen}
                    onClose={() => {
                        setIsModalOpen(false);
                        setTempPoints([]); // Reset line on cancel
                    }}
                    onConfirm={handleCalibrationConfirm}
                />
                <div className="flex justify-between items-center">
                    <div>
                        <h2 className="text-xl font-bold">Calibrar Medidas para o Documento</h2>
                        <p className="text-gray-500 dark:text-gray-400 max-w-2xl">
                          Para obter medidas precisas, clique em dois pontos na imagem (ex: as bordas de uma porta) e informe a medida real.
                        </p>
                    </div>
                     <button onClick={onCancel} className="p-2 -mt-2 -mr-2 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
                        <XIcon className="w-6 h-6" />
                    </button>
                </div>
                 <div ref={containerRef} className="relative w-full aspect-video bg-black rounded-lg overflow-hidden">
                    <canvas 
                        ref={canvasRef} 
                        onClick={handleCanvasClick} 
                        className="w-full h-full cursor-crosshair"
                    />
                </div>
                <div className="flex justify-between items-center gap-2">
                    <button onClick={() => setTempPoints([])} className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">Limpar Linha</button>
                    <button onClick={onCancel} className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">
                      Pular e Usar Estimativas da IA
                    </button>
                </div>
           </div>
      );
  }

  // Original component for full area editing (currently unused in main flow)
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4">
      <p>Area Editor (Full)</p>
    </div>
  );
};