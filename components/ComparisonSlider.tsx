import React, { useState, useRef, useCallback, useEffect } from 'react';

interface ComparisonSliderProps {
  before: string;
  after: string;
  mode?: 'slider' | 'side-by-side';
}

const loadImage = (src: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = src;
    });
};

const drawImageWithLetterbox = (ctx: CanvasRenderingContext2D, image: ImageBitmap) => {
    const canvas = ctx.canvas;
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    const imageWidth = image.width;
    const imageHeight = image.height;

    const canvasAspect = canvasWidth / canvasHeight;
    const imageAspect = imageWidth / imageHeight;

    let drawWidth, drawHeight, x, y;

    if (imageAspect > canvasAspect) {
        drawWidth = canvasWidth;
        drawHeight = drawWidth / imageAspect;
        x = 0;
        y = (canvasHeight - drawHeight) / 2;
    } else {
        drawHeight = canvasHeight;
        drawWidth = drawHeight * imageAspect;
        y = 0;
        x = (canvasWidth - drawWidth) / 2;
    }

    ctx.fillStyle = '#111827'; // Dark gray for letterbox
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    ctx.drawImage(image, x, y, drawWidth, drawHeight);
};


export const ComparisonSlider: React.FC<ComparisonSliderProps> = ({ before, after, mode = 'slider' }) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const beforeCanvasRef = useRef<HTMLCanvasElement>(null);
  const afterCanvasRef = useRef<HTMLCanvasElement>(null);
  const isDragging = useRef(false);

  const renderToCanvas = useCallback(async () => {
    setIsLoading(true);
    setError(false);
    
    const beforeCanvas = beforeCanvasRef.current;
    const afterCanvas = afterCanvasRef.current;
    const container = containerRef.current;

    if (mode !== 'slider' || !beforeCanvas || !afterCanvas || !container) {
      setIsLoading(false);
      return;
    }
    
    const beforeCtx = beforeCanvas.getContext('2d');
    const afterCtx = afterCanvas.getContext('2d');
    
    if (!beforeCtx || !afterCtx) {
      setError(true);
      setIsLoading(false);
      return;
    }

    try {
      const [beforeImg, afterImg] = await Promise.all([loadImage(before), loadImage(after)]);
      const [beforeBitmap, afterBitmap] = await Promise.all([
          createImageBitmap(beforeImg, { imageOrientation: 'from-image' }),
          createImageBitmap(afterImg, { imageOrientation: 'from-image' }),
      ]);

      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      
      if (containerWidth === 0 || containerHeight === 0) {
        setIsLoading(false);
        return; 
      }

      beforeCanvas.width = containerWidth;
      beforeCanvas.height = containerHeight;
      afterCanvas.width = containerWidth;
      afterCanvas.height = containerHeight;

      drawImageWithLetterbox(beforeCtx, beforeBitmap);
      drawImageWithLetterbox(afterCtx, afterBitmap);

      beforeBitmap.close();
      afterBitmap.close();

    } catch (e) {
      console.error("Failed to render images to canvas:", e);
      setError(true);
    } finally {
      setIsLoading(false);
    }
  }, [before, after, mode]);
  
  useEffect(() => {
    renderToCanvas();
    window.addEventListener('resize', renderToCanvas);
    return () => {
      window.removeEventListener('resize', renderToCanvas);
    };
  }, [renderToCanvas]);


  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPosition(percent);
  }, []);

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => { isDragging.current = true; e.preventDefault(); };
  const handleMouseUp = () => { isDragging.current = false; };
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => { if (!isDragging.current) return; handleMove(e.clientX); };
  const handleTouchStart = () => { isDragging.current = true; };
  const handleTouchEnd = () => { isDragging.current = false; };
  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => { if (!isDragging.current) return; handleMove(e.touches[0].clientX); };

  useEffect(() => {
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('touchend', handleTouchEnd);
    return () => {
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchend', handleTouchEnd);
    };
  }, []);

  const renderFallback = () => (
    <div className="flex flex-col md:flex-row gap-4 p-4">
      <div className="flex-1 space-y-2">
        <h3 className="font-bold text-center text-lg text-gray-800 dark:text-gray-200">Antes</h3>
        <img src={before} alt="Before" className="w-full h-auto rounded-lg shadow-md" />
      </div>
      <div className="flex-1 space-y-2">
        <h3 className="font-bold text-center text-lg text-gray-800 dark:text-gray-200">Depois</h3>
        <img src={after} alt="After" className="w-full h-auto rounded-lg shadow-md" />
      </div>
    </div>
  );

  if (mode === 'side-by-side') return renderFallback();
  if (error) {
    return (
      <div className="p-4">
        <p className="text-red-500 text-center mb-2">Não foi possível carregar o slider. Exibindo imagens lado a lado.</p>
        {renderFallback()}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden select-none cursor-ew-resize rounded-lg bg-gray-900"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onTouchMove={handleTouchMove}
    >
      <img src={before} alt="" className="block w-full h-auto opacity-0 pointer-events-none" aria-hidden="true" />
      
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/70">
          <p className="text-white">Carregando comparativo...</p>
        </div>
      )}

      <canvas ref={beforeCanvasRef} className="absolute inset-0 w-full h-full" />
      
      <div className="absolute inset-0 w-full h-full overflow-hidden" style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}>
        <canvas ref={afterCanvasRef} className="w-full h-full" />
      </div>

      <div
        className="absolute top-0 bottom-0 w-1 bg-white/80 shadow-md backdrop-blur-sm cursor-ew-resize"
        style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -left-5 w-10 h-10 bg-white/80 rounded-full flex items-center justify-center shadow-lg border-2 border-gray-300 backdrop-blur-sm">
          <svg className="w-6 h-6 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l4-4 4 4m0 6l-4 4-4-4" transform="rotate(90 12 12)" />
          </svg>
        </div>
      </div>
    </div>
  );
};