// Fix: Correct import paths with extensions.
import { TemplateSpec, Shape, TextBox } from './types.ts';
import { extractImageColors, getContrast, hexToRgb, rgbToHex, getLuminance } from '../utils/colors.ts';

type Texts = { title: string; phone: string; subtitle: string };
type PaletteOption = 'auto' | 'light' | 'dark';
type QuickAdjustments = {
    isUpper: boolean;
    hasShadow: boolean;
    hasStroke: boolean;
    palette: PaletteOption;
};

const FONT_FAMILY = "Inter, Rubik, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'";
const MAX_CANVAS_WIDTH = 1200;

interface Rect { x: number; y: number; width: number; height: number; }
interface ResolvedPalette { bg: string; fg: string; accent: string; }

// --- Main Render Function ---
export async function renderTemplate(
    canvas: HTMLCanvasElement,
    image: HTMLImageElement,
    spec: TemplateSpec,
    texts: Texts,
    adjustments: QuickAdjustments
) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // --- 1. Setup Canvas and Draw Image ---
    const canvasWidth = Math.min(MAX_CANVAS_WIDTH, image.width);
    const scale = canvasWidth / image.width;
    const canvasHeight = image.height * scale;
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    ctx.drawImage(image, 0, 0, canvasWidth, canvasHeight);

    // --- 2. Resolve Palette ---
    const palette = await resolvePalette(ctx, spec, adjustments.palette);

    // --- 3. Draw Shapes and Store their calculated rectangles ---
    const shapeRects = new Map<string, Rect>();
    for (const shape of spec.shapes) {
        const rect = drawShape(ctx, shape, palette, canvasWidth, canvasHeight, shapeRects);
        shapeRects.set(shape.id, rect);
    }
    
    // --- 4. Draw Text Boxes ---
    for (const textSpec of spec.text) {
        const textContent = texts[textSpec.id];
        if (!textContent) continue;
        
        let containerRect: Rect;
        if (textSpec.area === 'free') {
            containerRect = { x: 0, y: 0, width: canvasWidth, height: canvasHeight };
        } else {
            const parentRect = shapeRects.get(textSpec.area);
            if (!parentRect) continue;
            containerRect = parentRect;
        }

        await drawTextBox(ctx, textSpec, textContent, containerRect, adjustments, palette, image);
    }
}

// --- Shape Drawing Logic ---
function drawShape(ctx: CanvasRenderingContext2D, shape: Shape, palette: ResolvedPalette, cw: number, ch: number, existingRects: Map<string, Rect>): Rect {
    ctx.save();
    const color = (shape.kind === 'pill' && shape.id === 'phone-pill') ? palette.accent : palette.bg;
    ctx.fillStyle = color;
    ctx.globalAlpha = shape.opacity ?? 1;

    const w = typeof shape.width === 'string' ? (parseFloat(shape.width) / 100) * cw : shape.width ?? 0;
    const h = shape.height ?? 0;
    const r = shape.radius ?? 0;

    let x = shape.offsetX ?? 0;
    let y = shape.offsetY ?? 0;
    
    // Anchor calculations
    switch (shape.anchor) {
        case 'top':         x += (cw - w) / 2; y += 0; break;
        case 'bottom':      x += (cw - w) / 2; y += ch - h; break;
        case 'left':        x += 0; y += (ch - h) / 2; break;
        case 'right':       x += cw - w; y += (ch - h) / 2; break;
        case 'center':      x += (cw - w) / 2; y += (ch - h) / 2; break;
        case 'bottom-right':x += cw - w; y += ch - h; break;
        case 'top-right':   x += cw - w; y += 0; break;
    }
    
    // FIX: Removed `as any` since `parent` is now in the Shape type.
    if (shape.parent) {
      const parentRect = existingRects.get(shape.parent);
      if(parentRect) {
         x = parentRect.x + (parentRect.width - w) / 2 + (shape.offsetX ?? 0);
         y = parentRect.y + parentRect.height + (shape.offsetY ?? 0);
      }
    }


    // Draw path with rounded corners
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    
    return { x, y, width: w, height: h };
}


// --- Text Drawing Logic ---
async function drawTextBox(
    ctx: CanvasRenderingContext2D,
    spec: TextBox,
    text: string,
    container: Rect,
    adjustments: QuickAdjustments,
    palette: ResolvedPalette,
    bgImage: HTMLImageElement
) {
    ctx.save();
    
    const useUpper = adjustments.isUpper && spec.upper;
    const displayText = useUpper ? text.toUpperCase() : text;
    
    const paddedRect = {
        x: container.x + (spec.padding ?? 0),
        y: container.y + (spec.padding ?? 0),
        width: container.width - 2 * (spec.padding ?? 0),
        height: container.height - 2 * (spec.padding ?? 0),
    };

    // --- Font Size Calculation ---
    let fontSize = spec.maxSize;
    let lines: string[] = [];
    while (fontSize >= spec.minSize) {
        ctx.font = `${spec.weight ?? 400} ${fontSize}px ${FONT_FAMILY}`;
        lines = wrapText(ctx, displayText, paddedRect.width);
        if (lines.length <= spec.maxLines) break;
        fontSize--;
    }
    
    // --- Contrast Calculation & Correction ---
    const finalTextColor = await getContrastingTextColor(ctx, spec, paddedRect, palette, bgImage);
    ctx.fillStyle = finalTextColor;

    // --- Text Effects ---
    const hasShadow = adjustments.hasShadow && spec.shadow;
    const hasStroke = adjustments.hasStroke && spec.stroke;
    if (hasShadow) {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetX = 3;
        ctx.shadowOffsetY = 3;
    }
    if (hasStroke) {
        ctx.strokeStyle = getLuminance(hexToRgb(finalTextColor)) > 0.5 ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1;
    }

    // --- Positioning & Drawing ---
    const totalTextHeight = lines.length * fontSize * 1.2;
    let y_start: number;

    switch (spec.verticalAlign) {
        case 'top':    y_start = paddedRect.y + fontSize; break;
        case 'bottom': y_start = paddedRect.y + paddedRect.height - totalTextHeight + fontSize; break;
        case 'middle':
        default:       y_start = paddedRect.y + (paddedRect.height - totalTextHeight) / 2 + fontSize; break;
    }

    ctx.textAlign = spec.align;
    let x_start: number;
    switch (spec.align) {
        case 'left':   x_start = paddedRect.x; break;
        case 'right':  x_start = paddedRect.x + paddedRect.width; break;
        case 'center':
        default:       x_start = paddedRect.x + paddedRect.width / 2; break;
    }
    
    if(spec.orientation === 'vertical') {
      ctx.textAlign = 'center';
      const letters = displayText.split('');
      const verticalTextHeight = letters.length * fontSize * 1.1;
      const vertical_y_start = paddedRect.y + (paddedRect.height - verticalTextHeight) / 2 + fontSize;
      letters.forEach((letter, i) => {
         ctx.fillText(letter, x_start, vertical_y_start + i * fontSize * 1.1);
      });
    } else {
      lines.forEach((line, i) => {
          const y = y_start + i * fontSize * 1.2;
          if (hasStroke) ctx.strokeText(line, x_start, y);
          ctx.fillText(line, x_start, y);
      });
    }

    ctx.restore();
}

// --- Text Wrapping and Balancing ---
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
    const words = text.split(' ');
    const lines: string[] = [];
    let currentLine = '';

    for (const word of words) {
        const testLine = currentLine ? `${currentLine} ${word}` : word;
        const { width: testWidth } = ctx.measureText(testLine);
        if (testWidth > maxWidth && currentLine) {
            lines.push(currentLine);
            currentLine = word;
        } else {
            currentLine = testLine;
        }
    }
    lines.push(currentLine);
    
    // Simple line balancing for 2 lines to avoid orphans
    if (lines.length === 2 && words.length > 2) {
        const lastWord = words[words.length - 1];
        const firstLineWidth = ctx.measureText(lines[0]).width;
        const lastWordWidth = ctx.measureText(` ${lastWord}`).width;
        if (firstLineWidth > lastWordWidth * 3) { // If first line is much longer
            const lastWordOfFirstLine = words[words.length - 2];
            lines[0] = words.slice(0, words.length - 2).join(' ');
            lines[1] = `${lastWordOfFirstLine} ${lastWord}`;
        }
    }

    return lines;
}

// --- Palette and Color Logic ---
async function resolvePalette(ctx: CanvasRenderingContext2D, spec: TemplateSpec, option: PaletteOption): Promise<ResolvedPalette> {
    if (option === 'light' || (typeof spec.palette === 'object' && spec.palette.bg)) {
        return { bg: '#FFFFFF', fg: '#111827', accent: '#0891b2' };
    }
    if (option === 'dark') {
        return { bg: '#111827', fg: '#FFFFFF', accent: '#0891b2' };
    }
    
    // Auto mode
    const { light, dark } = await extractImageColors(ctx);
    return { bg: rgbToHex(dark), fg: rgbToHex(light), accent: '#0891b2' };
}

async function getContrastingTextColor(
    ctx: CanvasRenderingContext2D,
    spec: TextBox,
    rect: Rect,
    palette: ResolvedPalette,
    bgImage: HTMLImageElement
): Promise<string> {
    if (spec.color) return spec.color;

    const baseColor = palette.fg;
    let bgColorHex: string;

    if (spec.area === 'free') {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = bgImage.width;
        tempCanvas.height = bgImage.height;
        const tempCtx = tempCanvas.getContext('2d');
        if (!tempCtx) return baseColor;
        tempCtx.drawImage(bgImage, 0, 0);
        const { dark } = await extractImageColors(tempCtx, rect);
        bgColorHex = rgbToHex(dark);
    } else {
        bgColorHex = palette.bg;
    }
    
    const contrast = getContrast(hexToRgb(baseColor), hexToRgb(bgColorHex));
    
    // If contrast is good, return base color. Otherwise, return the opposite.
    if (contrast >= 4.5) {
        return baseColor;
    } else {
        // Return the opposite color from the palette
        return baseColor === palette.fg ? palette.bg : palette.fg;
    }
}