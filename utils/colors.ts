// --- Color Utility Functions ---

type RGB = { r: number; g: number; b: number };
type Rect = { x: number, y: number, width: number, height: number };

/** Converts a hex color string to an RGB object. */
export function hexToRgb(hex: string): RGB {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
        ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16),
        }
        : { r: 0, g: 0, b: 0 };
}

/** Converts an RGB object to a hex color string. */
export function rgbToHex({ r, g, b }: RGB): string {
  const toHex = (c: number) => `0${c.toString(16)}`.slice(-2);
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}


/** Calculates the relative luminance of an RGB color. */
export function getLuminance({ r, g, b }: RGB): number {
    const a = [r, g, b].map((v) => {
        v /= 255;
        return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return a[0] * 0.2126 + a[1] * 0.7152 + a[2] * 0.0722;
}

/** Calculates the contrast ratio between two RGB colors. */
export function getContrast(rgb1: RGB, rgb2: RGB): number {
    const lum1 = getLuminance(rgb1);
    const lum2 = getLuminance(rgb2);
    const brightest = Math.max(lum1, lum2);
    const darkest = Math.min(lum1, lum2);
    return (brightest + 0.05) / (darkest + 0.05);
}

/** Extracts the lightest and darkest color from a region of a canvas. */
export function extractImageColors(
    ctx: CanvasRenderingContext2D,
    region?: Rect
): Promise<{ light: RGB; dark: RGB }> {
    return new Promise((resolve) => {
        const r = region || { x: 0, y: 0, width: ctx.canvas.width, height: ctx.canvas.height };
        const imageData = ctx.getImageData(r.x, r.y, r.width, r.height).data;
        
        let light: RGB = { r: 0, g: 0, b: 0 };
        let dark: RGB = { r: 255, g: 255, b: 255 };
        let lightLum = -1;
        let darkLum = 2;

        // Sample a subset of pixels for performance
        const sampleRate = Math.max(1, Math.floor(imageData.length / 4 / 1000)); 

        for (let i = 0; i < imageData.length; i += 4 * sampleRate) {
            const r = imageData[i];
            const g = imageData[i + 1];
            const b = imageData[i + 2];
            
            const lum = getLuminance({r, g, b});

            if (lum > lightLum) {
                lightLum = lum;
                light = { r, g, b };
            }
            if (lum < darkLum) {
                darkLum = lum;
                dark = { r, g, b };
            }
        }
        resolve({ light, dark });
    });
}


/** Extracts dominant colors from a canvas region using a simple bucketing method. */
export const extractDominantColors = (
    ctx: CanvasRenderingContext2D,
    colorCount: number = 3,
    region?: Rect
): Promise<string[]> => {
    return new Promise((resolve) => {
        const r = region || { x: 0, y: 0, width: ctx.canvas.width, height: ctx.canvas.height };
        const imageData = ctx.getImageData(r.x, r.y, r.width, r.height).data;
        
        const colorCounts: { [hex: string]: { count: number, original: RGB } } = {};
        const sampleRate = Math.max(1, Math.floor(imageData.length / 4 / 2000)); // Sample ~2000 pixels

        for (let i = 0; i < imageData.length; i += 4 * sampleRate) {
            const r_val = imageData[i];
            const g_val = imageData[i + 1];
            const b_val = imageData[i + 2];
            const alpha = imageData[i + 3];

            // Skip transparent or near-transparent pixels
            if (alpha < 128) continue;
            
            // Skip grays/whites/blacks
            if (r_val > 245 && g_val > 245 && b_val > 245) continue;
            if (r_val < 10 && g_val < 10 && b_val < 10) continue;
            const max = Math.max(r_val, g_val, b_val);
            const min = Math.min(r_val, g_val, b_val);
            if (max - min < 20) continue; // Low saturation

            // Use a simplified key to bucket similar colors by rounding
            const r_key = Math.round(r_val / 32) * 32;
            const g_key = Math.round(g_val / 32) * 32;
            const b_key = Math.round(b_val / 32) * 32;
            
            const hex = rgbToHex({r: r_key, g: g_key, b: b_key});
            
            if (!colorCounts[hex]) {
                colorCounts[hex] = { count: 0, original: { r: r_val, g: g_val, b: b_val } };
            }
            colorCounts[hex].count++;
        }

        const dominantColors = Object.entries(colorCounts)
            .sort(([, a], [, b]) => b.count - a.count)
            .slice(0, colorCount)
            .map(([, data]) => rgbToHex(data.original)); // Use the first-seen original color for the bucket

        resolve(dominantColors);
    });
};