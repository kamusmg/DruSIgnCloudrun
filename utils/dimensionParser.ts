
// utils/dimensionParser.ts

/**
 * Parses a dimension string from the AI (e.g., "4.5m x 1.2m", "Aprox. 12m²")
 * and returns formatted strings for width, height, and area.
 * @param dimensionStr The string to parse.
 * @returns An object with formatted width, height, and area strings.
 */
export const parseAndFormatDimensions = (dimensionStr: string | null | undefined): { width: string; height: string; area: string } => {
    if (!dimensionStr) {
        return { width: '—', height: '—', area: '—' };
    }

    const formatNumber = (num: number | null | undefined): string => {
        if (num === null || num === undefined || isNaN(num) || num === 0) return '—';
        return num.toFixed(2).replace('.', ',');
    };
    
    // Case 1: "W m x H m" or "W x H"
    const matchesWH = dimensionStr.toLowerCase().match(/(\d+[\.,]?\d*)\s*m?\s*[xX]\s*(\d+[\.,]?\d*)/);
    if (matchesWH && matchesWH.length >= 3) {
        const width = parseFloat(matchesWH[1].replace(',', '.'));
        const height = parseFloat(matchesWH[2].replace(',', '.'));
        if (!isNaN(width) && !isNaN(height)) {
            return {
                width: formatNumber(width),
                height: formatNumber(height),
                area: formatNumber(width * height),
            };
        }
    }

    // Case 2: "A m²" or "A m"
    const areaMatch = dimensionStr.match(/(\d+[\.,]?\d*)\s*m/);
    if (areaMatch && areaMatch.length >= 2) {
        const area = parseFloat(areaMatch[1].replace(',', '.'));
        if(!isNaN(area)) {
            return {
                width: '—',
                height: '—',
                area: formatNumber(area),
            };
        }
    }

    // Fallback if no numbers found
    return { width: '—', height: '—', area: '—' };
};


/**
 * Parses a dimension string from the AI into raw numbers for calculations.
 * @param dimensionStr The string to parse.
 * @returns An object with width, height, and area as numbers. Returns 0 for non-parsable values.
 */
export const parseDimensionsToNumbers = (dimensionStr: string | null | undefined): { width: number; height: number; area: number } => {
    if (!dimensionStr) {
        return { width: 0, height: 0, area: 0 };
    }

    // Case 1: "W m x H m" or "W x H"
    const matchesWH = dimensionStr.toLowerCase().match(/(\d+[\.,]?\d*)\s*m?\s*[xX]\s*(\d+[\.,]?\d*)/);
    if (matchesWH && matchesWH.length >= 3) {
        const width = parseFloat(matchesWH[1].replace(',', '.'));
        const height = parseFloat(matchesWH[2].replace(',', '.'));
        if (!isNaN(width) && !isNaN(height)) {
            return {
                width,
                height,
                area: width * height,
            };
        }
    }

    // Case 2: "A m²" or "A m"
    const areaMatch = dimensionStr.match(/(\d+[\.,]?\d*)\s*m/);
    if (areaMatch && areaMatch.length >= 2) {
        const area = parseFloat(areaMatch[1].replace(',', '.'));
        if (!isNaN(area)) {
            return {
                width: 0,
                height: 0,
                area: area,
            };
        }
    }

    // Fallback if no numbers found
    return { width: 0, height: 0, area: 0 };
};
