// Fix: Correct the import path for types.
import { TechnicalPlanItem } from "../types.ts";
import { parseDimensionsToNumbers } from './dimensionParser.ts';


// NOTE: These prices are for demonstration purposes only.
const MATERIAL_PRICES_PER_SQ_METER: Record<string, number> = {
  'acm': 350,       // R$350/m²
  'letra caixa': 800, // R$800/m² (more complex)
  'adesivo': 90,    // R$90/m²
  'lona': 70,       // R$70/m²
  'pintura': 50,    // R$50/m²
};

const BASE_ITEM_COST: Record<string, number> = {
    'iluminação': 400, // Base cost for lighting installation
    'totem': 1500,     // Base cost for a totem structure
    'wind banner': 250,// Cost per unit
};

function parseDimensions(dimensionStr: string): { width: number, height: number, area: number } | null {
    if (!dimensionStr) return null;
    // Regex to find numbers, allowing for "x", ",", ".", and "m"
    const matches = dimensionStr.toLowerCase().match(/(\d+[\.,]?\d*)\s*m?\s*[xX]\s*(\d+[\.,]?\d*)/);
    if (matches && matches.length >= 3) {
        const width = parseFloat(matches[1].replace(',', '.'));
        const height = parseFloat(matches[2].replace(',', '.'));
        if (!isNaN(width) && !isNaN(height)) {
            return { width, height, area: width * height };
        }
    }
    // Fallback for area only (e.g., "5m²")
    const areaMatch = dimensionStr.match(/(\d+[\.,]?\d*)\s*m/);
    if (areaMatch && areaMatch.length >= 2) {
        const area = parseFloat(areaMatch[1].replace(',', '.'));
        if(!isNaN(area)) return { width: 0, height: 0, area };
    }
    return null;
}

export function calculatePriceEstimate(technicalPlan: TechnicalPlanItem[]): { min: number, max: number } {
    let totalCost = 0;

    for (const item of technicalPlan) {
        const dims = parseDimensions(item.dimensions);
        let itemCost = 0;

        const lowerMaterial = item.material.toLowerCase();
        const lowerItem = item.item.toLowerCase();
        
        // Check for base item costs
        for(const key in BASE_ITEM_COST) {
            if(lowerItem.includes(key)) {
                itemCost += BASE_ITEM_COST[key];
            }
        }

        // Check for material costs based on area
        if (dims && dims.area > 0) {
            let pricePerSqMeter = 0;
            for(const key in MATERIAL_PRICES_PER_SQ_METER) {
                if (lowerMaterial.includes(key) || lowerItem.includes(key)) {
                    pricePerSqMeter = MATERIAL_PRICES_PER_SQ_METER[key];
                    break;
                }
            }
            if(pricePerSqMeter > 0) {
                itemCost += dims.area * pricePerSqMeter;
            } else if (itemCost === 0) {
                // Add a default cost for unknown items with area
                itemCost += dims.area * 150; 
            }
        }
        
        if (itemCost === 0) {
            itemCost = 200; // Default cost for any item not priced
        }

        totalCost += itemCost;
    }

    if (totalCost === 0) return { min: 0, max: 0 };
    
    // Provide a range (e.g., +/- 15%)
    const min = totalCost * 0.85;
    const max = totalCost * 1.15;
    
    // Round to nearest 50
    return {
        min: Math.round(min / 50) * 50,
        max: Math.round(max / 50) * 50,
    };
}


/**
 * Provides an intelligent estimate for material costs, separating per-square-meter
 * pricing from fixed base costs. This is used to pre-populate the budget sheet.
 * @param item A single item from the AI-generated technical plan.
 * @returns An object containing the estimated price per square meter and any base cost.
 */
export function getEstimatedMaterialPrice(item: TechnicalPlanItem): { pricePerSqm: number, baseCost: number } {
    const lowerMaterial = item.material.toLowerCase();
    const lowerItem = item.item.toLowerCase();

    // Base costs for non-area items (return these first as they are specific)
    if (lowerItem.includes('iluminação') || lowerItem.includes('spots')) return { pricePerSqm: 0, baseCost: 450 };
    if (lowerItem.includes('wind banner')) return { pricePerSqm: 0, baseCost: 250 };
    if (lowerItem.includes('placa informativa')) return { pricePerSqm: 0, baseCost: 120 };


    // Prices per square meter (most common case)
    if (lowerItem.includes('letra caixa') || lowerMaterial.includes('letra caixa')) return { pricePerSqm: 850, baseCost: 0 };
    if (lowerItem.includes('luminoso') || lowerMaterial.includes('luminoso')) return { pricePerSqm: 1200, baseCost: 0 };
    if (lowerItem.includes('totem')) return { pricePerSqm: 950, baseCost: 1500 }; // Totems have both
    if (lowerMaterial.includes('acm') || lowerItem.includes('acm')) return { pricePerSqm: 380, baseCost: 0 };
    if (lowerMaterial.includes('adesivo')) return { pricePerSqm: 95, baseCost: 0 };
    if (lowerMaterial.includes('lona')) return { pricePerSqm: 75, baseCost: 0 };
    if (lowerItem.includes('pintura')) return { pricePerSqm: 50, baseCost: 0 };
    
    const dims = parseDimensionsToNumbers(item.dimensions);
    // Default for items that have dimensions but aren't recognized
    if (dims && dims.area > 0) return { pricePerSqm: 150, baseCost: 0 };

    // Default base cost for unknown items without clear area pricing
    return { pricePerSqm: 0, baseCost: 200 };
}