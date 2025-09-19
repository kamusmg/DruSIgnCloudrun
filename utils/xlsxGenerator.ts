import ExcelJS from 'exceljs';
import { RedesignResult, DetailedRequestData, Calibration, TechnicalPlanItem } from "../types.ts";
import { parseDimensionsToNumbers } from './dimensionParser.ts';
import { getEstimatedMaterialPrice } from './pricing.ts';

// --- STYLING & FORMATTING CONSTANTS ---
const currencyFormat = 'R$ #,##0.00;-R$ #,##0.00;"—"';
const numberFormat = '#,##0.00;-#,##0.00;"—"';

const setHeaderStyle = (row: ExcelJS.Row) => {
    row.font = { bold: true, color: { argb: 'FFFFFFFF' }, name: 'Calibri', size: 11 };
    row.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF444444' } }; // Dark Grey
    row.alignment = { vertical: 'middle', horizontal: 'center' };
    row.height = 22;
};


// --- MAIN GENERATOR FUNCTION ---
export const generateInteractiveBudgetSheetXlsx = async (
    result: RedesignResult,
    requestData: DetailedRequestData,
    calibration: Calibration | null // This is kept in the signature for potential future use, but not used now.
): Promise<Blob> => {
    const workbook = new ExcelJS.Workbook();
    workbook.creator = 'Dru Sign';
    workbook.created = new Date();
    
    // --- Create the single RESUMO Sheet ---
    const resumoSheet = workbook.addWorksheet('Resumo');
    resumoSheet.pageSetup = { paperSize: 9, orientation: 'landscape', fitToPage: true, fitToWidth: 1 };
    resumoSheet.views = [{ state: 'frozen', ySplit: 1 }];

    resumoSheet.columns = [
        { header: '#', key: 'num', width: 5 },
        { header: 'Tipo de área', key: 'item', width: 35 },
        { header: 'Largura (m)', key: 'width', width: 15 },
        { header: 'Altura (m)', key: 'height', width: 15 },
        { header: 'Área (m²)', key: 'area', width: 15 },
        { header: 'Preço por m² (R$)', key: 'pricePerSqm', width: 20 },
        { header: 'Total da área (R$)', key: 'total', width: 22 },
        { header: 'Observações', key: 'details', width: 45 },
    ];
    setHeaderStyle(resumoSheet.getRow(1));
    resumoSheet.autoFilter = 'A1:H1';

    // Populate Data Rows with formulas and styles
    const dataRowCount = result.technicalPlan.length;
    result.technicalPlan.forEach((item, index) => {
        const rowNum = index + 2;
        const dims = parseDimensionsToNumbers(item.dimensions);
        const price = getEstimatedMaterialPrice(item);

        const row = resumoSheet.addRow({
            num: index + 1,
            item: item.item,
            width: dims.width > 0 ? dims.width : undefined,
            height: dims.height > 0 ? dims.height : undefined,
            pricePerSqm: price.pricePerSqm > 0 ? price.pricePerSqm : undefined,
            details: item.details,
        });

        // Set Formulas
        row.getCell('E').value = { formula: `IFERROR(C${rowNum}*D${rowNum}, 0)` };
        row.getCell('G').value = { formula: `IFERROR(E${rowNum}*F${rowNum}, 0)` };

        // Set Cell Formats
        row.getCell('C').numFmt = numberFormat;
        row.getCell('D').numFmt = numberFormat;
        row.getCell('E').numFmt = numberFormat;
        row.getCell('F').numFmt = currencyFormat;
        row.getCell('G').numFmt = currencyFormat;
    });

    // Add Totals Row
    if (dataRowCount > 0) {
        const totalRowNum = dataRowCount + 2;
        const totalRow = resumoSheet.addRow([]); // Add an empty row for spacing
        const sumRow = resumoSheet.getRow(totalRowNum + 1);
        
        sumRow.getCell('D').value = 'TOTAIS';
        sumRow.getCell('E').value = { formula: `SUM(E2:E${totalRowNum - 1})` };
        sumRow.getCell('G').value = { formula: `SUM(G2:G${totalRowNum - 1})` };
        
        sumRow.font = { bold: true, size: 12 };
        sumRow.getCell('E').numFmt = numberFormat;
        sumRow.getCell('G').numFmt = currencyFormat;
    }

    // Protect sheet but unlock editable cells
    await resumoSheet.protect(Math.random().toString(36).substring(2, 15), {
        selectUnlockedCells: true,
        formatCells: false,
    });
    
    for (let i = 2; i <= dataRowCount + 1; i++) {
        resumoSheet.getCell(`C${i}`).protection = { locked: false }; // Largura
        resumoSheet.getCell(`D${i}`).protection = { locked: false }; // Altura
        resumoSheet.getCell(`E${i}`).protection = { locked: false }; // Área (can be overridden)
        resumoSheet.getCell(`F${i}`).protection = { locked: false }; // Preço/m²
    }


    const buffer = await workbook.xlsx.writeBuffer();
    return new Blob([buffer], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
};