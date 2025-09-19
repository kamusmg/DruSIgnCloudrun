import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { RedesignResult, DetailedRequestData, Calibration } from "../types.ts";
import { parseAndFormatDimensions } from './dimensionParser.ts';

// --- NEW DARK THEME PALETTE ---
const COLORS = {
    BACKGROUND: '#1a2233', // Deep Dark Blue
    TEXT_PRIMARY: '#F9FAFB', // White/Gray-50
    TEXT_SECONDARY: '#9CA3AF', // Gray-400
    ACCENT: '#38bdf8', // Medium Blue (sky-400)
    BORDER: '#374151', // Gray-700
};

const A4_LANDSCAPE = { width: 841.89, height: 595.28 };
const MARGIN = 42.5; // ~15mm in points

const addImageDataToDoc = async (
    dataUrl: string
): Promise<{ data: string; format: string; width: number; height: number; } | null> => {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                resolve(null);
                return;
            }
            ctx.drawImage(img, 0, 0);
            const format = dataUrl.includes('png') ? 'PNG' : 'JPEG';
            resolve({
                data: canvas.toDataURL(`image/${format.toLowerCase()}`),
                format,
                width: img.naturalWidth,
                height: img.naturalHeight,
            });
        };
        img.onerror = () => resolve(null);
        img.src = dataUrl;
    });
};

const drawPageTemplate = (doc: jsPDF, pageNumber: number, totalPages: number, pageTitle: string) => {
    // Background
    doc.setFillColor(COLORS.BACKGROUND);
    doc.rect(0, 0, A4_LANDSCAPE.width, A4_LANDSCAPE.height, 'F');

    // Simple Title Header
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);
    doc.setTextColor(COLORS.TEXT_PRIMARY);
    doc.text(pageTitle, MARGIN, MARGIN + 5);

    // Footer page number (will be updated at the end)
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(COLORS.TEXT_SECONDARY);
    doc.text(`Página ${pageNumber} de ${totalPages}`, A4_LANDSCAPE.width / 2, A4_LANDSCAPE.height - 20, { align: 'center' });
};

const addFullPageImage = (doc: jsPDF, imageData: any) => {
    const pageW = A4_LANDSCAPE.width;
    const pageH = A4_LANDSCAPE.height;
    const availableW = pageW - MARGIN * 2;
    const availableH = pageH - MARGIN * 2 - 40; // Space for title

    const imgRatio = imageData.width / imageData.height;
    
    let imgW = availableW;
    let imgH = imgW / imgRatio;

    if (imgH > availableH) {
        imgH = availableH;
        imgW = imgH * imgRatio;
    }
    
    const x = (pageW - imgW) / 2;
    const y = MARGIN + 40;

    doc.addImage(imageData.data, imageData.format, x, y, imgW, imgH);
}


export const generatePresentationPdf = async (
    result: RedesignResult,
    requestData: DetailedRequestData,
    originalImage: string,
    calibration: Calibration | null,
    coverImageBase64: string,
): Promise<Blob> => {
    const doc = new jsPDF('l', 'pt', 'a4');
    
    const coverImg = await addImageDataToDoc(coverImageBase64);
    const originalImg = await addImageDataToDoc(originalImage);
    const redesignedImg = await addImageDataToDoc(result.redesignedImage);
    
    // --- Set PDF Metadata ---
    doc.setProperties({
        title: `Relatório de Fachada - ${requestData.companyName || 'Projeto'}`,
        author: 'Dru Sign',
        subject: 'Proposta de Redesign de Fachada Gerada por IA',
    });

    // --- Page 1: Artistic Cover Page ---
    if (coverImg) {
        doc.addImage(coverImg.data, coverImg.format, 0, 0, A4_LANDSCAPE.width, A4_LANDSCAPE.height);
        
        // --- Add Text Overlay to Bottom-Right ---
        const x_pos = A4_LANDSCAPE.width - MARGIN;
        const y_pos_subtitle = A4_LANDSCAPE.height - MARGIN - 15;
        const y_pos_title = A4_LANDSCAPE.height - MARGIN - 38;
        const shadowOffset = 1.5;

        // Draw shadow first for legibility
        doc.setTextColor('#111111'); // Shadow color
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(28);
        doc.text("DRU SIGN", x_pos + shadowOffset, y_pos_title + shadowOffset, { align: 'right' });
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(14);
        doc.text("Proposta de Redesign de Fachada", x_pos + shadowOffset, y_pos_subtitle + shadowOffset, { align: 'right' });

        // Draw main text
        doc.setTextColor(COLORS.TEXT_PRIMARY);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(28);
        doc.text("DRU SIGN", x_pos, y_pos_title, { align: 'right' });
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(14);
        doc.text("Proposta de Redesign de Fachada", x_pos, y_pos_subtitle, { align: 'right' });
        
        // Date remains at the bottom center
        doc.setFontSize(10);
        doc.setTextColor(COLORS.TEXT_SECONDARY);
        const dateText = new Date().toLocaleDateString('pt-BR', { year: 'numeric', month: 'long', day: 'numeric' });
        doc.text(dateText, A4_LANDSCAPE.width / 2, A4_LANDSCAPE.height - 40, { align: 'center' });
    }

    // --- Page 2: "Antes" Image ---
    doc.addPage();
    let totalPages = result.technicalPlan.length > 5 ? 5 : 4; // Initial assumption
    drawPageTemplate(doc, 2, totalPages, 'Situação Atual (Antes)');
    if (originalImg) {
        addFullPageImage(doc, originalImg);
    }
    
    // --- Page 3: "Depois" Image ---
    doc.addPage();
    drawPageTemplate(doc, 3, totalPages, 'Proposta de Redesign (Depois)');
    if (redesignedImg) {
        addFullPageImage(doc, redesignedImg);
    }

    // --- Page 4+: Technical Plan ---
    doc.addPage();
    drawPageTemplate(doc, 4, totalPages, 'Plano Técnico e Medidas Estimadas');

    const tableData = result.technicalPlan.map((item, index) => {
        const dims = parseAndFormatDimensions(item.dimensions);
        return [
            (index + 1).toString(),
            item.item,
            dims.width,
            dims.height,
            dims.area,
            item.details
        ];
    });

    autoTable(doc, {
        startY: MARGIN + 40,
        margin: { left: MARGIN, right: MARGIN },
        head: [['#', 'Tipo de Área', 'Largura (m)', 'Altura (m)', 'Área (m²)', 'Observações']],
        body: tableData,
        theme: 'grid',
        headStyles: {
            fillColor: COLORS.ACCENT,
            textColor: '#FFFFFF',
            fontSize: 10,
            fontStyle: 'bold',
        },
        styles: {
            fontSize: 9,
            cellPadding: 6,
            textColor: COLORS.TEXT_PRIMARY,
            fillColor: COLORS.BORDER,
        },
        alternateRowStyles: {
            fillColor: COLORS.BACKGROUND,
        },
        columnStyles: {
            0: { cellWidth: 30, halign: 'center' },
            1: { cellWidth: 120 },
            2: { cellWidth: 60, halign: 'right' },
            3: { cellWidth: 60, halign: 'right' },
            4: { cellWidth: 70, halign: 'right' },
            5: { cellWidth: 'auto' },
        },
        willDrawPage: (data) => {
            // This hook is for pages created BY autotable (i.e., page 5+)
            drawPageTemplate(doc, data.pageNumber, doc.getNumberOfPages(), 'Plano Técnico e Medidas Estimadas');
        },
        didParseCell: (data) => {
            // Set the background correctly for every cell before drawing
            const isHeader = data.row.section === 'head';
            const isEven = data.row.index % 2 === 0;
            if (!isHeader) {
                data.cell.styles.fillColor = isEven ? COLORS.BORDER : COLORS.BACKGROUND;
            }
        },
        willDrawCell: (data) => {
            doc.setDrawColor(COLORS.BORDER); // Ensure grid lines have the right color
        }
    });
    
    // --- Calibration Summary & Disclaimer on the LAST page ---
    const finalY = (doc as any).autoTable.previous.finalY;
    let currentContentY = finalY;
    
    // Check if we need to add a new page for the summary
    if (A4_LANDSCAPE.height - currentContentY < 80) { 
        doc.addPage();
        drawPageTemplate(doc, doc.getNumberOfPages(), doc.getNumberOfPages(), 'Plano Técnico e Medidas Estimadas');
        currentContentY = MARGIN;
    }
    
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(10);
    doc.setTextColor(COLORS.TEXT_PRIMARY);
    doc.text('Referência de Medição', MARGIN, currentContentY + 25);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(COLORS.TEXT_SECONDARY);
    const calibrationText = calibration
        ? `As medidas foram calculadas com base na referência fornecida pelo usuário de ${calibration.realDistanceMeters.toFixed(2).replace('.', ',')} metros.`
        : 'Nenhuma referência de medida foi fornecida. As dimensões apresentadas são estimativas geradas pela Inteligência Artificial com base na análise da imagem.';
    
    const splitText = doc.splitTextToSize(calibrationText, A4_LANDSCAPE.width - MARGIN * 2);
    doc.text(splitText, MARGIN, currentContentY + 40);


    // Final loop to update footers with correct total pages
    const finalPageCount = doc.getNumberOfPages();
    for (let i = 1; i <= finalPageCount; i++) {
        doc.setPage(i);
        if (i > 1) { // No footer on cover
            // Redraw background to cover potential table overflow
            doc.setFillColor(COLORS.BACKGROUND);
            doc.rect(0, A4_LANDSCAPE.height - 35, A4_LANDSCAPE.width, 35, 'F');
            
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(9);
            doc.setTextColor(COLORS.TEXT_SECONDARY);
            doc.text(`Página ${i} de ${finalPageCount}`, A4_LANDSCAPE.width / 2, A4_LANDSCAPE.height - 20, { align: 'center' });
        }
        
        if (i === finalPageCount) {
             doc.setFontSize(8);
             doc.setTextColor('#a0aec0'); // Light gray
             const disclaimer = 'As medidas apresentadas são aproximações geradas automaticamente por inteligência artificial e podem conter imprecisões.';
             doc.text(
                 disclaimer,
                 A4_LANDSCAPE.width / 2, 
                 A4_LANDSCAPE.height - 35, 
                 { align: 'center', maxWidth: A4_LANDSCAPE.width - MARGIN * 4 }
             );
        }
    }


    return doc.output('blob');
};