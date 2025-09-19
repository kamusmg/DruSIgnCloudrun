import { DetailedRequestData, StickerDetail, LightingOption, TotemFeature, TotemSize, LonaInstallationType } from './types.ts';

/**
 * Generates a descriptive string for a list of stickers.
 */
export const generateStickerDescriptionText = (stickers: StickerDetail[]): string => {
  if (stickers.length === 0) return '';

  const stickerDescriptions = stickers.map(sticker => {
    switch (sticker.type) {
      case 'pattern':
        if (sticker.data.generatedPattern) {
          return `- Padrão Decorativo: Use a imagem de padrão fornecida como uma textura de papel de parede em pequena escala e repetitiva no seguinte local: ${sticker.data.placement}.`;
        }
        return `- Padrão Decorativo: Crie um padrão contínuo em pequena escala com o tema "${sticker.data.theme}" e aplique no seguinte local: ${sticker.data.placement}.`;
      case 'cut':
        return `- Adesivo de Recorte: Adicione um adesivo de vinil de recorte (plano, 2D, sem sombra) com o texto/forma "${sticker.data.description}" aplicado exclusivamente na seguinte superfície: ${sticker.data.placement}. O adesivo não deve ultrapassar os limites dessa superfície. O acabamento deve ser ${sticker.data.finish} na cor ${sticker.data.color}.`;
      case 'print':
        const printDetails = [
          sticker.data.hasContourCut ? 'recorte de contorno' : '',
          sticker.data.isPerforated ? 'perfurado' : ''
        ].filter(Boolean).join(', ');
        return `- Adesivo de Impressão: Use a arte fornecida como um adesivo de impressão digital no seguinte local: ${sticker.data.placement}${printDetails ? ` (com as seguintes propriedades: ${printDetails})` : ''}.`;
      default:
        return '';
    }
  }).filter(Boolean);

  if (stickerDescriptions.length === 0) return '';
  
  return `\n\n**Instruções de Adesivos:**\n${stickerDescriptions.join('\n')}`;
};


/**
 * Generates a simple, descriptive prompt from the request data, based on the user's preferred older style.
 */
export const generateInspirationText = (data: DetailedRequestData, isEditing: boolean = false): string => {
  let parts: string[] = [];

  if (isEditing) {
      parts.push("Use the provided image as the primary visual foundation. Preserve its existing architectural style, materials, and camera perspective. Only apply the following specific changes and additions. Do not radically alter the base image unless explicitly asked to do so.");
  }

  if (data.companyName) {
      parts.push(`Projeto para a empresa "${data.companyName}".`);
  }

  const signTypeMap: Record<Exclude<DetailedRequestData['mainSignType'], 'lona'>, string> = {
      'acm_adesivo': 'placa de ACM com adesivo aplicado',
      'acm_letra_caixa': 'placa de ACM com logo em letra caixa',
      'placa_iluminada': 'letreiro luminoso',
  };
  
  if (data.elements.includes('Placa Principal')) {
      if (data.mainSignType === 'lona') {
          const installationMap: Record<LonaInstallationType, string> = {
              'grommets': 'instalada em uma robusta estrutura metálica com ilhós visíveis, perfeitamente esticada e tensionada com cordas ou abraçadeiras de nylon',
              'hidden_finish': 'instalada em uma robusta estrutura metálica com acabamento embutido, onde a lona é esticada para cobrir as laterais, resultando em uma aparência limpa e sem bordas visíveis, perfeitamente tensionada'
          };
          parts.push(`O elemento principal é uma placa de lona impressa de alta resolução, ${installationMap[data.lonaInstallationType]}. O design na lona pode ser complexo, incluindo fotos, textos detalhados e múltiplos logos.`);
      } else {
          parts.push(`O elemento principal é uma ${signTypeMap[data.mainSignType]}.`);
      }
  }

  const acmColorToUse = data.acmColorName || data.acmColor;
  if (data.elements.includes('Revestimento em ACM') && acmColorToUse) {
      const { selections = [], custom = '' } = data.acmPlacement || {};
      const allPlacements = [...selections];
      if (custom.trim()) {
          allPlacements.push(custom.trim());
      }

      let instruction: string;
      if (allPlacements.length > 0) {
          instruction = `Usar revestimento de ACM na cor ${acmColorToUse} nas seguintes áreas: ${allPlacements.join(', ')}.`;
      } else {
          // Fallback for when color is selected but placement is not.
          instruction = `Usar revestimento de ACM na cor ${acmColorToUse}.`;
      }

      // Jurisdiction rule to prevent ACM from covering a canvas sign structure
      if (data.elements.includes('Placa Principal') && data.mainSignType === 'lona' && selections.includes('Fachada Completa (Tudo)')) {
          instruction += " IMPORTANTE: O revestimento de ACM deve ser aplicado em toda a fachada, EXCETO na área da estrutura metálica superior que sustenta a placa de lona.";
      }
      
      parts.push(instruction);
  } else if (data.elements.includes('Placa Principal') && acmColorToUse && (data.mainSignType.includes('acm') || data.mainSignType === 'placa_iluminada')) {
      parts.push(`A cor de fundo da placa deve ser ${acmColorToUse}.`);
  }

  if (data.elements.includes('Pintura Nova') && data.paintColor) {
      parts.push(`Aplicar nova pintura nas paredes na cor ${data.paintColor}.`);
  }
  
  if (data.elements.includes('Placa Principal') && data.mainSignType === 'acm_letra_caixa' && data.channelLetterColorMode === 'monochromatic' && data.channelLetterColor) {
      parts.push(`A letra caixa do logo deve ser renderizada na cor única ${data.channelLetterColor}.`);
  }
  
  if (data.elements.includes('Placa Principal')) {
      let logoDescription = 'Instruções do logo: ';
      switch (data.logoOption) {
          case 'use_from_photo':
              logoDescription += `Use o logo existente na foto.`;
              break;
          case 'reinvent_from_photo':
              logoDescription += `Modernize e reinvente o logo existente na foto.`;
              break;
          case 'generate':
              logoDescription += `Gere um novo logo com a descrição: "${data.logoPrompt}".`;
              break;
          case 'upload':
              logoDescription += `Use o arquivo de logo que foi fornecido.`;
              break;
      }
      parts.push(logoDescription);
  }
  
  if (data.additionalText && data.additionalText.text) {
    let textInstruction = `Incluir o texto adicional: "${data.additionalText.text}"`;
    if (data.additionalText.location) {
        textInstruction += ` no seguinte local: ${data.additionalText.location}`;
    }
    parts.push(textInstruction + ".");
  }
  
  if (data.elements.includes('Iluminação Especial') && data.lightingOptions && data.lightingOptions.length > 0) {
    const lightingTypeMap: Record<LightingOption, string> = {
        'spots': 'Spots de Destaque',
        'uplighting': 'Iluminação de Piso (Uplighting)',
        'contour_led': 'Fita de LED de Contorno'
    };
    const selectedLightingTypes = data.lightingOptions.map(opt => lightingTypeMap[opt]).join(', ');
    parts.push(`Adicionar iluminação especial com os seguintes tipos: ${selectedLightingTypes}.`);
  }

  if (data.elements.includes('Totem') && data.totemDetails) {
    const sizeMap: Record<TotemSize, string> = {
        small: 'Pequeno (informativo, ~1.0m de altura)',
        medium: 'Médio (padrão, ~2.2m de altura)',
        large: 'Grande (alta visibilidade, ~3.5m+ de altura)'
    };
    const featureMap: Record<TotemFeature, string> = {
        luminous: 'Luminoso',
        channel_letter: 'Letra Caixa'
    };
    let totemDescription = `Incluir um totem de tamanho ${sizeMap[data.totemDetails.size]}`;
    if (data.totemDetails.features.length > 0) {
        totemDescription += ` com as seguintes características: ${data.totemDetails.features.map(f => featureMap[f]).join(', ')}.`;
    } else {
        totemDescription += '.';
    }
    parts.push(totemDescription);
  }

  if (data.elements.includes('Banner ou Faixa')) {
    const details = data.bannerFaixaDetails;
    let sizeDesc = '';
    if (details.sizeType === 'custom' && details.customWidth && details.customHeight) {
        sizeDesc = `no tamanho personalizado de ${details.customWidth}cm x ${details.customHeight}cm`;
    } else {
        sizeDesc = `no tamanho padrão de ${details.presetSize}`;
    }

    let artDesc = '';
    if (details.artType === 'upload' && details.artFile) {
        artDesc = `usando a arte que foi fornecida`;
    } else if (details.artType === 'ai_prompt' && details.artPrompt.trim()) {
        artDesc = `com uma arte gerada pela IA com a seguinte descrição: "${details.artPrompt.trim()}"`;
    } else { 
        const prompt = `uma arte promocional genérica e visualmente atraente para a empresa ${data.companyName || ''}, alinhada com a identidade visual do projeto`;
        artDesc = `com uma arte gerada pela IA com a seguinte descrição: "${prompt}"`;
    }

    const type = details.type === 'banner' ? 'um banner (vertical)' : 'uma faixa (horizontal)';
    
    let placementHint = '';
    if (details.type === 'banner') {
        placementHint = 'Posicione o banner na calçada, próximo à entrada, em um nível visível para os pedestres (ao nível dos olhos).';
    } else { // 'faixa'
        placementHint = 'Posicione a faixa horizontalmente na fachada, tipicamente abaixo da placa principal ou acima da porta/janelas.';
    }

    parts.push(`Incluir ${type}, ${sizeDesc}, ${artDesc}. ${placementHint}`);
  }

  if (data.elements.includes('Placas Informativas') && data.informationalSignsText?.trim()) {
    parts.push(`Incluir pequenas placas informativas, posicionadas de forma inteligente perto da entrada, com o seguinte texto: "${data.informationalSignsText.trim()}".`);
  }

  if (data.elements.includes('Adicionar Pessoas')) {
      parts.push('Adicionar 1 funcionário homem e 1 mulher vestindo uniformes com as cores da empresa. Adicione também alguns clientes interagindo com o estabelecimento de forma natural.');
  }
  
  if (data.elements.includes('Veículos')) {
    const { selectedTypes, customName } = data.vehicleDetails;
    const allVehicleTypes = [...selectedTypes];
    if (customName) {
      allVehicleTypes.push(customName);
    }

    let vehicleDescription = 'um ou mais veículos da empresa';
    if (allVehicleTypes.length > 0) {
      vehicleDescription = `um ou mais veículos da empresa do(s) tipo(s): ${allVehicleTypes.join(', ')}`;
    }

    const instruction = `Incluir ${vehicleDescription}, estacionado(s) na rua em frente à fachada, de forma realista (não sobre a calçada). O(s) veículo(s) deve(m) ser plotado(s) com o logo e as cores da empresa.`;
    parts.push(instruction);
  }

  const otherElements = data.elements.filter(el => ![
    'Placa Principal', 'Adesivos', 'Revestimento em ACM', 'Pintura Nova', 
    'Texto Adicional', 'Veículos', 'Iluminação Especial', 'Totem', 
    'Banner ou Faixa', 'Placas Informativas', 'Adicionar Pessoas'
  ].includes(el));

  if (otherElements.length > 0) {
      parts.push(`Incluir também: ${otherElements.join(', ')}.`);
  }
  
  if (data.ambiance === 'night') {
      parts.push("A cena deve ser noturna, com a iluminação da fachada em destaque.");
  } else {
      parts.push("A cena deve ser diurna, com luz natural.");
  }

  return parts.join(' ');
};