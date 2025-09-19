import { GoogleGenAI, Type, Modality, GenerateContentResponse, HarmCategory, HarmBlockThreshold } from "@google/genai";
import { DetailedRequestData, Logo, TechnicalPlanItem, RedesignResult } from '../types.ts';

// Guideline: Always use new GoogleGenAI({apiKey: process.env.API_KEY});
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Helper to convert image base64 to Part
const fileToGenerativePart = (base64Data: string, mimeType: string) => {
    // remove data:image/jpeg;base64, prefix
    const data = base64Data.split(',')[1];
    return {
        inlineData: {
            data,
            mimeType
        },
    };
};

/**
 * Enhances a user-provided prompt with technical details for better AI image generation.
 */
export const enhancePrompt = async (originalPrompt: string): Promise<string> => {
    const model = 'gemini-2.5-flash';
    const fullPrompt = `
        Você é um especialista em comunicação visual e um engenheiro de prompts para IA generativa de imagens.
        Sua tarefa é reescrever a solicitação do usuário a seguir em um prompt detalhado e técnico em inglês, otimizado para uma IA de imagem como Midjourney ou Stable Diffusion.
        O prompt deve ser descritivo, evocativo e focado em aspectos visuais. Mantenha a intenção original, mas adicione detalhes sobre estilo de iluminação, composição, ângulo da câmera, tipo de lente e atmosfera para garantir um resultado fotorrealista e de alta qualidade.
        Retorne APENAS o prompt reescrito em inglês, sem nenhuma outra explicação ou texto adicional.

        Solicitação do Usuário: "${originalPrompt}"
    `;
    
    try {
        const response = await ai.models.generateContent({
            model,
            contents: fullPrompt,
        });
        return response.text.trim();
    } catch (error) {
        console.error("Error enhancing prompt:", error);
        throw new Error("Failed to enhance prompt.");
    }
};

/**
 * Refines a user's description of where to place a sticker on a facade.
 */
export const refinePlacementPrompt = async (
  facadeImage: string, // base64
  placementDescription: string
): Promise<string> => {
    const model = 'gemini-2.5-flash';
    const imagePart = fileToGenerativePart(facadeImage, 'image/jpeg');

    const prompt = `
        Analise a imagem da fachada. O usuário quer colocar um adesivo no seguinte local: "${placementDescription}".
        Com base na imagem, reescreva essa descrição de local para ser mais precisa e inequívoca para uma IA de imagem.
        Por exemplo, em vez de "na porta", seja mais específico como "na metade inferior da folha de vidro direita da porta de entrada principal".
        Retorne APENAS a descrição refinada, sem nenhum outro texto.
    `;

    try {
        const response = await ai.models.generateContent({
            model,
            contents: { parts: [imagePart, { text: prompt }] },
        });
        return response.text.trim();
    } catch (error) {
        console.error("Error refining placement prompt:", error);
        return placementDescription; // Fallback to original
    }
};

const technicalPlanSchema = {
    type: Type.ARRAY,
    items: {
        type: Type.OBJECT,
        properties: {
            item: {
                type: Type.STRING,
                description: 'O nome do componente ou serviço. Ex: "Placa Principal", "Revestimento de ACM", "Iluminação de Destaque".'
            },
            material: {
                type: Type.STRING,
                description: 'O material principal usado. Ex: "ACM 3mm com adesivo vinil", "Lona 440g com impressão digital", "Letra caixa em PVC expandido".'
            },
            dimensions: {
                type: Type.STRING,
                description: 'As dimensões estimadas em metros. Ex: "4.5m x 1.2m", "Aproximadamente 12m²".'
            },
            details: {
                type: Type.STRING,
                description: 'Detalhes adicionais sobre instalação, acabamento ou especificações. Ex: "Instalada em estrutura metálica com acabamento embutido.", "Pintura automotiva na cor #FFFFFF."'
            },
        },
        required: ["item", "material", "dimensions", "details"]
    }
};

/**
 * Generates the main facade redesign, including the image and technical plan.
 */
export const generateRedesign = async (
    originalImage: string, // base64
    prompt: string,
    requestData: DetailedRequestData
): Promise<Omit<RedesignResult, 'originalPrompt' | 'enhancedPrompt'>> => {
    
    const model = 'gemini-2.5-flash-image-preview'; // For image editing
    const textModel = 'gemini-2.5-flash';

    const imagePart = fileToGenerativePart(originalImage, 'image/jpeg');
    const allParts = [imagePart, { text: prompt }];

    // FIX: Check for an uploaded logo and add it to the request parts.
    if (requestData.logoFile && requestData.logoFile.base64) {
        try {
            const mimeType = requestData.logoFile.base64.split(';')[0].split(':')[1];
            if (mimeType) {
                const logoPart = fileToGenerativePart(requestData.logoFile.base64, mimeType);
                allParts.push(logoPart);
            }
        } catch (e) {
            console.warn("Could not parse and add uploaded logo file, skipping.", e);
        }
    }
    
    // Add uploaded banner/strip art file to the request parts.
    if (requestData.bannerFaixaDetails.artType === 'upload' && requestData.bannerFaixaDetails.artFile?.base64) {
        try {
            const mimeType = requestData.bannerFaixaDetails.artFile.base64.split(';')[0].split(':')[1];
            if (mimeType) {
                const artFilePart = fileToGenerativePart(requestData.bannerFaixaDetails.artFile.base64, mimeType);
                allParts.push(artFilePart);
            }
        } catch (e) {
            console.warn("Could not parse and add uploaded banner/strip art file, skipping.", e);
        }
    }
    
    // --- Step 1: Generate the new facade image with retry logic ---
    const MAX_RETRIES = 3;
    let successfulResponse: GenerateContentResponse | null = null;
    let lastError: any = null;

    for (let i = 0; i < MAX_RETRIES; i++) {
        try {
            console.log(`Image generation attempt ${i + 1}...`);
            // FIX: The 'safetySettings' property must be nested inside the 'config' object.
            const imageResponse = await ai.models.generateContent({
                model,
                contents: { parts: allParts },
                config: {
                    responseModalities: [Modality.IMAGE, Modality.TEXT],
                    // FIX: Use HarmCategory and HarmBlockThreshold enums for safety settings.
                    safetySettings: [
                        {
                            category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        },
                        {
                            category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        },
                    ],
                },
            });
            
            const hasImage = imageResponse.candidates?.[0]?.content?.parts?.some(p => p.inlineData);
            if (hasImage) {
                successfulResponse = imageResponse;
                break; // Success! Exit the loop.
            } else {
                lastError = new Error("AI did not return an image.");
                console.warn(`Attempt ${i + 1} succeeded but returned no image. Retrying... Full response:`, JSON.stringify(imageResponse));
            }

        } catch (error) {
            lastError = error;
            console.error(`Image generation attempt ${i + 1} failed:`, error);
        }
        
        if (i < MAX_RETRIES - 1) {
            const delay = Math.pow(2, i + 1) * 1000; // Exponential backoff: 2s, 4s
            console.log(`Waiting ${delay}ms before retrying...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }

    if (!successfulResponse) {
        console.error("All image generation attempts failed.", lastError);
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }

    const imageContentPart = successfulResponse.candidates?.[0]?.content?.parts?.find(p => p.inlineData);
    if (!imageContentPart || !imageContentPart.inlineData) {
        // This should be redundant due to the check in the loop, but it's a good safeguard.
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }
    const redesignedImage = `data:${imageContentPart.inlineData.mimeType};base64,${imageContentPart.inlineData.data}`;

    // --- Step 2: Generate the technical plan based on the prompt ---
    const technicalPlanPrompt = `
        Com base na seguinte solicitação de design para uma fachada, crie um plano técnico detalhado em formato JSON.
        Liste os principais componentes, materiais, dimensões estimadas e detalhes de instalação.
        A resposta DEVE ser um array de objetos JSON, seguindo o schema fornecido.
        Não inclua \`\`\`json ou qualquer outra formatação.

        Solicitação: "${prompt}"
    `;

    let technicalPlan: TechnicalPlanItem[];
    try {
        const planResponse = await ai.models.generateContent({
            model: textModel,
            contents: technicalPlanPrompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: technicalPlanSchema,
            },
        });
        const jsonText = planResponse.text.trim();
        technicalPlan = JSON.parse(jsonText);
    } catch (error) {
        console.error("Technical plan generation failed:", error);
        technicalPlan = [{ item: 'Erro', material: 'Não foi possível gerar o plano técnico.', dimensions: '-', details: 'Tente novamente.' }];
    }
    
    // --- Step 3: Handle the logo ---
    // The logo is now sent with the image generation request, but we still pass it to the result for display purposes.
    const finalLogo = requestData.logoFile;

    return {
        redesignedImage,
        technicalPlan,
        finalLogo,
    };
};

/**
 * Generates a logo image.
 */
export const generateLogo = async (prompt: string): Promise<Logo> => {
    const model = 'imagen-4.0-generate-001';
    const fullPrompt = `
      Create a clean, modern, vector-style logo for a company.
      The logo should be on a transparent background.
      The logo should be simple and easily readable.
      Do NOT include any text other than the company name if specified.
      Prompt: "${prompt}"
    `;
    
    try {
        const response = await ai.models.generateImages({
            model,
            prompt: fullPrompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/png', // PNG for transparency
                aspectRatio: '1:1',
            }
        });
        
        const base64ImageBytes = response.generatedImages[0].image.imageBytes;
        const base64 = `data:image/png;base64,${base64ImageBytes}`;
        return { base64, prompt };
    } catch (error) {
        console.error("Error generating logo:", error);
        throw new Error("Failed to generate logo.");
    }
};

/**
 * Reinvents a logo from an existing facade image.
 */
export const reinventLogo = async (facadeImage: string, companyName: string): Promise<Logo> => {
     // Note: imagen-4.0-generate-001 doesn't directly take an image input in the SDK.
    // We will generate a new logo based on the company name with a "reinvention" prompt.
    return generateLogo(`A modern, reinvented logo for "${companyName}"`);
};

/**
 * Generates a seamless pattern image.
 */
export const generatePattern = async (theme: string): Promise<Logo> => {
     const model = 'imagen-4.0-generate-001';
     const prompt = `
        Create a high-resolution, seamless, repeating pattern.
        The style should be modern and suitable for a decorative wall covering.
        Theme: "${theme}"
     `;
     
     try {
        const response = await ai.models.generateImages({
            model,
            prompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '1:1',
            }
        });
        
        const base64ImageBytes = response.generatedImages[0].image.imageBytes;
        const base64 = `data:image/jpeg;base64,${base64ImageBytes}`;
        return { base64, prompt: theme };
    } catch (error) {
        console.error("Error generating pattern:", error);
        throw new Error("Failed to generate pattern.");
    }
};


/**
 * Generates or re-uses a technical plan for deliverables.
 */
export const generateTechnicalPlanForDeliverables = async (prompt: string): Promise<{ technicalPlan: TechnicalPlanItem[] }> => {
    const textModel = 'gemini-2.5-flash';
    const technicalPlanPrompt = `
        Com base na seguinte solicitação de design para uma fachada, crie um plano técnico detalhado em formato JSON.
        Liste os principais componentes, materiais, dimensões estimadas e detalhes de instalação.
        A resposta DEVE ser um array de objetos JSON, seguindo o schema fornecido.
        Não inclua \`\`\`json ou qualquer outra formatação.

        Solicitação: "${prompt}"
    `;

    try {
        const planResponse = await ai.models.generateContent({
            model: textModel,
            contents: technicalPlanPrompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: technicalPlanSchema,
            },
        });
        const jsonText = planResponse.text.trim();
        const technicalPlan = JSON.parse(jsonText);
        return { technicalPlan };
    } catch (error) {
        console.error("Technical plan for deliverables failed:", error);
        return { technicalPlan: [{ item: 'Erro', material: 'Não foi possível gerar o plano técnico.', dimensions: '-', details: 'Tente novamente.' }] };
    }
};

/**
 * Generates markdown content for a specific deliverable type.
 */
export const generateDeliverable = async (
    type: 'Owner Package',
    technicalPlan: TechnicalPlanItem[],
    requestData: DetailedRequestData
): Promise<string> => {
    const model = 'gemini-2.5-flash';

    let prompt: string;
    
    const planString = technicalPlan.map(item => `- ${item.item}: ${item.material}, ${item.dimensions}. Detalhes: ${item.details}`).join('\n');

    switch (type) {
        case 'Owner Package':
            prompt = `
                Você é um especialista em comunicação visual da empresa Dru Sign.
                Crie um "Manual do Proprietário" para o cliente "${requestData.companyName}" com base no seguinte plano técnico.
                O manual deve ser escrito em Markdown e deve conter seções claras sobre:
                1.  **Visão Geral do Projeto**: Um parágrafo resumindo o projeto.
                2.  **Materiais Utilizados**: Uma lista dos materiais de cada item do plano e seus benefícios (durabilidade, aparência, etc.).
                3.  **Instruções de Cuidado e Limpeza**: Dicas práticas sobre como limpar e manter cada material (ACM, lona, adesivos, etc.) para maximizar a vida útil.
                4.  **Garantia e Suporte**: Um texto padrão informando sobre a garantia dos materiais e como contatar a Dru Sign para suporte.
                
                Seja claro, profissional e útil.
                
                Plano Técnico:
                ${planString}
            `;
            break;
        default:
            return Promise.reject("Invalid deliverable type");
    }

    try {
        const response = await ai.models.generateContent({
            model,
            contents: prompt,
        });
        return response.text.trim();
    } catch (error) {
        console.error(`Error generating ${type} content:`, error);
        throw new Error(`Failed to generate ${type}.`);
    }
};


const generateAbstractCover = async (facadeImage: string): Promise<string> => {
    const textModel = 'gemini-2.5-flash';
    const imageModel = 'imagen-4.0-generate-001';
    
    const imagePart = fileToGenerativePart(facadeImage, 'image/jpeg');
    const analysisPrompt = `
      Analyze the provided facade image. Identify the main architectural or design style (e.g., modern, rustic, industrial, classic, minimalist).
      Also, identify the two most dominant colors in the facade as hex codes.
      Return the answer ONLY as a simple JSON object with the keys "style" and "colors" (an array of hex strings).
      Example: {"style": "modern minimalist", "colors": ["#334155", "#f59e0b"]}
    `;
    
    let style = "modern";
    let colors = ["#1e3a8a", "#93c5fd"]; // Fallback colors

    try {
        const analysisResponse = await ai.models.generateContent({
            model: textModel,
            contents: { parts: [imagePart, { text: analysisPrompt }] },
            config: { responseMimeType: "application/json" }
        });
        const analysisResult = JSON.parse(analysisResponse.text);
        if (analysisResult.style) style = analysisResult.style;
        if (analysisResult.colors && analysisResult.colors.length > 0) colors = analysisResult.colors;
    } catch (error) {
        console.warn("Could not analyze facade for cover generation, using defaults.", error);
    }

    const prompt = `
      Create a sophisticated and futuristic cover page image for a design proposal, in an elegant, abstract, high-end advertising style.
      This image is for the cover only and should NOT contain any text.
      The aesthetic should be inspired by "${style}" design trends.
      Use a color palette based on these colors: ${colors.join(', ')}. The background must be a solid, deep, elegant version of one of these colors, possibly with a subtle gradient.
      The final image must be abstract and conceptual, reinforcing a professional brand. Do NOT include any photos of buildings or facades.
    `;
    
    const response = await ai.models.generateImages({
        model: imageModel,
        prompt,
        config: {
            numberOfImages: 1,
            outputMimeType: 'image/jpeg',
            aspectRatio: '16:9', // Widescreen for landscape A4
        }
    });
    
    const base64ImageBytes = response.generatedImages[0].image.imageBytes;
    return `data:image/jpeg;base64,${base64ImageBytes}`;
};


/**
 * Generates an artistic, thematic cover image for the PDF report.
 * If a logo is provided, it creates a "legendary" advertising-style cover.
 * If no logo is provided, it falls back to an abstract, branded cover.
 */
export const generatePdfCoverImage = async (
    logo: Logo | null,
    companyName: string,
    originalPrompt: string,
    facadeImage: string
): Promise<string> => {
    
    // --- LEGENDARY COVER (with Logo) ---
    if (logo) {
        try {
            const textModel = 'gemini-2.5-flash';
            const imageModel = 'gemini-2.5-flash-image-preview';

            // 1. Analyze business theme
            const analysisPrompt = `
                Analise o nome da empresa e a descrição do projeto.
                - Nome da Empresa: "${companyName}"
                - Descrição: "${originalPrompt}"
                Identifique a categoria principal do negócio (ex: "Bar", "Sorveteria", "Restaurante de frango frito").
                Em seguida, determine o produto ou ação principal que define esse negócio.
                Retorne um objeto JSON com "category" (a categoria) e "primary_prop" (uma descrição curta e visual do produto/ação principal, ex: "uma caneca de chopp gelado", "uma casquinha de sorvete colorida", "uma coxa de frango frito crocante").
            `;
            const analysisResponse = await ai.models.generateContent({ model: textModel, contents: analysisPrompt, config: { responseMimeType: "application/json" } });
            const { category, primary_prop } = JSON.parse(analysisResponse.text);
            
            // 2. Build the generation prompt
            const generationPrompt = `
                Você é um diretor de arte de classe mundial. Sua tarefa é criar uma imagem de publicidade fotorrealista e de alto impacto para a capa de uma proposta de design.
                A empresa é um(a) "${category}".
                A imagem deve ser uma foto de estúdio limpa, com uma mulher estilosa e feliz como protagonista.
                Ela deve estar vestindo um uniforme moderno que incorpora o LOGOTIPO FORNECIDO de forma proeminente e natural (no peito ou em um boné, por exemplo).
                Ela deve estar interagindo de forma criativa e feliz com o principal produto do negócio: ${primary_prop}. A imagem deve capturar a essência da(o) "${category}".
                O fundo deve ser minimalista, UMA COR SÓLIDA E VIBRANTE que complemente as cores do logotipo. Sem gradientes ou texturas.
                A iluminação deve ser de estúdio, brilhante e uniforme, com sombras mínimas, no estilo de fotografia de produto moderna para um visual "flat" e gráfico.
                Integrado artisticamente ao fundo, inclua o texto "DRU SIGN" de forma grande, ousada e em UMA COR SÓLIDA DE ALTO CONTRASTE com o fundo, como um elemento de design gráfico impactante.
                A imagem final deve ser uma obra-prima da fotografia publicitária, em formato widescreen 16:9. A imagem gerada NÃO deve conter nenhum outro texto.
            `;

            // 3. Generate the image
            const logoPart = fileToGenerativePart(logo.base64, logo.base64.split(';')[0].split(':')[1]);
            const response = await ai.models.generateContent({
                model: imageModel,
                contents: { parts: [logoPart, { text: generationPrompt }] },
                config: {
                    responseModalities: [Modality.IMAGE, Modality.TEXT],
                },
            });

            const imagePart = response.candidates?.[0]?.content?.parts?.find(p => p.inlineData);
            if (imagePart?.inlineData) {
                 return `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
            }
            throw new Error("Legendary cover generation failed to return an image.");

        } catch (error) {
            console.error("Error generating legendary PDF cover, falling back to abstract.", error);
            // Fallback to abstract if legendary fails
            return generateAbstractCover(facadeImage);
        }
    }

    // --- ABSTRACT COVER (Fallback) ---
    try {
        return await generateAbstractCover(facadeImage);
    } catch (error) {
        console.error("Error generating abstract PDF cover:", error);
        throw new Error("Failed to generate PDF cover image.");
    }
};