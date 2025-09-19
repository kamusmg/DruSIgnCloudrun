// @ts-nocheck
import { GoogleGenAI, GenerateContentResponse, Part, Type, Modality } from "@google/genai";
import { DetailedRequestData, Logo, TechnicalPlanItem } from '../types.ts';

// Initialize the Google Gemini AI client
// The API key is sourced from environment variables, as per the guidelines.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

/**
 * The result structure from the main redesign generation function.
 */
interface RedesignGenerationResult {
    redesignedImage: string; // base64 data URL
    finalLogo: Logo | null;
    technicalPlan: TechnicalPlanItem[];
}

/**
 * Generates a new facade design, including a photorealistic image and a technical plan.
 * This is a multi-modal function that can take multiple images (facade, logo, patterns) and text.
 * @returns A promise that resolves to a RedesignGenerationResult object.
 */
export const generateRedesign = async (
    originalImage: string,
    prompt: string,
    requestData: DetailedRequestData
): Promise<RedesignGenerationResult> => {
    // Use the multimodal model capable of generating both image and text in one response.
    const model = 'gemini-2.5-flash-image-preview';

    // Helper to extract base64 data from a data URL string
    const getBase64 = (dataUrl: string) => dataUrl.split(',')[1];
    
    // Start with the primary facade image
    const parts: Part[] = [{
        inlineData: { mimeType: 'image/jpeg', data: getBase64(originalImage) },
    }];
    
    // Conditionally add other images (logo, art, patterns) to the prompt parts
    if (requestData.logoFile) {
        parts.push({
            inlineData: { mimeType: 'image/png', data: getBase64(requestData.logoFile.base64) },
        });
    }

    if (requestData.bannerFaixaDetails.artFile) {
        parts.push({
            inlineData: {
                mimeType: requestData.bannerFaixaDetails.artFile.base64.startsWith('data:image/png') ? 'image/png' : 'image/jpeg',
                data: getBase64(requestData.bannerFaixaDetails.artFile.base64),
            }
        });
    }

    const patternSticker = requestData.stickerDetails.find(s => s.type === 'pattern' && s.data.generatedPattern);
    if (patternSticker?.type === 'pattern' && patternSticker.data.generatedPattern) {
         parts.push({
            inlineData: { mimeType: 'image/jpeg', data: getBase64(patternSticker.data.generatedPattern.base64) }
        });
    }

    const textPrompt = `
        **TASK:** You are an expert architect specializing in commercial facade redesigns.
        Your goal is to generate two things based on the user's request and the provided image(s):
        1.  A photorealistic redesigned image of the facade.
        2.  A structured technical plan in JSON format.

        **USER REQUEST:**
        ${prompt}

        **INSTRUCTIONS:**
        1.  **Redesigned Image:** Generate a new, photorealistic image that implements all the changes described in the user request. The new image must maintain the original camera angle and perspective. The result should look like a real photograph.
        2.  **Technical Plan (JSON):** After generating the image, create a detailed technical plan for the redesign. The plan should be a JSON array of objects, where each object represents a distinct element of the redesign (e.g., main sign, ACM paneling, lighting).
            - Each object must have these keys: "item" (string, e.g., "Placa Principal"), "material" (string, e.g., "ACM com letra caixa de acrílico"), "dimensions" (string, e.g., "Aproximadamente 4.5m x 1.2m"), and "details" (string, e.g., "Fundo em ACM preto fosco com iluminação interna. Logo aplicado em relevo.").
            - Be specific with materials and provide realistic, estimated dimensions in meters based on visual cues from the original image (e.g., doors, windows).

        **OUTPUT FORMAT:**
        You MUST provide two separate parts in your response:
        - The first part must be the redesigned image.
        - The second part must be the raw JSON text for the technical plan, enclosed in a single JSON code block (\`\`\`json ... \`\`\`). Do NOT include any other text or explanations before or after the JSON block.
        
        **---MANDATORY SAFETY FILTER---**
        **ABSOLUTELY DO NOT generate any real-world logos, copyrighted brands (like Coca-Cola, Nike, etc.), or identifiable human faces. If the user's prompt contains a brand or company name, you MUST create a completely new, generic, and legally safe logo and style that is only inspired by the name. For example, for "Paulo's Car Shop", create a generic symbol for a car shop, not the text "Paulo's Car Shop". This is a strict and non-negotiable rule to prevent copyright infringement.**
    `;
    
    parts.push({ text: textPrompt });

    try {
        const response = await ai.models.generateContent({
            model: model,
            contents: { parts: parts },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            },
        });

        // Check for explicit safety block
        if (response.promptFeedback?.blockReason) {
            console.error("AI prompt blocked due to safety settings. Reason:", response.promptFeedback.blockReason);
            throw new Error("AI_PROMPT_BLOCKED");
        }
        
        const finishReason = response.candidates?.[0]?.finishReason;
        if (finishReason === 'SAFETY' || finishReason === 'RECITATION') {
             console.error("AI prompt blocked, finish reason:", finishReason);
             throw new Error("AI_PROMPT_BLOCKED");
        }
        
        const responseParts = response.candidates?.[0]?.content?.parts;
        if (!responseParts || responseParts.length < 2) {
            throw new Error("AI response did not contain both an image and a technical plan.");
        }

        const imagePartResponse = responseParts.find(p => p.inlineData);
        const textPartResponse = responseParts.find(p => p.text);

        if (!imagePartResponse?.inlineData || !textPartResponse?.text) {
             throw new Error("AI response is missing the image or the technical plan part.");
        }
        
        const redesignedImage = `data:image/jpeg;base64,${imagePartResponse.inlineData.data}`;
        
        let technicalPlan: TechnicalPlanItem[];
        try {
            const jsonMatch = textPartResponse.text.match(/```json\s*([\s\S]*?)\s*```/);
            const jsonString = jsonMatch ? jsonMatch[1] : textPartResponse.text;
            technicalPlan = JSON.parse(jsonString.trim());

            if (!Array.isArray(technicalPlan)) {
                throw new Error("Parsed technical plan is not an array.");
            }
        } catch (e) {
            console.error("Failed to parse technical plan JSON:", e, "Raw text:", textPartResponse.text);
            throw new Error("AI returned an invalid technical plan format.");
        }

        return {
            redesignedImage,
            finalLogo: requestData.logoFile, // Pass through the logo that was used
            technicalPlan,
        };

    } catch (error) {
        console.error("Error generating redesign:", error);
        if (error instanceof Error && (error.message.includes('AI_PROMPT_BLOCKED') || error.message.includes('prompt was blocked') || error.message.includes('SAFETY'))) {
             throw new Error("AI_PROMPT_BLOCKED");
        }
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }
};

/**
 * Enhances a user's prompt by making it more detailed and technical for the image generation AI.
 */
export const enhancePrompt = async (originalPrompt: string): Promise<string> => {
    try {
        const systemInstruction = `You are a creative assistant for a sign-making company. Your task is to enhance a user's request for a facade redesign into a detailed, professional prompt for an image generation AI. The enhanced prompt should be descriptive, clear, and use technical terms where appropriate (e.g., "ACM paneling," "channel letters," "LED backlighting"). It must be in English. It should incorporate visual details that would lead to a high-quality, realistic image. Focus on materials, lighting, style, and overall ambiance. Do not add any conversational text or explanations, just output the enhanced prompt.`;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `Enhance this user request: "${originalPrompt}"`,
            config: {
                systemInstruction,
                temperature: 0.7,
            }
        });
        const enhancedPrompt = response.text.trim();
        return enhancedPrompt || originalPrompt; // Fallback to original if empty
    } catch (error) {
        console.error("Error enhancing prompt:", error);
        return originalPrompt; // Fallback on error
    }
};


/**
 * Refines a user's description of a location on an image into a more precise one.
 */
export const refinePlacementPrompt = async (originalImage: string, placement: string): Promise<string> => {
    const imagePart = {
        inlineData: {
            mimeType: 'image/jpeg',
            data: originalImage.split(',')[1],
        },
    };
    const textPart = {
        text: `Based on the provided image, refine the following user-provided placement description into a very precise instruction for an image generation AI. The output should be a short, clear phrase describing the exact location. For example, if the user says "on the window", a good refinement could be "on the large glass window pane to the right of the main entrance door". User placement description: "${placement}"`,
    };

    try {
        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: { parts: [imagePart, textPart] },
            config: { temperature: 0.4 }
        });
        const refinedPlacement = response.text.trim();
        return refinedPlacement || placement;
    } catch (error) {
        console.error('Error refining placement prompt:', error);
        return placement;
    }
};


/**
 * Generates a new logo image from a text prompt.
 */
export const generateLogo = async (prompt: string): Promise<Logo> => {
    const fullPrompt = `Create a modern, clean, minimalist logo suitable for a business sign. The logo should be on a transparent background. It should be a vector-style graphic. Description: "${prompt}"`;
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: fullPrompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/png',
                aspectRatio: '1:1',
            },
        });
        const image = response.generatedImages?.[0];
        if (!image?.image.imageBytes) throw new Error("AI did not return a valid logo image.");
        
        return {
            base64: `data:image/png;base64,${image.image.imageBytes}`,
            prompt: prompt,
        };
    } catch (error) {
        console.error("Error generating logo:", error);
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }
};

/**
 * Reinvents a logo by analyzing an existing one in a facade photo.
 */
export const reinventLogo = async (originalImage: string, companyName: string): Promise<Logo> => {
    const imagePart = {
        inlineData: { mimeType: 'image/jpeg', data: originalImage.split(',')[1] },
    };
    const textPart = {
        text: `Analyze the provided image of a storefront. Identify the existing logo for the company named "${companyName}". Create a new, modernized version of that logo. The new logo must be a clean, minimalist, vector-style graphic suitable for a high-end sign. It must have a transparent background and be presented on its own, without the original background image.`
    };
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image-preview',
            contents: { parts: [imagePart, textPart] },
            config: {
                responseModalities: [Modality.IMAGE, Modality.TEXT],
            }
        });
        const imagePartResponse = response.candidates?.[0]?.content?.parts.find(p => p.inlineData);
        if (!imagePartResponse?.inlineData) throw new Error("AI did not return a valid image for the reinvented logo.");

        return {
            base64: `data:image/png;base64,${imagePartResponse.inlineData.data}`,
            prompt: `Reinvented logo for ${companyName}`,
        };
    } catch (error) {
        console.error("Error reinventing logo:", error);
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }
};

/**
 * Generates a seamless, repeating pattern image from a text prompt.
 */
export const generatePattern = async (prompt: string): Promise<Logo> => {
    const fullPrompt = `Create a high-resolution, seamless, repeating pattern. The pattern should be modern and stylish. It must be suitable for use as a decorative vinyl sticker on a wall or window. Theme: "${prompt}"`;
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: fullPrompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '1:1',
            },
        });
        const image = response.generatedImages?.[0];
        if (!image?.image.imageBytes) throw new Error("AI did not return a valid pattern image.");
        
        return {
            base64: `data:image/jpeg;base64,${image.image.imageBytes}`,
            prompt: `Pattern: ${prompt}`,
        };
    } catch (error) {
        console.error("Error generating pattern:", error);
        throw new Error("AI_IMAGE_GENERATION_FAILED");
    }
};

/**
 * Generates an artistic cover image for the PDF presentation.
 */
export const generatePdfCoverImage = async (
    logo: Logo | null,
    companyName: string,
    prompt: string,
    originalImage: string
): Promise<string> => {
    const fullPrompt = `Create an artistic, abstract, visually stunning background image for a PDF presentation cover. The image should be professional, corporate, and modern. It should evoke the mood of an architectural redesign project. The image should NOT contain any text or logos. Use a color palette inspired by the facade redesign concept: "${prompt}". The style should be like a beautiful, abstract architectural rendering or a close-up of premium materials like brushed metal, dark wood, or polished concrete. The image must be in a 16:9 landscape aspect ratio.`;
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: fullPrompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '16:9',
            },
        });
        const image = response.generatedImages?.[0];
        if (!image?.image.imageBytes) throw new Error("AI did not return a valid PDF cover image.");
        
        return `data:image/jpeg;base64,${image.image.imageBytes}`;
    } catch (error) {
        console.error("Error generating PDF cover image:", error);
        return originalImage; // Fallback to original image
    }
};