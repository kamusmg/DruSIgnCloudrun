/**
 * Attempts to remove a simple, solid-color background from an image.
 * It samples the top-left pixel and makes pixels of a similar color transparent.
 * @param base64Image The base64 data URL of the image to process.
 * @param tolerance A value from 0 to 255 to determine how close a color must be to be removed.
 * @returns A promise that resolves with the base64 data URL of the processed image.
 */
export const removeSimpleBackground = (base64Image: string, tolerance: number = 30): Promise<string> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                return reject(new Error("Could not get canvas context"));
            }

            ctx.drawImage(img, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            // Get the color of the top-left pixel as the target color
            const r0 = data[0];
            const g0 = data[1];
            const b0 = data[2];

            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];

                // Calculate the color difference
                const diff = Math.sqrt(Math.pow(r - r0, 2) + Math.pow(g - g0, 2) + Math.pow(b - b0, 2));

                if (diff < tolerance) {
                    // Make the pixel transparent
                    data[i + 3] = 0;
                }
            }

            ctx.putImageData(imageData, 0, 0);
            resolve(canvas.toDataURL('image/png'));
        };
        img.onerror = (err) => {
            reject(err);
        };
        img.src = base64Image;
    });
};
