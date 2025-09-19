// Defines the declarative structure for sign templates.

export type TextBox = {
  id: 'title' | 'subtitle' | 'phone';
  /** 'banner' places text inside a shape with a matching id, 'free' places it on the image */
  area: string; 
  align: 'left' | 'center' | 'right';
  verticalAlign?: 'top' | 'middle' | 'bottom';
  maxLines: number;
  /** Font size in pixels, relative to a 1200px wide canvas */
  minSize: number; 
  /** Font size in pixels, relative to a 1200px wide canvas */
  maxSize: number;
  weight?: number; // 400–800
  upper?: boolean;
  shadow?: boolean; 
  stroke?: boolean;
  /** Padding in pixels, relative to a 1200px wide canvas */
  padding: number; 
  color?: string;
  /** For vertical text */
  orientation?: 'vertical';
}

export type Shape = {
  id: string; // Used to link text boxes
  kind: 'bar' | 'pill' | 'box';
  anchor: 'top' | 'bottom' | 'left' | 'right' | 'center' | 'bottom-right' | 'top-right';
  /** ID of another shape to anchor relative to. */
  parent?: string;
  /** Height in pixels, relative to a 1200px wide canvas */
  height?: number; 
  /** Width in pixels or percentage string (e.g., "30%"), relative to canvas */
  width?: number | string;
  radius?: number; // px
  opacity?: number; // 0–1
  /** Padding in pixels, relative to a 1200px wide canvas */
  paddingX?: number;
  /** Padding in pixels, relative to a 1200px wide canvas */
  paddingY?: number;
  offsetX?: number; // px
  offsetY?: number; // px
}

export type TemplateSpec = {
  id: string;
  name: string;
  palette: 'auto' | 'light' | 'dark' | {bg: string; fg: string; accent?: string};
  shapes: Shape[];
  text: TextBox[];
}