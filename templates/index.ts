// Fix: Correct import path with extension.
import { TemplateSpec } from "./types.ts";

export const TEMPLATES: TemplateSpec[] = [
  {
    id: "centralizado-premium",
    name: "Centralizado",
    palette: 'auto',
    shapes: [
      { id: 'banner', kind: 'bar', anchor: 'center', width: 800, height: 72, radius: 12, opacity: 0.8 },
      { id: 'phone-pill', kind: 'pill', anchor: 'center', width: 220, height: 40, radius: 20, offsetX: 270 }
    ],
    text: [
      { id: 'title', area: 'banner', align: 'center', verticalAlign: 'top', maxLines: 2, minSize: 24, maxSize: 48, weight: 700, upper: true, padding: 10 },
      { id: 'subtitle', area: 'banner', align: 'center', verticalAlign: 'bottom', maxLines: 1, minSize: 16, maxSize: 20, weight: 400, padding: 16 },
      { id: 'phone', area: 'phone-pill', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 16, maxSize: 18, weight: 600, padding: 0 }
    ]
  },
  {
    id: "tarja-superior-solida",
    name: "Tarja Superior",
    palette: 'dark',
    shapes: [
      { id: 'banner', kind: 'bar', anchor: 'top', width: "100%", height: 96, opacity: 0.95 }
    ],
    text: [
      { id: 'title', area: 'banner', align: 'left', verticalAlign: 'top', maxLines: 1, minSize: 32, maxSize: 56, weight: 800, upper: true, padding: 40 },
      { id: 'subtitle', area: 'banner', align: 'left', verticalAlign: 'bottom', maxLines: 1, minSize: 16, maxSize: 20, weight: 400, padding: 40 },
      { id: 'phone', area: 'banner', align: 'right', verticalAlign: 'middle', maxLines: 1, minSize: 18, maxSize: 22, weight: 600, padding: 40 }
    ]
  },
  {
    id: "tarja-lateral-esquerda",
    name: "Lateral",
    palette: 'auto',
    shapes: [
      { id: 'banner', kind: 'box', anchor: 'left', width: "30%", height: 650, radius: 16, opacity: 0.9 },
      { id: 'phone-pill', kind: 'pill', anchor: 'bottom', parent: 'banner', width: 240, height: 44, radius: 22, offsetY: -40 }
    ],
    text: [
      { id: 'title', area: 'banner', orientation: 'vertical', align: 'left', verticalAlign: 'top', maxLines: 2, minSize: 40, maxSize: 80, weight: 700, upper: true, padding: 40 },
      { id: 'phone', area: 'phone-pill', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 16, maxSize: 20, weight: 600, padding: 0 },
      { id: 'subtitle', area: 'banner', align: 'left', verticalAlign: 'bottom', maxLines: 2, minSize: 18, maxSize: 22, padding: 40 }
    ]
  },
  {
    id: "telefone-destaque",
    name: "Telefone Destaque",
    palette: 'light',
    shapes: [
      { id: 'phone-pill', kind: 'pill', anchor: 'bottom-right', width: 300, height: 60, radius: 30, opacity: 1, offsetX: -40, offsetY: -40 }
    ],
    text: [
      { id: 'title', area: 'free', align: 'left', verticalAlign: 'middle', maxLines: 2, minSize: 60, maxSize: 120, weight: 800, upper: true, padding: 80, shadow: true },
      { id: 'phone', area: 'phone-pill', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 20, maxSize: 28, weight: 700, padding: 0 },
      { id: 'subtitle', area: 'free', align: 'left', verticalAlign: 'top', maxLines: 2, minSize: 24, maxSize: 32, padding: 80, shadow: true }
    ]
  },
  {
    id: "slogan-inferior",
    name: "Slogan Inferior",
    palette: 'auto',
    shapes: [
      { id: 'banner', kind: 'bar', anchor: 'bottom', width: "100%", height: 80, opacity: 0.85 }
    ],
    text: [
      { id: 'subtitle', area: 'banner', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 24, maxSize: 40, weight: 700, padding: 0, upper: true },
      { id: 'title', area: 'free', align: 'center', verticalAlign: 'middle', maxLines: 2, minSize: 50, maxSize: 100, weight: 800, padding: 120, shadow: true },
      { id: 'phone', area: 'free', align: 'right', verticalAlign: 'top', maxLines: 1, minSize: 18, maxSize: 22, weight: 500, padding: 40, shadow: true }
    ]
  },
  {
    id: "caixa-de-info",
    name: "Caixa de Info",
    palette: 'dark',
    shapes: [
      { id: 'banner', kind: 'box', anchor: 'center', width: 600, height: 250, radius: 20, opacity: 0.75 }
    ],
    text: [
      { id: 'title', area: 'banner', align: 'center', verticalAlign: 'top', maxLines: 2, minSize: 40, maxSize: 64, weight: 700, upper: true, padding: 40 },
      { id: 'subtitle', area: 'banner', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 18, maxSize: 24, padding: 10 },
      { id: 'phone', area: 'banner', align: 'center', verticalAlign: 'bottom', maxLines: 1, minSize: 20, maxSize: 28, weight: 600, padding: 40 }
    ]
  },
  {
    id: "adesivo-vitrine",
    name: "Adesivo Vitrine",
    palette: 'light',
    shapes: [
      { id: 'title-pill', kind: 'pill', anchor: 'center', width: 500, height: 100, radius: 50, opacity: 0.9, offsetY: -30 },
      { id: 'phone-pill', kind: 'pill', anchor: 'center', width: 280, height: 50, radius: 25, opacity: 0.9, offsetY: 70 }
    ],
    text: [
      { id: 'title', area: 'title-pill', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 40, maxSize: 70, weight: 800, upper: true, padding: 0 },
      { id: 'subtitle', area: 'free', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 16, maxSize: 20, padding: 130, shadow: true, color: '#FFFFFF' },
      { id: 'phone', area: 'phone-pill', align: 'center', verticalAlign: 'middle', maxLines: 1, minSize: 18, maxSize: 22, weight: 600, padding: 0 }
    ]
  }
];