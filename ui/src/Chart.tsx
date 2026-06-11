import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Figure } from "./types";

// Bind react-plotly.js to the lightweight dist build (avoids bundling the full
// plotly.js source via the default `react-plotly.js` entry).
const Plot = createPlotlyComponent(Plotly);

const DARK_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e6e9f2", size: 12 },
  margin: { l: 50, r: 20, t: 40, b: 40 },
  legend: { orientation: "h", y: -0.2 },
};

export function Chart({ figure, title }: { figure?: Figure; title?: string }) {
  if (!figure) return null;
  const layout: Record<string, unknown> = { ...DARK_LAYOUT, ...(figure.layout || {}) };
  if (title) layout.title = { text: title };
  return (
    <div className="panel">
      <Plot
        data={figure.data}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "320px" }}
        useResizeHandler
      />
    </div>
  );
}
