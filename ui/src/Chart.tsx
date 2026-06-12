import type { ReactNode } from "react";
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

export function Chart({
  figure,
  image,
  title,
  bare = false,
  height = 320,
}: {
  figure?: Figure;
  image?: string; // base64 PNG (e.g. the exact frame sent to the VLM)
  title?: string;
  bare?: boolean; // skip the .panel wrapper (for embedding inside node cards)
  height?: number;
}) {
  // A raw base64 image takes priority — this is literally what the agent saw.
  let body: ReactNode = null;
  if (image) {
    body = (
      <figure className="agent-img">
        <img src={`data:image/png;base64,${image}`} alt={title || "agent view"} />
        {title && <figcaption>{title}</figcaption>}
      </figure>
    );
  } else if (figure) {
    const layout: Record<string, unknown> = { ...DARK_LAYOUT, ...(figure.layout || {}) };
    if (title) layout.title = { text: title };
    body = (
      <Plot
        data={figure.data}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: `${height}px` }}
        useResizeHandler
      />
    );
  }
  if (!body) return null;
  return bare ? body : <div className="panel">{body}</div>;
}
