// react-plotly.js resolves its factory against plotly.js types; we ship the
// minified dist build, so alias the module and relax the global Plotly types.
declare module "plotly.js-dist-min";

declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  // The factory returns a Plot component; props are loosely typed here.
  const createPlotlyComponent: (plotly: unknown) => ComponentType<Record<string, unknown>>;
  export default createPlotlyComponent;
}

declare namespace Plotly {
  type Data = Record<string, unknown>;
  type Layout = Record<string, unknown>;
}
