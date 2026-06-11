import type { ProgressEvent, RunRequest, RunResult } from "./types";

export async function fetchDevices(): Promise<{ devices: string[]; default: string }> {
  const r = await fetch("/api/devices");
  return r.json();
}

interface RunCallbacks {
  onProgress: (ev: ProgressEvent) => void;
  onResult: (res: RunResult) => void;
  onError: (message: string) => void;
}

/**
 * POST /api/run and parse the Server-Sent-Events stream, dispatching
 * progress / result / error frames to the callbacks. Returns when the stream
 * closes (the backend sends a final `done` event).
 */
export async function runStream(req: RunRequest, cb: RunCallbacks): Promise<void> {
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.body) {
    cb.onError("no response body");
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const parsed = JSON.parse(data);
      if (event === "progress") cb.onProgress(parsed as ProgressEvent);
      else if (event === "result") cb.onResult(parsed as RunResult);
      else if (event === "error") cb.onError(parsed.message ?? "run failed");
    }
  }
}
